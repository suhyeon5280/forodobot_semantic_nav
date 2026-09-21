"""D116 ① — heatmap 생산기. 3변종. **OmniVLA_edge 무수정.**

    frozen        frozen CLIP ViT-B-16-quickgelu
    adapter       사다리 최상점 어댑터 (act4_cl6159_s2)
    adapter_head  어댑터 + 국소화 헤드 (H1_linear lr 0.001, D112 채택)

입력 `current_img` 는 **ImageNet 정규화된 [B,3,224,224]** 다 (train.py:365 transform).
CLIP 은 다른 정규화를 쓰므로 **역정규화 → CLIP 정규화**한다 (결정적).
출력은 **[B,1,224,224]**, 프레임별 min-max 로 [0,1] 에 넣는다 (경계 있는 값).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from src.backbones.loaders import _clip_patch_forward                 # noqa: E402
from src.scoring.base import l2_normalize                             # noqa: E402

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class HeatmapProducer(nn.Module):
    """텍스트 토큰 × patch 특징 → heatmap. **전부 동결.**"""

    VARIANTS = ("frozen", "adapter", "adapter_head")

    def __init__(self, variant, backbone, pretrained, adapter_path=None,
                 head_path=None, head_name="H1_linear", mlp_rank=128, device="cuda"):
        super().__init__()
        assert variant in self.VARIANTS, variant
        import open_clip
        self.variant = variant
        m, _, pre = open_clip.create_model_and_transforms(backbone, pretrained=pretrained,
                                                          device=device)
        self.preprocess = pre                       # crop 판독용 (판정 세트와 같은 전처리)
        self.tokenizer = open_clip.get_tokenizer(backbone)
        if variant in ("adapter", "adapter_head"):
            ad = torch.load(adapter_path, map_location="cpu", weights_only=False)
            miss = m.load_state_dict(ad["trainable"], strict=False)
            assert not miss.unexpected_keys, f"어댑터 키 불일치 {miss.unexpected_keys[:3]}"
        m.eval()
        for p in m.parameters():
            p.requires_grad = False
        self.clip = m
        self.head = None
        if variant == "adapter_head":
            from loc_head_pilot_head import make_head_standalone
            h = make_head_standalone(head_name, 512, mlp_rank)
            h.load_state_dict(torch.load(head_path, map_location="cpu",
                                         weights_only=False)["state"])
            h.eval()
            for p in h.parameters():
                p.requires_grad = False
            self.head = h.to(device)
        self.register_buffer("im_m", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("im_s", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.register_buffer("cl_m", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1))
        self.register_buffer("cl_s", torch.tensor(CLIP_STD).view(1, 3, 1, 1))

    @torch.no_grad()
    def crop_cos(self, pil_img, boxes, phrase, margin=0.10, template="a photo of a {}."):
        """**판정 세트(67.46%) 와 같은 경로** — 후보 crop → 어댑터 pooled 임베딩 → 텍스트 cos.

        `text_gate.crops_of` (박스 10% 확장) · `act3_pilot.embed_rows`
        (patch l2 정규화 → 평균 → l2 정규화) 와 **같은 연산 순서**다.
        **국소화 헤드를 지나지 않는다** — 헤드는 카테고리 국소화용이고
        판정 세트 경로에 없다 (D112 · D129 §1-a).
        """
        W, Hh = pil_img.size
        dev = self.im_m.device
        px = []
        for b in boxes:
            x0, y0, x1, y1 = b
            mx, my = (x1 - x0) * margin, (y1 - y0) * margin
            c = pil_img.crop((int(max(0., x0 - mx) * W), int(max(0., y0 - my) * Hh),
                              int(min(1., x1 + mx) * W), int(min(1., y1 + my) * Hh)))
            if min(c.size) < 8: c = pil_img
            px.append(self.preprocess(c))
        if not px: return [], ""
        pv, _ = _clip_patch_forward(self.clip.visual, torch.stack(px).to(dev), maskclip=False)
        z = l2_normalize(l2_normalize(pv).mean(1))                       # [n, 512]
        txt = template.format(phrase)
        t = l2_normalize(self.clip.encode_text(self.tokenizer([txt]).to(dev)).float())[0]
        return [float(v) for v in (z @ t)], txt

    @torch.no_grad()
    def forward(self, current_img, tokens):
        """current_img [B,3,224,224] (ImageNet 정규화) · tokens [B,77] → [B,1,224,224]."""
        x = (current_img * self.im_s + self.im_m).clamp(0, 1)      # 역정규화
        x = (x - self.cl_m) / self.cl_s                            # CLIP 정규화
        pv, _ = _clip_patch_forward(self.clip.visual, x, maskclip=False)
        pv = l2_normalize(pv)
        if self.head is not None:
            pv = l2_normalize(self.head(pv))
        t = l2_normalize(self.clip.encode_text(tokens).float())     # [B,512]
        sim = torch.einsum("bnd,bd->bn", pv, t)                     # [B,N]
        B, N = sim.shape
        g = int(round(N ** 0.5))
        assert g * g == N, f"패치 격자가 정사각이 아니다 N={N}"
        hm = sim.view(B, 1, g, g)
        lo = hm.amin(dim=(2, 3), keepdim=True)
        hi = hm.amax(dim=(2, 3), keepdim=True)
        hm = (hm - lo) / (hi - lo).clamp_min(1e-6)                  # 프레임별 min-max
        return F.interpolate(hm, size=current_img.shape[-2:], mode="bilinear",
                             align_corners=False)
