"""백본 3종 로더 — 세션 2에서 검증된 경로를 그대로 감싼다.

세션 2 실측:
  - OpenAI 가중치는 **QuickGELU**로 학습됐다. 표준 GELU로 로드하면 CIFAR-10
    zero-shot 87.69% (정상 90.13%). `ViT-B-16-quickgelu`를 쓴다.
  - MaskCLIP은 CLIP과 **가중치를 공유**하고 pooled가 비트 단위로 같다.
    patch만 value-only readout으로 바뀐다 (norm 7.357 → 8.309).
  - 백본은 **raw feature**를 낸다. L2 정규화는 `Scorer.score()`에서만 (D10).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch
from torch import Tensor


@dataclass
class Backbone:
    """스코어링에 필요한 최소 인터페이스."""

    name: str
    embed_dim: int
    preprocess: Any
    encode_image_patches: Callable[[Tensor], tuple[Tensor, Tensor]]
    """`[B,3,H,W]` → (`patches [B,N,d]`, `pooled [B,d]`). 둘 다 raw."""
    encode_text_pooled: Callable[[Sequence[str]], Tensor]
    """문장 리스트 → `[n,d]` pooled(EOS). raw."""
    encode_text_tokens: Callable[[Sequence[str]], tuple[Tensor, Tensor]]
    """문장 리스트 → (`[n,T,d]` 토큰 임베딩, `[n,T]` 유효 마스크). raw.

    Phase 2의 슬롯 헤드가 cross-attention 할 대상. pooled 하나가 아니라
    시퀀스 전체가 필요하다 (D12 개정)."""


def _clip_patch_forward(v, px, *, maskclip: bool):
    x = v.conv1(px)
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    cls = v.class_embedding.to(x.dtype) + torch.zeros(
        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
    x = torch.cat([cls, x], dim=1) + v.positional_embedding.to(x.dtype)
    x = v.ln_pre(x)
    blocks = v.transformer.resblocks

    if not maskclip:
        x = v.transformer(x)
        x = v.ln_post(x)
        return x[:, 1:, :] @ v.proj, x[:, 0, :] @ v.proj

    for blk in blocks[:-1]:
        x = blk(x)
    std = blocks[-1](x)                       # pooled는 정상 경로 (CLIP과 동일)
    last, y = blocks[-1], blocks[-1].ln_1(x)
    attn = last.attn
    d = attn.embed_dim
    val = torch.nn.functional.linear(y, attn.in_proj_weight[2 * d:], attn.in_proj_bias[2 * d:])
    val = attn.out_proj(val)
    xm = x + val
    xm = xm + last.mlp(last.ln_2(xm))
    return v.ln_post(xm)[:, 1:, :] @ v.proj, v.ln_post(std)[:, 0, :] @ v.proj


def load_clip(device: str, *, maskclip: bool = False) -> Backbone:
    import open_clip
    tag = "ViT-B-16-quickgelu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        tag, pretrained="openai", device=device)
    model.eval()
    tok = open_clip.get_tokenizer(tag)

    def enc_img(px: Tensor):
        with torch.no_grad():
            return _clip_patch_forward(model.visual, px, maskclip=maskclip)

    def enc_txt(sents: Sequence[str]):
        with torch.no_grad():
            return model.encode_text(tok(list(sents)).to(device))

    def enc_tok(sents: Sequence[str]):
        with torch.no_grad():
            ids = tok(list(sents)).to(device)
            x = model.token_embedding(ids) + model.positional_embedding
            x = model.transformer(x, attn_mask=model.attn_mask)
            x = model.ln_final(x) @ model.text_projection
            return x, ids != 0

    return Backbone("maskclip" if maskclip else "clip_vitb16", 512,
                    preprocess, enc_img, enc_txt, enc_tok)


def load_siglip2(device: str) -> Backbone:
    from transformers import AutoModel, AutoProcessor
    mid = "google/siglip2-base-patch16-224"
    model = AutoModel.from_pretrained(mid).to(device).eval()
    proc = AutoProcessor.from_pretrained(mid)

    def preprocess(img):
        return proc(images=img, return_tensors="pt")["pixel_values"][0]

    def enc_img(px: Tensor):
        with torch.no_grad():
            out = model.vision_model(pixel_values=px)
            patches = out.last_hidden_state           # SigLIP은 CLS 없음
            pooled = getattr(out, "pooler_output", None)
            if pooled is None:
                pooled = patches.mean(dim=1)
            return patches, pooled

    def enc_txt(sents: Sequence[str]):
        with torch.no_grad():
            ti = proc(text=list(sents), return_tensors="pt",
                      padding="max_length", truncation=True).to(device)
            t = model.get_text_features(**ti)
            if isinstance(t, Tensor):
                return t
            pooled = getattr(t, "pooler_output", None)
            return pooled if pooled is not None else model.text_model(**ti).pooler_output

    def enc_tok(sents: Sequence[str]):
        with torch.no_grad():
            ti = proc(text=list(sents), return_tensors="pt",
                      padding="max_length", truncation=True).to(device)
            h = model.text_model(**ti).last_hidden_state
            return h, ti["input_ids"] != proc.tokenizer.pad_token_id

    return Backbone("siglip2", 768, preprocess, enc_img, enc_txt, enc_tok)


def load_backbone(name: str, device: str) -> Backbone:
    if name == "clip_vitb16":
        return load_clip(device, maskclip=False)
    if name == "maskclip":
        return load_clip(device, maskclip=True)
    if name == "siglip2":
        return load_siglip2(device)
    raise KeyError(f"알 수 없는 백본 {name!r}")
