"""1단계 — 목표(hard 판별 70~80%)가 물리적으로 가능한지 측정.

    python -m experiments.sweep_ceiling --set attr
    python -m experiments.sweep_ceiling --set attr --models dinov2_b14_r336 --probes attn

세 가지를 잰다.

  a. **백본 스윕** — DINOv2 B/14 · L/14 × register 유무 × 해상도 224/336/448.
     각각 두 crop 코사인 + 선형 프로브. CLIP 계열은 기준선으로 함께.
  b. **비선형 프로브** — 2층 MLP 와 attention 프로브. 이것이 실질 상한이다.
  c. **concat** — DINOv2 + SigLIP2 patch 이어붙이기.

**관문: 어느 구성이든 비선형 프로브 ≥ 75% 여야 2단계로 간다.**

---

세 가지 방법론 장치가 들어간다. 없으면 관문이 관문 노릇을 못 한다.

1. **group CV** — 대칭 쌍의 두 방향은 같은 이미지·같은 두 박스다. fold 가 갈리면
   train 에서 본 박스를 test 에서 다시 본다. `pair_id` 로 묶는다.

2. **대조군 상시 동반** — `geom`(박스 좌표만) 과 `text`(구문만). 시각 프로브가
   75% 를 넘어도 `geom` 이 73% 면 아무것도 측정하지 못한 것이다. 기존 위치 세트가
   좌표×관계어 교호작용만으로 **97.1%** 였다.

3. **박스 안 4×4 격자** — mean pooling 은 객체 내부 배치를 지운다. attention
   프로브가 잴 것이 있으려면 그 배치가 남아 있어야 한다. 격자 셀을 attention 의
   토큰으로 쓴다. 셀 평균이 곧 pooled 이므로 선형 프로브와 같은 캐시를 쓴다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backbones.loaders import load_backbone            # noqa: E402
from src.scoring.base import l2_normalize                  # noqa: E402
from src.teacher.grounding import box_to_grid              # noqa: E402
from src.utils.seed import set_seed                        # noqa: E402

CELL = 4          # 박스 안 4×4 격자
FOLDS = 5

# 스윕 대상. `res=None` 은 백본 기본 해상도.
MODELS = {}
for _sz, _tag in (("base", "b14"), ("large", "l14")):
    for _reg in (False, True):
        for _res in (224, 336, 448):
            MODELS[f"dinov2_{_tag}{'_reg' if _reg else ''}_r{_res}"] = {
                "kind": "dinov2",
                "timm": f"vit_{_sz}_patch14{'_reg4' if _reg else ''}_dinov2.lvd142m",
                "res": _res}
for _n in ("clip_vitb16", "maskclip", "siglip2"):
    MODELS[_n] = {"kind": "clip", "name": _n, "res": None}
MODELS["concat_dinov2_siglip2"] = {"kind": "concat",
                                   "parts": ["dinov2_b14_r336", "siglip2"]}

# **crop 판독** — 박스를 실제로 잘라 백본에 넣는다. 전체 이미지 patch 를 박스로
# pooling 하는 것과 갈리면 병목이 백본이 아니라 판독 방식이다. 실시간 프로파일은
# 후보마다 forward 가 필요해 깨지지만(N 후보 = N 회), 여기서는 상한 측정 장치다.
for _b in ("dinov2_b14_r224", "dinov2_l14_r224", "clip_vitb16", "siglip2"):
    MODELS[f"{_b}_crop"] = {**MODELS[_b], "crop": True}

# --------------------------------------------------------- Phase 4 트랙 1 추가분
# **region-text 감독으로 사전학습된 특징**이 이 세트에서 다른가 (phase4-assets §4a).
# 기존 스윕의 백본 넷은 아키텍처·규모가 달라도 region-text 감독이 없다는 점을
# 공유한다. 여기 추가하는 것은 그 축이 다른 모델들이다.
#
# 전처리는 각 모델 공식 규격의 **정규화 통계**를 쓴다. 다만 **정사각 리사이즈는
# 프로브 하네스의 제약**이다 — `side = sqrt(N)` 로 격자를 복원하므로 정사각이
# 아니면 성립하지 않는다. GroundingDINO 공식 전처리는 shortest_edge=800 의
# 비정사각이므로 이 부분은 의도적 이탈이고, 결과에 그대로 기록한다.
for _stage, _res in ((2, 640), (3, 640)):
    MODELS[f"gdino_swin_s{_stage}_r{_res}"] = {
        "kind": "gdino", "repo": "IDEA-Research/grounding-dino-tiny",
        "stage": _stage, "res": _res}
MODELS["sam_vitb"] = {"kind": "sam", "repo": "facebook/sam-vit-base", "res": 1024}
for _b in ("gdino_swin_s3_r640", "sam_vitb"):
    MODELS[f"{_b}_crop"] = {**MODELS[_b], "crop": True}


# --------------------------------------------------------------------- 특징 추출
def build_backbone(spec, device):
    if spec["kind"] == "dinov2":
        import timm
        m = timm.create_model(spec["timm"], pretrained=True, num_classes=0,
                              img_size=spec["res"]).to(device).eval()
        cfg = timm.data.resolve_model_data_config(m)
        cfg["input_size"] = (3, spec["res"], spec["res"])
        tf = timm.data.create_transform(**cfg, is_training=False)
        npre = m.num_prefix_tokens

        def enc(px):
            with torch.no_grad():
                f = m.forward_features(px)
            return f[:, npre:, :]
        return tf, enc, m
    if spec["kind"] == "gdino":
        from torchvision import transforms as T
        from transformers import (AutoImageProcessor,
                                  AutoModelForZeroShotObjectDetection)
        ip = AutoImageProcessor.from_pretrained(spec["repo"])
        full = AutoModelForZeroShotObjectDetection.from_pretrained(spec["repo"])
        swin = full.model.backbone.conv_encoder.model.to(device).eval()
        del full
        # 공식 정규화 통계. 리사이즈만 정사각으로 바꾼다(위 주석 참조)
        tf = T.Compose([T.Resize((spec["res"], spec["res"])), T.ToTensor(),
                        T.Normalize(ip.image_mean, ip.image_std)])
        idx = spec["stage"] - 1

        def enc(px):
            with torch.no_grad():
                fm = swin(px).feature_maps[idx]        # [B, C, h, w]
            return fm.flatten(2).transpose(1, 2)       # [B, N, C]
        return tf, enc, swin

    if spec["kind"] == "sam":
        from torchvision import transforms as T
        from transformers import AutoProcessor, SamModel
        proc = AutoProcessor.from_pretrained(spec["repo"])
        sam = SamModel.from_pretrained(spec["repo"]).vision_encoder.to(device).eval()
        ipr = proc.image_processor
        # SAM 공식은 longest_edge=1024 로 맞추고 패딩한다. 정사각 리사이즈는
        # 종횡비를 바꾸지만 격자 복원을 위해 필요하다 — gdino 와 같은 이탈
        tf = T.Compose([T.Resize((spec["res"], spec["res"])), T.ToTensor(),
                        T.Normalize(ipr.image_mean, ipr.image_std)])

        def enc(px):
            with torch.no_grad():
                h = sam(px).last_hidden_state          # [B, C, h, w]
            return h.flatten(2).transpose(1, 2)
        return tf, enc, sam

    bb = load_backbone(spec["name"], device)

    def enc(px):
        with torch.no_grad():
            p, _ = bb.encode_image_patches(px)
        return p
    return bb.preprocess, enc, bb


def cell_weights(box, side, device):
    """박스를 4×4 로 쪼개 각 셀의 patch 가중치. `[16, N]`, 각 행 합 1."""
    x0, y0, x1, y1 = box
    ws = []
    for i in range(CELL):
        for j in range(CELL):
            sub = (x0 + (x1 - x0) * j / CELL, y0 + (y1 - y0) * i / CELL,
                   x0 + (x1 - x0) * (j + 1) / CELL, y0 + (y1 - y0) * (i + 1) / CELL)
            ws.append(box_to_grid(sub, side, device))
    return torch.stack(ws)


def extract(name, spec, rows, img_root, device, cache_dir,
            enc_override=None, cache=True):
    """`[n, 2, 16, d]` 격자 특징을 캐시한다.

    캐시 키에 **행 수를 넣는다.** 데이터셋이 커졌는데 이름만으로 캐시를 맞히면
    조용히 낡은 특징을 쓰고, 결과는 그럴듯한 숫자로 나온다 (D11).

    `enc_override=(tf, enc, holder)` 를 주면 백본을 새로 만들지 않고 그것을 쓴다.
    **미세조정된 가중치로 같은 판독을 돌리기 위한 통로**이며, 이때 `cache=False`
    로 두어 가중치가 다른 특징이 기준선 캐시를 덮어쓰지 못하게 한다.
    """
    cp = cache_dir / f"{name}__n{len(rows)}.pt"
    if cache and cp.exists():
        return torch.load(cp, map_location="cpu")
    if spec["kind"] == "concat":
        parts = [extract(p, MODELS[p], rows, img_root, device, cache_dir,
                         cache=cache)
                 for p in spec["parts"]]
        g = torch.cat([l2_normalize(p["grid"]) for p in parts], dim=-1)
        out = {"grid": g, "dim": g.shape[-1],
               "n_patch": sum(p["n_patch"] for p in parts),
               "nparam": sum(p["nparam"] for p in parts),
               "ms_per_img": sum(p["ms_per_img"] for p in parts)}
        if cache:
            torch.save(out, cp)
        return out

    tf, enc, holder = (build_backbone(spec, device) if enc_override is None
                       else enc_override)
    grids = []
    # 비전 타워 파라미터 수. CLIP 래퍼는 dataclass + 클로저라 셀 수 없다 —
    # 기준선이므로 0 으로 두고 DINOv2 쪽 비용 비교에만 쓴다
    nparam = (sum(q.numel() for q in holder.parameters())
              if isinstance(holder, torch.nn.Module) else 0)
    t0 = time.time()
    crop = spec.get("crop", False)
    for i, r in enumerate(rows):
        im = Image.open(Path(img_root) / r["image"]).convert("RGB")
        pair = []
        if crop:
            W, H = im.size
            for b in (r["correct_box"], r["wrong_box"]):
                # 약간의 맥락을 남긴다 — 꽉 맞는 crop 은 경계 정보를 잃는다
                x0, y0, x1, y1 = b
                mx, my = (x1 - x0) * 0.1, (y1 - y0) * 0.1
                box = (max(0., x0 - mx) * W, max(0., y0 - my) * H,
                       min(1., x1 + mx) * W, min(1., y1 + my) * H)
                c = im.crop(tuple(int(v) for v in box))
                if min(c.size) < 8:
                    c = im
                p = l2_normalize(enc(tf(c).unsqueeze(0).to(device)))[0]
                side = int(round(p.shape[0] ** 0.5))
                g = p.reshape(side, side, -1).permute(2, 0, 1).unsqueeze(0)
                g = F.adaptive_avg_pool2d(g, CELL)[0].flatten(1).T   # [16, d]
                pair.append(F.normalize(g, dim=-1))
        else:
            px = tf(im).unsqueeze(0).to(device)
            p = l2_normalize(enc(px))[0]                   # [N, d]
            side = int(round(p.shape[0] ** 0.5))
            for b in (r["correct_box"], r["wrong_box"]):
                w = cell_weights(tuple(b), side, device)   # [16, N]
                pair.append(F.normalize(w @ p, dim=-1))    # [16, d]
        grids.append(torch.stack(pair).cpu())
        if i % 500 == 0:
            print(f"    {name} {i}/{len(rows)}  {time.time()-t0:.0f}s", flush=True)
    out = {"grid": torch.stack(grids), "dim": grids[0].shape[-1],
           "n_patch": int(side ** 2), "nparam": int(nparam),
           "ms_per_img": (time.time() - t0) / len(rows) * 1000}
    if cache:
        torch.save(out, cp)
    del holder, enc
    torch.cuda.empty_cache()
    return out


# ------------------------------------------------------------------------ 프로브
class MLP(nn.Module):
    def __init__(self, d, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Dropout(0.2),
                                 nn.Linear(h, h // 2), nn.GELU(), nn.Linear(h // 2, 1))

    def forward(self, x, *_):
        return self.net(x).squeeze(-1)


class AttnProbe(nn.Module):
    """텍스트가 질의가 되어 박스 안 16셀 중 볼 곳을 고른다.

    mean pooling 이 병목인지 시험한다 — 셀을 고를 수 있는데도 안 오르면 정보가
    애초에 없는 것이다.
    """

    def __init__(self, d, dt, h=256):
        super().__init__()
        self.q = nn.Linear(dt, h)
        self.k = nn.Linear(d, h)
        self.v = nn.Linear(d, h)
        self.out = nn.Sequential(nn.GELU(), nn.Linear(h, h // 2), nn.GELU(),
                                 nn.Linear(h // 2, 1))

    def pool(self, g, t):
        q = self.q(t).unsqueeze(1)                          # [B, 1, h]
        a = (q * self.k(g)).sum(-1) / q.shape[-1] ** 0.5    # [B, 16]
        return (a.softmax(-1).unsqueeze(-1) * self.v(g)).sum(1)

    def forward(self, x, g, t):
        ga, gb = g[:, 0], g[:, 1]
        return self.out(self.pool(ga, t) - self.pool(gb, t)).squeeze(-1)


def torch_cv(make, feats, y, groups, seed=0, epochs=120, lr=3e-4):
    """group CV. `make()` 가 모델을, `feats` 가 (x, grid, text) 를 준다."""
    g = np.array(groups)
    uniq = np.unique(g)
    rs = np.random.default_rng(seed)
    rs.shuffle(uniq)
    fold_of = {u: i % FOLDS for i, u in enumerate(uniq)}
    assign = np.array([fold_of[v] for v in g])
    accs = []
    for f in range(FOLDS):
        te = torch.tensor(assign == f)
        # **내부 검증으로 에폭을 고른다.** test fold 의 에폭별 최댓값을 취하면
        # 그 자체가 test 에 맞춘 선택이라 관문 임계값을 부풀린다. 다음 fold 를
        # 검증으로 떼고 test 는 마지막에 한 번만 본다.
        va = torch.tensor(assign == (f + 1) % FOLDS)
        torch.manual_seed(seed)
        m = make().cuda()
        opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-2)
        tr_i = (~te & ~va).nonzero().squeeze(-1)
        best_va, best_state = -1.0, None
        for ep in range(epochs):
            m.train()
            perm = tr_i[torch.randperm(len(tr_i))]
            for k in range(0, len(perm), 128):
                b = perm[k:k + 128]
                loss = F.binary_cross_entropy_with_logits(
                    m(*[x[b].cuda() for x in feats]), y[b].cuda())
                opt.zero_grad(); loss.backward(); opt.step()
            if ep % 10 == 9 or ep == epochs - 1:
                m.eval()
                with torch.no_grad():
                    pv = m(*[x[va].cuda() for x in feats])
                    a = float(((pv > 0).float().cpu() == y[va]).float().mean())
                if a > best_va:
                    best_va = a
                    best_state = {k_: v.detach().clone() for k_, v in m.state_dict().items()}
        m.load_state_dict(best_state)
        m.eval()
        with torch.no_grad():
            pr = m(*[x[te].cuda() for x in feats])
            accs.append(float(((pr > 0).float().cpu() == y[te]).float().mean()))
        del m
        torch.cuda.empty_cache()
    return float(np.mean(accs)), float(np.std(accs))


def logistic_cv_grouped(x, y, groups, seed=0):
    g = np.array(groups)
    uniq = np.unique(g)
    rs = np.random.default_rng(seed)
    rs.shuffle(uniq)
    fold_of = {u: i % FOLDS for i, u in enumerate(uniq)}
    assign = np.array([fold_of[v] for v in g])
    x = (x - x.mean(0)) / x.std(0).clamp_min(1e-8)
    accs = []
    for f in range(FOLDS):
        te = torch.tensor(assign == f)
        w = torch.zeros(x.shape[1], requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([w, b], max_iter=300)

        def closure():
            opt.zero_grad()
            l = F.binary_cross_entropy_with_logits(x[~te] @ w + b, y[~te])
            (l + 1e-3 * (w ** 2).sum()).backward()
            return l
        opt.step(closure)
        with torch.no_grad():
            accs.append((((x[te] @ w + b) > 0).float() == y[te]).float().mean().item())
    return float(np.mean(accs)), float(np.std(accs))


# -------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="attr", choices=["attr", "pos"])
    ap.add_argument("--models", default="")
    ap.add_argument("--probes", default="linear,mlp,attn")
    ap.add_argument("--out", default="results/ceiling")
    args = ap.parse_args()
    set_seed(0)
    dev = "cuda"

    if args.set == "attr":
        rows = [json.loads(l) for l in open("data/axes/vaw_train_referent_attr.jsonl")]
        rows += [json.loads(l) for l in open("data/axes/vaw_val_referent_attr.jsonl")]
        img_root = "data/vg"
    else:
        rows = [json.loads(l) for l in open("data/axes/train2017_referent_hard.jsonl")]
        rows += [json.loads(l) for l in open("data/axes/val2017_referent_hard.jsonl")]
        for i, r in enumerate(rows):
            r.setdefault("pair_id", f"p{i}")
        img_root = "data/coco"
    groups = [r["pair_id"] for r in rows]
    print(f"{args.set} 세트 {len(rows)}건  pair {len(set(groups))}개  "
          f"IoU 중앙값 {sorted(r['iou'] for r in rows)[len(rows)//2]:.3f}")

    cache_dir = Path("data/ceiling_cache") / args.set
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 라벨 — 좌우를 뒤집는다. **정확히 절반**을 뒤집는다: 무작위로 두면
    # 다수클래스 비율이 50% 를 벗어나고, 상수 예측으로 붕괴한 프로브가 그 값을
    # 내면서 학습한 것처럼 보인다 (실측: 53.8% 가 여러 백본에서 동일하게 나왔다).
    rng = np.random.default_rng(0)
    flip = torch.zeros(len(rows), dtype=torch.bool)
    flip[:len(rows) // 2] = True
    flip = flip[torch.from_numpy(rng.permutation(len(rows)))]
    Y = flip.float()
    print(f"라벨 균형 Y=1 {Y.mean()*100:.1f}%  → 상수 예측 상한 "
          f"{max(float(Y.mean()), 1-float(Y.mean()))*100:.1f}%")

    # 텍스트는 CLIP 것을 공유 — 어느 속성을 묻는지만 알면 되므로 정렬 불필요
    tp = cache_dir / f"_text__n{len(rows)}.pt"
    if tp.exists():
        TX = torch.load(tp)
    else:
        tb = load_backbone("clip_vitb16", dev)
        es = []
        for i in range(0, len(rows), 256):
            with torch.no_grad():
                es.append(l2_normalize(tb.encode_text_pooled(
                    [r["phrase"] for r in rows[i:i + 256]])).cpu())
        TX = torch.cat(es)
        torch.save(TX, tp)
        del tb
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------- 대조군
    def geo(b):
        x0, y0, x1, y1 = b
        return [x0, y0, x1, y1, (x0 + x1) / 2, (y0 + y1) / 2,
                (x1 - x0) * (y1 - y0), x1 - x0, y1 - y0]
    GA = torch.tensor([geo(r["correct_box"]) for r in rows], dtype=torch.float)
    GB = torch.tensor([geo(r["wrong_box"]) for r in rows], dtype=torch.float)
    gd = torch.where(flip[:, None], GB - GA, GA - GB)
    ctl = {
        "geom": logistic_cv_grouped(torch.cat([gd, (gd[:, :, None] * gd[:, None, :]).flatten(1)], -1), Y, groups)[0],
        "text": logistic_cv_grouped(TX.clone(), Y, groups)[0],
        "geom+text": logistic_cv_grouped(torch.cat([gd, TX], -1), Y, groups)[0],
    }
    print(f"\n대조군 (시각 특징 없음):  "
          + "  ".join(f"{k} {v*100:.1f}%" for k, v in ctl.items() if isinstance(v, float)))
    print(f"  chance 50.0%.  이 값들이 시각 프로브의 실질 기준선이다\n")

    # **양성 대조군.** 라벨을 담은 합성 특징. 프로브가 이걸 못 풀면 기계장치가
    # 고장난 것이고, 모든 "정보 없음" 판정이 무효다 (D11: 절대 기준점).
    if not args.models:
        gsyn = torch.randn(len(rows), 2, CELL * CELL, 64) * 0.5
        gsyn[:, 0, :, 0] += 1.2                       # 정답 박스에만 실리는 방향
        gs = torch.where(flip[:, None, None, None], gsyn[:, [1, 0]], gsyn)
        ps = F.normalize(gs.mean(2), dim=-1)
        pdz = ps[:, 0] - ps[:, 1]
        lin = logistic_cv_grouped(torch.cat([pdz, TX], -1), Y, groups)[0]
        xz = torch.cat([pdz, TX], -1)
        xz = (xz - xz.mean(0)) / xz.std(0).clamp_min(1e-8)
        ml = torch_cv(lambda d=xz.shape[1]: MLP(d), (xz, gs, TX), Y, groups)[0]
        at = torch_cv(lambda: AttnProbe(64, TX.shape[1]), (pdz, gs, TX), Y, groups)[0]
        print(f"양성 대조군 (합성 특징에 라벨 주입): 선형 {lin*100:.1f}%  "
              f"MLP {ml*100:.1f}%  attn {at*100:.1f}%   ← 셋 다 높아야 프로브가 유효")
        ctl["_positive_synth"] = {"linear": lin, "mlp": ml, "attn": at}
        if min(lin, ml, at) < 0.8:
            print("  **경고: 프로브 기계장치가 주입된 신호도 못 찾는다. 결과 해석 불가**")

    names = [n.strip() for n in args.models.split(",") if n.strip()] or list(MODELS)
    probes = args.probes.split(",")
    res = {"_control": ctl, "_n": len(rows), "_set": args.set}
    print(f"  {'백본':<26}{'patch':>6}{'M파라':>7}{'ms/장':>7}{'코사인':>8}"
          f"{'선형':>7}{'MLP':>7}{'attn':>7}{'최고':>8}")
    for name in names:
        spec = MODELS[name]
        try:
            f = extract(name, spec, rows, img_root, dev, cache_dir)
        except Exception as e:
            print(f"  {name:<26} 실패: {type(e).__name__} {e}")
            continue
        G = f["grid"]                                   # [n, 2, 16, d]
        P = F.normalize(G.mean(2), dim=-1)              # [n, 2, d] = pooled
        cos = float((P[:, 0] * P[:, 1]).sum(-1).mean())
        pd = torch.where(flip[:, None], P[:, 1] - P[:, 0], P[:, 0] - P[:, 1])
        Gf = torch.where(flip[:, None, None, None],
                         G[:, [1, 0]], G)               # 뒤집기를 격자에도 적용
        r = {"cos": cos, "n_patch": f.get("n_patch"), "nparam": f.get("nparam"),
             "ms_per_img": f.get("ms_per_img")}
        if "linear" in probes:
            r["linear"] = logistic_cv_grouped(torch.cat([pd, TX], -1), Y, groups)[0]
        if "mlp" in probes:
            x = torch.cat([pd, TX], -1)
            x = (x - x.mean(0)) / x.std(0).clamp_min(1e-8)
            a, s = torch_cv(lambda d=x.shape[1]: MLP(d), (x, Gf, TX), Y, groups)
            r["mlp"], r["mlp_std"] = a, s
        if "attn" in probes:
            a, s = torch_cv(lambda d=f["dim"]: AttnProbe(d, TX.shape[1]),
                            (pd, Gf, TX), Y, groups)
            r["attn"], r["attn_std"] = a, s
        r["best"] = max(r.get(k, 0) for k in ("linear", "mlp", "attn"))
        res[name] = r
        print(f"  {name:<26}{r['n_patch']:>6}{r['nparam']/1e6:>7.0f}"
              f"{r['ms_per_img']:>7.1f}{cos:>8.3f}{r.get('linear',0)*100:>6.1f}%"
              f"{r.get('mlp',0)*100:>6.1f}%{r.get('attn',0)*100:>6.1f}%"
              f"{r['best']*100:>7.1f}%", flush=True)
        (out_dir / f"sweep_{args.set}.json").write_text(
            json.dumps(res, indent=2, ensure_ascii=False))

    best_n = max((k for k in res if not k.startswith("_")),
                 key=lambda k: res[k]["best"], default=None)
    if best_n:
        b = res[best_n]["best"]
        margin = b - max(v for v in ctl.values() if isinstance(v, float))
        print(f"\n{'='*70}")
        print(f"관문 (비선형 프로브 ≥ 75%):  최고 {best_n} {b*100:.1f}%  "
              f"→ {'**통과**' if b >= 0.75 else '미달'}")
        print(f"  대조군 최고 대비 실질 이득 {margin*100:+.1f}p")


if __name__ == "__main__":
    main()
