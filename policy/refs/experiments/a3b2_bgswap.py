"""A3-b-2 — 배경 스왑 합성 구현 + 게이트 재실행 (D37).

    python -m experiments.a3b2_bgswap --n 1500 --erosion 2

파이프라인 (**순서 고정**: 지터가 먼저 — A3-c 와 정합)

    박스 지터 → mask 오려내기(erosion) → 뱅크 배경에 feather 합성

검사 셋

    1. 전경 불변식   합성 전후 mask 내부(erosion 후) 픽셀이 비트 동일한가.
                     feather 대역은 정의상 섞이므로 제외한다.
    2. 게이트 재실행 합성 뷰의 mask 여집합 저수준 통계로 pull/push 구분.
                     A3-b 와 **동일 프로브·동일 지표**. < 60% 통과 (예측 ~50%).
    3. 누출 재확인   합성 뷰에서 배경 난수 치환 코사인. 누출 메커니즘은 남되
                     나르는 내용이 무정보가 됐다는 것을 수치로 닫는다.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backbones.loaders import load_backbone            # noqa: E402
from src.scoring.base import l2_normalize                  # noqa: E402
from src.utils.provenance import collect, dump_result      # noqa: E402
from src.utils.seed import set_seed                        # noqa: E402

CFG_PATH = "configs/experiment/phase4_track2.yaml"
BINS = 8
MARGIN = 0.10
FEATHER = 2.0          # 가우시안 반경(px). 이 대역은 전경 불변식에서 제외


def jitter_box(b, W, H, jit, rng):
    x, y, w, h = b
    sc = 1.0 + rng.uniform(-jit, jit)
    cx, cy = x + w / 2 + rng.uniform(-jit / 2, jit / 2) * w, \
             y + h / 2 + rng.uniform(-jit / 2, jit / 2) * h
    nw, nh = w * sc, h * sc
    x0, y0, x1, y1 = cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2
    mx, my = nw * MARGIN, nh * MARGIN
    return (max(0, int(x0 - mx)), max(0, int(y0 - my)),
            min(W, int(x1 + mx)), min(H, int(y1 + my)))


def erode(m, k):
    """이진 마스크 침식. 경계 링의 원본 배경 잔존을 막는다."""
    if k <= 0:
        return m
    t = torch.tensor(m, dtype=torch.float32)[None, None]
    p = -F.max_pool2d(-t, kernel_size=2 * k + 1, stride=1, padding=k)
    return (p[0, 0].numpy() > 0.5).astype(np.uint8)


def compose(im, box, mask, bg_im, erosion, rng):
    """지터된 crop 을 오려내 뱅크 배경에 feather 합성."""
    x0, y0, x1, y1 = box
    fg = im.crop((x0, y0, x1, y1))
    w, h = fg.size
    if w < 8 or h < 8:
        return None, None, None
    m = erode(mask[y0:y1, x0:x1], erosion)
    if m.sum() < 16:
        return None, None, None
    # 배경 뱅크에서 같은 크기 crop
    BW, BH = bg_im.size
    if BW < w or BH < h:
        bg_im = bg_im.resize((max(w, BW), max(h, BH)))
        BW, BH = bg_im.size
    bx, by = rng.integers(0, BW - w + 1), rng.integers(0, BH - h + 1)
    bg = bg_im.crop((int(bx), int(by), int(bx) + w, int(by) + h))
    alpha = Image.fromarray((m * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(FEATHER))
    out = Image.composite(fg, bg, alpha)
    # **완전 불투명(alpha==255)만** feather 밖이다. 251~254 는 실제로 섞이므로
    # 여기에 넣으면 불변식이 거짓 위반을 낸다.
    return out, m, np.asarray(alpha) == 255


def compose_or_keep(im, box, mask, bg_im, erosion, rng, p_swap=1.0):
    """`p_swap` 확률로 배경 스왑, 아니면 **원본 배경을 유지**한 crop.

    `p_swap >= 1.0` 이면 난수를 소비하지 않고 `compose` 를 그대로 부른다 —
    기본 경로가 비트 단위로 불변이어야 D37 게이트 결과가 유지된다.
    """
    if p_swap >= 1.0 or rng.random() < p_swap:
        return compose(im, box, mask, bg_im, erosion, rng)
    x0, y0, x1, y1 = box
    fg = im.crop((x0, y0, x1, y1))
    w, h = fg.size
    if w < 8 or h < 8:
        return None, None, None
    m = erode(mask[y0:y1, x0:x1], erosion)
    if m.sum() < 16:
        return None, None, None
    alpha = Image.fromarray((m * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(FEATHER))
    return fg, m, np.asarray(alpha) == 255


def bg_hist(img, m):
    a = np.asarray(img.resize((64, 64)), dtype=np.float32) / 255
    mm = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize((64, 64))) > 127
    bg = ~mm
    if bg.sum() < 32:
        return None
    q = (a * (BINS - 1)).astype(int).clip(0, BINS - 1)
    idx = (q[..., 0] * BINS * BINS + q[..., 1] * BINS + q[..., 2])[bg]
    hh = np.bincount(idx, minlength=BINS ** 3).astype(np.float32)
    return hh / max(hh.sum(), 1)


def chi2(p, q):
    return float(0.5 * np.sum((p - q) ** 2 / (p + q + 1e-9)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jitter", type=float, default=0.10)
    ap.add_argument("--erosion", type=int, default=2)
    ap.add_argument("--p-swap", type=float, default=1.0,
                    help="배경 스왑 확률. 1.0 = D37 등록 구성 (기본, 비트 불변)")
    a = ap.parse_args()
    set_seed(a.seed)
    from pycocotools.coco import COCO
    cfg = yaml.safe_load(open(CFG_PATH))
    prov = collect(a.seed, cfg_path=CFG_PATH,
                   inputs=["data/coco/annotations/instances_train2017.json"],
                   extra={"effective_args": vars(a), "threshold": 0.60,
                          "feather_px": FEATHER, "p_swap": a.p_swap})
    coco = COCO("data/coco/annotations/instances_train2017.json")
    dims = {i["id"]: (i["width"], i["height"], i["file_name"])
            for i in coco.loadImgs(coco.getImgIds())}
    by_img, cat_imgs = defaultdict(list), defaultdict(set)
    for aid in coco.getAnnIds(iscrowd=False):
        an = coco.loadAnns([aid])[0]
        by_img[an["image_id"]].append(an)
        cat_imgs[an["category_id"]].add(an["image_id"])
    all_imgs = set(by_img)

    rng = np.random.default_rng(a.seed)
    ms = cfg["min_side"]
    X, Y, G, fg_ok, fg_bad, leak = [], [], [], 0, 0, []
    bb = load_backbone("clip_vitb16", "cuda")
    keys = list(by_img); rng.shuffle(keys)

    def bank_image(cid):
        """해당 카테고리 인스턴스가 **없는** 이미지."""
        for _ in range(20):
            cand = keys[int(rng.integers(0, len(keys)))]
            if cand not in cat_imgs[cid]:
                _, _, fn = dims[cand]
                p = Path("data/coco/train2017") / fn
                if p.exists():
                    return Image.open(p).convert("RGB")
        return None

    for img_id in keys:
        W, H, fn = dims[img_id]
        p = Path("data/coco/train2017") / fn
        if not p.exists():
            continue
        bycat = defaultdict(list)
        for an in by_img[img_id]:
            if min(an["bbox"][2], an["bbox"][3]) >= ms:
                bycat[an["category_id"]].append(an)
        pairs = [(cid, x, y) for cid, g in bycat.items() if len(g) >= 2
                 for x, y in combinations(g, 2)]
        if not pairs:
            continue
        im = Image.open(p).convert("RGB")
        for cid, P, Q in pairs[:2]:
            mP, mQ = coco.annToMask(P), coco.annToMask(Q)
            views = []
            for src, mk in ((P, mP), (P, mP), (Q, mQ)):
                bg = bank_image(cid)
                if bg is None:
                    break
                b = jitter_box(src["bbox"], W, H, a.jitter, rng)
                out, m, solid = compose_or_keep(im, b, mk, bg, a.erosion, rng,
                                                a.p_swap)
                if out is None:
                    break
                views.append((out, m, solid, b, mk))
            if len(views) < 3:
                continue
            # --- 1. 전경 불변식: feather 밖 mask 내부가 원본과 비트 동일한가
            for out, m, solid, b, mk in views:
                x0, y0, x1, y1 = b
                orig = np.asarray(im.crop((x0, y0, x1, y1)))
                comp = np.asarray(out)
                inside = (m > 0) & solid
                if inside.sum() < 8:
                    continue
                if np.array_equal(orig[inside], comp[inside]):
                    fg_ok += 1
                else:
                    fg_bad += 1
            # --- 2. 게이트: 합성 뷰의 배경 통계로 pull/push
            h1, h2, h3 = (bg_hist(v[0], v[1]) for v in views)
            if h1 is None or h2 is None or h3 is None:
                continue
            X.append([chi2(h1, h2), float(np.abs(h1 - h2).sum())]); Y.append(1); G.append(img_id)
            X.append([chi2(h1, h3), float(np.abs(h1 - h3).sum())]); Y.append(0); G.append(img_id)
            # --- 3. 누출 재확인
            if len(leak) < 200:
                out, m, solid, b, mk = views[0]
                arr = np.asarray(out).copy()
                mm = m > 0
                noisy = arr.copy()
                noisy[~mm] = rng.integers(0, 256, size=(int((~mm).sum()), 3))
                def pooled(x):
                    px = bb.preprocess(Image.fromarray(x)).unsqueeze(0).cuda()
                    with torch.no_grad():
                        pt, _ = bb.encode_image_patches(px)
                    pt = l2_normalize(pt)[0]
                    side = int(round(pt.shape[0] ** .5))
                    msk = torch.tensor(np.asarray(Image.fromarray(
                        (mm * 255).astype(np.uint8)).resize((side, side))) > 127,
                        device=pt.device).flatten().float()
                    return None if msk.sum() < 1 else F.normalize(
                        (pt * msk[:, None]).sum(0) / msk.sum(), dim=-1)
                v1, v2 = pooled(arr), pooled(noisy)
                if v1 is not None and v2 is not None:
                    leak.append(float((v1 @ v2).item()))
        if len(Y) >= a.n * 2:
            break

    X, Y, G = np.array(X), np.array(Y), np.array(G)
    from sklearn.ensemble import HistGradientBoostingClassifier
    ug = np.unique(G); rs = np.random.default_rng(a.seed); rs.shuffle(ug)
    fold = {u: i % 5 for i, u in enumerate(ug)}
    asg = np.array([fold[g] for g in G])
    accs = []
    for f in range(5):
        te = asg == f
        m_ = HistGradientBoostingClassifier(max_iter=200, random_state=a.seed)
        m_.fit(X[~te], Y[~te]); accs.append(float((m_.predict(X[te]) == Y[te]).mean()))
    acc = float(np.mean(accs))
    ok = acc < 0.60
    out = {"erosion": a.erosion, "jitter": a.jitter, "feather_px": FEATHER,
           "n_examples": len(Y), "bg_shortcut_acc": acc, "std": float(np.std(accs)),
           "threshold": 0.60, "passed": bool(ok),
           "baseline_a3b": 0.955,
           "foreground_invariant": {"identical": fg_ok, "differing": fg_bad,
                                    "tolerance": "alpha==255 인 화소만 검사(feather 대역 전부 제외), "
                                                 "erosion 후 mask 내부와 교집합"},
           "leak_after_swap": {"n": len(leak),
                               "cosine_mean": float(np.mean(leak)) if leak else None,
                               "baseline_before_swap": 0.8752,
                               "note": ("누출 메커니즘은 남는다. 달라진 것은 나르는 "
                                        "내용이 정체와 무상관이 됐다는 점이다.")}}
    sfx = "" if a.p_swap >= 1.0 else f"_p{a.p_swap:g}"
    dump_result(Path(f"results/phase4/a3b2_bgswap_e{a.erosion}{sfx}.json"),
                out, prov=prov)
    print(f"\n{'='*70}\nA3-b-2 — 배경 스왑 후 게이트 (erosion={a.erosion})\n")
    print(f"  전경 불변식  동일 {fg_ok} / 불일치 {fg_bad}  "
          f"({'통과' if fg_bad == 0 else '**위반**'})")
    print(f"  배경 단독 정확도  **{acc*100:.1f}%** ± {np.std(accs)*100:.1f}"
          f"   (A3-b 95.5% → )  임계 60%  → {'통과' if ok else '미통과'}")
    if leak:
        print(f"  누출 재확인  코사인 {np.mean(leak):.4f} (스왑 전 0.8752)")


if __name__ == "__main__":
    main()
