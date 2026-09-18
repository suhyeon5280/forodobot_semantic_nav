"""C 실행 1 — 사다리 점별 **판정 프로브**와 **무효 2종** (+ CIFAR 감시).

학습 스크립트(`c_train.py`)가 각 점 종료 시 호출한다. 이 파일은 새 판정 기계를
만들지 않는다 — 전부 이미 검증된 코드를 **가중치만 바꿔** 다시 부른다.

| 검사 | 재사용하는 것 | 기준 |
|---|---|---|
| 판정 프로브 | `a1_baseline_probe.judge` (라벨 규칙·분할·헤드·학습 루프 불변) | c < 65.2% / b / a ≥ 85.0% (사람 대비) |
| 속성 정렬 | A4-b 재구성 (crop-pool) | **무효** — ≥ 66.30% |
| geom 대조군 | `train_B.geom_probe` | **무효** — ≤ 55% |
| CIFAR-10 | `train_B.zeroshot_cifar` | **감시만** — 기록하되 무효 미발동 (D40) |

**미세조정 가중치를 판정 경로에 넣는 방법** — `sweep_ceiling.extract` 에
`enc_override` 로 `(preprocess, enc, model)` 을 준다. `enc` 는 기준선이 쓰는
`_clip_patch_forward(..., maskclip=False)` 그대로이므로 **판독 연산이 아니라
가중치만** 달라진다. `cache=False` 로 기준선 캐시를 덮어쓰지 않는다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.a1_baseline_probe import (features, judge,  # noqa: E402
                                           load_rows, oracle_split)
from src.backbones.loaders import _clip_patch_forward        # noqa: E402
from src.scoring.base import l2_normalize                    # noqa: E402

PROBE_MODEL = "clip_vitb16_crop"        # A1-c 프로토콜
JUDGE_HEAD = "mlp"                      # D35
HUMAN = 0.8833333333333333              # 홀드아웃 60쌍. 70쌍의 85.7% 와 다른 표본
BASELINE_PROBE, SIGMA = 0.5357142885526022, 0.020203063735798044
C_RATIO = (BASELINE_PROBE + 2 * SIGMA) / HUMAN          # 0.6522
A_RATIO = 0.85
FLOOR_CIFAR, FLOOR_ATTR, CEIL_GEOM = 0.8813, 0.6630, 0.55
TEMPLATE = "a photo of a {} {}."         # D12 3항


def override(model, preprocess):
    """기준선과 **같은 판독 연산**, 가중치만 미세조정된 것."""
    def enc(px):
        with torch.no_grad():
            p, _ = _clip_patch_forward(model.visual, px, maskclip=False)
        return p
    return (preprocess, enc, model)


def run_probe(model, preprocess, dev, seeds=(0, 1, 2)):
    """A1-c 판정. 명부 sha256 검증은 `oracle_split` 안에서 일어난다."""
    rows = load_rows()
    groups, te_mask = oracle_split(rows)
    G, TX = features(PROBE_MODEL, rows, dev,
                     enc_override=override(model, preprocess), cache=False)
    out = judge(rows, groups, G, TX, te_mask, list(seeds))
    acc = out[JUDGE_HEAD]["mean"]
    out["judge"] = {"head": JUDGE_HEAD, "acc": acc, "ratio": acc / HUMAN,
                    "verdict": verdict(acc / HUMAN)}
    return out


def verdict(ratio):
    """사전 등록된 3구간. 재량 없이 기계적으로 적용한다."""
    return "a" if ratio >= A_RATIO else ("c" if ratio < C_RATIO else "b")


# ------------------------------------------------------------------ 무효 3종
def cifar_zeroshot(model, preprocess, dev, batch=None):
    """`batch` 는 **추론 배치 크기**이고 결과를 바꾸지 않는다 — 이미지마다 argmax 를
    독립으로 내므로 배치 경계가 값에 개입하지 않는다. 공존으로 VRAM 이 얕을 때만
    낮춘다 (기본값은 종전 그대로)."""
    import open_clip
    from experiments.train_B import zeroshot_cifar
    kw = {} if batch is None else {"batch": batch}
    return zeroshot_cifar(model, preprocess,
                          open_clip.get_tokenizer("ViT-B-16-quickgelu"), dev, **kw)


def _attr_instances(n, seed):
    from experiments.a4_attr_zeroshot import load_instances
    man = json.loads(Path("results/phase4/holdout60_manifest.json").read_text())
    return load_instances(n, seed, set(man["pair_ids"]))


def attr_zeroshot(model, preprocess, dev, n=2000, seed=0):
    """A4-b — **crop-pool** 속성 정렬. step0 실측 70.3%, 무효 하한 66.3%.

    A4 본편(CLS-pool, 71.3%)과 갈리는 것은 pooling 하나다. 판정 경로가
    crop-pool 이므로 무효 조건도 그 경로에서 잰다 (D20).

    **정규화 순서가 결과를 바꾼다** — `norm → mean → norm` 이 기록값 70.2703% 와
    정확히 일치하고, `mean → norm` 은 70.8709% 로 0.6p 어긋난다. 무효 하한이
    step0 에서 유도되므로 이 순서를 틀리면 문턱이 조용히 움직인다. 판정 경로의
    `F.normalize(gs.mean(2))` 와도 같은 순서다."""
    import open_clip
    tok = open_clip.get_tokenizer("ViT-B-16-quickgelu")
    inst = _attr_instances(n, seed)
    ok = []
    for r in inst:
        p = Path("data/vg") / r["image"]
        if not p.exists():
            continue
        x, y, w, h = r["box"]
        if min(w, h) < 8:
            continue
        crop = Image.open(p).convert("RGB").crop(
            (int(x), int(y), int(x + w), int(y + h)))
        px = preprocess(crop).unsqueeze(0).to(dev)
        with torch.no_grad():
            patches, _ = _clip_patch_forward(model.visual, px, maskclip=False)
            v = l2_normalize(l2_normalize(patches).mean(1))
            t = l2_normalize(model.encode_text(tok(
                [TEMPLATE.format(r["pos"][0], r["obj"]),
                 TEMPLATE.format(r["neg"][0], r["obj"])]).to(dev)))
        s = (v @ t.T)[0]
        ok.append(bool(s[0] > s[1]))
    return float(np.mean(ok)), len(ok)


def geom_control(seed=0):
    """판정 세트가 기하만으로 풀리는가. 모델과 무관하므로 점마다 같은 값이다."""
    from experiments.train_B import geom_probe
    return float(geom_probe(load_rows(), seed))


def invalid_checks(model, preprocess, dev, geom=None, cifar_batch=None):
    """무효 **2종** (D40 승인, 2026-08-14) + CIFAR-10 감시.

    CIFAR-10 은 계속 측정·기록하되 **무효를 발동시키지 않는다.** 이 실험의
    목적함수가 CLIP 공간의 의도적 변형이므로 카테고리 zero-shot 유지는 과제
    요구가 아니다 — 감시 대상이 오이식돼 있었다는 것이 D40 의 판정이다.
    `watch` 에 남는 값은 논문 §8 의 관측(일반 표현 품질과 인스턴스 판별이 함께
    움직이지 않는다)에 유효 실행의 근거로 쓴다.
    """
    cif = cifar_zeroshot(model, preprocess, dev, batch=cifar_batch)
    attr, n_attr = attr_zeroshot(model, preprocess, dev)
    g = geom_control() if geom is None else geom
    fired = []
    if attr < FLOOR_ATTR:
        fired.append(f"속성 정렬 {attr*100:.2f}% < {FLOOR_ATTR*100:.2f}%")
    if g > CEIL_GEOM:
        fired.append(f"geom {g*100:.1f}% > {CEIL_GEOM*100:.0f}%")
    return {"cifar10": cif, "attr_align": attr, "attr_n": n_attr, "geom": g,
            "floors": {"attr_align": FLOOR_ATTR, "geom_ceiling": CEIL_GEOM},
            "watch": {"cifar10": {"value": cif, "ref_floor": FLOOR_CIFAR,
                                  "baseline": 0.9013,
                                  "role": "감시 지표 — 무효 미발동 (D40)",
                                  "below_ref_floor": bool(cif < FLOOR_CIFAR)}},
            "invalid": bool(fired), "fired": fired}
