"""D118 — E1(회귀) · E2(주 판정) 평가 세트 구성 + 학습 제외 명부. 학습 0회. 읽기 전용.

    python experiments/omnivla/build_eval_sets.py

E2 협의 = 동종 카테고리 >=2 **AND** COCO thing/person. **전량을 학습에서 제외**한다 (L1).
E1      = 그쪽 등록 test 에피소드의 비표면 객체 >=2 프레임.
|ΔL| < min_sep 쌍은 제외하고 **제외 수를 병기**한다 (D118 §2-b).
"""
from __future__ import annotations

import glob
import json
import os
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pointing import scorable_pairs                                   # noqa: E402
from src.utils.provenance import dump_result                          # noqa: E402

CFG = ROOT / "configs/experiment/omnivla_eval_sets.yaml"
OUT = ROOT / "results/phase4/omnivla/eval_sets.json"


def lemma(w):
    w = w.lower().strip()
    if len(w) > 3 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith("ses"):
        return w[:-2]
    if len(w) > 3 and w.endswith("es") and not w.endswith("sses"):
        return w[:-1]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def ptxt(p):
    return (p[0] if isinstance(p, (tuple, list)) and p else str(p)).strip().lower()


def head_noun(t):
    w = re.findall(r"[a-z]+", t)
    return lemma(w[-1]) if w else ""


def main():
    cfg = yaml.safe_load(open(CFG))
    D = cfg["dataset_root"]
    BL = {w.lower() for w in cfg["prompt_blocklist"]}
    TEST_EPS = set(cfg["test_episodes"])
    MIN_SEP = cfg["min_sep_m"]
    coco = {c["name"] for c in json.load(open(ROOT / cfg["coco_instances"]))["categories"]}
    coco_lem = {lemma(c) for c in coco}
    person_lem = {lemma(w) for w in cfg["person_synonyms"]}

    e1, e2, cats = [], [], Counter()
    n_frames = 0
    drop_e1 = drop_e2 = pair_e1 = pair_e2 = 0
    for ep in sorted(glob.glob(os.path.join(D, "episode_*"))):
        name = os.path.basename(ep)
        for f in sorted(glob.glob(os.path.join(ep, "pickle_nomad", "*.pkl"))):
            try:
                objs = pickle.load(open(f, "rb"))
            except Exception:
                continue
            if not isinstance(objs, list):
                objs = [objs]
            n_frames += 1
            items = []
            for o in objs:
                if not isinstance(o, dict):
                    continue
                keep = [p for p in (o.get("prompt") or [])
                        if not (set(re.findall(r"[a-z]+", ptxt(p))) & BL)]
                if not keep:
                    continue
                t = ptxt(keep[0])
                pm = np.asarray(o.get("pose_median")).reshape(-1).astype(float)
                if pm.size < 2:
                    continue
                items.append({"prompt": t, "cat": head_noun(t), "lat": float(pm[1])})
            if len(items) < 2:
                continue
            lat = [x["lat"] for x in items]
            pairs, drop = scorable_pairs(lat, MIN_SEP)
            rec = {"episode": name, "stem": os.path.basename(f)[:-4],
                   "items": items, "n_pairs": len(pairs), "n_dropped": drop}
            # E1 — 등록 test 에피소드. 비회귀용
            if name in TEST_EPS and pairs:
                e1.append(rec); pair_e1 += len(pairs); drop_e1 += drop
            # E2 협의 — 동종 카테고리 >=2 AND COCO thing/person
            c = Counter(x["cat"] for x in items)
            dup = [k for k, v in c.items() if v >= 2 and k]
            th = [k for k in dup if k in coco_lem or k in person_lem]
            if th:
                # **동종 쌍만** 채점한다 — 과제가 "동종 후보 둘 중 어느 쪽" 이다
                sp, dp = [], 0
                for i in range(len(items)):
                    for j in range(i + 1, len(items)):
                        if items[i]["cat"] != items[j]["cat"] or items[i]["cat"] not in th:
                            continue
                        if abs(lat[i] - lat[j]) < MIN_SEP:
                            dp += 1
                        else:
                            sp.append((i, j))
                r2 = dict(rec); r2["same_cat_pairs"] = sp
                r2["n_pairs"] = len(sp); r2["n_dropped"] = dp
                r2["dup_thing_categories"] = th
                e2.append(r2); pair_e2 += len(sp); drop_e2 += dp
                for k in th:
                    cats[k] += 1

    e2_keys = sorted({(r["episode"], r["stem"]) for r in e2})
    res = {"preregistered": "notes/decisions.md D118",
           "read_only_note": "OmniVLA_edge 레포를 수정하지 않았다",
           "min_sep_m": MIN_SEP, "min_sep_source": "grounding_test.py 기본값 (새 상수 아님)",
           "n_frames_scanned": n_frames,
           "E1": {"role": "회귀 — 비회귀 게이트만. 주 눈금 아님",
                  "source": f"등록 test 에피소드 {sorted(TEST_EPS)}",
                  "n_frames": len(e1), "n_scorable_pairs": pair_e1,
                  "n_dropped_pairs_below_min_sep": drop_e1},
           "E2": {"role": "주 판정 — arm3−arm2 · arm4−arm3",
                  "definition": "동종 카테고리 >=2 AND COCO thing/person · **동종 쌍만 채점**",
                  "n_frames": len(e2), "n_scorable_pairs": pair_e2,
                  "n_dropped_pairs_below_min_sep": drop_e2,
                  "category_frames_top15": cats.most_common(15)},
           "train_exclusion": {
               "rule": "L1 — E2 협의 프레임 **전량**을 학습에서 제외 (D118 §1)",
               "n_excluded_frames": len(e2_keys),
               "keys": [f"{a}/{b}" for a, b in e2_keys]},
           "note_4c": ("학습 측 협의 동종-후보는 제외 후 **0 건**이다. 1차는 **순수 전이 검사** — "
                       "광의(연속면 동종 후보)로 배운 heatmap 채널 읽기가 협의로 전이되는가 "
                       "(D118 §1-a). 미검출을 '어댑터 무용' 으로 쓰지 않는다")}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    dump_result(OUT, res, seed=0, cfg_path=CFG)
    print(f"E1  프레임 {len(e1)} · 채점 쌍 **{pair_e1}** · 제외(|ΔL|<{MIN_SEP}m) {drop_e1}")
    print(f"E2  프레임 {len(e2)} · 채점 쌍 **{pair_e2}** · 제외 {drop_e2}")
    print(f"    카테고리 상위: {cats.most_common(8)}")
    print(f"학습 제외 프레임 {len(e2_keys)}")
    print(f"저장 {OUT}")


if __name__ == "__main__":
    main()
