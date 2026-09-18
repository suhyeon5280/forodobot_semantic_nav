"""텍스트 경유 관문 — 텍스트 1개 × crop 2개. 설계안 `notes/text-gate-design.md` (D44 승인).

    python -m experiments.text_gate

이 측정이 답하는 것 — 사다리가 프로브를 +7.4p 올렸는데 그 이득을 **텍스트가
선택할 수 있는가**. 프로브는 "정보가 특징에 들어왔는가"(필요조건)만 답한다.

**학습이 없다.** 기존 가중치 4종에 zero-shot 채점을 돌릴 뿐이므로 설계 변경
예산을 쓰지 않는다 (D34 논리).

주 채점 — 행렬곱 1회

    t   = normalize(text("a photo of a {attr} {cat}."))          [d]
    z_i = normalize( normalize(patches_i).mean(0) )              [2, d]
    pred = argmax_i <z_i, t>

판독은 A1-c·D36 과 동일한 crop 판독(박스 10% 확장, 224 정사각, crop 전체 pool)
이고 정규화 순서 `norm → mean → norm` 을 고정한다 — `mean → norm` 은 같은
데이터에서 0.6p 어긋난다 (`c_eval.attr_zeroshot` 독스트링).

병기 — late interaction `max_n <patch_i(n), t>`. **판정에 쓰지 않는다** (D38).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.c_train import (gpu_exclusive_or_die,        # noqa: E402
                                 system_memory_or_die)
from src.backbones.loaders import _clip_patch_forward         # noqa: E402
from src.scoring.base import l2_normalize                     # noqa: E402
from src.utils.provenance import collect, dump_result         # noqa: E402
from src.utils.seed import set_seed                           # noqa: E402

CFG_PATH = "configs/experiment/text_gate.yaml"   # 규칙 3
MANIFEST = "results/phase4/holdout60_manifest.json"
VG_META = "data/vg/image_data.json"
IMG_ROOT = "data/vg"
POINTS = [(3500, "results/phase4/c_run1_sd_3500.pt"),
          (35000, "results/phase4/c_run1_sd_35000.pt"),
          (386802, "results/phase4/c_run1_sd_386802.pt")]


def load_rows():
    rows = [json.loads(l) for l in open("data/axes/vaw_train_referent_attr.jsonl")]
    rows += [json.loads(l) for l in open("data/axes/vaw_val_referent_attr.jsonl")]
    return rows


def overlap_images(rows):
    """선결 점검 — VG↔COCO train2017 겹침. 겹치는 이미지 집합을 낸다."""
    md = json.load(open(VG_META))
    vg2coco = {f"{d['image_id']}.jpg": d.get("coco_id") for d in md}
    tr = {r["image"] for r in (json.loads(l)
                               for l in open("data/axes/c_pairs.jsonl"))}
    ev = {r["image"] for r in rows}
    ov = {i for i in ev
          if vg2coco.get(i) and f"{int(vg2coco[i]):012d}.jpg" in tr}
    return ov, {"eval_images": len(ev),
                "unmapped": len([i for i in ev if i not in vg2coco]),
                "with_coco_id": len([i for i in ev if vg2coco.get(i)]),
                "overlap_images": len(ov)}


def build_model(dev, sd_path=None):
    import open_clip
    m, _, pre = open_clip.create_model_and_transforms(
        "ViT-B-16-quickgelu", pretrained="openai", device=dev)
    m.eval()
    if sd_path:
        blob = torch.load(sd_path, map_location="cpu", weights_only=False)
        miss = m.load_state_dict(blob["trainable"], strict=False)
        assert not miss.unexpected_keys, miss.unexpected_keys
        n_loaded = len(blob["trainable"])
        print(f"    가중치 주입 {sd_path} — {n_loaded} 텐서 "
              f"(step {blob['steps']:,}, cfg {blob['cfg_sha256'][:8]}…)", flush=True)
    for p in m.parameters():
        p.requires_grad_(False)
    return m, pre


MARGIN = 0.10          # main() 에서 config 값으로 덮어쓴다


def crops_of(im, r):
    """A1-c·sweep_ceiling 과 **같은** crop 파이프라인 (박스 10% 확장)."""
    W, H = im.size
    out = []
    for b in (r["correct_box"], r["wrong_box"]):
        x0, y0, x1, y1 = b
        mx, my = (x1 - x0) * MARGIN, (y1 - y0) * MARGIN
        box = (max(0., x0 - mx) * W, max(0., y0 - my) * H,
               min(1., x1 + mx) * W, min(1., y1 + my) * H)
        c = im.crop(tuple(int(v) for v in box))
        if min(c.size) < 8:
            c = im
        out.append(c)
    return out


def boot_ci(fn, groups, B=10000, seed=0):
    """군집(그룹) 부트스트랩 95% CI."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in uniq}
    vals = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        sel = np.concatenate([idx[uniq[k]] for k in pick])
        vals[b] = fn(sel)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="text_gate")
    ap.add_argument("--weights", default="",
                    help="label=path 쉼표 목록. 미지정이면 사다리 3점 (규모 곡선은 "
                         "사다리일 때만 낸다)")
    ap.add_argument("--project", default="",
                    help="γ 게이트 A — 텍스트 부분공간 P 로 투영해 채점 (D47 §2)")
    ap.add_argument("--no-exclude", action="store_true",
                    help="겹침 제외 없이 전체 1,620행. 제외 전/후 병기용 (설계안 §4)")
    a = ap.parse_args()
    import yaml
    cfg = yaml.safe_load(open(CFG_PATH))
    global MARGIN, TEMPLATE
    MARGIN, TEMPLATE = cfg["crop_margin"], cfg["template"]
    PROBE_HOLDOUT, PROBE_GAIN = cfg["probe_holdout"], cfg["probe_gain"]
    POS_SIGMA, POS_FLOOR = cfg["pos_sigma"], cfg["pos_floor"]
    set_seed(cfg["boot_seed"])
    dev = "cuda"
    mem = system_memory_or_die()
    gpu = gpu_exclusive_or_die()

    rows_all = load_rows()
    ov, ovstat = overlap_images(rows_all)
    man = json.load(open(MANIFEST))
    hold = set(man["pair_ids"])
    rows = rows_all if a.no_exclude else [r for r in rows_all if r["image"] not in ov]
    ovstat |= {"excluded": not a.no_exclude, "overlap_rows": len(rows_all) - len(rows),
               "overlap_groups": len({r["pair_id"] for r in rows_all}
                                     - {r["pair_id"] for r in rows}),
               "overlap_in_holdout60": len(hold - {r["pair_id"] for r in rows}),
               "kept_rows": len(rows),
               "kept_groups": len({r["pair_id"] for r in rows}),
               "kept_holdout_pairs": len(hold & {r["pair_id"] for r in rows})}
    print("선결 점검 — VG↔COCO train2017 겹침")
    for k, v in ovstat.items():
        print(f"    {k:<22} {v}")
    assert man["sha256"].startswith("26729b84"), "명부 sha256 불일치"
    print(f"명부 검증 통과 {man['n_pairs']}쌍 sha256 {man['sha256'][:16]}…\n")

    prov = collect(cfg["boot_seed"], cfg_path=CFG_PATH, inputs=["data/axes/vaw_train_referent_attr.jsonl",
                             "data/axes/vaw_val_referent_attr.jsonl",
                             MANIFEST, VG_META] + [p for _, p in POINTS],
                   extra={"template": TEMPLATE, "precheck": ovstat,
                          "gpu_exclusive": gpu, "system_memory_kb": mem,
                          "probe_holdout": PROBE_HOLDOUT, "probe_gain": PROBE_GAIN,
                          "preregistered": "notes/text-gate-design.md (D44)",
                          "projection": a.project or None})

    Pm = None
    if a.project:
        blob = torch.load(a.project, map_location=dev, weights_only=False)
        Pm = blob["P"].to(dev).float()
        print(f"투영 적용 — rank {blob['rank']} / d {blob['d']}  (K={blob['K']:,})",
              flush=True)
    import open_clip
    tok = open_clip.get_tokenizer("ViT-B-16-quickgelu")
    m0, pre = build_model(dev)
    with torch.no_grad():                      # 텍스트 타워는 학습되지 않았다 → 1회
        T = l2_normalize(m0.encode_text(
            tok([TEMPLATE.format(r["attr"], r["cat"]) for r in rows]).to(dev)))
    perm = np.random.default_rng(0).permutation(len(rows))     # null: 셔플 텍스트

    if a.weights:
        pts = [tuple(w.split("=", 1)) for w in a.weights.split(",") if w]
        cfgs = [("frozen", None)] + [(lb, pa) for lb, pa in pts]
        is_ladder = False
    else:
        cfgs = [("frozen", None)] + [(f"pt{n}", p) for n, p in POINTS]
        is_ladder = True
    score = {c: {k: np.zeros((len(rows), 2), np.float32)
                 for k in ("pool", "late", "pool_null")} for c, _ in cfgs}
    for cname, sd in cfgs:
        m = m0 if sd is None else build_model(dev, sd)[0]
        t0 = time.time()
        for i, r in enumerate(rows):
            im = Image.open(Path(IMG_ROOT) / r["image"]).convert("RGB")
            px = torch.stack([pre(c) for c in crops_of(im, r)]).to(dev)
            with torch.no_grad():
                p, _ = _clip_patch_forward(m.visual, px, maskclip=False)
                p = l2_normalize(p)                       # [2, N, d]  norm →
                z = l2_normalize(p.mean(1))               # mean → norm
                if Pm is not None:                        # D47 §2 게이트 A
                    z = l2_normalize(z @ Pm)
                    p = l2_normalize(p @ Pm)
            score[cname]["pool"][i] = (z @ T[i]).cpu().numpy()
            score[cname]["late"][i] = (p @ T[i]).max(1).values.cpu().numpy()
            score[cname]["pool_null"][i] = (z @ T[perm[i]]).cpu().numpy()
            if i % 400 == 0:
                print(f"    {cname} {i}/{len(rows)}  {time.time()-t0:.0f}s", flush=True)
        if sd is not None:
            del m
            torch.cuda.empty_cache()
        print(f"  {cname} 완료 {time.time()-t0:.0f}s", flush=True)

    # 정답 = correct_box 쪽 점수가 크다 (열 0 이 correct_box)
    ok = {c: {k: (v[:, 0] > v[:, 1]).astype(np.float64)
              for k, v in d.items()} for c, d in score.items()}
    groups = np.array([r["pair_id"] for r in rows])
    hmask = np.array([r["pair_id"] in hold for r in rows])

    # --- 양성 대조군: 시각 특징을 텍스트 임베딩으로 치환 (비전 타워만 우회)
    g = torch.Generator(device=dev).manual_seed(0)
    d = T.shape[1]
    eps = torch.randn(len(rows), 2, d, generator=g, device=dev) / d ** 0.5
    other = torch.tensor(perm, device=dev)
    zc = l2_normalize(T + POS_SIGMA * eps[:, 0])
    zw = l2_normalize(T[other] + POS_SIGMA * eps[:, 1])
    pos = ((zc * T).sum(-1) > (zw * T).sum(-1)).float().mean().item()
    print(f"\n양성 대조군 {pos*100:.1f}%  (하한 {POS_FLOOR*100:.0f}%) → "
          f"{'통과' if pos >= POS_FLOOR else '**무효**'}")

    def acc(v, mask=None):
        return float(v[mask].mean()) if mask is not None else float(v.mean())

    res = {"precheck": ovstat, "n_rows": len(rows),
           "n_groups": int(len(np.unique(groups))),
           "n_holdout_rows": int(hmask.sum()),
           "n_holdout_pairs": int(len(np.unique(groups[hmask]))),
           "positive_control": {"acc": pos, "floor": POS_FLOOR,
                                "passed": pos >= POS_FLOOR},
           "configs": {}, "template": TEMPLATE}
    for c, _ in cfgs:
        e = {}
        for setname, mask in (("full", None), ("holdout", hmask)):
            e[setname] = {}
            for k in ("pool", "late", "pool_null"):
                v = ok[c][k]
                gg = groups if mask is None else groups[mask]
                vv = v if mask is None else v[mask]
                lo, hi = boot_ci(lambda s, x=vv: x[s].mean(), gg, cfg["boot"])
                e[setname][k] = {"acc": acc(vv), "ci": [lo, hi]}
        res["configs"][c] = e

    # --- null 차단 판정: 부트스트랩 양측 p + BH 보정 (phase3-protocol §1-4)
    nulls, pvals = [], []
    for c, _ in cfgs:
        for setname, mask in (("full", None), ("holdout", hmask)):
            v = ok[c]["pool_null"] if mask is None else ok[c]["pool_null"][mask]
            gg = groups if mask is None else groups[mask]
            rr = np.random.default_rng(cfg["boot_seed"])
            uq = np.unique(gg); ix = {g: np.where(gg == g)[0] for g in uq}
            bs = np.array([v[np.concatenate([ix[uq[k]] for k in
                          rr.integers(0, len(uq), len(uq))])].mean()
                          for _ in range(cfg["boot"])])
            pv = 2 * min((bs <= 0.5).mean(), (bs >= 0.5).mean())
            nulls.append({"config": c, "set": setname, "acc": float(v.mean()),
                          "excess_p": float(v.mean() - 0.5), "p": float(pv)})
            pvals.append(pv)
    order = np.argsort(pvals); n = len(pvals)
    bh = np.zeros(n, bool)
    for rank, j in enumerate(order, 1):
        if pvals[j] <= 0.05 * rank / n:
            bh[order[:rank]] = True
    for k, e in enumerate(nulls):
        e["bh_significant"] = bool(bh[k])
        e["blocks"] = bool(bh[k] and abs(e["excess_p"]) > 0.03)   # 효과 크기 조건
    res["nulls"] = nulls
    res["null_blocking"] = [e for e in nulls if e["blocks"]]

    # --- 주 판정: Δ = pt386802 − frozen, 전체 세트
    target = cfgs[-1][0]
    dv = ok[target]["pool"] - ok["frozen"]["pool"]
    lo, hi = boot_ci(lambda s: dv[s].mean(), groups, cfg["boot"])
    delta = float(dv.mean())
    res["main"] = {"set": "full", "target": target, "delta": delta, "ci": [lo, hi],
                   "exists": lo > 0,
                   "verdict": "A. 텍스트로 선택 가능" if lo > 0 else "B. 구분 불가",
                   "recovery_ratio": delta / PROBE_GAIN,
                   "rule": "CI 하한 > 0 (존재 판정). 효과 크기 바닥 없음 (D44 수정 1)"}
    dh = dv[hmask]
    lo2, hi2 = boot_ci(lambda s: dh[s].mean(), groups[hmask], cfg["boot"])
    res["main_holdout_ref"] = {"delta": float(dh.mean()), "ci": [lo2, hi2],
                               "note": "참고. 주 판정은 전체 세트 (D44 수정 2)"}
    # --- 보조: 프로브 격차 (홀드아웃만, D44 수정 2)
    res["aux_probe_gap"] = {
        "set": "holdout", "text_gate": acc(ok[target]["pool"], hmask),
        "probe": PROBE_HOLDOUT,
        "gap": acc(ok[target]["pool"], hmask) - PROBE_HOLDOUT,
        "note": "프로브 61.0% 는 홀드아웃 60쌍에서 잰 값. 전체 세트와 섞지 않는다"}

    # --- 규모 곡선 (D44 격상). 사다리일 때만 — 임의 가중치에는 규모 축이 없다
    if not is_ladder:
        res["scale_curve"] = None
        res["raw"] = {"groups": groups.tolist(), "holdout_mask": hmask.tolist(),
                      "ok": {c: {k: v.astype(int).tolist() for k, v in ok[c].items()}
                             for c, _ in cfgs}}
        dump_result(Path(f"results/phase4/{a.tag}.json"), res, prov=prov)
        print(f"\n결과 저장 results/phase4/{a.tag}.json")
        return
    xs = np.log10([n for n, _ in POINTS])
    ys = np.array([acc(ok[f"pt{n}"]["pool"]) for n, _ in POINTS])
    rng = np.random.default_rng(0)
    uniq = np.unique(groups)
    idx = {gg: np.where(groups == gg)[0] for gg in uniq}
    sl = np.empty(cfg["boot"])
    for b in range(cfg["boot"]):
        pick = rng.integers(0, len(uniq), len(uniq))
        sel = np.concatenate([idx[uniq[k]] for k in pick])
        yb = np.array([ok[f"pt{n}"]["pool"][sel].mean() for n, _ in POINTS])
        sl[b] = np.polyfit(xs, yb, 1)[0]
    slope = float(np.polyfit(xs, ys, 1)[0])
    clo, chi = float(np.percentile(sl, 2.5)), float(np.percentile(sl, 97.5))
    reading = ("평평" if clo <= 0 <= chi else
               "단조 상승" if clo > 0 else "단조 하락")
    mono = bool(ys[0] <= ys[1] <= ys[2] or ys[0] >= ys[1] >= ys[2])
    res["scale_curve"] = {"x_log10_pairs": xs.tolist(), "y_acc": ys.tolist(),
                          "slope": slope, "ci": [clo, chi], "reading": reading,
                          "monotone_points": mono,
                          "probe_curve": {"y_ratio": [0.6253, 0.6442, 0.6900],
                                          "slope": 0.0317, "ci": [0.0100, 0.0535],
                                          "reading": "단조 상승"},
                          "rule": "CI 가 0 포함 → 평평 / 양수 → 단조 상승 / 비단조·역전 → 판정 불가"}
    res["raw"] = {"groups": groups.tolist(), "holdout_mask": hmask.tolist(),
                  "ok": {c: {k: v.astype(int).tolist() for k, v in ok[c].items()}
                         for c, _ in cfgs}}
    dump_result(Path(f"results/phase4/{a.tag}.json"), res, prov=prov)
    print(f"\n결과 저장 results/phase4/{a.tag}.json")


if __name__ == "__main__":
    main()
