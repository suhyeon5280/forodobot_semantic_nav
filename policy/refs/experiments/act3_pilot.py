"""D65 3막 파일럿 — 최소 질의-정렬 감독. μ 스윕 3점 × 3시드.

    python -m experiments.act3_pilot

**사전 등록** — `notes/act3-pilot-registration-draft.md` (커밋 968b603, 실행 전).
하이퍼파라미터는 `configs/experiment/act3_pilot.yaml` 에만 있다 (규칙 3).

    L  =  mu * L_align  +  lambda_anchor * L_distill

    L_align    VAW 양방향 대조 쌍의 지도 InfoNCE 2택.
               문구 임베딩 t 를 질의로 정답 crop / 하드 네거티브 crop 을 가른다.
               **판정 과제와 같은 형태**이므로 (a)가 오르는 것은 당연하고,
               재는 것은 (b) 로의 전이다.
    L_distill  원본 CLIP 임베딩과의 거리. c_train 의 후보 1 과 같은 형태.

**판정 경로 비트 동일** — 채점은 `text_gate.py` 가 쓰는 것과 **같은 함수**를
부른다(`_clip_patch_forward` → `l2_normalize` → `mean(1)` → `l2_normalize`).
그것이 실제로 같은지는 추정하지 않고 검사한다: frozen 을 이 경로로 다시 채점해
`text_gate.json` 의 acc 와 **비트 일치**하지 않으면 학습 전에 죽는다.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.c_eval import invalid_checks                        # noqa: E402
from experiments.c_train import (build_model, gpu_exclusive_or_die,   # noqa: E402
                                 system_memory_or_die)
from experiments.text_gate import crops_of, load_rows, overlap_images  # noqa: E402
import experiments.text_gate as TG                                     # noqa: E402
from src.backbones.loaders import _clip_patch_forward                  # noqa: E402
from src.scoring.base import l2_normalize                              # noqa: E402
from src.utils.provenance import collect, dump_result                  # noqa: E402
from src.utils.seed import set_seed                                    # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = "configs/experiment/act3_pilot.yaml"
TG_RESULT = "results/phase4/text_gate.json"
IMG_ROOT = "data/vg"
OUT = "results/phase4/act3_pilot.json"   # --outdir 로 덮을 수 있다


# ---------------- 채점 (text_gate 와 같은 경로) ----------------

def embed_rows(m, pre, rows, dev, bs=8):
    """각 행의 두 crop → [n,2,d]. text_gate 와 **같은 연산 순서**."""
    out = torch.zeros(len(rows), 2, 512, device=dev)
    for i0 in range(0, len(rows), bs):
        chunk = rows[i0:i0 + bs]
        px = []
        for r in chunk:
            im = Image.open(ROOT / IMG_ROOT / r["image"]).convert("RGB")
            px += [pre(c) for c in crops_of(im, r)]
        px = torch.stack(px).to(dev)
        with torch.no_grad():
            p, _ = _clip_patch_forward(m.visual, px, maskclip=False)
            p = l2_normalize(p)
            z = l2_normalize(p.mean(1))
        out[i0:i0 + len(chunk)] = z.reshape(len(chunk), 2, -1)
    return out


def encode_text(m, tok, texts, dev, bs=128):
    """청크 인코딩. 1,048개를 한 배치로 넣으면 QuickGELU 중간 활성이 976 MiB 라
    공존 VRAM 에서 OOM 난다 (2026-08-24 실측). 행 단위로 독립이라 값은 같다."""
    out = []
    for i in range(0, len(texts), bs):
        with torch.no_grad():
            out.append(l2_normalize(m.encode_text(tok(texts[i:i + bs]).to(dev))))
    return torch.cat(out)


def acc_of(z, T):
    """정답 = correct_box(열 0) 점수가 크다. text_gate 와 동일."""
    s = torch.einsum("nkd,nd->nk", z, T)
    return (s[:, 0] > s[:, 1]).double().cpu().numpy()


def boot_ci(ok, groups, B, seed):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in uniq}
    vals = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        sel = np.concatenate([idx[uniq[k]] for k in pick])
        vals[b] = ok[sel].mean()
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_ci(ok_a, ok_b, groups, B, seed):
    """Δ = ok_a − ok_b 의 군집 부트스트랩 CI (같은 행에서 짝지음)."""
    d = ok_a - ok_b
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in uniq}
    vals = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        sel = np.concatenate([idx[uniq[k]] for k in pick])
        vals[b] = d[sel].mean()
    return float(d.mean()), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ---------------- 학습 ----------------

def train_one(cfg, mu, seed, sup_rows, T_sup, dev, base_sd):
    set_seed(seed)
    m, pre, train_ps = build_model(cfg, dev)
    m.load_state_dict(base_sd, strict=True)
    m_ref, _, _ = build_model(cfg, dev)
    m_ref.load_state_dict(base_sd, strict=True)
    for p in m_ref.parameters():
        p.requires_grad_(False)
    m_ref.eval()

    opt = torch.optim.AdamW(train_ps, lr=cfg["lr_vision"], weight_decay=cfg["wd"])
    rng = np.random.default_rng(seed)
    n, bs = len(sup_rows), cfg["batch_pairs"]
    steps = 0
    traj = []          # (step, l_align, l_dist) — 두 항을 **분리** 저장한다.
    #   지금까지는 stdout 에만 찍혀 로그 텍스트로만 남았고 결과 파일에 없었다.
    #   "대조항이 정렬항을 압도하는가" 류의 가설은 이 궤적 없이는 검증 불가다.
    t0 = time.time()
    max_steps = cfg.get("max_steps")      # None 이면 종전대로 제한 없음 (D85)
    for ep in range(cfg["epochs"]):
        order = rng.permutation(n)
        for i0 in range(0, n - bs + 1, bs):
            sel = order[i0:i0 + bs]
            px = []
            for j in sel:
                r = sup_rows[j]
                im = Image.open(ROOT / IMG_ROOT / r["image"]).convert("RGB")
                px += [pre(c) for c in crops_of(im, r)]
            px = torch.stack(px).to(dev)
            p, _ = _clip_patch_forward(m.visual, px, maskclip=False)
            z = l2_normalize(l2_normalize(p).mean(1)).reshape(len(sel), 2, -1)
            t = T_sup[torch.as_tensor(sel, device=dev)]
            logits = torch.einsum("nkd,nd->nk", z, t) / cfg["temp"]
            l_align = F.cross_entropy(
                logits, torch.zeros(len(sel), dtype=torch.long, device=dev))
            with torch.no_grad():
                pr, _ = _clip_patch_forward(m_ref.visual, px, maskclip=False)
                zr = l2_normalize(l2_normalize(pr).mean(1)).reshape(len(sel), 2, -1)
            l_dist = (1 - (z * zr).sum(-1)).mean()
            loss = mu * l_align + cfg["lambda_anchor"] * l_dist
            loss.backward()
            opt.step(); opt.zero_grad(set_to_none=True)
            steps += 1
            traj.append([steps, float(l_align.item()), float(l_dist.item())])
            if max_steps is not None and steps >= max_steps:
                break
            if steps % 20 == 0:
                print(f"      step {steps:4d}  loss {loss.item():.4f} "
                      f"(align {l_align.item():.4f} dist {l_dist.item():.4f})",
                      flush=True)
        if max_steps is not None and steps >= max_steps:
            break
    dt = time.time() - t0
    trainable = {k: v.detach().cpu().clone()
                 for k, v in m.state_dict().items()
                 if any(k == n_ for n_, p in m.named_parameters() if p.requires_grad)}
    del m_ref
    torch.cuda.empty_cache()
    return m, pre, trainable, steps, dt, traj


def task_invalid_of(b_acc: float, frozen_b: float, cfg: dict) -> dict:
    """판정 과제 기반 무효 조건 (D77, M1 = 2 sigma).

    무효선 = frozen_b - k * sigma.  c 경계(기준선 + 2 sigma)와 **부호만 다른 대칭**이며
    sigma 는 D46 section 1 의 등록값이므로 새 상수를 만들지 않는다.
    """
    margin = cfg["task_invalid_sigma_p"] * cfg["task_invalid_k"] / 100.0
    floor = frozen_b - margin
    return {"floor": float(floor), "margin_p": float(margin * 100.0),
            "sigma_p": float(cfg["task_invalid_sigma_p"]), "k": float(cfg["task_invalid_k"]),
            "frozen_b": float(frozen_b), "b_acc": float(b_acc),
            "fired": bool(b_acc < floor), "basis": "M1 2sigma (D77)"}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="",
                    help="결과 디렉터리. 지정하면 <outdir>/act3_pilot.json 에 쓴다")
    ap.add_argument("--mu-grid", default="",
                    help="μ 격자 override (쉼표). 팩 A 용")
    ap.add_argument("--supply-fracs", default="",
                    help="학습 쌍 수 비율 (쉼표). 팩 B 하향 민감도용. μ 는 --mu-grid 단일값")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    out_path = (Path(a.outdir) / "act3_pilot.json") if a.outdir else Path(OUT)
    wdir = Path(a.outdir) if a.outdir else Path("results/phase4")
    (ROOT / wdir).mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(ROOT / CFG_PATH))
    if a.mu_grid:
        cfg["mu_grid"] = [float(x) for x in a.mu_grid.split(",")]
    supply_fracs = ([float(x) for x in a.supply_fracs.split(",")]
                    if a.supply_fracs else [1.0])
    TG.MARGIN = cfg["crop_margin"]
    dev = "cuda"
    mem = system_memory_or_die()
    gpu = gpu_exclusive_or_die()
    prov = collect(cfg["split_seed"], cfg_path=ROOT / CFG_PATH,
                   inputs=[ROOT / "data/axes/vaw_train_referent_attr.jsonl",
                           ROOT / "data/axes/vaw_val_referent_attr.jsonl",
                           ROOT / TG_RESULT],
                   extra={"preregistered": "notes/act3-pilot-registration-draft.md (968b603)",
                          "gpu_exclusive": gpu, "system_memory_kb": mem})

    # ---- 데이터: 판정 세트 (b) 와 감독 공급 A ----
    rows_all = load_rows()
    ov, ovstat = overlap_images(rows_all)   # **튜플이다** — 집합으로 받으면
    #   `not in ov` 가 항상 참이 되어 공급이 조용히 비고, 누수 검사가 공허하게
    #   통과한다 (2026-08-24 실제로 물렸다). 아래 assert 가 그것을 막는다.
    judge = [r for r in rows_all if r["image"] not in ov]      # (b) 1048행 / 405군집
    pool = [r for r in rows_all if r["image"] in ov]           # 공급안 A 572행 / 194군집
    pids = sorted({r["pair_id"] for r in pool})
    rng = np.random.default_rng(cfg["split_seed"])
    n_ho = max(1, round(len(pids) * cfg["holdout_frac"]))
    ho = set(np.array(pids)[rng.permutation(len(pids))[:n_ho]].tolist())
    sup = [r for r in pool if r["pair_id"] not in ho]          # 학습
    hold_a = [r for r in pool if r["pair_id"] in ho]           # (a)
    print(f"판정 (b)  {len(judge)}행 / {len({r['pair_id'] for r in judge})}군집")
    print(f"공급 A    {len(pool)}행 / {len(pids)}군집")
    print(f"  학습    {len(sup)}행 / {len(pids)-n_ho}군집")
    print(f"  (a)     {len(hold_a)}행 / {n_ho}군집   ← D6 미달. 기술통계로만 (초안 §3-e)")
    # 누수 검사 — **공허해지면 실패하게** 한다 (D32: 항상 참인 검사는 검사가 아니다)
    assert len(pool) > 0 and len(sup) > 0 and len(hold_a) > 0, \
        f"공급이 비었다 (pool {len(pool)} sup {len(sup)} a {len(hold_a)}) — 검사가 공허해진다"
    assert len(judge) == 1048, f"판정 세트 {len(judge)} != 1048 (등록값)"
    assert not ({r["image"] for r in pool} & {r["image"] for r in judge}), "이미지 누수"
    print(f"L1 이미지 분리 검사 통과 — 교집합 0 (공급 {len(pool)}행 비어 있지 않음)\n",
          flush=True)

    import open_clip
    tok = open_clip.get_tokenizer("ViT-B-16-quickgelu")
    m0, pre, _ = build_model(cfg, dev)
    base_sd = {k: v.detach().cpu().clone() for k, v in m0.state_dict().items()}
    fmt = lambda rs: [cfg["template"].format(r["attr"], r["cat"]) for r in rs]
    Tj = encode_text(m0, tok, fmt(judge), dev)
    Ta = encode_text(m0, tok, fmt(hold_a), dev)
    Ts = encode_text(m0, tok, fmt(sup), dev)

    gj = np.array([r["pair_id"] for r in judge])
    ga = np.array([r["pair_id"] for r in hold_a])

    # ---- 비트 동일성 검사: frozen 을 이 경로로 다시 채점 ----
    print("판정 경로 비트 동일성 검사 …", flush=True)
    zf = embed_rows(m0, pre, judge, dev)
    ok_frozen = acc_of(zf, Tj)
    got, want = float(ok_frozen.mean()), cfg["frozen_ref_acc"]
    print(f"  frozen 재채점 {got:.16f}\n  text_gate.json {want:.16f}")
    if abs(got - want) > 1e-12:
        raise SystemExit(f"정지 — 판정 경로가 비트 동일하지 않다 (Δ {got-want:.2e})")
    print("  일치. 판정 경로 비트 동일 확인\n", flush=True)
    fro_ci = boot_ci(ok_frozen, gj, cfg["boot"], cfg["boot_seed"])
    za = embed_rows(m0, pre, hold_a, dev)
    ok_fa = acc_of(za, Ta)

    # ---- null (셔플 텍스트) ----
    perm = np.random.default_rng(0).permutation(len(judge))
    ok_null = acc_of(zf, Tj[torch.as_tensor(perm, device=dev)])

    res = {"supply": {"judge_rows": len(judge), "judge_groups": int(len(set(gj))),
                      "pool_rows": len(pool), "pool_groups": len(pids),
                      "train_rows": len(sup), "holdout_rows": len(hold_a),
                      "holdout_groups": n_ho, "holdout_pair_ids": sorted(ho)},
           "bit_identity": {"frozen_reeval": got, "registered": want, "match": True},
           "frozen": {"b_acc": got, "b_ci": fro_ci, "a_acc": float(ok_fa.mean())},
           "null_shuffled_text": float(ok_null.mean()),
           "runs": []}

    geom_cache = None
    for frac in supply_fracs:
      for mu in cfg["mu_grid"]:
        for seed in cfg["seeds"]:
            # 공급 하향 — 군집 단위로 앞에서 자른다(시드 고정 순서). 팩 B 축.
            if frac < 1.0:
                spids = sorted({r["pair_id"] for r in sup})
                keep = set(np.array(spids)[
                    np.random.default_rng(cfg["split_seed"]).permutation(len(spids))
                    [:max(1, round(len(spids) * frac))]].tolist())
                sup_f = [r for r in sup if r["pair_id"] in keep]
                Ts_f = Ts[torch.as_tensor(
                    [i for i, r in enumerate(sup) if r["pair_id"] in keep], device=dev)]
            else:
                sup_f, Ts_f = sup, Ts
            print(f"── frac={frac} μ={mu} seed={seed} 학습 "
                  f"(감독 {len(sup_f)}행)", flush=True)
            m, pre_, trainable, steps, dt, traj = train_one(
                cfg, mu, seed, sup_f, Ts_f, dev, base_sd)
            zb = embed_rows(m, pre_, judge, dev)
            ok_b = acc_of(zb, Tj)
            zah = embed_rows(m, pre_, hold_a, dev)
            ok_a = acc_of(zah, Ta)
            d, lo, hi = paired_ci(ok_b, ok_frozen, gj, cfg["boot"], cfg["boot_seed"])
            inv = invalid_checks(m, pre_, dev, geom=geom_cache)
            geom_cache = inv["geom"]          # geom 은 모델 무관 — 1회만 계산
            wp = ROOT / wdir / f"act3_pilot_mu{mu}_f{frac}_s{seed}.pt"
            torch.save({"trainable": trainable, "steps": steps,
                        "cfg_sha256": prov["config"]["sha256"]}, wp)
            r = {"mu": mu, "supply_frac": frac, "supply_rows": len(sup_f),
                 "seed": seed, "steps": steps, "sec": dt,
                 "invalid_checks": inv,
                 "task_invalid": None,          # 아래에서 채운다 (frozen 기준 필요)
                 "loss_traj": traj,
                 "b_acc": float(ok_b.mean()), "b_ci": boot_ci(ok_b, gj, cfg["boot"], cfg["boot_seed"]),
                 "delta_vs_frozen": d, "delta_ci": [lo, hi],
                 "a_acc": float(ok_a.mean()),
                 "a_ci": boot_ci(ok_a, ga, cfg["boot"], cfg["boot_seed"]),
                 "weights": str(wp.relative_to(ROOT))}
            r["task_invalid"] = task_invalid_of(r["b_acc"], got, cfg)
            res["runs"].append(r)
            print(f"   (b) {r['b_acc']*100:.2f}%  Δ {d*100:+.2f}p [{lo*100:+.2f}, {hi*100:+.2f}]"
                  f"   (a) {r['a_acc']*100:.2f}%   무효 {'발동 ' + str(inv['fired']) if inv['invalid'] else '없음'}"
                  f"  과제무효 {'발동' if r['task_invalid']['fired'] else '없음'}"
                  f"  attr {inv['attr_align']*100:.2f}  CIFAR {inv['cifar10']*100:.2f}"
                  f"   {dt:.0f}s", flush=True)
            del m
            torch.cuda.empty_cache()

    # ---- 사전 판독 기계 적용 ----
    by_mu = {}
    for key in sorted({(r["mu"], r["supply_frac"]) for r in res["runs"]}):
        mu, frac = key
        rs = [r for r in res["runs"] if r["mu"] == mu and r["supply_frac"] == frac]
        ds = [r["delta_vs_frozen"] for r in rs]
        by_mu[f"mu{mu}_f{frac}"] = {"mu": mu, "supply_frac": frac,
                          "any_invalid": any(r["invalid_checks"]["invalid"] for r in rs),
                          "mean_attr_align": float(np.mean([r["invalid_checks"]["attr_align"] for r in rs])),
                          "mean_cifar10": float(np.mean([r["invalid_checks"]["cifar10"] for r in rs])),
                          "mean_b": float(np.mean([r["b_acc"] for r in rs])),
                          "mean_delta": float(np.mean(ds)),
                          "seed_sd": float(np.std([r["b_acc"] for r in rs])),
                          "any_task_invalid": any(r["task_invalid"]["fired"] for r in rs),
                          "all_task_invalid": all(r["task_invalid"]["fired"] for r in rs),
                          "any_ci_lo_gt0": any(r["delta_ci"][0] > 0 for r in rs),
                          "all_ci_lo_gt0": all(r["delta_ci"][0] > 0 for r in rs),
                          "mean_a": float(np.mean([r["a_acc"] for r in rs]))}
    breakthrough = any(v["any_ci_lo_gt0"] for v in by_mu.values())
    # D77 M1 — 설정 무효. 격자(μ)는 점끼리 독립이라 그 점만 무효,
    # 사다리(공급 비율, 점끼리 설정이 같아야 규모 축이 성립)는 한 점이라도 발동하면 전체 무효.
    is_ladder = len(supply_fracs) > 1
    any_fired = any(r["task_invalid"]["fired"] for r in res["runs"])
    res["task_invalid_condition"] = {
        "adopted": "M1 (2 sigma)", "decision": "D77",
        "sigma_p": cfg["task_invalid_sigma_p"], "k": cfg["task_invalid_k"],
        "margin_p": cfg["task_invalid_sigma_p"] * cfg["task_invalid_k"],
        "frozen_b": got,
        "floor": got - cfg["task_invalid_sigma_p"] * cfg["task_invalid_k"] / 100.0,
        "scope": "(b) 판정 세트 전용. (a) hold-out 에는 걸지 않는다 (D75 결정 3)",
        "application": "전향 적용 (D77). D65 소급 정식 적용 없음 — 주석 부기만",
        "fired_runs": [f"mu{r['mu']}_f{r['supply_frac']}_s{r['seed']}"
                       for r in res["runs"] if r["task_invalid"]["fired"]],
        "is_ladder": is_ladder,
        "pack_invalid": bool(is_ladder and any_fired)}
    res["by_mu"] = by_mu
    res["preread"] = {
        "rule": "어느 μ든 (b) Δ CI 하한 > 0 → frozen 첫 돌파 확정 / 전 μ 겹침 → 소량 감독 불가",
        "breakthrough": breakthrough,
        "verdict": ("frozen 첫 돌파 확정 — 3막 개시" if breakthrough
                    else "전 μ 가 frozen CI 와 겹침 — '소량 감독으로도 불가' 등재")}
    dump_result(ROOT / out_path, res, prov=prov)
    print(f"\n사전 판독 — {res['preread']['verdict']}")
    print(f"저장 {out_path}")


if __name__ == "__main__":
    main()
