"""D60 ① FILIP 게이트 — 질의 **토큰** × 패치 토큰 max-sim 으로 텍스트 관문 재실행.

    python -m experiments.filip_gate

**무엇이 새로운가.** `text_gate.py` 의 `late` 병기는 패치 토큰을 쓰지만 텍스트
쪽은 **풀링된 벡터 하나**다. FILIP 의 정렬은 그것이 아니라 **텍스트 토큰마다**
패치 최대 유사도를 구해 평균하는 것이다. 이 파일이 그것을 처음 잰다.

    score(text, crop) = mean_k  max_n  <t_k, v_n>        (합이 아니라 평균)

**SOT·EOT 를 뺀다 (사전 등록).** EOT 는 CLIP 이 전역 표현으로 쓰는 바로 그
위치이므로, 넣으면 max-sim 이 전역 풀링을 부분적으로 다시 재게 된다 — 이
게이트가 가르려는 두 가지가 섞인다. 포함 변형은 진단으로 함께 낸다.

사전 판독 (D60 ①, 결과 보기 전 고정)

    frozen max-sim 이 frozen 전역 대비 **Δ CI 하한 > 0**
        → 결손의 소재가 "감독 부재" 에서 "전역 풀링" 으로 이동.
          §7.12 결론 개정 + 3막 재설계 (라벨링 전에 채점 변경이 먼저다)
    상승 없음
        → "풀링 탓" 반론 소거. 기존 결론 강화
    어느 쪽이든 §7 등재.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import experiments.text_gate as TG                                    # noqa: E402
from experiments.c_train import system_memory_or_die                  # noqa: E402
from experiments.text_gate import (MANIFEST, POINTS, boot_ci,         # noqa: E402
                                   build_model, crops_of, load_rows,
                                   overlap_images)
from src.backbones.loaders import _clip_patch_forward                 # noqa: E402
from src.scoring.base import l2_normalize                             # noqa: E402
from src.utils.provenance import collect, dump_result                 # noqa: E402
from src.utils.seed import set_seed                                   # noqa: E402

CFG_PATH = "configs/experiment/text_gate.yaml"     # 관문과 **같은** 눈금 (규칙 3)
IMG_ROOT = TG.IMG_ROOT
PROBE_N = 20            # VRAM 실측용 예열 행 수
VRAM_MARGIN = 2.0       # 실측 피크의 몇 배를 여유로 요구하는가


def vram_guard(peak_gib: float) -> dict:
    """**학습용 4 GiB 문턱을 이 스크립트에 이식하지 않는다** (D20).

    `c_train.gpu_exclusive_or_die` 의 문턱은 수 시간짜리 학습이 시스템 정지를
    내지 않게 하려고 세운 값이다. 이 게이트는 배치 2 짜리 추론이고 수 분이면
    끝난다. 그래서 **이 스크립트가 실제로 쓰는 피크를 먼저 재고** 그 배수를
    요구한다 — 임계값을 발명하는 것이 아니라 측정에서 유도한다.
    """
    import subprocess
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=30)
    used, total = (int(x) for x in out.stdout.strip().split("\n")[0].split(","))
    free_gib = (total - used) / 1024
    need = peak_gib * VRAM_MARGIN
    ok = free_gib >= need
    st = {"free_gib": round(free_gib, 2), "measured_peak_gib": round(peak_gib, 2),
          "required_gib": round(need, 2), "margin_x": VRAM_MARGIN, "passed": ok,
          "note": "학습용 4 GiB 문턱 대신 실측 피크 기준 (D60 ① 단서)"}
    if not ok:
        raise SystemExit(f"VRAM 부족 — 여유 {free_gib:.2f} GiB < 필요 {need:.2f} GiB")
    return st


def token_embed(m, tok, sents, dev, chunk=64):
    """모든 위치를 joint space 로 투영한 텍스트 토큰 (`loaders.enc_tok` 과 동일).

    **청크로 자른다** — 1,101문장 × 77토큰을 한 번에 태우면 OOM 이다(실측).
    수학은 문장마다 독립이라 청크 크기가 결과를 바꾸지 않는다.
    """
    xs, idl = [], []
    for i in range(0, len(sents), chunk):
        with torch.no_grad():
            ids = tok(list(sents[i:i + chunk])).to(dev)
            x = m.token_embedding(ids) + m.positional_embedding
            x = m.transformer(x, attn_mask=m.attn_mask)
            x = m.ln_final(x) @ m.text_projection
        xs.append(x); idl.append(ids)
        del x
    torch.cuda.empty_cache()
    return torch.cat(xs), torch.cat(idl)


def masks_of(ids):
    """`content` = pad·SOT·EOT 제외 (주). `with_eot` = EOT 포함 (진단)."""
    pad = ids != 0
    eot = ids == int(ids.max())
    sot = torch.zeros_like(pad)
    sot[:, 0] = True
    return pad & ~eot & ~sot, pad & ~sot


def maxsim(tv, tm, pv):
    """`mean_k max_n <t_k, v_n>` → `[2]`. `tv [T,d]` · `tm [T]` · `pv [2,N,d]`."""
    sim = torch.einsum("td,cnd->ctn", tv, pv)          # [2, T, N]
    mx = sim.max(-1).values                            # [2, T]
    return (mx * tm).sum(-1) / tm.sum().clamp(min=1)


def main() -> None:
    import argparse, yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="filip_gate")
    ap.add_argument("--limit", type=int, default=0, help="스모크용")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(CFG_PATH))
    TG.MARGIN = cfg["crop_margin"]
    tpl, boot = cfg["template"], cfg["boot"]
    set_seed(cfg["boot_seed"])
    dev = "cuda"
    mem = system_memory_or_die()

    rows_all = load_rows()
    ov, ovstat = overlap_images(rows_all)
    rows = [r for r in rows_all if r["image"] not in ov]
    if a.limit:
        rows = rows[:a.limit]
    man = json.load(open(MANIFEST))
    hold = set(man["pair_ids"])
    assert man["sha256"].startswith("26729b84"), "명부 sha256 불일치"
    groups = np.array([r["pair_id"] for r in rows])
    hmask = np.array([r["pair_id"] in hold for r in rows])
    print(f"명부 검증 통과 · 행 {len(rows):,} · 군집 {len(np.unique(groups)):,} · "
          f"홀드아웃 {len(np.unique(groups[hmask]))}쌍", flush=True)

    import open_clip
    tok = open_clip.get_tokenizer("ViT-B-16-quickgelu")
    m0, pre = build_model(dev)
    sents = [tpl.format(r["attr"], r["cat"]) for r in rows]
    T_tok, ids = token_embed(m0, tok, sents, dev)
    T_tok = l2_normalize(T_tok)
    m_content, m_witheot = masks_of(ids)
    with torch.no_grad():
        T_pool = l2_normalize(torch.cat([m0.encode_text(tok(sents[i:i + 64]).to(dev))
                                         for i in range(0, len(sents), 64)]))
    perm = np.random.default_rng(0).permutation(len(rows))
    print(f"텍스트 토큰 — 내용 토큰 중앙 {int(m_content.sum(1).median())}개 "
          f"(EOT 포함 시 {int(m_witheot.sum(1).median())}개)", flush=True)

    torch.cuda.reset_peak_memory_stats()
    cfgs = [("frozen", None), ("pt3", POINTS[2][1])]
    KEYS = ("pool", "maxsim", "maxsim_eot", "maxsim_null")
    score = {c: {k: np.zeros((len(rows), 2), np.float32) for k in KEYS}
             for c, _ in cfgs}
    ms = {}
    guard = None
    for cname, sd in cfgs:
        m = m0 if sd is None else build_model(dev, sd)[0]
        t0, t_pool, t_max = time.time(), 0.0, 0.0
        for i, r in enumerate(rows):
            im = Image.open(Path(IMG_ROOT) / r["image"]).convert("RGB")
            px = torch.stack([pre(c) for c in crops_of(im, r)]).to(dev)
            with torch.no_grad():
                p, _ = _clip_patch_forward(m.visual, px, maskclip=False)
                p = l2_normalize(p)                          # [2, N, d]
                torch.cuda.synchronize(); s0 = time.time()
                z = l2_normalize(p.mean(1))                  # norm → mean → norm
                sp = (z @ T_pool[i])
                torch.cuda.synchronize(); t_pool += time.time() - s0
                s0 = time.time()
                sm = maxsim(T_tok[i], m_content[i].float(), p)
                torch.cuda.synchronize(); t_max += time.time() - s0
                se = maxsim(T_tok[i], m_witheot[i].float(), p)
                sn = maxsim(T_tok[perm[i]], m_content[perm[i]].float(), p)
            score[cname]["pool"][i] = sp.cpu().numpy()
            score[cname]["maxsim"][i] = sm.cpu().numpy()
            score[cname]["maxsim_eot"][i] = se.cpu().numpy()
            score[cname]["maxsim_null"][i] = sn.cpu().numpy()
            if i == PROBE_N and guard is None:
                guard = vram_guard(torch.cuda.max_memory_allocated() / 2 ** 30)
                print(f"VRAM 가드 통과 — 여유 {guard['free_gib']} GiB ≥ "
                      f"필요 {guard['required_gib']} GiB "
                      f"(실측 피크 {guard['measured_peak_gib']})", flush=True)
            if i % 400 == 0:
                print(f"    {cname} {i}/{len(rows)}  {time.time()-t0:.0f}s", flush=True)
        ms[cname] = {"pool_ms_per_pair": t_pool / len(rows) * 1000,
                     "maxsim_ms_per_pair": t_max / len(rows) * 1000}
        if sd is not None:
            del m; torch.cuda.empty_cache()
        print(f"  {cname} 완료 {time.time()-t0:.0f}s", flush=True)

    ok = {c: {k: (v[:, 0] > v[:, 1]).astype(np.float64) for k, v in d.items()}
          for c, d in score.items()}

    # 양성 대조군 — 패치를 정답 텍스트 토큰으로 치환. max-sim 기계가 살아 있는가
    g = torch.Generator(device=dev).manual_seed(0)
    pos_hit = []
    for i in range(len(rows)):
        tv, tm = T_tok[i], m_content[i].float()
        good = l2_normalize(tv + cfg["pos_sigma"] * 0.1
                            * torch.randn(tv.shape, generator=g, device=dev))
        bad = l2_normalize(T_tok[perm[i]] + cfg["pos_sigma"] * 0.1
                           * torch.randn(tv.shape, generator=g, device=dev))
        pv = torch.stack([good, bad])
        pos_hit.append(float(maxsim(tv, tm, pv)[0] > maxsim(tv, tm, pv)[1]))
    pos = float(np.mean(pos_hit))
    print(f"\n양성 대조군 {pos*100:.1f}%  (하한 {cfg['pos_floor']*100:.0f}%) → "
          f"{'통과' if pos >= cfg['pos_floor'] else '**무효**'}")

    res = {"n_rows": len(rows), "n_groups": int(len(np.unique(groups))),
           "n_holdout_pairs": int(len(np.unique(groups[hmask]))),
           "precheck": ovstat, "vram_guard": guard, "timing_ms": ms,
           "positive_control": pos, "pos_floor": cfg["pos_floor"],
           "content_tokens_median": int(m_content.sum(1).median()),
           "acc": {}, "delta": {}}
    print(f"\n{'구성':<22}{'전체 405G':>12}{'홀드아웃':>12}")
    for c, _ in cfgs:
        res["acc"][c] = {}
        for k in KEYS:
            v = ok[c][k]
            lo, hi = boot_ci(lambda s, v=v: v[s].mean(), groups, boot)
            hv = v[hmask]
            hlo, hhi = boot_ci(lambda s, hv=hv: hv[s].mean(), groups[hmask], boot)
            res["acc"][c][k] = {"all": float(v.mean()), "all_ci": [lo, hi],
                                "holdout": float(hv.mean()), "holdout_ci": [hlo, hhi]}
            print(f"{c+'/'+k:<22}{v.mean()*100:>10.1f}%"
                  f"{hv.mean()*100:>11.1f}%   [{lo*100:.1f}, {hi*100:.1f}]")

    print(f"\n{'짝지은 Δ':<34}{'Δ':>8}{'CI95':>20}{'판정':>10}")
    for c, _ in cfgs:
        for k in ("maxsim", "maxsim_eot"):
            d = ok[c][k] - ok[c]["pool"]
            lo, hi = boot_ci(lambda s, d=d: d[s].mean(), groups, boot)
            up = lo > 0
            res["delta"][f"{c}_{k}_vs_pool"] = {"delta": float(d.mean()),
                                                "ci95": [lo, hi],
                                                "ci_lower_positive": bool(up)}
            print(f"{c+' '+k+' − pool':<34}{d.mean()*100:>+7.1f}p"
                  f"  [{lo*100:+6.1f}, {hi*100:+6.1f}]p"
                  f"{('상승' if up else '상승 없음'):>10}")

    main_up = res["delta"]["frozen_maxsim_vs_pool"]["ci_lower_positive"]
    res["verdict"] = {
        "frozen_maxsim_beats_pool": main_up,
        "reading": ("결손의 소재가 '감독 부재' 에서 '전역 풀링' 으로 이동 — "
                    "§7.12 결론 개정 + 3막 재설계" if main_up else
                    "'풀링 탓' 반론 소거 — 기존 결론 강화"),
        "preregistered": "D60 ①"}
    print(f"\n사전 판독 → {res['verdict']['reading']}")
    print(f"\n[채점 비용] " + "  ".join(
        f"{c}: pool {ms[c]['pool_ms_per_pair']:.3f} ms / "
        f"maxsim {ms[c]['maxsim_ms_per_pair']:.3f} ms (쌍당)" for c, _ in cfgs))
    prov = collect(cfg["boot_seed"], cfg_path=CFG_PATH,
                   inputs=["data/axes/vaw_train_referent_attr.jsonl",
                           "data/axes/vaw_val_referent_attr.jsonl",
                           MANIFEST, TG.VG_META, POINTS[2][1]],
                   extra={"preregistered": "D60 ①", "template": tpl,
                          "token_mask": "pad·SOT·EOT 제외 (주). EOT 포함은 진단",
                          "gpu_shared": True, "vram_guard": guard,
                          "system_memory_kb": mem, "precheck": ovstat})
    dump_result(Path(f"results/phase4/{a.tag}.json"), res, prov=prov)


if __name__ == "__main__":
    main()
