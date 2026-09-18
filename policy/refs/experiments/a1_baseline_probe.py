"""A1 — 판정 눈금 재앵커. 실행 1 출발점 CLIP 의 **오라클 70쌍 프로브** 실측.

    python -m experiments.a1_baseline_probe --seeds 0,1,2

문제 (사용자 지적) — D30 의 3구간은 `FROZEN_BEST=64.8%`(warm-start 슬롯 헤드
학습 시스템)에 앵커돼 있는데 C 계획의 판정은 **오라클 세트 프로브**다. D28 의
프로브 기준선은 같은 70쌍에서 60.0% 라 눈금이 어긋난다. 이대로면 프로브 +6~8p
실질 회수가 c 구간(잡음)으로 판정된다.

**재현 범위의 한계를 먼저 적는다** — D28 이 "프로브 60.0%" 를 낸 코드는 저장소에
없다(그 세션의 임시 계산). 그래서 **동일 코드가 아니라 동일 기계장치**로 재구성한다:
`sweep_ceiling` 의 특징 추출(4×4 셀)·라벨 뒤집기·프로브 클래스(linear/MLP/attn)·
학습 루프를 그대로 import 해서 쓰고, 다른 것은 **평가 분할뿐**이다.

    sweep    group CV 5-fold, 전체 1,620행에서 평균
    A1       오라클 70쌍을 test 로 고정, 나머지로 학습(내부 val 분리)

따라서 A1 값은 D28 의 60.0% 와 **직접 비교하지 않는다.** A1 이 만드는 것은
이번 판정에 쓸 **자체 눈금**이고, 그것이 이 게이트의 목적이다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.sweep_ceiling import (CELL, MLP, MODELS,  # noqa: E402
                                       AttnProbe, extract)
from src.backbones.loaders import load_backbone            # noqa: E402
from src.scoring.base import l2_normalize                  # noqa: E402
from src.utils.provenance import collect, dump_result      # noqa: E402

CFG_PATH = "configs/experiment/phase4_track2.yaml"
MANIFEST = "results/phase4/holdout60_manifest.json"
JUDGE_HEAD = "mlp"          # D35 — attn 은 상수예측 붕괴, linear 는 앵커 헤드 아님

_man = json.loads(Path(MANIFEST).read_text())
HUMAN = _man["human_acc"]   # 88.3% (홀드아웃 60쌍). 70쌍의 85.7% 와 다른 표본이다


def load_rows():
    rows = [json.loads(l) for l in open("data/axes/vaw_train_referent_attr.jsonl")]
    rows += [json.loads(l) for l in open("data/axes/vaw_val_referent_attr.jsonl")]
    return rows


def oracle_pair_ids():
    """동결 명부에서 읽고 sha256 을 검증한다 (D35). 재계산하지 않는다."""
    import hashlib
    ids = sorted(_man["pair_ids"])
    h = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    if h != _man["sha256"]:
        raise SystemExit(f"명부 해시 불일치 {h[:16]} != {_man['sha256'][:16]} — 판정 무효")
    print(f"명부 검증 통과  {len(ids)}쌍  sha256 {h[:16]}…")
    return set(ids)


def train_eval(make, feats, y, tr_i, va_i, te_i, seed, epochs=120, lr=3e-4):
    """sweep_ceiling.torch_cv 와 같은 루프. 분할만 밖에서 준다."""
    torch.manual_seed(seed)
    m = make().cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-2)
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
                pv = m(*[x[va_i].cuda() for x in feats])
                a = float(((pv > 0).float().cpu() == y[va_i]).float().mean())
            if a > best_va:
                best_va = a
                best_state = {k_: v.detach().clone() for k_, v in m.state_dict().items()}
    m.load_state_dict(best_state)
    m.eval()
    with torch.no_grad():
        pr = m(*[x[te_i].cuda() for x in feats])
        acc = float(((pr > 0).float().cpu() == y[te_i]).float().mean())
    del m
    torch.cuda.empty_cache()
    return acc


def features(model, rows, dev, enc_override=None, cache=True):
    """격자 특징 `[n,2,16,d]` 와 문구 텍스트 임베딩. 미세조정 모델은 `enc_override`.

    텍스트 타워는 **학습 대상이 아니므로**(트랙 2 목적함수에 텍스트가 없다) 캐시를
    공유한다. 비전 쪽만 갈린다."""
    cache_dir = Path("data/ceiling_cache/attr")
    feat = extract(model, MODELS.get(model, MODELS["clip_vitb16_crop"]), rows,
                   "data/vg", dev, cache_dir, enc_override=enc_override, cache=cache)
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
        TX = torch.cat(es); torch.save(TX, tp); del tb; torch.cuda.empty_cache()
    return feat["grid"], TX


def judge(rows, groups, G, TX, te_mask, seeds, heads=("linear", "mlp", "attn")):
    """A1-c 판정 본체. **특징만 밖에서 받는다** — 라벨 규칙·분할·헤드·학습 루프는
    기준선과 비트 단위로 같은 코드다. C 실행 1 의 점별 판정이 이것을 그대로 쓴다."""
    # 라벨: sweep 과 **완전히 같은 규칙**(rng 0, 정확히 절반)
    rng = np.random.default_rng(0)
    flip = torch.zeros(len(rows), dtype=torch.bool)
    flip[:len(rows) // 2] = True
    flip = flip[torch.from_numpy(rng.permutation(len(rows)))]
    Y = flip.float()

    gs = torch.where(flip[:, None, None, None], G[:, [1, 0]], G)
    ps = F.normalize(gs.mean(2), dim=-1)
    pdz = ps[:, 0] - ps[:, 1]
    xz = torch.cat([pdz, TX], -1)
    xz = (xz - xz.mean(0)) / xz.std(0).clamp_min(1e-8)

    te_i = torch.tensor(np.where(te_mask)[0])
    rest = np.where(~te_mask)[0]
    out = {"n_rows": len(rows), "n_oracle_rows": int(te_mask.sum()),
           "n_oracle_pairs": len(set(groups[te_mask])), "human": HUMAN, "per_seed": {}}

    for head in heads:
        accs = []
        for s in seeds:
            # 학습/내부검증을 pair 단위로 가른다 — 같은 pair 가 양쪽에 가면 누수
            rs = np.random.default_rng(s)
            ug = np.unique(groups[rest]); rs.shuffle(ug)
            va_g = set(ug[:max(1, len(ug) // 5)])
            va_i = torch.tensor([i for i in rest if groups[i] in va_g])
            tr_i = torch.tensor([i for i in rest if groups[i] not in va_g])
            if head == "linear":
                mk = lambda d=xz.shape[1]: torch.nn.Sequential(
                    torch.nn.Linear(d, 1), torch.nn.Flatten(0))
                fe = (xz, gs, TX)
                mk_ = lambda: _Lin(xz.shape[1])
                acc = train_eval(mk_, fe, Y, tr_i, va_i, te_i, s)
            elif head == "mlp":
                acc = train_eval(lambda d=xz.shape[1]: MLP(d), (xz, gs, TX),
                                 Y, tr_i, va_i, te_i, s)
            else:
                acc = train_eval(lambda: AttnProbe(G.shape[-1], TX.shape[1]),
                                 (pdz, gs, TX), Y, tr_i, va_i, te_i, s)
            accs.append(acc)
            print(f"  {head:<6} seed {s}  오라클 70쌍 {acc*100:.1f}%  "
                  f"(사람 대비 {acc/HUMAN*100:.1f}%)", flush=True)
        out["per_seed"][head] = accs
        out[head] = {"mean": float(np.mean(accs)), "std": float(np.std(accs)),
                     "ratio_mean": float(np.mean(accs)) / HUMAN}
    return out


def oracle_split(rows):
    """동결 명부로 판정 분할을 만든다. 매칭이 60쌍 미만이면 판정 무효."""
    groups = np.array([r["pair_id"] for r in rows])
    opids = oracle_pair_ids()
    te_mask = np.isin(groups, list(opids))
    print(f"세트 {len(rows)}행 / pair {len(set(groups))}개")
    print(f"오라클 판정 pair {len(opids)}개 → 세트 내 매칭 행 {int(te_mask.sum())} "
          f"/ 매칭 pair {len(set(groups[te_mask]))}", flush=True)
    if len(set(groups[te_mask])) < 60:
        raise SystemExit("**중단** — 오라클 pair 가 세트에 충분히 매칭되지 않는다.")
    return groups, te_mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="clip_vitb16")
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    dev = "cuda"

    rows = load_rows()
    prov = collect(seeds[0], cfg_path=CFG_PATH,
                   inputs=["data/axes/vaw_train_referent_attr.jsonl",
                           "data/axes/vaw_val_referent_attr.jsonl",
                           "results/oracle/items.json",
                           "results/oracle/responses.jsonl"],
                   extra={"effective_args": vars(a), "seeds": seeds})
    groups, te_mask = oracle_split(rows)
    G, TX = features(a.model, rows, dev)
    out = judge(rows, groups, G, TX, te_mask, seeds)
    out["model"] = a.model

    best_head = JUDGE_HEAD          # D35 로 고정. 최고치 선택 아님
    base, sd = out[best_head]["mean"], out[best_head]["std"]
    out["anchor"] = {
        "head": best_head, "BASELINE_PROBE": base, "sigma": sd,
        "rule": "c 경계 = BASELINE_PROBE + 2σ (미만이면 c). a = 사람 대비 ≥85%",
        "c_boundary_abs": base + 2 * sd,
        "c_boundary_ratio": (base + 2 * sd) / HUMAN,
        "a_boundary_abs": 0.85 * HUMAN, "a_boundary_ratio": 0.85,
        "head_selection_note": ("D35 로 mlp 고정. 최고치 선택이 아니다 — attn 은 "
                               "상수예측 붕괴(σ=0), linear 는 앵커 헤드가 아니다."),
        "human_source": f"{MANIFEST} (홀드아웃 60쌍, 85.7%/70쌍과 다른 표본)",
    }
    dump_result(Path("results/phase4/a1_baseline_probe.json"), out, prov=prov)

    print(f"\n{'='*68}\nA1 — 재앵커 결과 ({a.model}, 시드 {seeds})\n")
    for h in ("linear", "mlp", "attn"):
        print(f"  {h:<6} {out[h]['mean']*100:5.1f}% ± {out[h]['std']*100:.1f}  "
              f"사람 대비 {out[h]['ratio_mean']*100:5.1f}%")
    print(f"\n  BASELINE_PROBE = {base*100:.1f}% ({best_head}), σ = {sd*100:.1f}p")
    print(f"  c 구간: 사람 대비 < {(base+2*sd)/HUMAN*100:.1f}%  (절대 {(base+2*sd)*100:.1f}%)")
    print(f"  b 구간: {(base+2*sd)/HUMAN*100:.1f}% ~ 85.0%")
    print(f"  a 구간: >= 85.0%  (절대 {0.85*HUMAN*100:.1f}%)")


class _Lin(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.f = torch.nn.Linear(d, 1)

    def forward(self, x, *_):
        return self.f(x).squeeze(-1)


if __name__ == "__main__":
    main()
