"""D88 큐 2·4 — 맥락 특정 **조합 채점**. 학습 0회 (기존 가중치 재사용).

    python -m experiments.context_score --split grid     # 큐 2: 격자 선정
    python -m experiments.context_score --split judge --lambda-d L --t-soft T   # 큐 4: 본 채점

채점 (D84 §1-b, 사전 등록)
    프레임 1회 forward → 패치 격자 [196, d]        ← **dense readout.** crop N-forward 아님
    score(k) = mean_tok max_{n ∈ box_k} <t_tok, v_n>       (D60 maxsim 과 같은 연산)
    c_a      = Σ_n softmax(H_a / T_soft)(n) · coord(n)     soft-argmax 앵커 좌표
    선택     = argmax_k [ score_t(k) − λ_d · dist(center_k, c_a) ]   dist 는 대각선 정규화

arm (D84 §1-c · D85 §2)
    T-only   argmax_k score_t(k)                  속성만 — 설계상 chance 여야 한다
    G-only   argmax_k (−dist_k)                   **주 판정의 분모**
    A-only   argmax_k score_a(k)                  앵커 문구만
    조합     위 식                                 **본 판정**
    GT앵커   c_a 를 주석 앵커 박스 중심으로 대체    앵커 국소화 실패 분리 (D84 §1-d)

**두 arm 이 같은 c_a 를 쓴다** — D85 §2 가 요구한 전제. 구현에 고정돼 있다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import os
import torch
from PIL import Image

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.act3_pilot import boot_ci, paired_ci                  # noqa: E402
from experiments.c_train import system_memory_or_die                   # noqa: E402
from experiments.filip_gate import masks_of, maxsim, token_embed       # noqa: E402
from src.backbones.loaders import _clip_patch_forward                  # noqa: E402
from src.scoring.base import l2_normalize                              # noqa: E402
from src.utils.provenance import collect, dump_result                  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = "configs/experiment/context_score.yaml"
SET_JSONL = "data/axes/context_set.jsonl"
MANIFEST = "results/phase4/context_set_manifest.json"
IMG_ROOT = "data/vg"


def patch_coords(g):
    """패치 중심의 정규화 좌표 [N,2] (x, y)."""
    j, i = np.meshgrid(np.arange(g), np.arange(g))
    return np.stack([(j.ravel() + .5) / g, (i.ravel() + .5) / g], 1)


def box_mask(box, pc):
    """박스 안에 중심이 든 패치. 없으면 가장 가까운 패치 1개."""
    x0, y0, x1, y1 = box
    m = (pc[:, 0] >= x0) & (pc[:, 0] <= x1) & (pc[:, 1] >= y0) & (pc[:, 1] <= y1)
    if not m.any():
        c = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        m = np.zeros(len(pc), bool); m[int(np.argmin(((pc - c) ** 2).sum(1)))] = True
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["grid", "judge", "anchor"])
    ap.add_argument("--lambda-d", type=float, default=None)
    ap.add_argument("--t-soft", type=float, default=None)
    ap.add_argument("--out", default="")
    # 세트 경로 인자화 — 기본값은 v1 이라 기존 산출의 재현성이 불변이다 (D92)
    ap.add_argument("--set-jsonl", default=SET_JSONL)
    ap.add_argument("--manifest", default=MANIFEST)
    a = ap.parse_args()
    import yaml
    cfg = yaml.safe_load(open(ROOT / CFG_PATH))
    dev = "cuda"
    mem = system_memory_or_die(profile="no_worker")

    man = json.load(open(ROOT / a.manifest))
    sha = hashlib.sha256((ROOT / a.set_jsonl).read_bytes()).hexdigest()
    assert sha == man["jsonl_sha256"], "세트 파일이 명부와 다르다"
    rows = [json.loads(l) for l in open(ROOT / a.set_jsonl)]
    rows = [r for r in rows if r["split"] == a.split]
    groups = np.array([r["pair_id"] for r in rows])
    assert len(rows) == man["splits"][a.split]["n_rows"], "분할 행 수 불일치"
    print(f"명부 검증 통과 · {a.split} {len(rows)}행 / {len(np.unique(groups))}군집",
          flush=True)

    out_path = Path(a.out) if a.out else Path(
        f"results/phase4/context_score_{a.split}.json")
    prov = collect(cfg["boot_seed"], cfg_path=ROOT / CFG_PATH,
                   inputs=[ROOT / a.set_jsonl, ROOT / a.manifest],
                   extra={"preregistered": "notes/decisions.md D84 §1 · D85 §2 · D88",
                          "manifest_sha256": man["sha256"], "system_memory": mem})

    import open_clip
    from torchvision import transforms as TF
    m, _, _ = open_clip.create_model_and_transforms(
        cfg["backbone"], pretrained=cfg["pretrained"], device=dev)
    m.eval()
    tok = open_clip.get_tokenizer(cfg["backbone"])
    S = cfg["image_size"]
    # CenterCrop 없이 전체 프레임을 정사각 리사이즈 — 좌표가 패치 격자에 선형 대응
    pre = TF.Compose([TF.Resize((S, S), interpolation=TF.InterpolationMode.BICUBIC),
                      TF.ToTensor(),
                      TF.Normalize((0.48145466, 0.4578275, 0.40821073),
                                   (0.26862954, 0.26130258, 0.27577711))])

    # 토큰 임베딩을 **CPU 에 둔다** — 공존 시 GPU OOM 회피. 문장마다 독립이라
    #   수학은 동일하고 배치 경계가 값에 개입하지 않는다 (D90 선례)
    Tt, ids_t = token_embed(m, tok, [r["phrase_target"] for r in rows], dev)
    Tt = l2_normalize(Tt).cpu(); mt, _ = masks_of(ids_t); mt = mt.cpu()
    del ids_t; torch.cuda.empty_cache()
    Ta, ids_a = token_embed(m, tok, [r["phrase_anchor"] for r in rows], dev)
    Ta = l2_normalize(Ta).cpu(); ma, _ = masks_of(ids_a); ma = ma.cpu()
    del ids_a; torch.cuda.empty_cache()

    g = S // 16
    pc = patch_coords(g)
    n = len(rows)
    st = np.zeros((n, 2), np.float32)      # 대상 문구 점수 (정답, 오답)
    sa = np.zeros((n, 2), np.float32)      # 앵커 문구 점수
    Ha = np.zeros((n, len(pc)), np.float32)  # 앵커 heatmap (전 패치)
    cen = np.zeros((n, 2, 2), np.float32)  # 후보 박스 중심
    gt_a = np.zeros((n, 2), np.float32)    # GT 앵커 중심

    for i, r in enumerate(rows):
        im = Image.open(ROOT / IMG_ROOT / r["image"]).convert("RGB")
        px = pre(im).unsqueeze(0).to(dev)
        with torch.no_grad():
            pv, _ = _clip_patch_forward(m.visual, px, maskclip=False)
            pv = l2_normalize(pv)[0]                       # [N, d]
            sim_t = torch.einsum("td,nd->tn", Tt[i].to(dev), pv)   # [T, N]
            sim_a = torch.einsum("td,nd->tn", Ta[i].to(dev), pv)
        s_t = sim_t.cpu().numpy(); s_a = sim_a.cpu().numpy()
        wt = mt[i].float().numpy(); wa = ma[i].float().numpy()
        for k, key in enumerate(("correct_box", "wrong_box")):
            bm = box_mask(r[key], pc)
            st[i, k] = ((s_t[:, bm].max(1) * wt).sum() / max(wt.sum(), 1))
            sa[i, k] = ((s_a[:, bm].max(1) * wa).sum() / max(wa.sum(), 1))
            b = r[key]; cen[i, k] = [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]
        Ha[i] = (s_a * wa[:, None]).sum(0) / max(wa.sum(), 1)   # 토큰 평균 heatmap
        b = r["anchor_box"]; gt_a[i] = [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{n}", flush=True)

    def c_from(T_soft):
        z = Ha / T_soft
        z = z - z.max(1, keepdims=True)
        w = np.exp(z); w /= w.sum(1, keepdims=True)
        return w @ pc                                        # [n, 2]

    DIAG = float(np.sqrt(2.0))                               # 정규화 좌표의 대각선
    def dists(ca):
        return np.linalg.norm(cen - ca[:, None, :], axis=2) / DIAG   # [n, 2]

    def acc(sel):
        return (sel == 0).astype(np.float64)                 # 0 = correct_box

    res = {"preregistered": "notes/decisions.md D84 §1 · D85 §2 · D87 §1 · D88",
           "split": a.split, "rows": len(rows),
           "groups": int(len(np.unique(groups))),
           "manifest_sha256": man["sha256"],
           "readout": "dense — 프레임당 forward 1회, 패치 격자 채점 (crop N-forward 아님)"}

    if a.split == "grid":
        # ---- 큐 2: 격자 선정. **judge 를 쓰지 않는다** ----
        tab = []
        for ts in cfg["t_soft_grid"]:
            ca = c_from(ts); d = dists(ca)
            for ld in cfg["lambda_d_grid"]:
                sel = np.argmin(-(st - ld * d), axis=1)
                tab.append({"t_soft": ts, "lambda_d": ld,
                            "combined_acc": float(acc(sel).mean())})
        best = max(tab, key=lambda x: (x["combined_acc"], -x["lambda_d"], x["t_soft"]))
        res["grid_table"] = tab
        res["selected"] = {k: best[k] for k in ("lambda_d", "t_soft", "combined_acc")}
        res["tie_break"] = cfg["tie_break"]
        res["note"] = ("이 분할은 **격자 선정 전용**이며 판정에 쓰지 않는다. "
                       "재열람은 새 사전 등록이다 (D87 병행 B)")
        print(f"\n선정 λ_d={best['lambda_d']} · T_soft={best['t_soft']}", flush=True)
    else:
        ld, ts = a.lambda_d, a.t_soft
        assert ld is not None and ts is not None, "--lambda-d · --t-soft 필수 (grid 등록값)"
        ca = c_from(ts); d = dists(ca)
        d_gt = np.linalg.norm(cen - gt_a[:, None, :], axis=2) / DIAG
        arms = {
            "T_only":   np.argmin(-st, axis=1),
            "G_only":   np.argmin(d, axis=1),
            "A_only":   np.argmin(-sa, axis=1),
            "combined": np.argmin(-(st - ld * d), axis=1),
            "combined_gt_anchor": np.argmin(-(st - ld * d_gt), axis=1)}
        ok = {k: acc(v) for k, v in arms.items()}
        res["params"] = {"lambda_d": ld, "t_soft": ts, "source": "grid 분할 등록값"}
        res["arms"] = {k: {"acc": float(v.mean()),
                           "ci": boot_ci(v, groups, cfg["boot"], cfg["boot_seed"])}
                       for k, v in ok.items()}
        def pc_(x, y):
            dd, lo, hi = paired_ci(ok[x], ok[y], groups, cfg["boot"], cfg["boot_seed"])
            return {"delta": dd, "ci": [lo, hi], "ci_lower_positive": bool(lo > 0)}
        res["main_judgment"] = {
            "rule": "Δ = acc(조합) − acc(G-only), 군집 부트스트랩 CI 하한 > 0 (D85 §2)",
            "combined_vs_G_only": pc_("combined", "G_only"),
            "combined_vs_T_only": pc_("combined", "T_only"),
            "combined_vs_A_only": pc_("combined", "A_only"),
            "gt_anchor_vs_pred_anchor": pc_("combined_gt_anchor", "combined")}
        # 앵커 국소화 진단 — soft-argmax 좌표가 GT 앵커 박스 안에 드는가
        inb = np.array([1.0 if (r["anchor_box"][0] <= ca[i, 0] <= r["anchor_box"][2]
                                and r["anchor_box"][1] <= ca[i, 1] <= r["anchor_box"][3])
                        else 0.0 for i, r in enumerate(rows)])
        res["anchor_localization"] = {
            "soft_argmax_in_gt_box": float(inb.mean()),
            "ci": boot_ci(inb, groups, cfg["boot"], cfg["boot_seed"])}
        res["preread_rule"] = {
            "source": "D87 §1 — 적용은 사용자 확인 후. 내가 갈래를 선언하지 않는다",
            "갈래1_결합성립": "Δ(조합−G_only) CI 하한 > 0 그리고 조합이 A_only·T_only 모두 초과",
            "갈래2_결합불성립": "Δ CI 가 0 포함 또는 음수",
            "갈래3_앵커병목": "GT앵커 arm 과 예측앵커 arm 의 격차가 유의 (GT ≫ 예측)",
            "동시발동": "1·3 또는 2·3 이 함께 발동하면 **3 을 먼저 처리**",
            "문구규칙": ("조합 정확도 X% (기하 단독 상한 Y%, Δ +Zp [CI]) — "
                        "기하 단독 상한 없이 조합 절대 성적만 인용 금지 (D85 §2)")}
    dump_result(ROOT / out_path, res, prov=prov)
    print(f"\n저장 {out_path}")


if __name__ == "__main__":
    main()
