"""C 실행 1 — 라벨-프리 인스턴스 대조 미세조정. 규모 사다리 3점.

    python -m experiments.c_train --measure-only          # 승인 조건: VRAM·시간 실측
    python -m experiments.c_train --ladder 3500,35000,0   # 0 = 전량

**사전 등록된 하나의 실험이다** (D34: 사다리 3점 = 설계 변경 1회).
설정은 `notes/plan-C-labelfree-v2.md` 에 고정돼 있고 학습 중 궤적을 보고 바꾸지
않는다. 조기 중단은 무효 조건 발동 시에만.

구성 (v2 §2~§4)

    readout   인스턴스별 독립 crop → 224 정사각 → crop 내부 mask-pool   (D36)
    증강      지터 0.30 → mask 오려내기(erosion 2) → 배경 스왑 p=1.0    (D37, A3-c)
    negative  같은 이미지의 같은 카테고리 다른 인스턴스 **전용**
    범위      비전 타워 마지막 2블록

부수 장치 (2026-08-14 추가, 사전 등록 **학습 설정**은 불변)

    착수 가드       VRAM 여유 < 4 GiB 이거나 시스템 메모리가 얕으면(SwapFree
                    < 1 GiB, MemAvailable < 4 GiB) **시작을 거부**한다. 둘 다
                    fail-closed — 확인 불가를 통과로 치지 않는다 (v2 §9, D42).
    점별 가중치 저장  점마다 학습 종료 시점의 **학습 대상 파라미터만** 1회 저장.
                    **사후 진단 전용이다 — 이어-학습에 쓰지 않는다.** 각 점은
                    반드시 `base_sd` 에서 다시 출발해야 하며(규모 축이 스텝 축과
                    섞이는 것을 막는다), 중단된 점은 처음부터 재실행한다 (D31).
                    이 파일이 없어서 D-1 진단이 재학습을 요구할 뻔했다.
    워커 회수       `persistent_workers=False` + 점 전환 시 `shutdown_loader()`.
                    로더를 루프 밖으로 뺄 수 없는 이유는 `make_loader` 참조 (D42).

## 실행은 tmux 안에서 — Claude Code 세션과 학습의 생존을 분리한다 (D42)

    tmux new -s edge_vlm        # 세션 만들기 (없을 때)
    tmux ls                     # 세션 목록
    tmux attach -t edge_vlm     # 붙기
    Ctrl+b 누른 뒤 d            # 분리 (학습은 계속 돈다)

세션 안에서 기동하면 터미널이나 에이전트 세션이 끊겨도 학습이 살아남는다.
이 프로젝트는 세션 중단으로 학습을 두 번 잃었다.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.a3b2_bgswap import (compose_or_keep,        # noqa: E402
                                     jitter_box)
from src.utils.provenance import collect, dump_result      # noqa: E402
from src.utils.seed import set_seed                        # noqa: E402

CFG_PATH = "configs/experiment/phase4_c.yaml"
PAIRS = "data/axes/c_pairs.jsonl"
SEGS = "data/axes/c_segs.pkl"
BANK = "data/axes/c_bank.json"
MANIFEST = "results/phase4/holdout60_manifest.json"
IMG_ROOT = "data/coco/train2017"


MIN_FREE_GIB = 4.0
"""공존 허용 문턱 (v2 §9, 2026-08-14 재개정 · 사용자 결정).

**`configs/` 에 두지 않는다.** 실험 하이퍼파라미터가 아니라 착수 가드의
운영 상수이고, 사전 등록된 `phase4_c.yaml` 의 sha256(`135a6521…`)을 바꾸면
그 자체로 사전 등록이 깨지기 때문이다. 결과에는 영향이 없다 — 통과 여부만
가른다. 실측 peak 는 1.63 GiB 이므로 4 GiB 는 약 2.5배 여유다.
"""


def make_loader(rows, cfg, preprocess, segs, bank, seed):
    """사다리 한 점의 DataLoader. **`persistent_workers=False` 가 규약이다 (D42).**

    스테이지 루프 **밖으로 뺄 수 없다** — 점마다 데이터 부분집합이 다르고
    (`rows[order[:n_pairs]]`), 그 차이가 곧 규모 축이다. 하나의 로더를 공유하면
    사다리가 규모를 재지 않는다. 그래서 대안 조항(`persistent_workers=False` +
    명시적 정리)을 쓴다.
    """
    return DataLoader(PairData(rows, cfg, preprocess, segs, bank, seed),
                      batch_size=cfg["pairs_per_batch"], shuffle=True,
                      num_workers=cfg["workers"], drop_last=True,
                      persistent_workers=False)


def shutdown_loader(*objs) -> None:
    """워커를 명시적으로 정리한다 (D42). 이터레이터를 먼저, 로더를 나중에."""
    import gc
    for o in objs:
        sd = getattr(o, "_shutdown_workers", None)
        if callable(sd):
            try:
                sd()
            except Exception:                      # 이미 정리된 경우
                pass
    gc.collect()


PSI_FULL_CEIL = 1.0
"""`no_worker` 프로파일의 스래싱 상한 (memory PSI full avg60, %). D81."""


def system_memory_or_die(profile: str = "worker") -> dict:
    """시스템 메모리 가드 (D42 · D81 재유도). **fail-closed.**

    GPU VRAM 만 보던 가드가 못 잡는 실패가 있다 — 워커 8개가 배경 합성을 하며
    호스트 RAM 과 swap 을 먹는다. 실행 1 점 3 을 끊은 것도 VRAM 부족이 아니라
    시스템 정지였다.

    프로파일 (D81, 사용자 결정)

        worker      SwapFree >= 1 GiB · MemAvailable >= 4 GiB      **D42 원안. 기본값**
                    DataLoader 워커가 배경 합성을 하는 스크립트용 (`c_train`)
        no_worker   MemAvailable >= 4 GiB · memory PSI full avg60 <= 1.0%
                    워커도 배경 합성도 없는 스크립트용 (`act4_ladder`)

    **`no_worker` 는 완화가 아니라 교체다.** `SwapFree` 하한을 빼는 대신 스래싱을
    **직접 재는** PSI 를 넣는다. 이 호스트는 swap 총량이 2 GiB 라 "SwapFree >= 1"
    이 사실상 "swap 절반이 비어 있을 것" 이 되고, swappiness 60 에서 그것은
    메모리 압박이 아니라 **평상시 운영의 결과**다(실측: PSI full avg10/60/300 = 0
    인데도 SwapFree 0.63 GiB). D20 이 금지한 "다른 조건에서 나온 임계값 이식" 의
    사례이며, D75 §4 선택지 (나)를 집행한 것이다.

    PSI 를 못 읽으면 **거부한다** — 확인 불가를 통과로 치지 않는다.
    """
    if profile not in ("worker", "no_worker"):
        raise SystemExit(f"알 수 없는 메모리 가드 프로파일: {profile!r}")
    need = ({"SwapFree": 1 * 1024 ** 2, "MemAvailable": 4 * 1024 ** 2}
            if profile == "worker" else {"MemAvailable": 4 * 1024 ** 2})   # kB
    try:
        info = {}
        for line in open("/proc/meminfo"):
            k, _, v = line.partition(":")
            info[k] = int(v.split()[0])            # kB
    except OSError as e:
        raise SystemExit(f"/proc/meminfo 를 읽을 수 없다 — 시작을 거부한다: {e}")
    low = {k: info.get(k, 0) for k, v in need.items() if info.get(k, 0) < v}
    if low:
        raise SystemExit(
            "시스템 메모리 부족 — 시작을 거부한다 (D42).\n"
            + "".join(f"    {k} {info.get(k,0)/1024**2:.2f} GiB "
                      f"< 하한 {need[k]/1024**2:.0f} GiB\n" for k in low)
            + "  다른 작업이 끝나기를 기다린다. 남의 프로세스는 죽이지 않는다.")
    psi = None
    if profile == "no_worker":
        try:
            for line in open("/proc/pressure/memory"):
                if line.startswith("full "):
                    psi = float(dict(
                        kv.split("=") for kv in line.split()[1:])["avg60"])
        except (OSError, ValueError, KeyError) as e:
            raise SystemExit(
                f"/proc/pressure/memory 를 읽을 수 없다 — 시작을 거부한다: {e}\n"
                "  no_worker 프로파일은 PSI 로 스래싱을 재므로 확인 불가는 거부다.")
        if psi is None:
            raise SystemExit("PSI full 행을 찾지 못했다 — 시작을 거부한다.")
        if psi > PSI_FULL_CEIL:
            raise SystemExit(
                f"메모리 스래싱 — 시작을 거부한다 (D81).\n"
                f"    memory PSI full avg60 {psi:.2f}% > 상한 {PSI_FULL_CEIL}%\n"
                "  다른 작업이 끝나기를 기다린다. 남의 프로세스는 죽이지 않는다.")
    print(f"시스템 메모리 확인 [{profile}] — "
          f"MemAvailable {info['MemAvailable']/1024**2:.1f} GiB / "
          f"SwapFree {info['SwapFree']/1024**2:.1f} GiB"
          + (f" / PSI full avg60 {psi:.2f}%" if psi is not None else ""), flush=True)
    out = {k: info.get(k, 0) for k in ("MemAvailable", "SwapFree", "SwapTotal")}
    out["profile"] = profile
    out["psi_full_avg60"] = psi
    out["psi_ceiling"] = PSI_FULL_CEIL if profile == "no_worker" else None
    return out


def gpu_exclusive_or_die(override_reason: str | None = None) -> dict:
    """착수 가드 — VRAM 여유가 문턱 미만이면 시작을 거부한다 (v2 §9).

    **fail-closed 다** — `nvidia-smi` 를 못 읽으면 통과시키지 않고 멈춘다.
    확인할 수 없는 것을 확인된 것으로 취급하지 않는다.

    판정 기준의 이력 (v2 §9 에 근거가 있다)

        원      "IsaacLab 공존 여유 4 GiB"           메모리 기준
        1차     "GPU 단독 — 프로세스 있으면 거부"     점 3 이 시스템 정지로 날아간 뒤
        2차     메모리 기준으로 복귀 (현재)           사용자 결정. 위험을 감수하고 착수

    1차 개정의 근거("여유 VRAM 은 시스템 정지를 막지 못한다")는 **취소되지
    않았다.** 공존 중 정지가 재발하면 그 점은 외부 무효이며 처음부터 재실행한다
    (D39). 다른 프로세스는 감지해 기록하되 죽이지 않는다.

    `override_reason` — **사용자가 문턱 미달을 알고 착수를 명시한 경우에만** 준다.
    문턱(`MIN_FREE_GIB`)은 **바꾸지 않는다.** 우회 사실과 사유가 반환 dict 에
    남고 provenance 를 통해 결과 파일에 기록되므로, 나중에 "가드를 지나갔다" 와
    "가드를 우회했다" 가 구분된다. 인자가 없으면 종전대로 거부한다.
    """
    import os
    import subprocess

    def smi(query: str) -> list[str]:
        out = subprocess.run(
            ["nvidia-smi", f"--query-{query}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30)
        if out.returncode != 0:
            raise RuntimeError(f"nvidia-smi 실패 (rc={out.returncode}): "
                               f"{out.stderr.strip()}")
        return [l.strip() for l in out.stdout.splitlines() if l.strip()]

    try:
        apps = smi("compute-apps=pid,used_memory,process_name")
        mem = smi("gpu=memory.used,memory.total")[0]
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as e:
        raise SystemExit(
            f"GPU 단독 사용을 확인할 수 없다 — 시작을 거부한다.\n  {e}\n"
            "  확인 불가를 통과로 취급하지 않는다 (v2 §9, fail-closed).") from e

    mine = str(os.getpid())
    others = [a for a in apps if a.split(",")[0].strip() != mine]
    used, total = (float(x.split()[0]) for x in mem.split(","))
    free_gib = (total - used) / 1024
    overridden = False
    if free_gib < MIN_FREE_GIB:
        if not override_reason:
            raise SystemExit(
                f"VRAM 여유 {free_gib:.2f} GiB < 문턱 {MIN_FREE_GIB} GiB — "
                "시작을 거부한다 (v2 §9).\n"
                + "".join(f"    {o}\n" for o in others)
                + "  남의 프로세스는 죽이지 않는다 — 끝나기를 기다리거나\n"
                  "  사용자에게 확인받는다.")
        overridden = True
        print(f"** 가드 우회 ** VRAM 여유 {free_gib:.2f} GiB < 문턱 "
              f"{MIN_FREE_GIB} GiB.\n"
              f"   사유: {override_reason}\n"
              "   문턱은 바꾸지 않았다. 우회 사실이 결과 파일에 기록된다.",
              flush=True)
    print(f"착수 가드 {'우회' if overridden else '통과'} — VRAM 여유 "
          f"{free_gib:.2f} GiB (문턱 {MIN_FREE_GIB}, 학습 peak 1.63) / 메모리 {mem}",
          flush=True)
    if others:
        print(f"  공존 {len(others)}건 — 죽이지 않는다. 정지 시 그 점은 "
              "외부 무효로 처음부터 재실행 (D39)", flush=True)
        for o in others:
            print(f"    {o}", flush=True)
    return {"compute_apps": apps, "memory": mem, "free_gib": free_gib,
            "min_free_gib": MIN_FREE_GIB, "coexisting": len(others),
            "exclusive": not others, "guard_overridden": overridden,
            "override_reason": override_reason if overridden else None}


def ann_mask(seg, h, w):
    """`coco.annToMask` 와 같은 경로. 워커마다 COCO 전량(2.88 GiB)을 띄우지 않으려고
    segmentation 만 담은 색인(`c_prep.py`)에서 복원한다."""
    from pycocotools import mask as maskutil
    if isinstance(seg, list):
        rle = maskutil.merge(maskutil.frPyObjects(seg, h, w))
    elif isinstance(seg["counts"], list):
        rle = maskutil.frPyObjects(seg, h, w)
    else:
        rle = seg
    return maskutil.decode(rle)


class PairData(Dataset):
    """한 쌍 → 4 crop (P 두 뷰, Q 두 뷰). 전부 배경 스왑된다."""

    def __init__(self, rows, cfg, preprocess, segs, bank, seed=0):
        self.rows, self.cfg, self.pre = rows, cfg, preprocess
        self.segs, self.bank = segs, bank
        self.seed = seed

    def __len__(self):
        return len(self.rows)

    def _bg(self, cat, rng):
        """D37 — 해당 카테고리 인스턴스가 **없는** 이미지에서만 뽑는다.
        게이트(A3-b-2)의 `bank_image` 와 같은 조건이다. 카테고리를 무시하고
        뽑으면 같은 카테고리가 배경에 섞여 지름길이 부분 복귀한다."""
        cand = self.bank["bank"][cat]
        fn = self.bank["images"][cand[int(rng.integers(0, len(cand)))]]
        return Image.open(Path(IMG_ROOT) / fn).convert("RGB")

    def __getitem__(self, i):
        r = self.rows[i]
        rng = np.random.default_rng((self.seed * 1000003 + i) % (2 ** 31))
        im = Image.open(Path(IMG_ROOT) / r["image"]).convert("RGB")
        W, H = im.size
        out = []
        for aid, bbox in ((r["a_id"], r["a"]), (r["b_id"], r["b"])):
            m = ann_mask(*self.segs[int(aid)])
            for _ in range(2):
                bg = self._bg(r["cat"], rng)
                b = jitter_box(bbox, W, H, self.cfg["jitter"], rng)
                comp, mk, _ = compose_or_keep(im, b, m, bg,
                                              self.cfg["erosion"], rng,
                                              self.cfg.get("p_swap", 1.0))
                if comp is None:
                    comp = im.crop((0, 0, min(64, W), min(64, H)))
                    mk = np.ones((comp.size[1], comp.size[0]), np.uint8)
                if self.cfg["flip"] and rng.random() < 0.5:
                    comp = comp.transpose(Image.FLIP_LEFT_RIGHT)
                    mk = mk[:, ::-1].copy()
                side = self.cfg["grid"]
                mg = np.asarray(Image.fromarray((mk * 255).astype(np.uint8))
                                .resize((side, side))) > 127
                if mg.sum() < 1:
                    mg[:] = True
                out.append((self.pre(comp), torch.tensor(mg.reshape(-1), dtype=torch.float32)))
        px = torch.stack([o[0] for o in out])
        mk = torch.stack([o[1] for o in out])
        return px, mk


def mask_pool(patches, mask):
    """crop 내부 mask-pool. `patches [B,N,d]`, `mask [B,N]`."""
    p = F.normalize(patches, dim=-1)
    w = mask / mask.sum(-1, keepdim=True).clamp_min(1e-6)
    return F.normalize((p * w[..., None]).sum(1), dim=-1)


def infonce(z, temp):
    """z: [B,4,d] — 0,1 = P 두 뷰 / 2,3 = Q 두 뷰. in-pair negative 전용."""
    loss = 0.0
    for anc, pos, negs in ((0, 1, (2, 3)), (1, 0, (2, 3)),
                           (2, 3, (0, 1)), (3, 2, (0, 1))):
        a = z[:, anc]
        sims = torch.stack([(a * z[:, pos]).sum(-1)]
                           + [(a * z[:, n]).sum(-1) for n in negs], dim=1) / temp
        loss = loss + F.cross_entropy(sims, torch.zeros(len(a), dtype=torch.long,
                                                        device=a.device))
    return loss / 4


def build_model(cfg, device):
    import open_clip
    m, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16-quickgelu", pretrained="openai", device=device)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    blocks = m.visual.transformer.resblocks
    train_ps = []
    for b in blocks[-cfg["n_blocks"]:]:
        for p in b.parameters():
            p.requires_grad_(True)
            train_ps.append(p)
    return m, preprocess, train_ps


def encode(m, px):
    """patch token 을 joint 공간으로. sweep 의 crop 경로와 같은 연산."""
    v = m.visual
    x = v.conv1(px).reshape(px.shape[0], v.conv1.out_channels, -1).permute(0, 2, 1)
    x = torch.cat([v.class_embedding.to(x.dtype)
                   + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                                 device=x.device), x], dim=1)
    x = x + v.positional_embedding.to(x.dtype)
    x = v.ln_pre(x)
    x = v.transformer(x)
    x = v.ln_post(x)
    if v.proj is not None:
        x = x @ v.proj
    return x[:, 1:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", default="3500,35000,0")
    ap.add_argument("--measure-only", action="store_true")
    ap.add_argument("--tag", default="c_run1",
                    help="결과 파일명. 스모크는 다른 tag 로 — 사전 등록 결과를 덮지 않는다")
    ap.add_argument("--seed", type=int, default=0)
    # --- 큐 3 파일럿 전용 스위치. **기본값은 전부 사전 등록 구성이다** —
    #     끄면 사다리와 같은 경로를 그대로 돈다 (후보 기계장치가 기준선을
    #     바꾸지 않는다는 것이 비교의 전제다).
    ap.add_argument("--pilot-anchor-lambda", type=float, default=0.0,
                    help="후보 1. 원본 CLIP 임베딩과의 distillation 가중 (0=off)")
    ap.add_argument("--pilot-bgswap-p", type=float, default=1.0,
                    help="후보 2. 배경 스왑 확률 (1.0=D37 등록 구성)")
    ap.add_argument("--pilot-temp", type=float, default=None,
                    help="후보 3. InfoNCE 온도 (미지정=config 값)")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(CFG_PATH))
    cfg["p_swap"] = a.pilot_bgswap_p
    if a.pilot_temp is not None:
        cfg["temp"] = a.pilot_temp
    set_seed(a.seed)
    dev = "cuda"
    mem = system_memory_or_die()      # D42 — 호스트 RAM/swap 이 얕으면 멈춘다
    gpu = gpu_exclusive_or_die()      # v2 §9 — VRAM 여유가 얕으면 멈춘다
    prov = collect(a.seed, cfg_path=CFG_PATH, inputs=[PAIRS, SEGS, BANK, MANIFEST],
                   extra={"effective_args": vars(a), "preregistered": True,
                          "gpu_exclusive": gpu, "system_memory_kb": mem})

    rows = [json.loads(l) for l in open(PAIRS)]
    import pickle
    segs = pickle.load(open(SEGS, "rb"))
    bank = json.load(open(BANK))
    nb = min(len(v) for v in bank["bank"].values())
    print(f"쌍 {len(rows):,}  mask {len(segs):,}  "
          f"배경 뱅크 카테고리별 최소 {nb:,}장", flush=True)
    m, preprocess, train_ps = build_model(cfg, dev)
    n_train = sum(p.numel() for p in train_ps)
    trainable_names = [n for n, p in m.named_parameters() if p.requires_grad]
    m_ref = None
    if a.pilot_anchor_lambda > 0:
        # 후보 1 — 원본(미세조정 전) CLIP 을 얼려 두고 그 임베딩으로 끌어당긴다.
        m_ref, _, _ = build_model(cfg, dev)
        for prm in m_ref.parameters():
            prm.requires_grad_(False)
        m_ref.eval()
        print(f"후보 1 — 텍스트 앵커(원본 CLIP distillation) λ={a.pilot_anchor_lambda}",
              flush=True)
    print(f"학습 파라미터 {n_train/1e6:.2f}M (마지막 {cfg['n_blocks']}블록)", flush=True)
    # 실측 20스텝이 가중치를 건드리므로 **그 전에** 원본을 뜬다. 사다리 각 점은
    # 여기서 다시 출발한다 — 점 사이에 학습이 누적되면 규모 축이 스텝 축과 섞인다.
    base_sd = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}

    # ---------------------------------------------- 승인 조건: 실효 구성 실측
    dl = make_loader(rows, cfg, preprocess, segs, bank, a.seed)
    opt = torch.optim.AdamW(train_ps, lr=cfg["lr_vision"], weight_decay=cfg["wd"])
    torch.cuda.reset_peak_memory_stats()

    it = iter(dl)
    t_data = t_step = 0.0
    n_meas = cfg["measure_steps"]
    for i in range(n_meas + 1):
        t0 = time.time()
        px, mk = next(it)
        t1 = time.time()
        B, V = px.shape[:2]
        px = px.flatten(0, 1).to(dev, non_blocking=True)
        mk = mk.flatten(0, 1).to(dev, non_blocking=True)
        z = mask_pool(encode(m, px), mk).reshape(B, V, -1)
        loss = infonce(z, cfg["temp"])
        loss.backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        if i:                                    # 첫 스텝은 warmup
            t_data += t1 - t0
            t_step += time.time() - t0
    shutdown_loader(it, dl)                       # D42: 워커 명시적 정리
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    ms_step, ms_data = t_step / n_meas * 1000, t_data / n_meas * 1000
    crops = cfg["pairs_per_batch"] * 4
    meas = {"pairs_per_batch": cfg["pairs_per_batch"], "crops_per_step": crops,
            "peak_gib": peak, "ms_per_step": ms_step, "ms_data_wait": ms_data,
            "dataloader_bottleneck": bool(ms_data > ms_step * 0.5),
            "workers": cfg["workers"], "trainable_M": n_train / 1e6,
            "v1_reference": {"ms_per_step": 147, "peak_gib": 1.40,
                             "note": "b32 난수·독립 샘플. 실효 구성은 crop 4배"}}
    print(f"\n실효 구성 실측 — 쌍 {cfg['pairs_per_batch']}/배치 = crop {crops}개")
    print(f"  peak {peak:.2f} GiB   {ms_step:.0f} ms/step   "
          f"데이터 대기 {ms_data:.0f} ms ({'병목' if meas['dataloader_bottleneck'] else '비병목'})")
    for n_pairs, tag in ((3500, "3.5k"), (35000, "35k"), (len(rows), "전량")):
        steps = n_pairs // cfg["pairs_per_batch"]
        print(f"  사다리 {tag:>4}: {steps:,} step × {ms_step:.0f} ms = "
              f"{steps*ms_step/60000:.1f} 분")
    if a.measure_only:
        # `--tag` 를 따른다. 이전 판은 경로를 하드코딩해 **승인 조건 실측을
        # 조용히 덮어썼다** — 그 파일이 커밋돼 있지 않았으면 복구 불가였다.
        # `--tag` 의 존재 이유가 "사전 등록 결과를 덮지 않는다" 이므로 그것을
        # 무시하는 경로가 있어서는 안 된다. (D42, 2026-08-14 수정)
        name = "c_measure" if a.tag == "c_run1" else a.tag
        dump_result(Path(f"results/phase4/{name}.json"), {"measure": meas}, prov=prov)
        print("\n--measure-only 이므로 여기서 멈춘다. 학습하지 않았다.")
        return

    # -------------------------------------------------------------- 사다리 3점
    from experiments import c_eval
    ladder = [len(rows) if n == 0 else n for n in
              (int(x) for x in a.ladder.split(","))]
    geom = c_eval.geom_control()          # 모델 무관 — 1회 계산해 점마다 공유
    print(f"\ngeom 대조군 {geom*100:.1f}% (상한 {c_eval.CEIL_GEOM*100:.0f}%)", flush=True)

    order = np.random.default_rng(a.seed).permutation(len(rows))
    points = []

    def save():
        dump_result(Path(f"results/phase4/{a.tag}.json"),
                    {"measure": meas, "ladder": ladder, "geom": geom,
                     "n_pairs_total": len(rows), "points": points,
                     "readout": ladder_readout(points) if len(points) == len(ladder)
                     else None,
                     "anchor": {"BASELINE_PROBE": c_eval.BASELINE_PROBE,
                                "sigma": c_eval.SIGMA, "human": c_eval.HUMAN,
                                "c_ratio": c_eval.C_RATIO, "a_ratio": c_eval.A_RATIO,
                                "rule": "c < 사람대비 65.2% / b / a >= 85.0%"}},
                    prov=prov)

    for n_pairs in ladder:
        # **점마다 원본 가중치에서 다시 시작한다.** 이어 학습하면 규모 축이 누적
        # 스텝 축과 섞여 곡선이 규모를 재지 않는다.
        m.load_state_dict(base_sd)
        sub = [rows[i] for i in order[:n_pairs]]
        dl_p = make_loader(sub, cfg, preprocess, segs, bank, a.seed)
        opt = torch.optim.AdamW(train_ps, lr=cfg["lr_vision"],
                                weight_decay=cfg["wd"])
        m.train()
        t0, losses = time.time(), []
        for step, (px, mk) in enumerate(dl_p):
            B, V = px.shape[:2]
            pxf = px.flatten(0, 1).to(dev, non_blocking=True)
            mkf = mk.flatten(0, 1).to(dev, non_blocking=True)
            zf = mask_pool(encode(m, pxf), mkf)
            loss = infonce(zf.reshape(B, V, -1), cfg["temp"])
            if m_ref is not None:
                with torch.no_grad():
                    zr = mask_pool(encode(m_ref, pxf), mkf)
                loss = loss + a.pilot_anchor_lambda * (1 - (zf * zr).sum(-1)).mean()
            loss.backward()
            opt.step(); opt.zero_grad(set_to_none=True)
            losses.append(float(loss))
            if step % 500 == 0:
                print(f"  [{n_pairs:,}] step {step:,}  loss "
                      f"{np.mean(losses[-500:]):.4f}  {time.time()-t0:.0f}s", flush=True)
        mins = (time.time() - t0) / 60
        shutdown_loader(dl_p)                     # D42: 다음 점 전에 워커 회수
        del dl_p
        print(f"  [{n_pairs:,}] 학습 종료 {len(losses):,} step / {mins:.1f} 분 — 판정",
              flush=True)

        # 점별 가중치 — **사후 진단 전용.** 이어-학습 금지 (독스트링 참조).
        # 학습 대상 파라미터만 담는다. 나머지는 openai 원본과 같으므로
        # `base_sd` 와 합치면 그 시점 모델이 그대로 복원된다.
        sd_path = Path(f"results/phase4/{a.tag}_sd_{n_pairs}.pt")
        sd = m.state_dict()
        torch.save({"trainable": {k: sd[k].detach().cpu().clone()
                                  for k in trainable_names},
                    "n_pairs": n_pairs, "steps": len(losses), "seed": a.seed,
                    "cfg_sha256": prov["config"]["sha256"],
                    "git_sha": prov["git"]["sha"],
                    "base": "open_clip ViT-B-16-quickgelu / openai",
                    "note": "사후 진단 전용. 이어-학습에 쓰지 않는다 (D31)."},
                   sd_path)
        print(f"  [{n_pairs:,}] 진단용 가중치 저장 {sd_path} "
              f"({len(trainable_names)}개 텐서)", flush=True)

        m.eval()
        inv = c_eval.invalid_checks(m, preprocess, dev, geom=geom)
        pr = c_eval.run_probe(m, preprocess, dev)
        points.append({
            "n_pairs": n_pairs, "steps": len(losses), "minutes": mins,
            "loss_first50": float(np.mean(losses[:50])),
            "loss_last50": float(np.mean(losses[-50:])),
            "probe": {h: pr[h] for h in ("linear", "mlp", "attn")},
            "per_seed": pr["per_seed"], "judge": pr["judge"], "invalid": inv})
        print(f"  [{n_pairs:,}] 프로브 {pr['judge']['acc']*100:.1f}%  "
              f"사람 대비 {pr['judge']['ratio']*100:.1f}%  →  구간 "
              f"**{pr['judge']['verdict']}**   무효 "
              f"{'발동 — ' + '; '.join(inv['fired']) if inv['invalid'] else '아님'}",
              flush=True)
        save()

    # ---------------------------------------------------- 사다리 판독 (§12.1)
    read = ladder_readout(points)
    save()
    if "slope" not in read:            # 점이 2개 미만 — 기울기가 정의되지 않는다
        print(f"\n{'='*72}\n사다리 판독 — **{read['reading']}**  ({read.get('note','')})")
        for p_ in points:
            print(f"  {p_['n_pairs']:>7,}쌍  프로브 {p_['judge']['acc']*100:5.1f}%  "
                  f"사람 대비 {p_['judge']['ratio']*100:5.1f}%  구간 {p_['judge']['verdict']}"
                  f"   무효 {'발동' if p_['invalid']['invalid'] else '아님'}")
        return
    print(f"\n{'='*72}\n사다리 판독 — **{read['reading']}**   기울기 "
          f"{read['slope']*100:+.2f} p/decade   CI [{read['ci'][0]*100:+.2f}, "
          f"{read['ci'][1]*100:+.2f}]")
    for p in points:
        print(f"  {p['n_pairs']:>7,}쌍  프로브 {p['judge']['acc']*100:5.1f}%  "
              f"사람 대비 {p['judge']['ratio']*100:5.1f}%  구간 {p['judge']['verdict']}"
              f"   무효 {'발동' if p['invalid']['invalid'] else '아님'}")
    worst = min(p["judge"]["ratio"] for p in points)
    best = max(p["judge"]["ratio"] for p in points)
    print(f"\n  최고 점 사람 대비 {best*100:.1f}% → 구간 "
          f"**{c_eval.verdict(best)}**  (최저 {worst*100:.1f}%)")
    if c_eval.verdict(best) == "c":
        print(f"  c 분기 사전 바인딩(§12.1): {read['c_branch_if_c']}")


def ladder_readout(points, n_boot=10000, seed=0):
    """§12.1 — `log10(쌍 수)` 대 `사람 대비 %` 회귀. 기울기 CI 로 3갈래.

    시드가 1개이므로 사전 등록대로 **각 점의 프로브 3시드 표준편차를 회귀 잔차**로
    써서 부트스트랩한다. 판독은 기계적이다 — 서술로 고르지 않는다."""
    from experiments.c_eval import HUMAN
    if len(points) < 2:
        return {"reading": "판독 불가", "n_points": len(points),
                "note": "점이 2개 미만이면 기울기가 정의되지 않는다"}
    x = np.log10([p["n_pairs"] for p in points])
    y = np.array([p["judge"]["ratio"] for p in points])
    sd = np.array([np.std(p["per_seed"]["mlp"]) / HUMAN for p in points])
    rng = np.random.default_rng(seed)
    slopes = [np.polyfit(x, y + rng.normal(0, np.maximum(sd, 1e-9)), 1)[0]
              for _ in range(n_boot)]
    lo, hi = float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))
    reading = "평평" if lo <= 0 <= hi else ("단조 상승" if lo > 0 else "비단조")
    # 기울기가 유의해도 점들이 단조가 아니면 비단조로 읽는다 (학습 불안정 우선)
    if reading == "단조 상승" and not np.all(np.diff(y) >= 0):
        reading = "비단조"
    branch = {"단조 상승": "(a) 학습 범위 확대 last2 → full + checkpointing",
              "평평": "(b) 손실 재설계 (온도·negative 가중)",
              "비단조": "시드 확대 후 두 갈래 재적용"}[reading]
    return {"x_log10_pairs": x.tolist(), "y_ratio": y.tolist(),
            "resid_sd": sd.tolist(), "slope": float(np.polyfit(x, y, 1)[0]),
            "ci": [lo, hi], "reading": reading, "c_branch_if_c": branch,
            "rule": "CI 가 0 포함 → 평평 / 양수 → 단조 상승 / 그 외 → 비단조"}


if __name__ == "__main__":
    main()
