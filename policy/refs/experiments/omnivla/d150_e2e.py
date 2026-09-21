"""D150 — 지정 3프레임 x 어순 교환 6문장의 **end-to-end 시각화**. 합성 세트·통계가 아니다.

**2단계로 돌린다** (환경이 갈린다. D131 §6-a 규칙 — 스크립트는 실행 환경을 명시한다)

    ① 검출   edge_vlm .venv  (ultralytics 가 여기 있다)
       .venv/bin/python experiments/omnivla/d150_e2e.py --stage det
    ② 본체   frodo_lan + .omni_deps  (정책·CLIP 이 여기 있다)
       PYTHONPATH=/home/shy/suhyeon/edge_vlm/.omni_deps \
         /home/shy/anaconda3/envs/frodo_lan/bin/python \
         experiments/omnivla/d150_e2e.py --stage main

**런타임 형태다 — 주석 박스를 쓰지 않는다.** 후보는 검출기에서만 온다.
**판정은 사용자 육안.** 이 스크립트는 수치를 해석하지 않는다 (D74-b).
"""
from __future__ import annotations
import argparse, base64, io, json, os, re, sys
from pathlib import Path
import numpy as np, yaml
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
CFG = ROOT / "configs/experiment/d150_e2e.yaml"
ECFG = ROOT / "configs/experiment/omnivla_eval_sets.yaml"
ICFG = ROOT / "configs/experiment/omnivla_integration.yaml"
DET = ROOT / "results/phase4/omnivla/d150_det.json"
OUTD = ROOT / "results/phase4/omnivla/d150_e2e"
OUT = ROOT / "results/phase4/omnivla/d150_e2e.json"
OUTD_M = ROOT / "results/phase4/omnivla/d153_e2e_mask"      # **마스킹 모드 기본 산출**
OUT_M = ROOT / "results/phase4/omnivla/d153_e2e_mask.json"
MWS = 0.125          # pointing.py 와 같은 눈금
cen = lambda b: ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


# ────────────────────────── 파싱 (내 규칙. 되돌리려면 지시가 필요하다) ──────────────────────────
def parse(sent):
    """'{A} next to {B}' → (A구, B구). 관사만 떼고 **구는 그대로 둔다**."""
    s = sent.strip().lower()
    if " next to " not in s:
        return None, None
    a, b = s.split(" next to ", 1)
    strip = lambda t: re.sub(r"^(the|a|an)\s+", "", t.strip())
    return strip(a), strip(b)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1: return 0.0
    i = (x2 - x1) * (y2 - y1)
    return i / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i + 1e-9)


def match_coco_phrase(phrase, coco_names, head=None):
    """**D157 ① + D161 정정** — 문구에서 COCO 이름을 찾되 **머리명사를 포함하는 이름을 우선**한다.

    'a potted plant' → 'potted plant'  (머리명사 'plant' 를 포함한다)
    'the orange chair' → **'chair'**   ('orange' 도 COCO 이름이지만 머리명사가 아니다.
                                        D161 전에는 더 긴 'orange' 가 이겨 매핑이 뒤집혔다)
    포함하는 이름이 없으면 **머리명사 경로로 넘긴다**(None) — 속성어가 대상을 가로채지 않게.
    """
    import sys as _s
    from build_eval_sets import lemma as _lem
    t = " " + " ".join(re.findall(r"[a-z]+", phrase.lower())) + " "
    hits = [nm for nm in coco_names if " " + nm.lower() + " " in t]
    if not hits:
        return None
    h = _lem(head if head is not None else head_of(phrase))
    pref = [nm for nm in hits if h and h in {_lem(w) for w in nm.lower().split()}]
    if not pref:
        return None
    return max(pref, key=lambda x: (len(x.split()), len(x)))


def head_of(phrase):
    """구의 머리명사 — **전치사 앞**에서 마지막 단어를 딴다.

    'boy in blue clothes' → 'boy'  (그냥 마지막 단어를 따면 'clothes' 가 된다)
    **내 규칙이다** (D97 §0 · D147 §6-b 와 같은 지위).
    """
    p = re.split(r"\s+(?:in|with|on|wearing|holding)\s+", phrase)[0]
    w = re.findall(r"[a-z]+", p)
    return w[-1] if w else ""


# ────────────────────────── ① 검출 단계 ──────────────────────────
def stage_det(cfg, frames=None, outpath=None):
    import cv2
    from ultralytics import YOLO
    D = cfg["dataset_root"]; dc = cfg["detector"]
    det = YOLO(dc["weights"]); names = det.model.names if hasattr(det, "model") else det.names
    out = {}
    for key in (frames or cfg["frames"]):
        ep, st = key.split("/")
        p = os.path.join(D, ep, "image", st + ".jpg")
        im = cv2.imread(p)
        assert im is not None, f"프레임을 못 읽는다: {p}"
        H, W = im.shape[:2]
        r = det.predict(im, imgsz=int(dc["imgsz"]), conf=float(dc["conf"]),
                        device=0, verbose=False)[0]
        bs = []
        for b, k, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int),
                           r.boxes.conf.cpu().numpy()):
            bs.append({"cat": str(names[int(k)]), "conf": float(c),
                       "box": [float(b[0] / W), float(b[1] / H),
                               float(b[2] / W), float(b[3] / H)]})
        bs.sort(key=lambda d: -d["conf"])
        out[key] = {"img": p, "wh": [W, H], "dets": bs}
        print(f"{key}  검출 {len(bs)}  " + ", ".join(f'{d["cat"]}:{d["conf"]:.2f}' for d in bs[:8]))
    dp = Path(outpath) if outpath else DET
    dp.parent.mkdir(parents=True, exist_ok=True)
    dp.write_text(json.dumps({"config": dc, "frames": out}, ensure_ascii=False, indent=2))
    print(f"저장 {dp}")


# ────────────────────────── 그리기 ──────────────────────────
def _heat_rgb(h):
    """[0,1] 지도 → 흑-적-황 색표 (matplotlib 없이)."""
    h = np.clip(h, 0, 1)
    r = np.clip(h * 3, 0, 1); g = np.clip(h * 3 - 1, 0, 1); b = np.clip(h * 3 - 2, 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def panel_left(img, dets, sel, peak, bcen, P):
    im = img.resize((P, P)); d = ImageDraw.Draw(im)
    for i, dt in enumerate(dets):
        x1, y1, x2, y2 = [max(1.0, v * P) for v in dt["box"]]
        d.rectangle([x1 - 2, y1 - 2, x2 + 2, y2 + 2], outline=(0, 0, 0), width=4)   # 대비용 검은 테두리
        d.rectangle([x1, y1, x2, y2], outline=(255, 255, 255), width=2)
        lb = f'c{i}:{dt["cat"]} {dt["conf"]:.2f}'
        d.rectangle([x1, y1, x1 + 7 * len(lb) + 6, y1 + 14], fill=(0, 0, 0))
        d.text((x1 + 3, y1 + 2), lb, fill=(255, 255, 255))
    if sel is not None:
        x1, y1, x2, y2 = [max(1.0, v * P) for v in dets[sel]["box"]]
        d.rectangle([x1 - 3, y1 - 3, x2 + 3, y2 + 3], outline=(0, 0, 0), width=6)
        d.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=5)
        d.rectangle([x1, y2 - 15, x1 + 66, y2 - 1], fill=(0, 0, 0))
        d.text((x1 + 3, y2 - 14), "SELECTED", fill=(0, 255, 0))
    if peak is not None:
        px, py = peak[0] * P, peak[1] * P
        d.line([px - 11, py - 11, px + 11, py + 11], fill=(80, 160, 255), width=4)
        d.line([px - 11, py + 11, px + 11, py - 11], fill=(80, 160, 255), width=4)
        d.text((px + 13, py - 6), "B peak", fill=(80, 160, 255))
    if bcen is not None:
        bx, by = bcen[0] * P, bcen[1] * P
        d.ellipse([bx - 7, by - 7, bx + 7, by + 7], outline=(80, 160, 255), width=3)
        d.text((bx + 9, by + 4), "B det-center", fill=(80, 160, 255))
    d.rectangle([0, P - 18, P, P], fill=(0, 0, 0))
    d.text((6, P - 15), "(1) detector boxes  (4) B peak  (5) selection", fill=(200, 200, 200))
    return im


def panel_heat(h224, box, P, title):
    im = Image.fromarray(_heat_rgb(h224)).resize((P, P)); d = ImageDraw.Draw(im)
    if box is not None:
        d.rectangle([box[0] * P, box[1] * P, box[2] * P, box[3] * P],
                    outline=(0, 255, 0), width=3)
    d.text((6, 6), title, fill=(255, 255, 255))
    return im


def panel_traj(t4, t1, cand_x, sel, P, rng_m):
    im = Image.new("RGB", (P, P), (18, 18, 22)); d = ImageDraw.Draw(im)
    LAT, FWD = rng_m
    to_px = lambda y, x: (P / 2 - (y / LAT) * (P / 2), P - (x / FWD) * (P - 30) - 15)
    for m in np.arange(-LAT, LAT + .01, 1.0):          # 횡 격자
        px = to_px(m, 0)[0]; d.line([px, 0, px, P], fill=(45, 45, 52))
        d.text((px + 2, P - 14), f"{m:+.0f}m", fill=(90, 90, 100))
    for m in np.arange(0, FWD + .01, 1.0):             # 전방 격자
        py = to_px(0, m)[1]; d.line([0, py, P, py], fill=(45, 45, 52))
        d.text((3, py - 12), f"{m:.0f}m", fill=(90, 90, 100))
    for xs, col, nm in ((t1, (90, 150, 255), "arm-1 (full sentence)"),
                        (t4, (255, 150, 40), "arm-4' (heatmap ch)")):
        if xs is None: continue
        pts = [to_px(float(p[1]) * MWS, float(p[0]) * MWS) for p in xs]
        d.line([(0 + P / 2, P - 15)] + pts, fill=col, width=4)
        for p in pts: d.ellipse([p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3], fill=col)
        d.ellipse([pts[-1][0] - 6, pts[-1][1] - 6, pts[-1][0] + 6, pts[-1][1] + 6],
                  outline=col, width=3)
    for i, cx in enumerate(cand_x):                    # 후보의 **이미지 x 대리 위치**
        y = (0.5 - cx) * 2 * LAT
        px = to_px(y, 0)[0]
        col = (0, 255, 0) if i == sel else (150, 150, 150)
        d.polygon([(px, 22), (px - 7, 8), (px + 7, 8)], fill=col)
        d.text((px + 8, 8), f"c{i}", fill=col)
    d.text((6, 34), "arm-4' orange / arm-1 blue", fill=(220, 220, 220))
    d.text((6, 48), "triangles = candidate image-x PROXY (not robot coords)", fill=(160, 160, 160))
    d.text((6, P - 16), "(6)(7) trajectories", fill=(200, 200, 200))
    return im


def wrap(d, text, x, y, w, fill=(230, 230, 230), lh=13):
    for ln in text.split("\n"):
        while len(ln) > w:
            cut = ln.rfind(" ", 0, w); cut = cut if cut > 40 else w
            d.text((x, y), ln[:cut], fill=fill); y += lh; ln = "  " + ln[cut:].lstrip()
        d.text((x, y), ln, fill=fill); y += lh
    return y


# ────────────────────────── ② 본체 ──────────────────────────
def stage_main(cfg, mask_rgb=True, frames=None, pairs=None,
               outdir=None, outjson=None, det_path=None):
    """`mask_rgb=True` 가 **기본값**이다 (D153 §1 — 배포 모드).
    정책 입력의 current_img RGB 를 정규화 공간 0 으로 채운다. heatmap·obs_img·텍스트 무수정."""
    import torch
    from torchvision.transforms import functional as TF, Normalize
    OMNI = Path("/home/shy/suhyeon/OmniVLA_edge/train"); sys.path.insert(0, str(OMNI))
    from vint_train.models.il.il import IL_gps_map_mask3_lan2
    import arm_model as AM
    from arm_model import ArmModel, _patch_first_conv
    from heatmap import HeatmapProducer
    from build_eval_sets import lemma
    from experiments.context_score import box_mask, patch_coords
    from src.utils.provenance import dump_result
    import clip as oc
    IMG = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    det = json.load(open(det_path or DET)); ec = yaml.safe_load(open(ECFG))
    ic = yaml.safe_load(open(ICFG))
    FRAMES = frames or cfg["frames"]; PAIRS = pairs or cfg["sentence_pairs"]
    LD = float(cfg["lambda_d"]); P = int(cfg["panel_px"]); G = 14
    SCORE_MODE = cfg.get("score_mode", "crop_cos")          # **D161 기본값**
    CROP_MARGIN = float(cfg.get("crop_margin", 0.10))
    SCORE_TMPL = cfg.get("score_template", "a photo of a {}.")
    DIAG = float(np.sqrt(2.0)); dev = "cuda"
    H = ic["image_size"]; cs = ic["model"]["context_size"]
    mk = {k: ic["model"][k] for k in ("context_size", "len_traj_pred", "learn_angle",
          "obs_encoder", "obs_encoding_size", "late_fusion", "mha_num_attention_heads",
          "mha_num_attention_layers", "mha_ff_dim_factor")}
    person = {lemma(w) for w in ec["person_synonyms"]}
    coco = {lemma(c) for c in {d["cat"] for f in det["frames"].values() for d in f["dets"]}}
    COCO_NAMES = [c["name"] for c in json.load(open(ROOT / ec["coco_instances"]))["categories"]]
    COCO80 = {lemma(n) for n in COCO_NAMES}

    hp = HeatmapProducer("adapter_head", "ViT-B-16-quickgelu", "openai",
        adapter_path=str(ROOT / "results/phase4/d79_ladder/act4_cl6159_s2.pt"),
        head_path=str(ROOT / "results/phase4/loc_head_full/full_H1_linear_lr0.001_s0.pt"),
        device=dev).to(dev).eval()

    m1 = IL_gps_map_mask3_lan2(**mk)                    # **arm-1 원본** (3채널)
    sd1 = torch.load(ic["arm1_ckpt"], map_location="cpu", weights_only=False)
    mi1, un1 = m1.load_state_dict(sd1, strict=False)
    print(f"[ckpt] arm-1 원본 · 누락 {len(mi1)} · 버려진 키 {len(un1)} "
          f"({sorted({k.split('.')[0] for k in un1})})", flush=True)
    m1.to(dev).eval()
    m4 = ArmModel(**mk); m4._new_conv = _patch_first_conv(m4, 1)   # arm-4' (D128)
    sd4 = torch.load(ROOT / "results/phase4/omnivla/arm4p_s0/latest.pth",
                     map_location="cpu", weights_only=False)
    mi4, un4 = IL_gps_map_mask3_lan2.load_state_dict(m4, sd4, strict=False)
    assert not mi4 and not un4, "arm-4' 로드 이상"
    m4.to(dev).eval()
    w3 = float(m4._new_conv.weight[:, 3:].norm().item())
    print(f"[arm-4'] ‖W[:,3]‖ = {w3:.6f}", flush=True)

    class Fixed:
        def __init__(self): self.m = None
        def __call__(self, cur, tok): return self.m.to(cur.device, cur.dtype)
    fm = Fixed(); ArmModel.HEATMAP = fm; ArmModel.MASK_P = 0.0
    txt, _ = oc.load("ViT-B/32", device=dev); txt.to(torch.float32).eval()
    pc = patch_coords(G)
    load224 = lambda p: TF.resize(TF.to_tensor(Image.open(p).convert("RGB")), (224, 224))

    def pack(ep, st):
        imd = os.path.join(cfg["dataset_root"], ep, "image")
        sl = sorted(f[:-4] for f in os.listdir(imd) if f.endswith(".jpg"))
        k = sl.index(st)
        cur = load224(os.path.join(imd, st + ".jpg"))
        ctx = [cur] + [load224(os.path.join(imd, sl[max(0, k - h)] + ".jpg")) for h in range(1, cs + 1)]
        obs = torch.cat([TF.resize(im, (H, H)) for im in ctx[::-1]]).unsqueeze(0).to(dev)
        ol = torch.split(obs, 3, dim=1)
        z = torch.zeros(1, 3, H, H, device=dev)
        return (torch.cat([IMG(x) for x in ol], 1), torch.zeros(1, 4, device=dev),
                torch.cat((IMG(z), IMG(z), ol[-1]), 1), IMG(z),
                torch.full((1,), 7, dtype=torch.long, device=dev),
                IMG(cur.unsqueeze(0).to(dev)), cur, (k < cs))

    (Path(outdir) if outdir else (OUTD_M if mask_rgb else OUTD)).mkdir(parents=True, exist_ok=True)
    rec, cards = [], []
    for key, pair in zip(FRAMES, PAIRS):
        ep, st = key.split("/")
        fr = det["frames"][key]; dets_all = fr["dets"]
        img = Image.open(fr["img"]).convert("RGB")
        obs_t, gp, mp, gimg, gm, clg, cur, short_ctx = pack(ep, st)
        rows = []
        for sent in pair:
            a, b = parse(sent)
            step = {"sentence": sent, "A": a, "B": b}
            fail = None
            if a is None:
                fail = "(2) PARSE FAILED - no 'next to'"
            ha_head = head_of(a or ""); hb_head = head_of(b or "")
            # ① 후보 — A 카테고리 매핑. 실패하면 **전 박스**
            want = match_coco_phrase(a or "", COCO_NAMES)          # **D157 ① 다단어 우선**
            mw = want is not None and " " in want
            if want is None:
                if lemma(ha_head) in person: want = "person"
                elif lemma(ha_head) in COCO80: want = ha_head
            cands = [d for d in dets_all if lemma(d["cat"]) == lemma(want)] if want else list(dets_all)
            map_note = (f"A phrase -> COCO '{want}' (multiword)" if mw else
                        f"A head '{ha_head}' -> COCO '{want}'" if want
                        else f"A head '{ha_head}' -> NO COCO class (mapping FAILED) -> ALL boxes")
            # **D156 1-a 폴백 개정** — 매핑된 클래스의 검출이 0 이면 **전 박스로 완화**한다.
            # 이전에는 곧바로 (1) ZERO CANDIDATES 로 멈췄다. **내 규칙이다** (되돌리려면 지시 필요)
            relaxed = False
            if want and not cands and dets_all:
                cands = list(dets_all); relaxed = True
                map_note += f"  ** {want} 0 detected -> RELAXED to ALL boxes ({len(dets_all)}) **"
            if not cands:
                fail = fail or f"(1) ZERO CANDIDATES - {map_note}"
            # ③④ heatmap
            tok = oc.tokenize([a or sent, b or sent], truncate=True).to(dev)
            with torch.no_grad():
                hfull = hp(clg, tok[:1])                       # A 구 heatmap (224)
                hb = hp(clg, tok[1:2])
                hA = torch.nn.functional.adaptive_avg_pool2d(hfull, G)[0, 0].cpu().numpy().reshape(-1)
                hB = torch.nn.functional.adaptive_avg_pool2d(hb, G)[0, 0].cpu().numpy().reshape(-1)
            peak = tuple(float(v) for v in pc[int(np.argmax(hB))])
            bwant = match_coco_phrase(b or "", COCO_NAMES)        # **D157 ① 다단어 우선 (B 에도 적용 — 내 선택)**
            if bwant is None:
                bwant = ("person" if lemma(hb_head) in person else
                         (hb_head if lemma(hb_head) in COCO80 else None))
            bdets = [d for d in dets_all if bwant and lemma(d["cat"]) == lemma(bwant)]
            bcen = cen(bdets[0]["box"]) if bdets else None
            # **D157 ② — A 후보에서 B 를 뺀다** (B 검출 박스 또는 B 피크를 품은 박스와 IoU > 0.5)
            # B 박스는 **B 피크로 고른다** — A·B 가 같은 클래스일 때 첫 검출을 집으면 엉뚱한
            # 박스를 뺀다 (ep0030/681 처럼 person 둘). **내 규칙이다.**
            def _pick_b(pool):
                inb = [bx for bx in pool
                       if bx[0] <= peak[0] <= bx[2] and bx[1] <= peak[1] <= bx[3]]
                if inb: return min(inb, key=lambda bx: (bx[2]-bx[0])*(bx[3]-bx[1]))
                if pool: return min(pool, key=lambda bx: np.hypot((bx[0]+bx[2])/2-peak[0],
                                                                 (bx[1]+bx[3])/2-peak[1]))
                return None
            bbox_ex = _pick_b([d["box"] for d in bdets]) if bdets else None
            if bbox_ex is None:
                inb = [d["box"] for d in dets_all
                       if d["box"][0] <= peak[0] <= d["box"][2] and d["box"][1] <= peak[1] <= d["box"][3]]
                if inb: bbox_ex = min(inb, key=lambda bx: (bx[2]-bx[0])*(bx[3]-bx[1]))
            n_before_bx = len(cands); b_excluded = 0
            if bbox_ex is not None:
                keep = [d for d in cands if iou(d["box"], bbox_ex) <= 0.5]
                b_excluded = len(cands) - len(keep)
                cands = keep
            if b_excluded:
                map_note += f"  ** B excluded {b_excluded} box(es) (IoU>0.5) **"
            if not cands and not fail:
                fail = f"(1) ZERO CANDIDATES after B-exclusion - {map_note}"

            # ③ 후보별 속성 점수 · ⑤ 규칙
            # **D161 — 기본 경로가 crop_cos 로 바뀌었다** (판정 세트 67.46% 와 같은 경로).
            # heat_max(구 구현)도 같이 재서 파일에 남긴다 — 두 경로를 나란히 볼 수 있게.
            s_crop, cos_txt = hp.crop_cos(img, [d["box"] for d in cands], a or sent,
                                          CROP_MARGIN, SCORE_TMPL)
            scored = []
            for i, d in enumerate(cands):
                m = box_mask(d["box"], pc)
                s_heat = float(hA[m].max()) if m.any() else float("-inf")
                s = s_crop[i] if SCORE_MODE == "crop_cos" else s_heat
                c = cen(d["box"])
                dist = float(np.hypot(c[0] - peak[0], c[1] - peak[1])) / DIAG
                scored.append({"i": i, "cat": d["cat"], "conf": d["conf"],
                               "score_A": s, "score_crop_cos": s_crop[i], "score_heat_max": s_heat,
                               "dist_to_Bpeak": dist, "final": s - LD * dist})
            sel = int(np.argmax([x["final"] for x in scored])) if scored else None
            # ⑥ 선택 박스 안만 남긴 1채널 → arm-4'  /  ⑦ arm-1 (문장 전체)
            t4 = t1 = None
            if sel is not None:
                m = box_mask(cands[sel]["box"], pc)
                h = np.zeros(G * G, np.float32); h[m] = np.clip(hA[m], 0, None)
                h = h.reshape(G, G); h = (h - h.min()) / max(h.max() - h.min(), 1e-8)
                fm.m = torch.nn.functional.interpolate(torch.from_numpy(h)[None, None],
                        size=(224, 224), mode="bilinear", align_corners=False)
                chan = fm.m[0, 0].numpy()
                tk4 = oc.tokenize([a], truncate=True).to(dev)
                tkS = oc.tokenize([sent], truncate=True).to(dev)
                clgm = torch.zeros_like(clg)          # **언어 분기 RGB 마스킹** (D153 §1)
                with torch.no_grad():
                    AM._STASH["tokens"] = tk4
                    f4 = txt.encode_text(tk4).float()
                    a4m, _, _ = m4(obs_t, gp, mp, gimg, gm, f4, clgm)    # 배포 모드 (주)
                    a4u, _, _ = m4(obs_t, gp, mp, gimg, gm, f4, clg)     # 참고 (무마스킹)
                    a1_, _, _ = m1(obs_t, gp, mp, gimg, gm, txt.encode_text(tkS).float(), clg)
                t4m = a4m[0, :, :2].detach().cpu().numpy()
                t4u = a4u[0, :, :2].detach().cpu().numpy()
                t4 = t4m if mask_rgb else t4u
                t1 = a1_[0, :, :2].detach().cpu().numpy()
            else:
                chan = np.zeros((224, 224), np.float32); t4m = t4u = None
            e4 = float(t4[-1, 1] * MWS) if t4 is not None else None
            e4u = float(t4u[-1, 1] * MWS) if t4u is not None else None
            e1 = float(t1[-1, 1] * MWS) if t1 is not None else None
            step.update({"policy_mask_rgb": bool(mask_rgb),
                         "endpoint_lateral_nomask_m": e4u, "A_head": ha_head, "B_head": hb_head, "mapping": map_note,
                         "n_dets_all": len(dets_all), "n_candidates": len(cands),
                         "candidate_relaxed": bool(relaxed),
                         "coco_multiword": bool(mw), "b_excluded_boxes": int(b_excluded),
                         "n_cands_before_b_exclusion": int(n_before_bx),
                         "candidates": scored, "B_peak_xy": list(peak),
                         "B_det_center": list(bcen) if bcen else None,
                         "selected": sel, "endpoint_lateral_m": {"arm4p": e4, "arm1": e1},
                         "failure_stage": fail, "short_context": bool(short_ctx)})
            rec.append(dict(step, frame=key))
            # ── 패널 ──
            L = panel_left(img, cands, sel, peak, bcen, P)
            M = panel_heat(chan, cands[sel]["box"] if sel is not None else None, P,
                           "(6) policy input channel")
            cxs = [cen(d["box"])[0] for d in cands]
            R = panel_traj(t4, t1, cxs, sel, P, cfg["traj_range_m"])
            R2 = panel_traj(t4u, t4m, cxs, sel, P, cfg["traj_range_m"])
            dd = ImageDraw.Draw(R); dd.rectangle([0, 0, P, 18], fill=(0, 0, 0))
            dd.text((6, 3), ("(7) MASKED mode (deploy) orange | arm-1 blue" if mask_rgb
                             else "(7) NO-MASK orange | arm-1 blue"), fill=(255, 200, 120))
            d2 = ImageDraw.Draw(R2); d2.rectangle([0, 0, P, 18], fill=(0, 0, 0))
            d2.text((6, 3), "(ref) no-mask orange | masked blue", fill=(200, 200, 200))
            txtlines = [
                f'SENT: "{sent}"',
                f'(2) parse  A="{a}"  B="{b}"   heads: A={ha_head} / B={hb_head}',
                f'(1) detector: {len(dets_all)} boxes -> {len(cands)} candidates.  {map_note}',
                f'(3) attr score [{SCORE_MODE}] / (5) final = score - {LD}*dist:  ' + " | ".join(
                    f'c{x["i"]}({x["cat"]}) s={x["score_A"]:.3f} (heat {x["score_heat_max"]:.3f})'
                    f' d={x["dist_to_Bpeak"]:.3f} f={x["final"]:.3f}'
                    for x in scored) if scored else "(3)(5) no candidate",
                f'(4) B peak=({peak[0]:.3f},{peak[1]:.3f})' +
                (f'  B det-center=({bcen[0]:.3f},{bcen[1]:.3f})' if bcen else '  B det-center=none (not COCO / not detected)'),
                f'(5) selected = c{sel}' if sel is not None else '(5) selected = NONE',
                f'(6)(7) endpoint lateral:  arm-4\' {e4:+.3f} m   arm-1 {e1:+.3f} m'
                if e4 is not None else '(6)(7) no trajectory',
            ]
            if fail: txtlines.insert(0, f'*** FAILURE AT {fail} ***')
            if short_ctx: txtlines.append('note: context frames clamped (frame index < context size)')
            rows.append((L, M, R, R2, "\n".join(txtlines)))
        # ── 프레임 1장 ──
        TXT = 116
        W = P * 4; Hh = (P + TXT) * 2 + 26
        canvas = Image.new("RGB", (W, Hh), (12, 12, 14)); d = ImageDraw.Draw(canvas)
        d.text((8, 7), f"D150/D153 end-to-end  |  frame {key}  |  runtime form: NO annotation "
                       f"boxes  |  policy RGB mask = {'ON (deploy mode)' if mask_rgb else 'OFF'}",
               fill=(255, 255, 255))
        for r, (L, M, R, R2, t) in enumerate(rows):
            y = 26 + r * (P + TXT)
            for j, im in enumerate((L, M, R, R2)): canvas.paste(im, (j * P, y))
            wrap(d, t, 8, y + P + 4, 200)
        od = Path(outdir) if outdir else (OUTD_M if mask_rgb else OUTD)
        od.mkdir(parents=True, exist_ok=True)
        fn = od / f"{key.replace('/', '_')}.png"
        canvas.save(fn)
        buf = io.BytesIO(); canvas.save(buf, "PNG")
        cards.append((key, base64.b64encode(buf.getvalue()).decode()))
        print(f"그림 {fn}", flush=True)

    html = ["<meta charset='utf-8'><style>body{background:#111;color:#ddd;"
            "font-family:sans-serif;max-width:1320px;margin:20px auto}img{width:100%}"
            "h2{font-size:16px;margin:28px 0 6px}.w{background:#2a1f10;border-left:4px solid #c80;"
            "padding:10px;margin:12px 0;font-size:13px;line-height:1.6}</style>",
            "<h1>D150/D153 — 지정 3프레임 × 어순 교환 6문장 end-to-end</h1>",
            "<div class='w'><b>이것은 합성 세트 통계가 아니라 지정 프레임의 실행 그림이다.</b><br>"
            "런타임 형태 — <b>주석 박스를 쓰지 않는다.</b> 후보는 YOLOv8n(conf "
            f"{det['config']['conf']})에서만 온다.<br>"
            "각 행 = 문장 하나. <b>위/아래가 어순 교환 쌍</b>이다. 좌=검출·기준피크·선택 · "
            "중=정책 입력 1채널 · 우=궤적(arm-4′ 주황 / arm-1 파랑).<br>"
            "<b>정책 RGB 마스킹 = 배포 모드(D153 §1)가 기본값이다.</b> 3열이 <b>주 열(마스킹)</b>, "
            "4열은 참고(무마스킹 주황 대 마스킹 파랑).<br>"
            "삼각형은 후보의 <b>이미지 x 대리 위치</b>이고 로봇 좌표가 아니다.<br>"
            "<b>판정은 육안이다.</b> 이 문서에 정확도·통계는 없다 (D74-b)."
            "</div>"]
    for key, b64 in cards:
        html.append(f"<h2>{key}</h2><img src='data:image/png;base64,{b64}'>")
    _od = Path(outdir) if outdir else (OUTD_M if mask_rgb else OUTD)
    _od.joinpath("index.html").write_text("\n".join(html), encoding="utf-8")

    res = {"preregistered": "notes/decisions.md D150 — 사용자 지시 (지정 프레임 육안 시험)",
           "read_only_note": "추론만 · 학습 0회 · OmniVLA_edge 무수정 · **주석 박스 미사용**",
           "env": "frodo_lan + .omni_deps (검출 단계는 edge_vlm .venv)",
           "pipeline": {"1_detector": f'YOLOv8n conf {det["config"]["conf"]} imgsz {det["config"]["imgsz"]}',
                        "2_parse": "'{A} next to {B}' · 관사만 제거 · 머리명사는 전치사 앞 (내 규칙)",
                        "3_score": f"**{SCORE_MODE}** — crop_cos = 후보 crop(10% 확장) → "
                                   "어댑터 pooled 임베딩 → A 텍스트 cos (판정 세트 67.46% 경로, "
                                   "D161) · heat_max = 헤드 heatmap 박스 내 max (구 구현, 병기)",
                        "4_anchor": "B 구 heatmap **피크**. B 가 COCO 면 검출 박스 중심 병기",
                        "5_rule": f"argmax_k [score_k − {LD}·dist(box_k 중심, B peak)/대각선]",
                        "6_channel": "선택 박스 안만 남긴 hA · 프레임별 min-max → arm-4' (D134 h2box)",
                        "7_baseline": "arm-1 **원본** · 텍스트는 **문장 전체**"},
           "arm4p_new_channel_norm": w3,
           "arm1_dropped_keys": len(un1),
           "caveat": ["**육안 판정용이다. 수치 해석·판정 서술 없음** (D74-b)",
                      "우 패널의 후보 표시는 **이미지 x 대리**이고 로봇 좌표가 아니다",
                      "정책 텍스트는 arm-4' = A 구 · arm-1 = 문장 전체 (지시 ⑦)"],
           "steps": rec}
    res["policy_mask_rgb_default"] = bool(mask_rgb)
    o = Path(outjson) if outjson else (OUT_M if mask_rgb else OUT)
    dump_result(o, res, seed=0, cfg_path=CFG)
    print(f"저장 {o}\n갤러리 {_od/'index.html'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", choices=["det", "main"], required=True)
    ap.add_argument("--no-mask", action="store_true",
                    help="배포 모드(RGB 마스킹)를 끄고 D150 원형으로 돌린다")
    a = ap.parse_args(); c = yaml.safe_load(open(CFG))
    if a.stage == "det": stage_det(c)
    else: stage_main(c, mask_rgb=not a.no_mask)
