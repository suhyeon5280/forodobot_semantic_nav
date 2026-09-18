"""D120 §6-a — (가) 경로용 런타임 패치. **OmniVLA_edge 레포 무수정.**

① 모델   IL_gps_map_mask3_lan2 서브클래스. `load_state_dict` 를 가로채
         **ckpt 로드 → FiLM 첫 conv 4ch(0 초기화) → 동결 정책** 순으로 처리한다.
         (그쪽 main 이 로드 **후** optimizer 를 만들므로 이 시점이 맞다)
② 텍스트 text_encoder 를 얇게 감싸 `encode_text(tokens)` 의 **tokens 를 stash**.
         CLIP 토큰 ID 는 B/32·B/16 이 같으므로 그대로 B/16 타워에 넣는다.
③ 데이터 LeLaN_Dataset_multi 서브클래스 — E2 프레임을 **풀 자체에서 제거**한다
         (`_getitem_frodo_lan` 의 무작위 재추출 경로까지 닫는다)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OMNI = Path("/home/shy/suhyeon/OmniVLA_edge/train")
if str(OMNI) not in sys.path:
    sys.path.insert(0, str(OMNI))

from vint_train.models.il.il import IL_gps_map_mask3_lan2              # noqa: E402
from vint_train.data.lelan_dataset import LeLaN_Dataset_multi          # noqa: E402

# 모델과 텍스트 래퍼가 공유하는 stash (같은 이터레이션 안에서만 쓴다)
_STASH = {"tokens": None}

# D128 — RGB 지름길 차단. 학습 전용 · 표본 단위 Bernoulli(p)
_MASK = {"n_seen": 0, "n_masked": 0}


def _mask_rgb(model, current_img):
    """`current_img` 의 RGB 3채널을 확률 p 로 0 채움. **학습 중에만.**

    0 은 **정규화 공간의 0** = 데이터셋 평균이다 (train.py 가 ImageNet 정규화를 건다).
    픽셀 0 을 넣으면 정규화 후 채널마다 값이 달라 인공 신호가 된다.

    **heatmap 은 이 함수 **전에** 원본에서 뽑는다** — 마스킹된 이미지로 뽑으면
    채널 자체가 죽어 개입의 목적이 사라진다 (D128 §1).
    """
    p = float(getattr(model, "MASK_P", 0.0) or 0.0)
    if not model.training or p <= 0.0:
        return current_img
    b = current_img.shape[0]
    m = torch.rand(b, device=current_img.device) < p
    _MASK["n_seen"] += b
    _MASK["n_masked"] += int(m.sum().item())
    if m.any():
        current_img = current_img.clone()
        current_img[m] = 0.0
    return current_img


class TextEncoderStash(nn.Module):
    """encode_text 의 tokens 를 stash 하고 원 출력을 그대로 돌려준다."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def encode_text(self, tokens):
        _STASH["tokens"] = tokens
        return self.inner.encode_text(tokens)

    def eval(self):
        self.inner.eval(); return self

    def to(self, *a, **k):
        self.inner.to(*a, **k); return self

    def __getattr__(self, n):
        try:
            return super().__getattr__(n)
        except AttributeError:
            return getattr(self.inner, n)


def _patch_first_conv(model, n_extra=1):
    seq = model.film_model.initial_feature_extractor.layers
    holder = idx = old = None
    for i, m in enumerate(seq):
        if isinstance(m, nn.Conv2d):
            holder, idx, old = seq, i, m; break
        if isinstance(m, nn.Sequential):
            for j, mm in enumerate(m):
                if isinstance(mm, nn.Conv2d):
                    holder, idx, old = m, j, mm; break
        if old is not None:
            break
    assert old is not None and old.in_channels == 3, "FiLM 첫 conv(3ch)를 찾지 못했다"
    new = nn.Conv2d(3 + n_extra, old.out_channels, old.kernel_size, old.stride,
                    old.padding, old.dilation, old.groups, old.bias is not None)
    with torch.no_grad():
        new.weight.zero_(); new.weight[:, :3].copy_(old.weight)   # [:,3] = 0
        if old.bias is not None:
            new.bias.copy_(old.bias)
    holder[idx] = new
    return new


class ArmModel(IL_gps_map_mask3_lan2):
    """heatmap 을 FiLM 입력 4번째 채널로 넣는 arm. 학습 = 새 채널 + 융합부."""

    HEATMAP = None          # HeatmapProducer. 런처가 주입한다
    REPORT = {}             # 기동 직전 검사 결과
    MASK_P = 0.0            # D128 — RGB 마스킹 확률. 런처가 주입한다

    def load_state_dict(self, sd, strict=True):
        out = super().load_state_dict(sd, strict=strict)
        missing, unexpected = out if isinstance(out, tuple) else (out.missing_keys,
                                                                 out.unexpected_keys)
        lgx = [k for k in unexpected if "lgx" in k.lower()]
        ArmModel.REPORT["ckpt"] = {"n_keys": len(sd), "n_missing": len(missing),
                                   "n_unexpected": len(unexpected),
                                   "unexpected": list(unexpected),
                                   "unexpected_lgx": lgx}
        print(f"[ckpt] 키 {len(sd)} · 누락 {len(missing)} · **버려진 키 {len(unexpected)} "
              f"(lgx {len(lgx)})**", flush=True)
        print(f"[ckpt] 버려진 키 전부: {list(unexpected)}", flush=True)
        assert len(lgx) == len(unexpected) == 12, \
            f"버려진 키가 12(lgx) 가 아니다: {list(unexpected)[:5]} — 전제가 달라졌다"
        self._new_conv = _patch_first_conv(self, 1)
        self._apply_freeze()
        return out

    def _apply_freeze(self):
        """학습 = 새 채널 conv + 융합부(compress_goal_enc_lan). 나머지 전부 동결."""
        for p in self.parameters():
            p.requires_grad = False
        train_mods = [self._new_conv, self.compress_goal_enc_lan]
        for m in train_mods:
            for p in m.parameters():
                p.requires_grad = True
        # RGB 3채널은 **동결 유지** — 기울기를 0 으로 막는다 (새 채널만 배운다)
        def _mask(g):
            g = g.clone(); g[:, :3] = 0.0; return g
        self._new_conv.weight.register_hook(_mask)
        allowed = set()
        for m in train_mods:
            allowed |= {id(p) for p in m.parameters()}
        bad = [n for n, p in self.named_parameters() if p.requires_grad and id(p) not in allowed]
        assert not bad, f"허용 밖 파라미터가 학습 대상이다: {bad[:5]}"
        for pref in ("obs_encoder", "goal_encoder", "goal_encoder_img"):
            nb = [n for n, p in self.named_parameters()
                  if n.startswith(pref) and p.requires_grad]
            assert not nb, f"{pref} 가 동결되지 않았다: {nb[:3]}"
        ArmModel.REPORT["freeze_check"] = {
            "trainable_outside_allowed": 0,
            "obs_encoder_trainable": 0, "goal_encoder_trainable": 0,
            "goal_encoder_img_trainable": 0}
        n_tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        ArmModel.REPORT["trainable_params"] = int(n_tr)
        ArmModel.REPORT["trainable_modules"] = ["film first conv (new channel only)",
                                                "compress_goal_enc_lan"]
        print(f"[freeze] 학습 파라미터 {n_tr:,d} (새 채널 conv + 융합부)", flush=True)

    def forward(self, obs_img, goal_pose, map_images, goal_img, goal_mask,
                feat_text, current_img, *a, **k):
        tok = _STASH.get("tokens")
        ArmModel._nfwd = getattr(ArmModel, "_nfwd", 0) + 1
        if self.training and ArmModel._nfwd % 100 == 0:
            _w = self._new_conv.weight
            _g = _w.grad
            _gs = "None" if _g is None else ("%.3e" % _g[:, 3].norm().item())
            print("[dbg] fwd %d W3=%.6e grad=%s" % (
                ArmModel._nfwd, _w[:, 3].norm().item(), _gs), flush=True)
        if not getattr(ArmModel, "_dbg_done", False):
            ArmModel._dbg_done = True
            print(f"[dbg] HEATMAP={self.HEATMAP is not None} tok={None if tok is None else tuple(tok.shape)} "
                  f"cur={tuple(current_img.shape)} training={self.training}", flush=True)
        if self.HEATMAP is None or tok is None or tok.shape[0] != current_img.shape[0]:
            hm = torch.zeros(current_img.shape[0], 1, *current_img.shape[-2:],
                             device=current_img.device, dtype=current_img.dtype)
        else:
            hm = self.HEATMAP(current_img, tok).to(current_img.dtype)
            if not getattr(ArmModel, "_dbg2_done", False):
                ArmModel._dbg2_done = True
                print(f"[dbg] heatmap mean={hm.mean().item():.4f} min={hm.min().item():.4f} "
                      f"max={hm.max().item():.4f}", flush=True)
        cur = _mask_rgb(self, current_img)      # **heatmap 을 뽑은 뒤** 마스킹한다
        return super().forward(obs_img, goal_pose, map_images, goal_img, goal_mask,
                               feat_text, torch.cat([cur, hm], dim=1), *a, **k)


def make_excluded_dataset_cls(excluded_keys):
    """E2 프레임을 **표집 대상에서** 제외한다.

    **풀에서 제거하지 않는다** — `ep_lo`/`ep_hi` 가 원본 절대 색인이고 context·미래
    프레임(`iv-h`, `iv+8`)이 리스트 위치로 계산되므로, 중간 원소를 빼면 **시간적 이웃이
    조용히 바뀐다** (D123 §7 에서 실측으로 확인했다).

    대신 `__len__`/`__getitem__` 을 **허용 색인으로 사상**한다. 내부 배열은 불변이다.
    `_getitem_frodo_lan` 의 무작위 재추출(`random.randint(0, len(image_path)-1)`)도
    **그 호출만** 허용 색인으로 돌려 경로를 닫는다.
    """

    class ExcludedLeLaN(LeLaN_Dataset_multi):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            paths = list(getattr(self, "image_path", []) or [])
            self._n_all = len(paths)
            self._allowed = []
            for idx, q in enumerate(paths):
                parts = Path(str(q)).parts
                ep = next((x for x in parts if x.startswith("episode_")), None)
                if ep is not None and f"{ep}/{Path(str(q)).stem}" in excluded_keys:
                    continue
                self._allowed.append(idx)
            self._allowed_set = set(self._allowed)
            assert len(self.image_path) == self._n_all, "내부 배열을 건드리면 안 된다"
            bad = [i for i in self._allowed
                   if f"{next((x for x in Path(str(paths[i])).parts if x.startswith('episode_')), '')}"
                      f"/{Path(str(paths[i])).stem}" in excluded_keys]
            assert not bad, f"허용 목록에 E2 프레임이 있다 {len(bad)}건"
            ArmModel.REPORT.setdefault("exclusion", {})
            ArmModel.REPORT["exclusion"].update(
                {"applied": True, "pool_total": self._n_all,
                 "sampled_from": len(self._allowed),
                 "excluded": self._n_all - len(self._allowed),
                 "mechanism": ("표집 사상 — 내부 배열 불변. 풀 제거는 시간적 이웃을 "
                               "바꾸므로 쓰지 않는다 (D123 §7)")})
            print(f"[exclude] 표집 대상 {self._n_all} → {len(self._allowed)} "
                  f"(제외 {self._n_all - len(self._allowed)}) · 내부 배열 불변", flush=True)

        def __len__(self):
            return len(self._allowed)

        def __getitem__(self, i):
            import random as _r
            real = self._allowed[int(i) % len(self._allowed)]
            _orig = _r.randint
            hi = self._n_all - 1
            allowed = self._allowed

            def _patched(a, b):
                # 재추출 호출만 가로챈다. 다른 randint(예: goal_id)는 원본 그대로
                if a == 0 and b == hi:
                    return allowed[_orig(0, len(allowed) - 1)]
                return _orig(a, b)

            _r.randint = _patched
            try:
                return super().__getitem__(real)
            finally:
                _r.randint = _orig

    return ExcludedLeLaN


class ControlModel(IL_gps_map_mask3_lan2):
    """D128 대조 arm — **3채널 + 같은 마스킹.** heatmap 채널이 없다.

    증강 효과를 채널 효과에서 분리하는 것이 유일한 목적이다 (D127 §4).
    학습 대상은 arm-4' 에서 **새 채널 conv 를 뺀 것**과 같다 — `compress_goal_enc_lan`
    하나. (arm-4' 도 첫 conv 의 RGB 가중치는 기울기 훅으로 막혀 있다.)

    **마스킹된 step 에서 FiLM 분기 입력이 전부 0 이 되어 그 분기는 텍스트만의
    prior(β·γ)로 퇴화한다.** 결함이 아니라 대조군의 정의다 (초안 §1-c).
    """

    REPORT = None           # ArmModel.REPORT 를 공유한다
    MASK_P = 0.0

    def load_state_dict(self, sd, strict=True):
        out = super().load_state_dict(sd, strict=strict)
        missing, unexpected = out if isinstance(out, tuple) else (out.missing_keys,
                                                                 out.unexpected_keys)
        lgx = [k for k in unexpected if "lgx" in k.lower()]
        ArmModel.REPORT["ckpt"] = {"n_keys": len(sd), "n_missing": len(missing),
                                   "n_unexpected": len(unexpected),
                                   "unexpected": list(unexpected),
                                   "unexpected_lgx": lgx}
        print(f"[ckpt] 키 {len(sd)} · 누락 {len(missing)} · **버려진 키 {len(unexpected)} "
              f"(lgx {len(lgx)})**", flush=True)
        print(f"[ckpt] 버려진 키 전부: {list(unexpected)}", flush=True)
        assert len(lgx) == len(unexpected) == 12, \
            f"버려진 키가 12(lgx) 가 아니다: {list(unexpected)[:5]} — 전제가 달라졌다"
        self._apply_freeze()
        return out

    def _apply_freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        for p in self.compress_goal_enc_lan.parameters():
            p.requires_grad = True
        allowed = {id(p) for p in self.compress_goal_enc_lan.parameters()}
        bad = [n for n, p in self.named_parameters()
               if p.requires_grad and id(p) not in allowed]
        assert not bad, f"허용 밖 파라미터가 학습 대상이다: {bad[:5]}"
        for pref in ("obs_encoder", "goal_encoder", "goal_encoder_img"):
            nb = [n for n, p in self.named_parameters()
                  if n.startswith(pref) and p.requires_grad]
            assert not nb, f"{pref} 가 동결되지 않았다: {nb[:3]}"
        n_tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        ArmModel.REPORT["trainable_params"] = int(n_tr)
        ArmModel.REPORT["trainable_modules"] = ["compress_goal_enc_lan"]
        ArmModel.REPORT["freeze_check"] = {"trainable_outside_allowed": 0,
                                           "obs_encoder_trainable": 0,
                                           "goal_encoder_trainable": 0,
                                           "goal_encoder_img_trainable": 0}
        print(f"[freeze] 학습 파라미터 {n_tr:,d} (융합부만 — 새 채널 없음)", flush=True)

    def forward(self, obs_img, goal_pose, map_images, goal_img, goal_mask,
                feat_text, current_img, *a, **k):
        if not getattr(ControlModel, "_dbg_done", False):
            ControlModel._dbg_done = True
            print(f"[dbg] CONTROL(3ch) cur={tuple(current_img.shape)} "
                  f"mask_p={self.MASK_P} training={self.training}", flush=True)
        cur = _mask_rgb(self, current_img)
        return super().forward(obs_img, goal_pose, map_images, goal_img, goal_mask,
                               feat_text, cur, *a, **k)
