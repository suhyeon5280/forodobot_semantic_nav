"""Grounding teacher — Grounding DINO로 (이미지, 구문) → 박스 감독 신호 생성.

**Grounding DINO를 쓰는 이유는 공유 편향 회피다.** OWLv2·OWL-ViT는 vision tower가
CLIP 계열이라, 그것으로 만든 라벨로 CLIP patch를 학습시키면 순환이 된다 — D8이
τ_L 라벨원에서 OWL-ViT를 기각한 것과 같은 논리다. Grounding DINO는 vision이 Swin,
text가 BERT라 이 문제가 없다.

출력은 patch 격자 위의 목표 분포다. 박스를 그대로 쓰지 않고 격자로 래스터화하는
이유는 학생이 예측하는 것이 박스가 아니라 patch 분포이기 때문이다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

MODEL_ID = "IDEA-Research/grounding-dino-tiny"


@dataclass(frozen=True)
class GroundedPhrase:
    """구문 하나에 대한 teacher 출력.

    Attributes:
        phrase: 원 구문.
        box: `[x0, y0, x1, y1]` 정규화 좌표 (0~1). 검출 실패면 None.
        score: teacher 신뢰도. 실패면 0.
    """

    phrase: str
    box: tuple[float, float, float, float] | None
    score: float


class GroundingTeacher:
    """Grounding DINO 래퍼. 배치로 (이미지, 구문 목록) → 구문별 최고 점수 박스."""

    def __init__(self, device: str, model_id: str = MODEL_ID, box_threshold: float = 0.25):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
        self.device = device
        self.box_threshold = box_threshold

    @torch.no_grad()
    def ground(self, image, phrases: Sequence[str]) -> list[GroundedPhrase]:
        """이미지 하나 + 구문 여러 개 → 구문별 박스.

        Grounding DINO는 `"a. b. c."` 형태의 마침표 구분 프롬프트를 받는다. 구문마다
        따로 호출하면 N배 느려지므로 한 번에 넣고 반환된 라벨로 되매핑한다.
        """
        if not phrases:
            return []
        clean = [p.strip().rstrip(".").lower() for p in phrases]
        prompt = ". ".join(clean) + "."
        inputs = self.proc(images=image, text=prompt, return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        w, h = image.size
        res = self.proc.post_process_grounded_object_detection(
            out, inputs.input_ids, threshold=self.box_threshold,
            text_threshold=self.box_threshold, target_sizes=[(h, w)])[0]

        best: dict[str, tuple[float, tuple[float, float, float, float]]] = {}
        labels = res.get("text_labels", res.get("labels", []))
        for box, score, lab in zip(res["boxes"], res["scores"], labels):
            lab = str(lab).strip().lower()
            hit = next((c for c in clean if c and (lab in c or c in lab)), None)
            if hit is None:
                continue
            sc = float(score)
            if hit not in best or sc > best[hit][0]:
                x0, y0, x1, y1 = [float(v) for v in box]
                best[hit] = (sc, (x0 / w, y0 / h, x1 / w, y1 / h))

        return [GroundedPhrase(p, best[c][1] if c in best else None,
                               best[c][0] if c in best else 0.0)
                for p, c in zip(phrases, clean)]


def box_to_grid(box: tuple[float, float, float, float], side: int,
                device: str = "cpu") -> Tensor:
    """정규화 박스 → `[side*side]` 목표 분포. patch 셀의 박스 피복률에 비례.

    이진 마스크가 아니라 피복률을 쓰는 이유: 박스 경계에 걸친 셀이 절반만 포함되는데
    이진화하면 그 정보가 버려지고, 작은 박스는 셀 하나로 붕괴한다.
    """
    x0, y0, x1, y1 = box
    xs = torch.linspace(0, 1, side + 1, device=device)
    ov_x = (torch.clamp(torch.minimum(xs[1:], torch.tensor(x1, device=device)), min=0)
            - torch.maximum(xs[:-1], torch.tensor(x0, device=device))).clamp_min(0)
    ov_y = (torch.clamp(torch.minimum(xs[1:], torch.tensor(y1, device=device)), min=0)
            - torch.maximum(xs[:-1], torch.tensor(y0, device=device))).clamp_min(0)
    grid = ov_y[:, None] * ov_x[None, :]          # [side, side]
    total = grid.sum()
    if total <= 0:                                 # 박스가 격자보다 작아 전부 0인 경우
        cx = int(min(side - 1, max(0, (x0 + x1) / 2 * side)))
        cy = int(min(side - 1, max(0, (y0 + y1) / 2 * side)))
        grid = torch.zeros(side, side, device=device)
        grid[cy, cx] = 1.0
        total = grid.sum()
    return (grid / total).flatten()


def box_to_mask(box: tuple[float, float, float, float], side: int,
                device: str = "cpu", thresh: float = 0.5) -> Tensor:
    """정규화 박스 → `[side*side]` bool. 셀 면적의 `thresh` 이상 덮이면 True.

    `L_max`(in-box sim > out-of-box sim margin)가 이 마스크를 쓴다.
    """
    g = box_to_grid(box, side, device)
    cell = g * (side * side)                       # 균등 대비 배율
    return cell >= thresh * cell.max().clamp_min(1e-8) if cell.max() > 0 else g > 0
