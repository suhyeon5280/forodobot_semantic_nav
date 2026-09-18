"""스코어러 프로토콜 + **정규화의 유일한 지점**.

ablation의 각 행이 이 인터페이스의 구현체 하나다. 행을 바꾸는 것이 스코어러를
바꾸는 것과 정확히 같아야, 2행 대 3행의 격차가 다른 변화에 오염되지 않는다.

    1행   single_vector.SingleVectorScorer
    2행   late_interaction.LateInteractionScorer
    3a/3b polarity.PolarityScorer          ← 기여가 측정되는 지점

**L2 정규화는 여기서만 한다** (decisions.md D10). 백본 래퍼는 raw feature를
그대로 내놓고, 정규화는 `Scorer.score()`가 `_score()`를 부르기 전에 단 한 번
적용한다. 두 군데서 하면 조용히 이중 정규화되고, 이중 정규화는 값이 여전히
norm 1이라 assert로도 안 잡힌다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor

from ..backbones.base import ImageEncoding, TextEncoding
from ..slots.extract import SlotSet
from ..utils.shapes import assert_finite, assert_normalized, assert_shape


def l2_normalize(t: Tensor, eps: float = 1e-8) -> Tensor:
    """마지막 축 L2 정규화. `eps`는 zero-norm 벡터에서의 0 나눗셈 방지용."""
    return t / t.norm(dim=-1, keepdim=True).clamp_min(eps)


def normalize_inputs(
    image: ImageEncoding, text: TextEncoding, slots: SlotSet
) -> tuple[ImageEncoding, TextEncoding, SlotSet]:
    """`patches` · `pooled` · `tokens` · `slots`를 모두 L2 정규화한다.

    **이 함수를 `Scorer.score()` 밖에서 부르지 말 것.** 정규화 지점이 하나라는
    것이 D10의 요구사항이다.

    백본 실측(세션 2)에서 raw norm이 CLIP 7.4 / MaskCLIP 8.3 / SigLIP2 49.6으로
    6배 이상 벌어져 있었다. 정규화하지 않으면 `⟨s_k, v_n⟩`의 절대 스케일이
    백본마다 달라 τ가 백본 간 비교 불가능한 양이 된다.
    """
    return (
        replace(image, patches=l2_normalize(image.patches), pooled=l2_normalize(image.pooled)),
        replace(text, tokens=l2_normalize(text.tokens), pooled=l2_normalize(text.pooled)),
        replace(slots, slots=l2_normalize(slots.slots)),
    )


def similarity(slots: Tensor, patches: Tensor) -> Tensor:
    """`⟨s_k, v_n⟩` 전체 → `[B_img, B_txt, K, N]`.

    세 항(긍정 · 전역 부정 · 국소 부정)이 **이 하나를 공유**한다. 두 번 계산하면
    비용이 배가 되고 항 간 수치가 어긋날 여지가 생긴다.
    """
    return torch.einsum("tkd,ind->itkn", slots, patches)


def attention_map(sim: Tensor, temperature: float) -> Tensor:
    """`A = softmax(sim / T)` — patch 축(N) 기준. `[B_img, B_txt, K, N]`.

    `sim`에서 파생시키는 이유: A는 (S, V, T)의 결정적 함수라 별도 인자로 받으면
    `sim`과 다른 정규화에서 나온 A가 들어올 수 있다. `spatial.py`(Phase 2)도
    이 함수를 쓴다.
    """
    return F.softmax(sim / temperature, dim=-1)


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    """`[B_img, B_txt, K]`를 마스크된 슬롯 축 평균으로 → `[B_img, B_txt]`.

    유효 슬롯이 하나도 없으면 **정확히 0**을 낸다. 0으로 나누지 않는다 —
    3a↔3b 동치 불변식이 여기서 깨진다.
    """
    m = mask.to(values.dtype)
    total = (values * m).sum(dim=-1)
    count = m.sum(dim=-1)
    return total / count.clamp_min(1.0)


def hinge(values: Tensor, tau: float) -> Tensor:
    """`[x − τ]₊`. `x == τ`이면 정확히 0 (경계는 닫힌 쪽이 0)."""
    return (values - tau).clamp_min(0.0)


class Scorer(ABC):
    """(이미지 인코딩, 텍스트 인코딩, 슬롯 집합) → 쌍별 점수 행렬.

    세 인자를 모두 받는 이유: 1행은 `TextEncoding.pooled`만 쓰고 2·3행은
    `SlotSet`만 쓴다. 어느 한쪽만 넘기는 인터페이스로 두면 나머지 행이 우회
    경로를 만들게 되고, 그러면 행 간 비교가 스코어러 교체만으로 성립하지 않는다.
    """

    name: str

    uses_polarity: bool
    """극성을 실제로 소비하는가. False면 `SlotSet.polarity`를 무시한다.

    2행이 극성을 무시한다는 사실을 명시적으로 들고 있어야, 3행과의 비교가
    "극성을 켰다/껐다"임이 코드에서 드러난다.
    """

    def score(self, image: ImageEncoding, text: TextEncoding, slots: SlotSet) -> Tensor:
        """`[B_img, B_txt]` 점수 행렬. **하위 클래스가 재정의하지 않는다.**

        정규화 → 사전 assert → `_score()` → 사후 assert 순서를 고정한다.
        재정의하면 정규화가 우회되므로, 스코어러는 `_score()`만 구현한다.

        전체 쌍을 한 번에 내는 것이 인터페이스인 이유: N개 후보 채점이 행렬곱
        1회라는 성질이 dual encoder를 고집하는 근거 중 하나다. 쌍마다 호출하는
        형태로 두면 그 성질이 코드에서 사라진다.
        """
        image, text, slots = normalize_inputs(image, text, slots)

        # D10: 정규화가 실제로 걸렸는지 여기서 확인한다 (절대 규칙 4)
        assert_normalized(image.patches, dim=-1, name="patches")
        assert_normalized(slots.slots, dim=-1, name="slots", mask=slots.mask)

        out = self._score(image, text, slots)

        assert_shape(out, (image.patches.shape[0], slots.slots.shape[0]), "score")
        assert_finite(out, "score")
        return out

    @abstractmethod
    def _score(self, image: ImageEncoding, text: TextEncoding, slots: SlotSet) -> Tensor:
        """실제 채점. 입력은 **이미 L2 정규화된 상태**로 들어온다.

        구현은 절대 다시 정규화하지 않는다. `slots.mask`가 False인 위치가
        max와 평균의 분모에 참여하지 않게 하는 것은 구현 책임이다.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "Scorer":
        raise NotImplementedError
