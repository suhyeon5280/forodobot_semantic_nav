"""텍스트 → K개 슬롯 S ∈ R^(K×d).

Phase 1의 기본값은 `TokenSlots` — FILIP처럼 토큰 임베딩을 그대로 슬롯으로 쓴다.
`NounPhraseSlots`는 명사구 단위로 묶는 변형이고, 어느 쪽을 쓰는지에 따라 결과가
꽤 달라질 수 있으므로 설정으로 고른다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from torch import Tensor

from ..backbones.base import TextEncoding

GLOBAL = 0
LOCAL = 1
"""`SlotSet.scope`의 값. decisions.md D7."""


@dataclass(frozen=True)
class SlotSet:
    """한 배치의 슬롯 집합.

    Attributes:
        slots: `[B, K, d]`. 수식의 S.
        polarity: `[B, K]`. 값은 +1 또는 -1. 수식의 π_k.
            **부정 슬롯도 내용은 긍정과 동일하게 인코딩된다** — 부재는 시각
            특징이 아니므로 부호만 뒤집는다. 이 덕분에 다중 부정이 자동으로
            일반화된다.
        scope: `[B, K]`. `GLOBAL`(0) 또는 `LOCAL`(1). 부정 슬롯에만 의미가 있고
            긍정 슬롯에서는 무시된다 (D7).
        anchor: `[B, K]` int64. 국소 부정 슬롯이 참조하는 앵커 슬롯 인덱스.
            수식의 α(k). 전역이거나 무의미한 위치는 -1.
        mask: `[B, K]` bool. 유효 슬롯 위치. 패딩 슬롯에 max가 걸리지 않게 한다.
        spans: 슬롯이 원문의 어느 토큰 구간에서 왔는지. 디버깅·오류 분석용.
        texts: 슬롯별 원문 조각. 결과 파일에 함께 남겨 정성 분석에 쓴다.
    """

    slots: Tensor
    polarity: Tensor
    scope: Tensor
    anchor: Tensor
    mask: Tensor
    spans: Sequence[Sequence[tuple[int, int]]] = field(default=())
    texts: Sequence[Sequence[str]] = field(default=())

    @property
    def pos_idx(self) -> Tensor:
        """`[B, K]` bool. K⁺ — 긍정이면서 유효한 슬롯."""
        return (self.polarity > 0) & self.mask

    @property
    def neg_idx(self) -> Tensor:
        """`[B, K]` bool. K⁻ 전체 — 부정이면서 유효한 슬롯."""
        return (self.polarity < 0) & self.mask

    @property
    def neg_global_idx(self) -> Tensor:
        """`[B, K]` bool. K⁻_G — 전역 부재 부정 (D7)."""
        return self.neg_idx & (self.scope == GLOBAL)

    @property
    def neg_local_idx(self) -> Tensor:
        """`[B, K]` bool. K⁻_L — 국소 속성 부정 (D7)."""
        return self.neg_idx & (self.scope == LOCAL)


class SlotExtractor(ABC):
    """`TextEncoding` → `SlotSet`."""

    name: str

    @abstractmethod
    def extract(self, texts: Sequence[str], encoding: TextEncoding) -> SlotSet:
        """토큰 임베딩을 슬롯으로 축약하고 극성·스코프·앵커를 붙인다.

        극성 태깅 자체는 `polarity.py`에 위임한다.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "SlotExtractor":
        raise NotImplementedError


class TokenSlots(SlotExtractor):
    """토큰 임베딩을 그대로 슬롯으로 쓴다 (FILIP 방식). Phase 1 기본값.

    K = 유효 토큰 수. 축약이 없으므로 슬롯 분해가 결과에 개입하지 않는다 —
    ablation 2행을 순수하게 측정하려면 이쪽이 맞다.
    """


class NounPhraseSlots(SlotExtractor):
    """명사구 단위로 토큰을 묶어 슬롯을 만든다.

    K가 훨씬 작아지고 슬롯 하나가 "빨간 큐브"처럼 속성+객체를 함께 담는다.
    속성 결합을 슬롯 내부에 가두므로 bag-of-words 실패 유형 중 하나를 구조적으로
    차단할 수 있으나, 청킹 품질에 결과가 종속된다.
    """
