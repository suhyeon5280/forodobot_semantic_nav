"""백본 추상 인터페이스.

하위 모듈 전체가 백본에 무관하게 동작하도록 만드는 지점. 스코어러·평가·진단은
구체 백본을 절대 직접 임포트하지 않는다.

표기는 CLAUDE.md 표를 따른다: `patches` = V ∈ R^(N×d), `slots` = S ∈ R^(K×d).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from torch import Tensor


@dataclass(frozen=True)
class ImageEncoding:
    """이미지 한 배치의 인코딩 결과.

    Attributes:
        patches: `[B, N, d]`. pooling 하지 않은 patch token. late interaction의 V.
        pooled: `[B, d]`. 단일 벡터 baseline(ablation 1행)이 쓰는 값.
        grid: `(H, W)`. `H * W == N`. soft-argmax로 patch 인덱스를 좌표로 되돌릴 때
            필요하다. 공간 관계 판정이 여기에 의존한다.
    """

    patches: Tensor
    pooled: Tensor
    grid: tuple[int, int]


@dataclass(frozen=True)
class TextEncoding:
    """텍스트 한 배치의 인코딩 결과.

    슬롯 분해 이전의 원시 토큰 수준 출력이다. K개 슬롯으로의 축약은 `slots/`가 한다.

    Attributes:
        tokens: `[B, T, d]`. 토큰별 임베딩.
        pooled: `[B, d]`. 단일 벡터 baseline이 쓰는 값 (CLIP의 EOS 임베딩 등).
        mask: `[B, T]` bool. 유효 토큰 위치. 패딩에 max가 걸리는 사고를 막는다.
    """

    tokens: Tensor
    pooled: Tensor
    mask: Tensor


class VisionTower(ABC):
    """이미지 → patch tokens."""

    embed_dim: int
    """공유 공간의 차원. `TextTower.embed_dim`과 같아야 한다."""

    @abstractmethod
    def preprocess(self, images: Sequence[Any]) -> Tensor:
        """PIL 이미지들을 이 백본 전용 전처리로 `[B, 3, H, W]` 텐서화한다.

        해상도·정규화 상수는 백본마다 다르므로 절대 공유하지 않는다.
        """
        raise NotImplementedError

    @abstractmethod
    def encode_image(self, pixel_values: Tensor) -> ImageEncoding:
        """`[B, 3, H, W]` → `ImageEncoding`.

        **`patches`를 정규화하지 않고 raw 상태로 반환한다.** L2 정규화는
        `scoring.base.Scorer.score()`에서만 한다 (decisions.md D10). 백본에서도
        하면 이중 정규화가 되고, 이중 정규화는 결과가 여전히 norm 1이라
        assert로 잡히지 않는다.
        """
        raise NotImplementedError


class TextTower(ABC):
    """텍스트 → 토큰 임베딩. Phase 1에서 에피소드당 1회만 호출된다."""

    embed_dim: int

    @abstractmethod
    def tokenize(self, texts: Sequence[str]) -> Mapping[str, Tensor]:
        """문자열들을 이 백본의 토크나이저로 인코딩한다."""
        raise NotImplementedError

    @abstractmethod
    def encode_text(self, tokens: Mapping[str, Tensor]) -> TextEncoding:
        """토큰 → `TextEncoding`."""
        raise NotImplementedError


class DualEncoder(ABC):
    """vision tower + text tower 한 쌍.

    `shares_embedding_space`가 이 클래스의 존재 이유다. DINOv2 + CLIP text처럼
    함께 학습된 적 없는 조합은 차원을 맞춰도 내적이 의미를 갖지 않는다. 그런
    조합이 스코어링 경로로 조용히 흘러드는 것을 타입 수준에서 막는다
    (decisions.md D2).
    """

    name: str

    shares_embedding_space: bool
    """vision·text 임베딩의 내적이 의미를 갖는가.

    False면 스코어링에 쓸 수 없다. `assert_scorable()`이 이를 강제한다.
    """

    vision: VisionTower
    text: TextTower

    @classmethod
    @abstractmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "DualEncoder":
        """설정에서 백본을 구성한다. 하이퍼파라미터의 유일한 출처 (절대 규칙 3)."""
        raise NotImplementedError

    def assert_scorable(self) -> None:
        """이 백본을 스코어링에 쓸 수 있는지 검사한다.

        Raises:
            SharedSpaceError: `shares_embedding_space`가 False일 때.
        """
        raise NotImplementedError


class SharedSpaceError(RuntimeError):
    """공유 임베딩 공간이 없는 백본을 스코어링에 쓰려 했을 때 (decisions.md D2)."""
