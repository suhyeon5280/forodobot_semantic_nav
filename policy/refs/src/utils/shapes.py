"""텐서 shape · 수치 상태 검증 헬퍼.

CLAUDE.md 절대 규칙 4를 코드로 강제하기 위한 모듈. 텐서 연산 전후로 여기의
함수를 호출한다. 규칙이 문서에만 있으면 지켜지지 않으므로 실행 가능한 형태로 둔다.

`expected`의 `None`은 "임의 크기 허용"을 뜻한다:

    assert_shape(patches, (None, None, backbone.embed_dim), "patches")
"""

from __future__ import annotations

import torch
from torch import Tensor


def assert_shape(t: Tensor, expected: tuple[int | None, ...], name: str) -> None:
    """`t.shape`가 `expected`와 맞는지 검사한다. `None` 항은 임의 크기.

    Raises:
        AssertionError: 랭크가 다르거나 `None`이 아닌 축의 크기가 다를 때.
            메시지에 `name`, 기대 shape, 실제 shape를 모두 포함한다.
    """
    actual = tuple(t.shape)
    if len(actual) != len(expected):
        raise AssertionError(
            f"{name}: 랭크 불일치 — 기대 {expected} (랭크 {len(expected)}), "
            f"실제 {actual} (랭크 {len(actual)})"
        )
    for axis, (want, got) in enumerate(zip(expected, actual)):
        if want is not None and want != got:
            raise AssertionError(
                f"{name}: 축 {axis} 크기 불일치 — 기대 {expected}, 실제 {actual}"
            )


def assert_finite(t: Tensor, name: str) -> None:
    """`t`에 NaN·inf가 없는지 검사한다.

    max 축약과 hinge는 NaN을 조용히 전파시키므로 스코어링 경로에서 특히 중요하다.
    """
    if not torch.isfinite(t).all():
        n_nan = int(torch.isnan(t).sum())
        n_inf = int(torch.isinf(t).sum())
        raise AssertionError(f"{name}: 비유한 값 — NaN {n_nan}개, inf {n_inf}개")


def assert_normalized(
    t: Tensor,
    dim: int = -1,
    atol: float = 1e-3,
    *,
    name: str,
    mask: Tensor | None = None,
) -> None:
    """`t`가 `dim` 축으로 L2 정규화되어 있는지 검사한다.

    τ는 절대 임계값이라 정규화 규약이 어긋나면 의미가 바뀐다. 백본 간 τ 비교
    가능성이 여기에 걸려 있다 (decisions.md D10).

    Args:
        mask: 주어지면 True 위치만 검사한다. 패딩 슬롯은 영벡터일 수 있고
            영벡터는 정규화해도 노름이 0이라, 마스크 없이 검사하면 오탐이 난다.
    """
    norms = t.float().norm(dim=dim)
    if mask is not None:
        if mask.shape != norms.shape:
            raise AssertionError(
                f"{name}: mask shape {tuple(mask.shape)} != norm shape {tuple(norms.shape)}"
            )
        norms = norms[mask]
    if norms.numel() == 0:
        return
    dev = (norms - 1.0).abs().max().item()
    if dev > atol:
        raise AssertionError(
            f"{name}: L2 정규화 안 됨 — |‖v‖-1| 최대 {dev:.4f} > atol {atol}. "
            f"노름 범위 [{norms.min().item():.4f}, {norms.max().item():.4f}]"
        )


def assert_same_space(*tensors_with_names: tuple[Tensor, str]) -> None:
    """여러 텐서가 같은 임베딩 차원을 갖는지 검사한다.

    차원이 같다는 것은 공유 공간의 필요조건일 뿐 충분조건이 아니다. 공유 공간
    자체는 `backbones.base.DualEncoder.shares_embedding_space`로 판정한다
    (decisions.md D2).
    """
    if not tensors_with_names:
        return
    ref_d = tensors_with_names[0][0].shape[-1]
    for t, name in tensors_with_names[1:]:
        if t.shape[-1] != ref_d:
            ref_name = tensors_with_names[0][1]
            raise AssertionError(
                f"임베딩 차원 불일치 — {ref_name}: {ref_d}, {name}: {t.shape[-1]}"
            )
