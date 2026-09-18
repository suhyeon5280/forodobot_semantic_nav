"""시드 고정.

CLAUDE.md 절대 규칙 5의 앞쪽 절반. 뒤쪽 절반(실행 명령 기록)은 `eval/report.py`.
"""

from __future__ import annotations


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """python·numpy·torch(CPU/CUDA)의 난수 상태를 고정한다.

    Args:
        seed: 시드 값. `configs/`에서만 온다 (절대 규칙 3).
        deterministic: True면 cuDNN 결정론 모드를 켠다. 느려지지만 Phase 1은
            학습이 없어 비용이 작다.
    """
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_state() -> dict[str, object]:
    """현재 난수 상태 스냅샷을 반환한다. 결과 파일 메타데이터에 동봉한다."""
    import random
    import torch
    return {"python": random.getstate()[1][0], "torch": int(torch.initial_seed())}
