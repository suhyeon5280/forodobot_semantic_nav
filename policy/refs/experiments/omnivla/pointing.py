"""D118 §2 — 지목 정확도 정의. **한 곳에 고정.** E1·E2 가 같은 함수를 부른다.

정의 (D118 §2, 눈금 출처 = `OmniVLA_edge/train/eval/grounding_test.py` target-selection)

    동종 후보 둘의 프롬프트 각각에 대해 모델이 낸 **궤적 끝점 횡위치** e_i, e_j 가
    두 객체의 **실제 횡위치** L_i, L_j (pose_median) 와 **같은 순서**인가
        correct  <=>  sign(e_i - e_j) == sign(L_i - L_j)

단서 — 1차원 눈금이라 **전후 배치는 판별 불가**. |L_i - L_j| < min_sep 인 쌍은
제외하고 **제외 수를 병기한다** (D118 §2-b). min_sep 은 grounding_test 기본값 0.3 m.
"""
from __future__ import annotations

import numpy as np

MWS = 0.125          # metric waypoint scaling — grounding_test.py 의 mws
LATERAL_IDX = 1      # action_pred[:, -1, 1] = 끝점 횡위치 (+ = 좌)


def endpoint_lateral(action_pred):
    """action_pred [B, T, 4] → 끝점 횡위치 [B] (m). grounding_test 와 같은 식."""
    a = action_pred.detach().cpu().numpy() if hasattr(action_pred, "detach") else np.asarray(action_pred)
    return a[:, -1, LATERAL_IDX] * MWS


def scorable_pairs(lat_true, min_sep):
    """횡위치 차가 min_sep 이상인 쌍의 인덱스. 제외 수도 낸다."""
    n = len(lat_true)
    keep, drop = [], 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(lat_true[i] - lat_true[j]) < min_sep:
                drop += 1
            else:
                keep.append((i, j))
    return keep, drop


def pair_correct(e, lat_true, pairs):
    """쌍별 정오 벡터 [len(pairs)]. **E1·E2 가 이 함수만 쓴다.**"""
    out = np.empty(len(pairs), dtype=np.float64)
    for k, (i, j) in enumerate(pairs):
        out[k] = 1.0 if np.sign(e[i] - e[j]) == np.sign(lat_true[i] - lat_true[j]) else 0.0
    return out


def score_frame(action_pred, lat_true, min_sep):
    """한 프레임의 (정오 벡터, 제외 쌍 수). action_pred 행 순서 = 프롬프트 순서."""
    e = endpoint_lateral(action_pred)
    pairs, drop = scorable_pairs(lat_true, min_sep)
    return (pair_correct(e, lat_true, pairs) if pairs else np.empty(0)), drop, len(pairs)
