"""헤드 정의 재사용 — edge_vlm 쪽 정의를 frodo_lan 환경에서도 쓸 수 있게 분리.

`experiments/head_cost_survey.py` 의 H1/H2 와 **같은 구조**다. 항등 초기화는
`loc_head_pilot.make_head` 와 동일하나, 여기서는 **학습된 state_dict 를 덮어쓰므로**
초기화는 무관하다.
"""
import torch
import torch.nn as nn


class H1Linear(nn.Module):
    def __init__(self, d):
        super().__init__(); self.w = nn.Linear(d, d)

    def forward(self, x):
        return self.w(x)


class H2MLP(nn.Module):
    def __init__(self, d, r):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, r), nn.GELU(), nn.Linear(r, d))

    def forward(self, x):
        return x + self.f(x)


def make_head_standalone(name, d, mlp_rank):
    if name == "H1_linear":
        return H1Linear(d)
    if name.startswith("H2_mlp"):
        return H2MLP(d, mlp_rank)
    raise ValueError(name)
