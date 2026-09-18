"""결과 파일 provenance — CLAUDE.md 절대 규칙 5 를 구조로 강제한다.

    seed · argv · git sha · config 해시가 없는 결과 파일은 무효다.

규칙은 처음부터 있었으나 `experiments/train_B.py` 가 지키지 않았고, B 경로의
**최종 결과 2건이 재현 불가 상태로 커밋됐다**(2026-08-12, D32 정리 중 발견).
소급 기입은 `experiments/backfill_provenance.py` 로 했지만 `argv` 원문은 영영
복원되지 않았다. 규칙을 문서로 두면 다섯 번째가 나온다 — 저장 경로에서 **없으면
실패**하게 만든다.

사용법 — 결과를 쓰는 모든 스크립트에서:

    from src.utils.provenance import collect, dump_result

    dump_result(Path("results/x/y.json"), payload, cfg_path="configs/experiment/x.yaml")

`dump_result` 가 provenance 를 붙이고 완결성을 검사한 뒤에만 쓴다. 검사에 걸리면
`ProvenanceError` 로 **결과를 저장하지 않고** 죽는다. 결과가 안 남는 편이 재현
불가능한 결과가 남는 것보다 낫다.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

REQUIRED = ("seed", "argv", "git", "config", "started_at", "env")


class ProvenanceError(RuntimeError):
    """provenance 가 불완전하다. 결과를 쓰지 않는다."""


def _git(*a: str) -> str | None:
    try:
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def sha256_file(p: str | Path) -> str | None:
    p = Path(p)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(seed: int, cfg_path: str | Path | None = None,
            inputs: list[str | Path] | None = None,
            extra: dict[str, Any] | None = None) -> dict:
    """실행 시점에 부른다. `dump_result` 가 알아서 부르므로 보통 직접 안 쓴다."""
    sha = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain")
    prov: dict[str, Any] = {
        "backfilled": False,
        "seed": seed,
        "argv": sys.argv,
        "cmdline": " ".join(sys.argv),
        "git": {
            "sha": sha,
            "dirty": bool(dirty),
            # dirty 면 sha 만으로 코드가 복원되지 않는다. 무엇이 더러운지 남긴다.
            "dirty_files": dirty.splitlines() if dirty else [],
            "describe": _git("describe", "--always", "--dirty"),
        },
        "config": None,
        "inputs": {},
        "started_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "env": {"python": platform.python_version()},
    }
    if cfg_path is not None:
        prov["config"] = {"path": str(cfg_path), "sha256": sha256_file(cfg_path)}
    for p in inputs or []:
        prov["inputs"][str(p)] = sha256_file(p)
    try:
        import torch
        prov["env"] |= {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception:                                    # torch 없는 스크립트도 있다
        pass
    try:
        import open_clip
        prov["env"]["open_clip"] = open_clip.__version__
    except Exception:
        pass
    if extra:
        prov |= extra
    return prov


def assert_complete(prov: dict | None, *, path: str | Path = "?") -> None:
    """규칙 5 검사. 소급 기입본(`backfilled: True`)은 예외로 통과시킨다."""
    if prov is None:
        raise ProvenanceError(
            f"{path}: provenance 가 없다. 규칙 5 위반이므로 결과를 쓰지 않는다. "
            f"src.utils.provenance.dump_result 를 쓸 것.")
    if prov.get("backfilled"):
        return                                           # 소급 기입본은 지위가 명시돼 있다
    missing = [k for k in REQUIRED if prov.get(k) in (None, {}, [])]
    if missing:
        raise ProvenanceError(
            f"{path}: provenance 필드 누락 {missing}. 규칙 5 위반이므로 결과를 "
            f"쓰지 않는다.")
    if prov["git"].get("sha") is None:
        raise ProvenanceError(f"{path}: git sha 를 못 읽었다. 저장소 밖에서 돌렸는가?")
    if prov.get("config") is not None and prov["config"].get("sha256") is None:
        raise ProvenanceError(
            f"{path}: config 경로 {prov['config']['path']} 의 해시가 None 이다. "
            f"파일이 없다.")


def dump_result(path: str | Path, payload: dict, *, seed: int | None = None,
                cfg_path: str | Path | None = None,
                inputs: list[str | Path] | None = None,
                prov: dict | None = None, indent: int = 2) -> None:
    """provenance 를 붙이고 검사한 뒤에만 쓴다. 검사 실패 시 파일을 만들지 않는다.

    `prov` 를 직접 주면 그것을 쓴다(실행 시작 시점에 `collect` 해 둔 경우).
    아니면 `seed` 로 지금 만든다 — 다만 `started_at` 이 저장 시각이 되므로
    긴 학습에서는 시작 시점에 `collect` 를 부르는 쪽을 권한다.
    """
    if prov is None:
        if seed is None:
            raise ProvenanceError(f"{path}: seed 없이 저장할 수 없다 (규칙 5).")
        prov = collect(seed, cfg_path=cfg_path, inputs=inputs)
    assert_complete(prov, path=path)
    out = dict(payload)
    out["provenance"] = prov
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=indent, ensure_ascii=False))
