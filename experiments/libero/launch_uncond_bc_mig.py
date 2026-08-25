"""Launch one local UNCOND-BC DDP rank per explicitly assigned MIG UUID."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path


def _rank_environment(
    base: dict[str, str],
    *,
    rank: int,
    world_size: int,
    mig_uuid: str,
    master_addr: str,
    master_port: int,
) -> dict[str, str]:
    environment = dict(base)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": mig_uuid,
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            # Each process sees exactly one MIG, so its process-local ordinal is 0.
            "LOCAL_RANK": "0",
            "LOCAL_WORLD_SIZE": str(world_size),
            "MASTER_ADDR": master_addr,
            "MASTER_PORT": str(master_port),
        }
    )
    return environment


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mig-uuids",
        required=True,
        help="Comma-separated MIG UUIDs in global-rank order.",
    )
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", required=True, type=int)
    parser.add_argument(
        "--entrypoint",
        type=Path,
        default=Path(__file__).with_name("train_uncond_lora_bc.py"),
    )
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    mig_uuids = [item.strip() for item in args.mig_uuids.split(",") if item.strip()]
    if not mig_uuids or len(set(mig_uuids)) != len(mig_uuids):
        raise ValueError("MIG UUIDs must be a non-empty unique sequence.")
    if not 1 <= args.master_port <= 65535:
        raise ValueError("master-port must lie in [1, 65535].")
    if not args.entrypoint.is_file():
        raise FileNotFoundError(args.entrypoint)

    command = [sys.executable, str(args.entrypoint), *args.overrides]
    processes: list[subprocess.Popen[bytes]] = []
    try:
        for rank, mig_uuid in enumerate(mig_uuids):
            environment = _rank_environment(
                os.environ,
                rank=rank,
                world_size=len(mig_uuids),
                mig_uuid=mig_uuid,
                master_addr=args.master_addr,
                master_port=args.master_port,
            )
            processes.append(subprocess.Popen(command, env=environment))

        first_failure = 0
        for process in processes:
            returncode = process.wait()
            if returncode and not first_failure:
                first_failure = returncode
                for peer in processes:
                    if peer.poll() is None:
                        peer.send_signal(signal.SIGTERM)
        return first_failure
    finally:
        for process in processes:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
        for process in processes:
            if process.poll() is None:
                process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
