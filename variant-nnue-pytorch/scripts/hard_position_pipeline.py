#!/usr/bin/env python3
"""Run hard-position mining/rescoring before NNUE training.

This wrapper keeps preprocessing outside the trainer data loader: the C++ tools
binary performs packed-SFEN mining/rescoring, and train.py consumes the resulting
.bin files exactly like any other dataset.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def main():
    parser = argparse.ArgumentParser(description="Mine hard positions, optionally deep-rescore them, then launch train.py.")
    parser.add_argument("--tools", required=True, help="Path to the variant-nnue-tools executable.")
    parser.add_argument("--train-input", required=True, help="Original training .bin file.")
    parser.add_argument("--val", required=True, help="Validation .bin file passed to train.py.")
    parser.add_argument("--mine-output", required=True, help="Mined .bin output path.")
    parser.add_argument("--rescored-output", help="If set, deep-rescore mined data to this .bin and train on it.")
    parser.add_argument("--mode", default="search-gap", choices=["search-gap", "eval-disagree"], help="Mining mode.")
    parser.add_argument("--net", help="NNUE for search-gap mining.")
    parser.add_argument("--old-net", help="Old NNUE for eval-disagree mining.")
    parser.add_argument("--new-net", help="New NNUE for eval-disagree mining.")
    parser.add_argument("--variant", help="Variant to set before NNUE loading; if omitted, tools infer it from the NNUE filename when possible.")
    parser.add_argument("--keep-count", type=int, default=1_000_000, help="Positions to keep after mining.")
    parser.add_argument("--min-gap", type=int, default=0, help="Minimum disagreement needed to keep a position.")
    parser.add_argument("--rescore-depth", type=int, default=8, help="Depth for optional deep rescoring.")
    parser.add_argument("--rescore-nodes", type=int, default=0, help="Node cap for optional deep rescoring; 0 means depth-only.")
    parser.add_argument("--rescore-research-count", type=int, default=0, help="Additional searches before final rescoring search.")
    parser.add_argument("--train-py", default=str(Path(__file__).resolve().parents[1] / "train.py"), help="Path to train.py.")
    parser.add_argument("train_args", nargs=argparse.REMAINDER, help="Arguments after -- are forwarded to train.py.")
    args = parser.parse_args()

    mine_cmd = [args.tools, "transform", "mine", "--mode", args.mode,
                "--input", args.train_input, "--output", args.mine_output,
                "--keep-count", args.keep_count, "--min-gap", args.min_gap]
    if args.variant:
        mine_cmd.extend(["--variant", args.variant])

    if args.mode == "search-gap":
        if not args.net:
            parser.error("--net is required for --mode search-gap")
        mine_cmd.extend(["--net", args.net])
    else:
        if not args.old_net or not args.new_net:
            parser.error("--old-net and --new-net are required for --mode eval-disagree")
        mine_cmd.extend(["--old-net", args.old_net, "--new-net", args.new_net])
    run(mine_cmd)

    train_data = args.mine_output
    if args.rescored_output:
        run([args.tools, "transform", "rescore", "--input", args.mine_output,
             "--output", args.rescored_output, "--depth", args.rescore_depth,
             "--nodes", args.rescore_nodes, "--research-count", args.rescore_research_count])
        train_data = args.rescored_output

    forwarded = args.train_args[1:] if args.train_args[:1] == ["--"] else args.train_args
    run([sys.executable, args.train_py, train_data, args.val, *forwarded])


if __name__ == "__main__":
    main()
