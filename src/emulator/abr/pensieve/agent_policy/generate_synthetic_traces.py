import argparse
import os
import sys
sys.path.append('/users/janechen/Genet/src')
sys.path.append('/users/janechen/Genet/src/emulator/abr')
print(sys.path)
from pensieve import Pensieve
from simulator.abr_simulator.schedulers import (
    UDRTrainScheduler,
)
# from pensieve import create_mask
from common.utils import set_seed, save_args

def parse_args():
    parser = argparse.ArgumentParser("Generating synthetic traces.")
    parser.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to the output directory.",
    )
    parser.add_argument(
        "--num-traces",
        type=int,
        default=100,
        help="Number of traces to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Generating {args.num_traces} synthetic traces.")
    set_seed(args.seed)
    config_file = args.config_file
    train_scheduler = UDRTrainScheduler(config_file, [], 0.0)

    for i in range(args.num_traces):
        # print("Generating synthetic trace", i)
        abr_trace = train_scheduler.get_trace()
        trace_path = os.path.join(args.output_dir, f"{args.seed}_synthetic_trace_{i}")
        abr_trace.convert_to_mahimahi_format(trace_path)

if __name__ == "__main__":
    main()