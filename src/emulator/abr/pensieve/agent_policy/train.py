import argparse
import os
import time
import warnings
import subprocess
import numpy as np
import torch.multiprocessing as mp
import sys
sys.path.append('/users/janechen/Genet/src')
sys.path.append('/users/janechen/Genet/src/emulator/abr')
print(sys.path)
from pensieve import Pensieve
from common.utils import set_seed, save_args

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = ''

UP_LINK_SPEED_FILE = "pensieve/data/12mbps"  # example
VIDEO_SIZE_DIR = "pensieve/data/video_sizes"  # example

def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("Training code.")
    parser.add_argument(
        "--jump-action", action="store_true",
        help="Use jump action when specified."
    )
    parser.add_argument(
        "--exp-name", type=str, default="", help="Experiment name."
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="direcotry to save the model.",
    )
    parser.add_argument("--seed", type=int, default=20, help="seed")
    parser.add_argument(
        "--total-epoch",
        type=int,
        default=100,
        help="Total number of epoch to be trained.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="",
        help="Path to a pretrained Tensorflow checkpoint.",
    )
    parser.add_argument(
        "--video-size-file-dir",
        type=str,
        default="",
        help="Path to video size files.",
    )
    parser.add_argument(
        "--nagent",
        type=int,
        default=2,
        help="Path to a pretrained Tensorflow checkpoint.",
    )
    # parser.add_argument(
    #     "--validation",
    #     action="store_true",
    #     help="specify to enable validation.",
    # )
    parser.add_argument(
        "--val-freq",
        type=int,
        default=1000,
        help="specify to enable validation.",
    )
    parser.add_argument(
        "--model-save-interval",
        type=int,
        default=100,
        help="Interval to save the model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size.",
    )
    parser.add_argument(
        "--train-trace-dir",
        type=str,
        default="",
        help="A directory contains the training trace files.",
    )
    subparsers = parser.add_subparsers(dest="curriculum", help="CL parsers.")
    udr_parser = subparsers.add_parser("udr", help="udr")
    udr_parser.add_argument(
        "--real-trace-prob",
        type=float,
        default=0.0,
        help="Probability of picking a real trace in training",
    )
    udr_parser.add_argument(
        "--val-trace-dir",
        type=str,
        default="",
        help="A directory contains the validation trace files.",
    )
    udr_parser.add_argument(
        "--config-file",
        type=str,
        default="",
        help="A json file which contains a list of randomization ranges with "
        "their probabilites.",
    )
    # parser.add_argument(
    #     "--dataset",
    #     type=str,
    #     default="pantheon",
    #     choices=("pantheon", "synthetic"),
    #     help="dataset name",
    # )
    cl1_parser = subparsers.add_parser("cl1", help="cl1")
    cl1_parser.add_argument(
        "--config-files",
        type=str,
        nargs="+",
        help="A list of randomization config files.",
    )
    cl1_parser.add_argument(
        "--val-trace-dir",
        type=str,
        default="",
        help="A directory contains the validation trace files.",
    )
    cl2_parser = subparsers.add_parser("cl2", help="cl2")
    cl2_parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        choices=("mpc", "bba"),
        help="Baseline used to sort environments.",
    )
    cl2_parser.add_argument(
        "--config-file",
        type=str,
        default="",
        help="A json file which contains a list of randomization ranges with "
        "their probabilites.",
    )
    cl2_parser.add_argument(
        "--val-trace-dir",
        type=str,
        default="",
        help="A directory contains the validation trace files.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    save_args(args, args.save_dir)
    set_seed(args.seed)

    ############################################################################
    # 1) Start the video server on port 8000 (or whatever).
    ############################################################################
    video_server_dir = "src/emulator/abr/pensieve/video_server"
    video_server_port = 8000

    # cd into the video_server folder, run python server
    # We run it in background so the main script can proceed.
    print("Starting video server on port={}".format(video_server_port))
    video_server_proc = subprocess.Popen(
        ["python", "video_server.py", "--port={}".format(video_server_port)],
        cwd=video_server_dir
    )
    time.sleep(1.5)  # give it a moment to start

    
    ############################################################################
    # 2) Build the list of (delay, trace_file) pairs from --train-trace-dir
    #    plus a default delay_list = [5, 10, 20, 40, 80].
    ############################################################################
    delay_list = [5, 10, 20, 40, 80]
    trace_dir = args.train_trace_dir
    # print("Trace dir: {}".format(trace_dir))
    all_trace_files = []
    if trace_dir and os.path.exists(trace_dir):
        all_trace_files = [
            f for f in os.listdir(trace_dir)
            if os.path.isfile(os.path.join(trace_dir, f))
        ]
    train_envs = []
    # print("All trace files: {}".format(all_trace_files))
    for trace_file in all_trace_files:
        trace_path = os.path.join(trace_dir, trace_file)
        # print("Trace path: {}".format(trace_path))
        for d in delay_list:
            train_envs.append({"trace_file": trace_path, "delay": d})

    print("Built train_envs with {} total configurations.".format(len(train_envs)))

    
    ############################################################################
    # 3) Initialize Pensieve agent with the training environments.
    ############################################################################
    pensieve = Pensieve(
        num_agents=args.nagent,
        log_dir=args.save_dir,
        actor=None,               # If you have a loaded model, pass it here
        critic_path=None,
        model_save_interval=args.model_save_interval,
        batch_size=args.batch_size,
        randomization='',
        randomization_interval=1,
        video_size_file_dir=args.video_size_file_dir,
        val_traces=args.train_trace_dir,
    )

    ###########################################################################
    # 4) Actually train the model with real networking. 
    #    This calls the multi-process logic that spawns agent processes.
    ###########################################################################
    # For simplicity, we’re not passing “val_envs” or “test_envs” here.
    # They could be built similarly if you want validation / test sets.
    # Also we skip replay_buffer or advanced curriculum logic to keep it short.
    ###########################################################################
    try:
        pensieve.train(train_envs=train_envs,
                       iters=args.total_epoch,
                       save_dir=args.save_dir,
                       use_replay_buffer=False)
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        print("Training failed: {}".format(e))
    finally:
        # 5) When training is done or fails, kill the video_server process
        print("Terminating video server ...")
        video_server_proc.terminate()
        video_server_proc.wait()


if __name__ == "__main__":
    t_start = time.time()
    main()
    print("time used: {:.2f}s".format(time.time() - t_start))
