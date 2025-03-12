import subprocess
import time
import os
import numpy as np
import torch
from models import create_mask
from collections import defaultdict
import math

AGGREGATION_WINDOW_MS = 80 
WINDOW = 10
EMBEDDING_SIZE = 16
S_LEN = 6 
S_INFO = 6
DEVICE = 'cpu'



transformer = torch.load("/users/janechen/Genet/src/emulator/abr/pensieve/agent_policy/Checkpoint-Combined_10RTT_6col_Transformer3_64_5_5_16_4_lr_1e-05-999iter.p", map_location='cpu')
transformer.eval()


class TCPStatsAggregator:
    def __init__(self):
        self.window_data = defaultdict(list)
        self.window_start_time = None
        
    def aggregate_window_data(self):
        # if not self.window_data:
        #     return None
            
        stats = {}
        
        print(len(self.window_data['srtt_us']))
        rtts = [float(x) / 100000.0 / 8  for x in self.window_data['srtt_us']]
        stats['rtt_ms'] = np.mean(rtts) if rtts else 0
        
        rttvars = [float(x) / 1000.0 for x in self.window_data['rttvar']]
        stats['rttvar_ms'] = np.mean(rttvars) if rttvars else 0

        cwnd_rate = [float(x) for x in self.window_data['cwnd_rate']]
        stats['cwnd_rate'] = np.mean(cwnd_rate) if cwnd_rate else 0

        l_w_mbps = [float(x) * 8.0 for x in self.window_data['l_w_mbps']]
        stats['l_w_mbps'] = np.mean(l_w_mbps) if cwnd_rate else 0

        delivery_rate = [float(x) / 125000.0 / 100 for x in self.window_data['delivery_rate']]
        stats['delivery_rate'] = np.mean(delivery_rate) if cwnd_rate else 0
        
        stats['window_start'] = self.window_start_time
        stats['num_packets'] = len(rtts)
        
        return stats

################################################################################
# Helper functions
################################################################################

def get_last_line(file_path):
    # If the file is empty, return an empty string (or handle however you prefer)
    if os.path.getsize(file_path) == 0:
        return ""
    
    with open(file_path, 'rb') as f:
        # Move back from the end until we find a newline or reach the beginning
        f.seek(-2, os.SEEK_END)
        while f.tell() > 0:
            char = f.read(1)
            if char == b'\n':
                break
            f.seek(-2, os.SEEK_CUR)
        return f.readline().decode()


def get_bftrace_out_path(agent_id):
    return f"/mydata/logs/bpftrace_out_{agent_id}.txt"


################################################################################
# Embedding functions
################################################################################

def compute_token(bpftrace_path):
    """
    Spawns the TokenAggregator, which continuously reads bpftrace_output.txt
    and puts tokens into `token_queue`.
    """
    aggregator = TCPStatsAggregator()
    metrics_data = []
    window_duration = 80
    
    try:
        last_line = get_last_line(bpftrace_path)
        last_time_ms = float(last_line.strip().split(',')[0]) / 1000000
        # print("last_time_ms: ", last_time_ms)
    except Exception as e:
        print(f"Error reading the last line: {e}")
        return np.array(metrics_data)
    first_time_ms = None
    with open(bpftrace_path, "r") as f:
        f.readline()
        rtt_window_count = 10 
        lost_packets = 0
        pre_cwnd = 0
        for line in f:
            if line.startswith('A') or line.startswith('t'):
                print(f"Skipping line with invalid data")
                continue
            parts = line.strip().split(',')
            if len(parts) < 9:
                # Ensure there are enough parts to prevent IndexError
                print(f"Skipping line with insufficient data")
                continue
            try:
                current_time_ns = float(parts[0])
                current_time_ms = current_time_ns / 1000000
                time_diff = last_time_ms - current_time_ms
                if not first_time_ms:
                    first_time_ms = current_time_ms
                    print("first_time_ms: ", first_time_ms)
            except ValueError:
                # Skip lines with invalid numerical data
                print(f"Skipping line with invalid numerical data")
                continue
            if time_diff < rtt_window_count * window_duration and current_time_ms <= last_time_ms:
                if not aggregator.window_start_time:
                    aggregator.window_start_time = current_time_ms
                
                # Determine if we've moved into a new window
                if time_diff <= (rtt_window_count - 1) * window_duration:
                    rtt_window_count -= 1
                    print("time_diff: ", time_diff)
                    print("rtt_window_count: ", rtt_window_count)
                    print("window_duration: ", window_duration)
                    # Aggregate data for the current window
                    stats = aggregator.aggregate_window_data()
                    if stats:
                        metrics_data.append([
                            0.7/1.5,
                            stats['rtt_ms'],
                            stats['rttvar_ms'],
                            stats['delivery_rate'],
                            stats['l_w_mbps'],
                            stats['cwnd_rate']
                        ])

                        print(f"Window starts {stats['window_start']:.2f}ms "
                            f"({stats['num_packets']} packets):\n"
                            f"  Average RTT: {stats['rtt_ms']:.2f}ms\n"
                            f"  RTT Variance: {stats['rttvar_ms']:.2f}ms\n"
                            f"  Delivery Rate: {stats['delivery_rate']:.2f}\n"
                            f"  L W mbps: {stats['l_w_mbps']:.2f}\n"
                            f"  Cwnd rate: {stats['cwnd_rate']:.2f}\n")
                    
                    aggregator = TCPStatsAggregator()
                    aggregator.window_start_time = current_time_ms
                
                lost_packets += float(parts[6])
                
                if current_time_ms - aggregator.window_start_time > WINDOW:
                    aggregator.window_data['time_ms'].append(current_time_ms)
                    aggregator.window_data['srtt_us'].append(parts[1])
                    aggregator.window_data['rttvar'].append(parts[2])
                    rate_interval = float(parts[4])
                    if rate_interval == 0:
                        rate_interval = 1
                    aggregator.window_data['delivery_rate'].append(float(parts[3])*float(parts[5])*1000000 / rate_interval)
                    aggregator.window_data['l_w_mbps'].append(lost_packets/window_duration)
                    cwnd = float(parts[-3])
                    if pre_cwnd>0:
                        target_cwnd = cwnd/pre_cwnd
                        cwnd_rate = round(math.log2(target_cwnd)*1000)/1000
                    else:
                        cwnd_rate = math.log2(0.0001)
                    aggregator.window_data['cwnd_rate'].append(cwnd_rate)
                    lost_packets = 0
                    pre_cwnd = cwnd
    
    # Convert to numpy array and save
    metrics_array = np.array(metrics_data)
    # np.save('tcp_metrics_reno_trace_file_1.npy', metrics_array)
    # open(bpftrace_path, 'w').close()  # simple truncation
    return metrics_array


def add_embedding(state, tokens, embeddings):
    """
    Appends the Transformer embedding of tokens to the state.
    
    Args:
        state (np.ndarray): The current state array. Shape: [S_INFO, S_LEN]
        tokens (list or np.ndarray): List of the latest 10 tokens, each with 6 features. Shape: [10, 6]
        embeddings (np.ndarray): Current embeddings array. Shape: [EMBEDDING_SIZE, S_LEN]
        transformer (nn.Module): The Transformer model.
        create_mask_func (function): Function to create masks.
        WINDOW (int): Required number of tokens to compute embedding.
        S_LEN (int): Length of the embedding window.
        DEVICE (str): Device to perform computations on.
        
    Returns:
        np.ndarray: Updated state with the embedding appended. Shape: [S_INFO + EMBEDDING_SIZE, S_LEN]
        np.ndarray: Updated embeddings array. Shape: [EMBEDDING_SIZE, S_LEN]
    """
    if len(tokens) < WINDOW:
        # Not enough tokens to compute embedding
        print("Not enough tokens to compute embedding. Using zero embeddings.")
        # if state has 3 dimensions, squeeze
        print("Not enough tokens state shape before embedding:", state.shape)  # [6, 6]
        if len(state.shape) == 3 and state.shape[1] == S_INFO:
            updated_state = np.concatenate((state.squeeze(0), embeddings), axis=0)  # [6 + 64, 6] = [70, 6]
        elif state.shape[0] == S_INFO:
            updated_state = np.concatenate((state, embeddings), axis=0)
        return updated_state, embeddings
    
    # Convert tokens to NumPy array if not already
    tokens_np = np.array(tokens)  # Shape: [10, 6]
    tokens_tensor = torch.from_numpy(tokens_np).unsqueeze(0).float().to(DEVICE)  # Shape: [1, 10, 6]
    print("tokens_tensor:", tokens_tensor.shape)
    
    with torch.no_grad():
        enc_input = tokens_tensor[:, :, :].to(DEVICE)
        dec_input = (1.5 * torch.ones((tokens_tensor.shape[0], 10, tokens_tensor.shape[2]))).to(DEVICE)
        src_mask, tgt_mask, _, _ = create_mask(enc_input, dec_input, pad_idx=2, device=DEVICE)
        
        # Transformer forward pass
        encoder_out = transformer(
            enc_input, dec_input, 
            src_mask=src_mask, 
            tgt_mask=tgt_mask,
            src_padding_mask=None, 
            tgt_padding_mask=None, 
            memory_key_padding_mask=None
        )
        
        # Compute the mean embedding across the sequence length
        embedding_tensor = encoder_out.mean(dim=1)  # Shape: [1, EMBEDDING_SIZE]
        print("embedding_tensor:", embedding_tensor.shape)
        
        # Move to CPU and convert to NumPy
        embedding = embedding_tensor.squeeze(0).cpu().numpy().astype(np.float32)  # Shape: [64]
        
        # Shift embeddings to the left and insert the new embedding at the end
        embeddings = np.roll(embeddings, -1, axis=1)  # Shift left along S_LEN axis
        embeddings[:, -1] = embedding  # Insert new embedding
        
    # Concatenate embeddings to state
    print("embedding shape: ", embeddings.shape)  # [64, 6]
    print("add_embedding state shape before embedding:", state.shape)  # [6, 6]
    if len(state.shape) == 3:
        updated_state = np.concatenate((state.squeeze(0), embeddings), axis=0)  # [6 + 64, 6] = [70, 6]
    else:
        updated_state = np.concatenate((state, embeddings), axis=0)
    print("state shape after embedding:", updated_state.shape)  # [70, 6]
    
    return updated_state, embeddings



################################################################################
# ABR SERVER API
################################################################################


def launch_video_server_and_bftrace(agent_id, agent_logger=None, run_video_server=True, wait=False, redis_client=None):
    # Launch a video server for the agent, log info in agent_logger, if wait, wait for the server to finish

    ################################################################################
    # 1) Start the custom video server on a unique port for each agent
    ################################################################################

    
    video_server_dir = "/users/janechen/Genet/src/emulator/abr/pensieve/video_server"
    video_server_port = 6626 + int(agent_id)  # unique server port for each agent
    video_server_proc = None

    if run_video_server:
        if agent_logger:
            agent_logger.info("Starting video server on port=%d", video_server_port)
        video_server_proc = subprocess.Popen(
            ["python", "video_server.py", f"--port={video_server_port}"],
            cwd=video_server_dir
        )
        time.sleep(1.5)  # give it a moment to start

    ################################################################################
    # 2) Launch bpftrace in the background, capturing traffic on `video_server_port`.
    ################################################################################
    bpftrace_script = "/users/janechen/Genet/src/emulator/abr/pensieve/virtual_browser/check.bt"
    bpftrace_out_path = get_bftrace_out_path(agent_id)

    agent_logger.info(f"Starting bpftrace for agent {agent_id}, port={video_server_port}, output={bpftrace_out_path}")
    bpftrace_outfile = open(bpftrace_out_path, "w")
    bpftrace_cmd = ["sudo", "bpftrace", bpftrace_script, str(video_server_port)]
    bpftrace_proc = subprocess.Popen(bpftrace_cmd, stdout=bpftrace_outfile, stderr=subprocess.STDOUT)
    # Optional: short sleep so bpftrace can initialize
    time.sleep(1.0)

    return video_server_proc, bpftrace_proc

    if wait:
        stop_flag = redis_client.get(f"{agent_id}_stop_flag")
        while(stop_flag and int(stop_flag) == 0):
            sleep(60)
            stop_flag = redis_client.get(f"{agent_id}_stop_flag")
        video_server_proc.terminate()
        bftrace_proc.terminate()

def null_embedding_and_token():
    embedding = np.zeros((EMBEDDING_SIZE, S_LEN), dtype=np.float32)
    tokens = np.array([])
    return embedding, tokens


def transform_state_and_add_embedding(agent_id, state, embeddings, tokens):
    bpftrace_out_path = get_bftrace_out_path(agent_id)
    state = np.array(state)
    print(f"State shape before embedding: {state.shape}")
    
    # Compute tokens and embed
    if tokens.shape[0] > WINDOW:
        tokens = np.concatenate((tokens, compute_token(bpftrace_out_path)), axis=0)
        # only keep the last WINDOW size of tokens
        tokens = tokens[-WINDOW:]
    else:
        tokens = compute_token(bpftrace_out_path)
    print(f"add_embedding1 state shape: {state.shape}")
    # print(f"Tokens: {tokens}")
    state, embeddings = add_embedding(state, tokens, embeddings)
    return state, embeddings, tokens
