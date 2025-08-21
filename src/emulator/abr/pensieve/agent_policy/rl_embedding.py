import subprocess
import time
import os
import numpy as np
import torch
from models import create_mask
from collections import defaultdict
import math
from bpftrace_reader import BPFTraceTokenCache

AGGREGATION_WINDOW_MS = 80 
WINDOW = 10
EMBEDDING_SIZE = 16
HIDDEN_SIZE = 3
S_LEN = 6 
S_INFO = 6
DEVICE = 'cpu'



transformer = torch.load("/users/janechen/Genet/src/emulator/abr/pensieve/agent_policy/Checkpoint-Combined_10RTT_6col_Transformer3_256_4_4_32_4_lr_0.0001_boundaries-quantile50-merged_multi-559iter.p", map_location='cpu')
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


def get_bftrace_out_path(agent_id, collection=False, summary_dir=None, trace_name=None):
    if collection and summary_dir and trace_name:
        # Modify filename correctly
        # if trace_name.endswith('.log'):
        #     output_filename = trace_name.replace('.log', '.out')
        # else:
        output_filename = trace_name + '.out'

        # Ensure summary_dir is properly formatted
        summary_dir = os.path.abspath(summary_dir)  # Normalize path

        # Create the directory if it doesn't exist
        os.makedirs(summary_dir, exist_ok=True)
        return os.path.join(summary_dir, output_filename)

    return f"/mydata/logs/bpftrace_out_{agent_id}.txt"


################################################################################
# Embedding functions
################################################################################

import numpy as np

AGGREGATION_WINDOW_MS = 80
WINDOW = 10  # sample every 10ms
S_INFO = 6
S_LEN = 6
EMBEDDING_SIZE = 32
FEATURE_DIM = 6

boundaries_dict = {
    1: [0.17970980077981952, 0.2477400004863739, 0.3431900143623352, 0.4394623124599457, 0.5199699997901917, 0.5591899752616882, 0.664075837135315, 0.8117193746566773, 0.8589123022556305, 0.9806600213050842, 1.0453529238700867, 1.1441732501983644, 1.3060192465782166, 1.517687382698061, 1.586526298522949, 1.6478514432907105, 1.745526261329651, 1.8974904060363769, 2.0416860580444336, 2.162302017211914, 2.3284184312820435, 2.5949565315246588, 2.9315665817260745, 3.072933483123779, 3.345926284790039, 3.7394452285766597, 4.04695195198059, 4.346436538696292, 4.812195568084716, 5.366202068328857, 6.018584337234497, 6.7387018966674805, 7.579217300415039, 8.52665195465088, 9.62692012786865, 10.986263847351076, 12.524017887115479, 14.386759757995605, 16.63687934875488, 19.408872985839864, 22.90160530090331, 27.062631301879854, 32.15915336608886, 38.12343765258791, 45.52385025024414, 54.971268005371094, 67.31158157348622, 83.75869232177739, 107.49414642333988],
    2: [0.29899999499320984, 0.3546000123023987, 0.393533319234848, 0.42340001463890076, 0.4477333426475525, 0.4687333405017853, 0.4875999987125397, 0.5053333044052124, 0.522266685962677, 0.538937509059906, 0.5557500123977661, 0.5731250047683716, 0.5914999842643738, 0.6112619996070876, 0.6334666609764099, 0.6590666770935059, 0.6904666423797607, 0.7289999723434448, 0.7777500152587891, 0.8388500213623047, 0.9150000214576721, 1.0096999406814575, 1.1299333572387695, 1.2836250066757202, 1.4804999828338623, 1.7339999675750732, 2.0407114744186394, 2.413186016082768, 2.855799913406372, 3.3714001178741455, 3.9744690465927066, 4.684124946594238, 5.513999938964844, 6.499000072479248, 7.665569019317625, 9.042400360107422, 10.74500633239746, 12.85283763885498, 15.441013832092283, 18.600000381469727, 22.39473651885986, 27.171252136230468, 33.37250564575195, 41.365531158447325, 51.64250411987305, 66.98408111572273, 90.70465042114252, 128.4860766601563, 210.4208187866211],
    3: [0.002559774396941066, 0.01374848645180464, 0.023857793733477593, 0.03573869913816452, 0.04676923900842666, 0.058328409641981126, 0.07351147383451462, 0.09181121304631232, 0.10452739179134371, 0.11561187133193009, 0.13634089231491087, 0.1626020714640617, 0.18855283677578005, 0.22310818642377864, 0.2505026543140412, 0.281861475110054, 0.30442466378211974, 0.3239287030696869, 0.35384442925453186, 0.3957134610414505, 0.44626529216766364, 0.46199937283992765, 0.5147144532203672, 0.5932496166229247, 0.6770075297355653, 0.7629529356956486, 0.8523847126960762, 0.9143763852119445, 1.017856621742249, 1.4104804039001464],
    4: [0.0, 0.011620868295431134, 0.026965714767575276, 0.08147502973675737],
    5: [0.9821428656578064, 0.9932490539550781, 1.0035088062286377, 1.021384003162384]
}

def bucketize_value(value, boundaries):
        if not boundaries:
            return 0
        idx = np.searchsorted(boundaries, value, side='left')
        return idx

def compute_token_from_parsed_lines(parsed_lines, boundaries_dict):
    """
    Takes in parsed bpftrace lines (list of dicts) and computes a (1, 10, 6) token array.
    """
    if not parsed_lines:
        return np.empty((0, 20, 6), dtype=np.float32)

    last_time_ms = parsed_lines[-1]['time_ms']
    rtt_window_count = 10
    total_window_start = last_time_ms - (AGGREGATION_WINDOW_MS * rtt_window_count)

    parsed_lines = [l for l in parsed_lines if l['time_ms'] >= total_window_start]
    if len(parsed_lines) == 0:
        return np.empty((0, 20, 6), dtype=np.float32)

    metrics_data = []

    pre_pkt_lost = parsed_lines[0]['lost']
    pre_delivered = parsed_lines[0]['delivered']
    pre_cwnd = parsed_lines[0]['snd_cwnd']
    dt_pre = parsed_lines[0]['time_ms']

    for window_idx in range(rtt_window_count):
        interval_start = total_window_start + window_idx * AGGREGATION_WINDOW_MS
        interval_end = interval_start + AGGREGATION_WINDOW_MS
        block = [l for l in parsed_lines if interval_start <= l['time_ms'] < interval_end]

        if not block:
            return np.empty((0, 20, 6), dtype=np.float32)

        block_sorted = sorted(block, key=lambda x: x['time_ms'])

        samples = []
        t = interval_start + WINDOW
        while t <= interval_end:
            candidates = [l for l in block_sorted if l['time_ms'] >= t]
            if candidates:
                samples.append(candidates[0])
            t += WINDOW

        if not samples:
            return np.empty((0, 20, 6), dtype=np.float32)

        srtt_vals, rttvar_vals, rate_vals, f5_vals, f6_vals = [], [], [], [], []

        for s in samples:
            srtt_vals.append(s['srtt'] / 100000.0)
            rttvar_vals.append(s['rttvar'] / 1000.0)

            dt = (s['time_ms'] - dt_pre) * 1000 if dt_pre > 0 else 1
            delivered_rate = (s['delivered'] - pre_delivered) * s['mss'] * 80 / dt if s['delivered'] > pre_delivered else 0
            rate_vals.append(delivered_rate)
            dt_pre = s['time_ms']
            pre_delivered = s['delivered']

            l_db = (s['lost'] - pre_pkt_lost) * s['mss'] if s['lost'] > pre_pkt_lost else 0
            f5_vals.append(8.0 * l_db / dt / 100.0 if dt > 0 else 0.0)
            pre_pkt_lost = s['lost']

            cwnd = s['snd_cwnd']
            f6_vals.append(cwnd / pre_cwnd if pre_cwnd > 0 else 0.0)
            pre_cwnd = cwnd

        # Compute the 6 features
        avg_features = [
            0.8,
            np.mean(srtt_vals),
            np.mean(rttvar_vals),
            np.mean(rate_vals),
            np.mean(f5_vals),
            np.mean(f6_vals)
        ]
        # metrics_data.append(avg_features)

        # # Now perform bucketization on features 1-5 (index 1 to 5)
        # token = []
        # for feat_idx in range(1, 6):
        #     val = avg_features[feat_idx]
        #     bds = boundaries_dict.get(feat_idx, [])
        #     b_idx = bucketize_value(val, bds)
        #     token.append(b_idx)

    # Final output
    return np.array(avg_features, dtype=np.float32)

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
        return embeddings
    
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

    return embedding



################################################################################
# ABR SERVER API
################################################################################


def launch_video_server_and_bftrace(agent_id, agent_logger=None, run_video_server=True, wait=False, redis_client=None, collection=False, summary_dir=None, trace_name=None):
    # Launch a video server for the agent, log info in agent_logger, if wait, wait for the server to finish

    ################################################################################
    # 1) Start the custom video server on a unique port for each agent
    ################################################################################

    
    video_server_dir = "/users/janechen/Genet/src/emulator/abr/pensieve/video_server"
    video_server_port = 6626 + int(agent_id)  # unique server port for each agent
    video_server_proc = None

    if run_video_server:
        print(f"Starting video server on port={video_server_port}")
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
    if collection:
        bpftrace_out_path = get_bftrace_out_path(agent_id, collection, summary_dir, trace_name)
    else:
        bpftrace_out_path = get_bftrace_out_path(agent_id)

    agent_logger.info(f"Starting bpftrace for agent {agent_id}, port={video_server_port}, output={bpftrace_out_path}")
    print(f"Starting bpftrace for agent {agent_id}, port={video_server_port}, output={bpftrace_out_path}")
    bpftrace_outfile = open(bpftrace_out_path, "w")
    bpftrace_cmd = ["sudo", "bpftrace", bpftrace_script, str(video_server_port)]
    bpftrace_proc = subprocess.Popen(bpftrace_cmd, stdout=bpftrace_outfile, stderr=subprocess.STDOUT)
    # Optional: short sleep so bpftrace can initialize
    time.sleep(1.0)

    return video_server_proc, bpftrace_proc, video_server_port

    if wait:
        stop_flag = redis_client.get(f"{agent_id}_stop_flag")
        while(stop_flag and int(stop_flag) == 0):
            sleep(60)
            stop_flag = redis_client.get(f"{agent_id}_stop_flag")
        video_server_proc.terminate()
        bftrace_proc.terminate()

def null_embedding_and_token():
    embedding = np.zeros((EMBEDDING_SIZE), dtype=np.float32)
    tokens = np.array([])
    return embedding, tokens

def transform_state_and_add_embedding(agent_id, state, embeddings, tokens, token_reader):
    """
    For a given agent, update state with embedding from recent RTT tokens.

    Args:
        agent_id (int): The ID of the agent.
        state (np.ndarray): The current state [S_INFO, S_LEN].
        embeddings (np.ndarray): Current embedding vector.
        tokens (np.ndarray): Previously collected RTT token vectors.
        token_reader (BPFTraceTokenCache): Per-agent token reader instance.

    Returns:
        (np.ndarray, np.ndarray, np.ndarray): updated state, embeddings, and tokens
    """
    state = np.array(state)
    print(f"[Agent {agent_id}] State shape before embedding: {state.shape}")

    parsed_lines = token_reader.get_recent_parsed_lines()
    new_tokens = compute_token_from_parsed_lines(parsed_lines, boundaries_dict)

    if new_tokens.size > 0:
        tokens = np.concatenate((tokens, new_tokens), axis=0) if tokens.size > 0 else new_tokens
        tokens = tokens[-WINDOW:]

        print(f"[Agent {agent_id}] add_embedding state shape: {state.shape}")
        # embeddings = add_embedding(state, tokens, embeddings)
        print(f"[Agent {agent_id}] tokens shape after adding new token: {tokens.shape}")

    return state, embeddings, tokens

