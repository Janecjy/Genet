import os
import pickle
import numpy as np
import random
from multiprocessing import Pool, cpu_count

# 80ms in nanoseconds
RTT_INTERVAL_NS = 80_000_000
TOTAL_INTERVAL_NS = 20 * RTT_INTERVAL_NS
SAMPLE_INTERVAL_NS = 10_000_000  # 10 ms in nanoseconds

def sample_block(block, block_start, block_end, pre_pkt_lost, dt_pre, pre_cwnd):
    """
    Within [block_start, block_end), collect lines every 10 ms, 
    then compute f2..f6 from those sampled lines. For f5, f6, we also average 
    over all sampled lines in this block.
    """
    # Sort block by time_us so we process lines in ascending time
    block_sorted = sorted(block, key=lambda x: x['time_us'])

    sampled = []
    next_collect_time = block_start + SAMPLE_INTERVAL_NS

    # We'll step through time from block_start in increments of 10 ms
    while next_collect_time <= block_end:
        # find the first line >= next_collect_time
        line_candidates = [p for p in block_sorted if p['time_us'] >= next_collect_time]
        if not line_candidates:
            break
        chosen_line = line_candidates[0]
        sampled.append(chosen_line)
        next_collect_time += SAMPLE_INTERVAL_NS

    # If we didn't sample any lines in this block, can't compute features
    if not sampled:
        return None, pre_pkt_lost, dt_pre, pre_cwnd

    # We'll store all per-line computations in lists
    srtt_vals = []
    rttvar_vals = []
    rate_vals = []
    f5_vals = []  # each line's instantaneous loss-based bandwidth
    f6_vals = []  # each line's cwnd ratio

    for line in sampled:
        # f2, f3, f4
        srtt_vals.append(line['srtt'] / 100000.0)
        rttvar_vals.append(line['rttvar'] / 1000.0)
        rate_vals.append(line['rate_delivered'] / 125000.0 / 100.0)

        # Compute line-based f5 (loss-based bandwidth)
        lost = line['lost']
        mss = line['mss_cache']
        time_us = line['time_us']

        dt = time_us - dt_pre if dt_pre > 0 else 1
        dt_pre = time_us

        l_db = (lost - pre_pkt_lost) * mss if lost > pre_pkt_lost else 0
        pre_pkt_lost = lost

        if dt > 0:
            line_f5 = 8.0 * l_db / dt / 100.0
        else:
            line_f5 = 0.0
        f5_vals.append(line_f5)

        # Compute line-based f6 (cwnd ratio)
        cwnd = line['snd_cwnd']
        if pre_cwnd > 0:
            line_f6 = cwnd / pre_cwnd
        else:
            line_f6 = 0.0
        f6_vals.append(line_f6)

        # Update pre_cwnd to the current line's cwnd for the next line's ratio
        pre_cwnd = cwnd

    # f1 is constant
    f1 = 0.8

    # Average the sample-based lists
    srtt_avg = np.mean(srtt_vals)
    rttvar_avg = np.mean(rttvar_vals)
    rate_avg = np.mean(rate_vals)
    f5_avg = np.mean(f5_vals)
    f6_avg = np.mean(f6_vals)

    return [f1, srtt_avg, rttvar_avg, rate_avg, f5_avg, f6_avg], pre_pkt_lost, dt_pre, pre_cwnd


def build_sample(lines, start_idx):
    """
    For each of 20 RTT intervals (80ms each), gather lines in 10ms increments, 
    then compute average features (including f5, f6) from those samples.
    """
    parsed = []
    for line in lines[start_idx:]:
        if line.strip() == "" or line.startswith("time_us") or line.startswith("Attaching"):
            continue
        cols = line.strip().split(',')
        if len(cols) < 11:
            continue
        try:
            time_us = int(cols[0])
            parsed.append({
                'time_us': time_us,
                'srtt': float(cols[1]),
                'rttvar': float(cols[2]),
                'rate_delivered': float(cols[3]),
                'rate_interval_us': float(cols[4]),
                'mss_cache': int(cols[5]),
                'lost': int(cols[6]),
                'snd_cwnd': int(cols[8])
            })
            # Stop once we've covered 20 RTT intervals from the first line
            if len(parsed) > 1 and (time_us - parsed[0]['time_us']) > TOTAL_INTERVAL_NS:
                break
        except:
            continue

    if not parsed:
        return None

    sample = []
    pre_pkt_lost = 0
    dt_pre = 0
    pre_cwnd = 0

    t0 = parsed[0]['time_us']
    rtt_idx = 0
    interval_start = t0
    interval_end = interval_start + RTT_INTERVAL_NS

    while rtt_idx < 20:
        block = [p for p in parsed if interval_start <= p['time_us'] < interval_end]
        if not block:
            return None

        block_features, pre_pkt_lost, dt_pre, pre_cwnd = sample_block(
            block, interval_start, interval_end, pre_pkt_lost, dt_pre, pre_cwnd
        )

        if block_features is None:
            return None

        sample.append(block_features)

        rtt_idx += 1
        interval_start = interval_end
        interval_end += RTT_INTERVAL_NS

    return np.array(sample, dtype=np.float32)


def process_one_file(filepath):
    fname = os.path.basename(filepath)
    with open(filepath, "r") as f:
        lines = f.readlines()

    # Collect time_us to quickly check if the trace can cover 20 RTT intervals
    time_us_list = [
        int(l.split(',')[0]) for l in lines
        if l.strip() and not l.startswith("time_us") and not l.startswith("Attaching")
    ]
    if not time_us_list:
        return None

    min_time = min(time_us_list)
    max_time = max(time_us_list)
    trace_duration = max_time - min_time

    if trace_duration < TOTAL_INTERVAL_NS:
        return None

    max_start_time = max_time - TOTAL_INTERVAL_NS
    file_data = []
    attempts = 0
    max_attempts = 500
    seen = set()

    while len(file_data) < 50 and attempts < max_attempts:
        start_idx = random.randint(0, len(lines) - 1)
        if lines[start_idx].strip() == "" or lines[start_idx].startswith("time_us") or lines[start_idx].startswith("Attaching"):
            continue
        start_time = int(lines[start_idx].split(',')[0])
        if start_idx in seen or start_time > max_start_time:
            attempts += 1
            continue
        seen.add(start_idx)
        # print(f"[{fname}] Attempting to build sample starting at index {start_idx} (time_us: {start_time})")

        sample = build_sample(lines, start_idx)
        if sample is not None:
            file_data.append(sample)
        attempts += 1

    if file_data:
        print(f"[{fname}] Collected {len(file_data)} valid samples")
        return np.stack(file_data, axis=0)
    else:
        print(f"[{fname}] Skipped (no valid samples)")
        return None

def build_dataset_rtt_50_sample(trace_dirs, save_path):
    all_out_files = []
    for d in trace_dirs:
        files = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".out")]
        all_out_files.extend(files)

    print(f"Found {len(all_out_files)} .out files. Starting parallel processing...")

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_one_file, all_out_files)

    all_data = [r for r in results if r is not None]

    if all_data:
        dataset = np.concatenate(all_data, axis=0)
    else:
        dataset = np.empty((0, 20, 6), dtype=np.float32)

    print("Final dataset shape:", dataset.shape)

    with open(save_path, "wb") as f:
        pickle.dump(dataset, f)

    print(f"Dataset saved to {save_path}")


if __name__ == "__main__":
    TRACE_DIRS = [
        "/users/janechen/Genet/data/abr/unum/BBA_0_60_40",
        "/users/janechen/Genet/data/abr/unum/RobustMPC_0_60_40"
    ]
    OUTPUT_PATH = "/mydata/pensieve_rtt_random.p"

    build_dataset_rtt_50_sample(TRACE_DIRS, OUTPUT_PATH)
