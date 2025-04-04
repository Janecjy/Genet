import os
import pickle
import numpy as np
import random
from multiprocessing import Pool, cpu_count

# 80ms in nanoseconds
RTT_INTERVAL_NS = 80_000_000
TOTAL_INTERVAL_NS = 20 * RTT_INTERVAL_NS

def build_sample(lines, start_idx):
    sample = []
    pre_pkt_lost = 0
    dt_pre = 0
    pre_cwnd = 0

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
            if len(parsed) > 1 and (time_us - parsed[0]['time_us']) > TOTAL_INTERVAL_NS:
                break  # Stop once we’ve reached 20 RTT intervals
        except:
            continue

    if len(parsed) == 0:
        return None

    t0 = parsed[0]['time_us']
    rtt_idx = 0
    interval_start = t0
    interval_end = interval_start + RTT_INTERVAL_NS

    while rtt_idx < 20:
        block = [p for p in parsed if interval_start <= p['time_us'] < interval_end]
        if not block:
            return None

        f1 = 80 / 100.0

        srtt = np.mean([b['srtt'] / 100000.0 for b in block])
        rttvar = np.mean([b['rttvar'] / 1000.0 for b in block])
        rate_delivered = np.mean([b['rate_delivered'] / 125000.0 / 100.0 for b in block])

        # Loss bandwidth calculation
        loss_db = []
        sent_dt = []
        for b in block:
            lost = b['lost']
            mss = b['mss_cache']
            time_us = b['time_us']
            l_db = (lost - pre_pkt_lost) * mss if lost > pre_pkt_lost else 0
            dt = time_us - dt_pre if dt_pre > 0 else 1
            loss_db.append(l_db)
            sent_dt.append(dt)
            pre_pkt_lost = lost
            dt_pre = time_us

        f5 = 8.0 * sum(loss_db) / sum(sent_dt) / 100.0 if sum(sent_dt) > 0 else 0.0

        # CWND ratio
        snd_cwnds = [b['snd_cwnd'] for b in block]
        snd_cwnd = snd_cwnds[-1] if snd_cwnds else pre_cwnd
        f6 = snd_cwnd / pre_cwnd if pre_cwnd > 0 else 0.0
        pre_cwnd = snd_cwnd

        sample.append([f1, srtt, rttvar, rate_delivered, f5, f6])

        rtt_idx += 1
        interval_start = interval_end
        interval_end += RTT_INTERVAL_NS

    return np.array(sample, dtype=np.float32)


def process_one_file(filepath):
    fname = os.path.basename(filepath)
    with open(filepath, "r") as f:
        lines = f.readlines()

    # Extract all valid time_us values to find min and max
    time_us_list = [
        int(l.split(',')[0]) for l in lines
        if l.strip() and not l.startswith("time_us") and not l.startswith("Attaching")
    ]
    if len(time_us_list) == 0:
        return None

    min_time = min(time_us_list)
    max_time = max(time_us_list)
    trace_duration = max_time - min_time

    if trace_duration < RTT_INTERVAL_NS * 20:
        return None

    max_start_time = max_time - RTT_INTERVAL_NS * 20
    file_data = []
    attempts = 0
    max_attempts = 500
    seen = set()
    # print("max_start_time: ", max_start_time)

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
