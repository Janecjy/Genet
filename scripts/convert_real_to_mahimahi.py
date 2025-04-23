import os
import numpy as np

# Constants
BYTES_PER_PKT = 1500.0  # Packet size in bytes
BITS_IN_BYTE = 8.0      # Bits per byte
MILLISECONDS_IN_SECOND = 1000  # Number of milliseconds in a second

# Directories
TRACE_DIR = os.path.expanduser("/home/jane/Genet/abr_trace/testing_trace/")
OUTPUT_DIR = os.path.expanduser("/home/jane/Genet/abr_trace/testing_trace_mahimahi/")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def convert_plaintext_to_mahimahi(trace_file, output_file):
    """Convert plain-text time-bandwidth trace file to a Mahimahi trace."""
    print(f"Processing {trace_file} -> {output_file}")
    
    timestamps = []
    bandwidths = []

    with open(trace_file, 'r') as f:
        for line in f:
            if line.strip() == "":
                continue
            parts = line.strip().split()
            if len(parts) != 2:
                continue  # skip malformed lines
            time_val, bw_val = float(parts[0]), float(parts[1])
            timestamps.append(time_val)
            bandwidths.append(bw_val)
    
    if not timestamps or not bandwidths:
        print(f"Skipping {trace_file} due to missing or invalid data.")
        return
    
    with open(output_file, 'w') as mf:
        for i in range(len(timestamps) - 1):
            start_time = int(timestamps[i] * MILLISECONDS_IN_SECOND)
            end_time = int(timestamps[i + 1] * MILLISECONDS_IN_SECOND)
            bandwidth_mbps = bandwidths[i]
            
            mbps_to_bps = bandwidth_mbps * 1e6  # Convert Mbps to bps
            bps_to_Bps = mbps_to_bps / BITS_IN_BYTE  # Convert bps to Bps
            Bps_to_pkts = bps_to_Bps / BYTES_PER_PKT  # Convert Bps to packets per sec
            pkt_per_millisec = Bps_to_pkts / MILLISECONDS_IN_SECOND  # Convert to packets per ms
            
            for t in range(start_time, end_time):
                num_packets = int(np.floor((t + 1) * pkt_per_millisec)) - int(np.floor(t * pkt_per_millisec))
                for _ in range(num_packets):
                    mf.write(f"{t}\n")
    
    print(f"Converted {trace_file} -> {output_file}")

if __name__ == "__main__":    
    # Process all trace files in the directory (non-JSON, e.g., .txt or no extension)
    for root, _, files in os.walk(TRACE_DIR):
        for file in files:
            if file.startswith("test"):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, TRACE_DIR)
                leaf_dir = os.path.basename(root)
                output_filename = f"{leaf_dir}_{file.replace('.json', '').replace('.txt', '')}"
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                convert_plaintext_to_mahimahi(input_path, output_path)
