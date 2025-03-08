import os
import sys

# BASE_PATH = sys.argv[1]
# DATA_PATH = os.path.join(BASE_PATH, 'Cellular-Traces-NYC/')
# OUTPUT_PATH = os.path.join(BASE_PATH, 'ns-traces-orca/')
BYTES_PER_PKT = 1500.0
MILLISEC_IN_SEC = 1000.0
BITS_IN_BYTE = 8.0

def process_file(file_path, output_path):
    with open(file_path, 'r') as f:
        packet_times = [int(line.strip()) for line in f]

    interval = 1000  # ms
    max_time = max(packet_times)
    throughput_results = []

    for start_time in range(0, max_time + interval, interval):
        end_time = start_time + interval
        packets_in_interval = [pkt_time for pkt_time in packet_times if start_time <= pkt_time < end_time]
        throughput = len(packets_in_interval) * BYTES_PER_PKT * BITS_IN_BYTE / interval  # kbps
        throughput_results.append((start_time, int(throughput)))

    with open(output_path, 'w') as mf:
        for time, throughput in throughput_results:
            mf.write(f"{time} {throughput}\n")

def main():
    # Ensure the output directory exists
    # os.makedirs(OUTPUT_PATH, exist_ok=True)

    # # Traverse all files under the data directory
    # for f in os.listdir(DATA_PATH):
    #     if len(f.split('-')) == 3:
    #         file_path = os.path.join(DATA_PATH, f)
    #         output_path = os.path.join(OUTPUT_PATH, f)

    #         print(file_path, output_path)
    process_file("wire192", "wire192-out")

if __name__ == '__main__':
    main()