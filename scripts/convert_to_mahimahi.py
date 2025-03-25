import os
import json
import random
import argparse

MILLISECONDS_IN_SECOND = 1000.0

class AbrTrace:
    def __init__(self, timestamps, bandwidths, link_rtt, buffer_thresh, name=""):
        """Network trace used in ABR applications.

        Args:
            timestamps: List of timestamps (seconds).
            bandwidths: List of bandwidth values (Mbps).
            link_rtt: Network link base RTT (milliseconds).
            buffer_thresh: Length of playback buffer in client video player (seconds).
        """
        assert len(timestamps) == len(bandwidths)
        self.timestamps = timestamps
        self.bandwidths = bandwidths
        self.link_rtt = link_rtt
        self.buffer_thresh = buffer_thresh * MILLISECONDS_IN_SECOND
        self.name = name

    def convert_to_mahimahi_format(self, output_file):
        """Convert the trace to Mahimahi format and save to output file."""
        BYTES_PER_PKT = 1500.0
        BITS_IN_BYTE = 8.0

        with open(output_file, 'w') as mf:
            millisec_time = 0
            mf.write(str(millisec_time) + '\n')

            for i in range(len(self.bandwidths)):
                throughput = self.bandwidths[i]

                if i < len(self.timestamps) - 1:
                    duration_sec = self.timestamps[i+1] - self.timestamps[i]
                else:
                    duration_sec = 1.0

                duration_ms = duration_sec * MILLISECONDS_IN_SECOND

                mbps_to_bps = throughput * 1e6  # Convert Mbps to bps
                bps_to_Bps = mbps_to_bps / BITS_IN_BYTE
                Bps_to_pkts = bps_to_Bps / BYTES_PER_PKT
                pkt_per_millisec = Bps_to_pkts / MILLISECONDS_IN_SECOND

                millisec_count = 0
                pkt_count = 0

                while millisec_count < duration_ms:
                    millisec_count += 1
                    millisec_time += 1

                    to_send = (millisec_count * pkt_per_millisec) - pkt_count
                    to_send = int(to_send)

                    for _ in range(to_send):
                        mf.write(str(millisec_time) + '\n')

                    pkt_count += to_send

    @staticmethod
    def load_from_file(filename):
        """Load an AbrTrace object from a space-separated trace file."""
        timestamps = []
        bandwidths = []
        
        with open(filename, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue  # Skip malformed lines
                
                timestamp, bandwidth = map(float, parts)
                timestamps.append(timestamp)
                bandwidths.append(bandwidth)

        if not timestamps or not bandwidths:
            raise ValueError(f"Error: Failed to load valid data from {filename}")

        # Assign default values for link_rtt and buffer_thresh (modify as needed)
        link_rtt = 50.0  # Default base RTT in ms
        buffer_thresh = 4.0  # Default buffer threshold in seconds

        return AbrTrace(timestamps, bandwidths, link_rtt, buffer_thresh)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Path to the directory containing trace files")
    parser.add_argument("--output-dir", required=True, help="Path to save Mahimahi-formatted traces")
    parser.add_argument("--node-id", type=int, required=True, help="Node ID")
    parser.add_argument("--num-nodes", type=int, required=True, help="Total number of nodes")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    trace_files = sorted(os.listdir(args.input_dir))
    
    # Distribute traces evenly across nodes
    selected_traces = [trace_files[i] for i in range(args.node_id - 1, len(trace_files), args.num_nodes)]

    for trace_file in selected_traces:
        input_path = os.path.join(args.input_dir, trace_file)
        output_path = os.path.join(args.output_dir, trace_file)

        trace = AbrTrace.load_from_file(input_path)
        trace.convert_to_mahimahi_format(output_path)

        print(f"Node {args.node_id} converted {trace_file} -> {output_path}")

    print(f"Node {args.node_id}: Conversion completed!")

if __name__ == "__main__":
    main()
