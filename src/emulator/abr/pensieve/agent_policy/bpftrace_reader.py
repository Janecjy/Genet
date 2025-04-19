class BPFTraceTokenCache:
    def __init__(self, bpftrace_path, window_ms=800):
        self.bpftrace_path = bpftrace_path
        self.window_ms = window_ms
        self.offset = 0
        self.buffer = []

    def _read_new_lines(self):
        with open(self.bpftrace_path, 'r') as f:
            f.seek(self.offset)
            lines = f.readlines()
            self.offset = f.tell()
        return [l.strip() for l in lines if l.strip() and not l.startswith(('A', 't', 'time_us'))]

    def _parse_lines(self, lines):
        parsed = []
        for line in lines:
            parts = line.split(',')
            if len(parts) < 9:
                continue
            try:
                parsed.append({
                    'time_ms': float(parts[0]) / 1e6,
                    'srtt': float(parts[1]),
                    'rttvar': float(parts[2]),
                    'delivered': float(parts[3]),
                    'rate_interval': float(parts[4]),
                    'mss': float(parts[5]),
                    'lost': float(parts[6]),
                    'snd_cwnd': float(parts[-3])
                })
            except:
                continue
        return parsed

    def get_recent_parsed_lines(self):
        new_lines = self._read_new_lines()
        new_data = self._parse_lines(new_lines)
        self.buffer.extend(new_data)

        if not self.buffer:
            return []

        last_time = self.buffer[-1]['time_ms']
        cutoff = last_time - self.window_ms
        self.buffer = [x for x in self.buffer if x['time_ms'] >= cutoff]
        return self.buffer
