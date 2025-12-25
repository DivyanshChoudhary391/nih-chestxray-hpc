import csv
import os
import time

class PerformanceLogger:
    def __init__(self, path):
        self.path = path
        self.start_time = None

        # create file with header if not exists
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "chunk",
                    "num_images",
                    "train_time_sec",
                    "images_per_sec"
                ])

    def start(self):
        self.start_time = time.perf_counter()

    def end_and_log(self, chunk_idx, num_images):
        elapsed = time.perf_counter() - self.start_time
        throughput = num_images / elapsed if elapsed > 0 else 0

        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                chunk_idx,
                num_images,
                round(elapsed, 3),
                round(throughput, 2)
            ])

        return elapsed, throughput
