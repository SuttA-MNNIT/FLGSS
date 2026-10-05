import os
import csv
from typing import Dict, Any

class CSVLogger:
    """
    Thread-safe, appendable CSV logger for federated experiment tracking.
    """
    def __init__(self, log_dir: str, filename: str = "metrics.csv"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.filepath = os.path.join(log_dir, filename)
        self.headers_written = os.path.exists(self.filepath)

    def log(self, metrics: Dict[str, Any]):
        if not self.headers_written:
            with open(self.filepath, mode='w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
                writer.writeheader()
                writer.writerow(metrics)
            self.headers_written = True
        else:
            with open(self.filepath, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
                writer.writerow(metrics)
