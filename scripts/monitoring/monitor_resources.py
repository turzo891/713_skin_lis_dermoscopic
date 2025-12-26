#!/usr/bin/env python3
"""
Real-time Resource Monitor for Training

Monitors CPU, GPU, and RAM usage during training.
Displays utilization and identifies bottlenecks.

Usage:
    python3 monitor_resources.py

    or run in background while training:
    python3 monitor_resources.py --log resource_log.csv
"""

import time
import argparse
import csv
from datetime import datetime
import subprocess
import sys

try:
    import psutil
except ImportError:
    print("Installing psutil...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
    import psutil

try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False


class ResourceMonitor:
    """Monitor CPU, GPU, and RAM usage."""

    def __init__(self, log_file=None, interval=1.0):
        self.log_file = log_file
        self.interval = interval
        self.csv_writer = None
        self.csv_file = None

        if log_file:
            self.csv_file = open(log_file, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                'timestamp', 'cpu_percent', 'ram_percent', 'ram_used_gb',
                'gpu_percent', 'gpu_memory_used_gb', 'gpu_memory_total_gb'
            ])

    def get_cpu_usage(self):
        """Get CPU utilization percentage."""
        return psutil.cpu_percent(interval=0.1)

    def get_ram_usage(self):
        """Get RAM usage."""
        memory = psutil.virtual_memory()
        return {
            'percent': memory.percent,
            'used_gb': memory.used / (1024**3),
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3)
        }

    def get_gpu_usage(self):
        """Get GPU utilization and memory usage."""
        try:
            # Use nvidia-smi for reliable GPU stats
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,name',
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=2
            )

            if result.returncode == 0:
                # Parse output: "85, 12345, 24576, NVIDIA GeForce RTX 3090"
                parts = result.stdout.strip().split(',')
                gpu_util = float(parts[0].strip())
                gpu_mem_used = float(parts[1].strip()) / 1024  # MB to GB
                gpu_mem_total = float(parts[2].strip()) / 1024  # MB to GB
                gpu_name = parts[3].strip()

                return {
                    'percent': gpu_util,
                    'memory_used_gb': gpu_mem_used,
                    'memory_total_gb': gpu_mem_total,
                    'memory_percent': (gpu_mem_used / gpu_mem_total) * 100,
                    'name': gpu_name
                }
        except Exception as e:
            pass

        # Fallback to torch if nvidia-smi fails
        if TORCH_AVAILABLE:
            try:
                gpu_memory_used = torch.cuda.memory_allocated() / (1024**3)
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)

                return {
                    'percent': 0,  # Can't get utilization without pynvml
                    'memory_used_gb': gpu_memory_used,
                    'memory_total_gb': gpu_memory_total,
                    'memory_percent': (gpu_memory_used / gpu_memory_total) * 100,
                    'name': torch.cuda.get_device_name(0)
                }
            except Exception:
                pass

        return None

    def print_bar(self, label, value, max_value=100, width=40):
        """Print a progress bar."""
        filled = int(width * value / max_value)
        bar = '█' * filled + '░' * (width - filled)

        # Color based on utilization
        if value >= 90:
            color = '\033[92m'  # Green (good utilization)
        elif value >= 60:
            color = '\033[93m'  # Yellow (moderate)
        else:
            color = '\033[91m'  # Red (underutilized)

        reset = '\033[0m'

        print(f"{label:15} {color}[{bar}]{reset} {value:5.1f}%")

    def monitor_loop(self):
        """Main monitoring loop."""
        print("\n" + "="*80)
        print("RESOURCE MONITOR - CPU + GPU + RAM Usage".center(80))
        print("="*80)
        print("\nPress Ctrl+C to stop\n")

        try:
            while True:
                # Clear screen
                print("\033[2J\033[H", end="")

                # Header
                print("="*80)
                print(f"Resource Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(80))
                print("="*80)
                print()

                # Get metrics
                cpu_usage = self.get_cpu_usage()
                ram_usage = self.get_ram_usage()
                gpu_usage = self.get_gpu_usage()

                # Display CPU
                print("CPU USAGE:")
                self.print_bar("CPU", cpu_usage)
                print(f"  Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
                print()

                # Display RAM
                print("RAM USAGE:")
                self.print_bar("RAM", ram_usage['percent'])
                print(f"  Used: {ram_usage['used_gb']:.2f} GB / {ram_usage['total_gb']:.2f} GB")
                print(f"  Available: {ram_usage['available_gb']:.2f} GB")
                print()

                # Display GPU
                if gpu_usage:
                    print("GPU USAGE:")
                    self.print_bar("GPU Compute", gpu_usage['percent'])
                    self.print_bar("GPU Memory", gpu_usage['memory_percent'])
                    print(f"  Memory: {gpu_usage['memory_used_gb']:.2f} GB / {gpu_usage['memory_total_gb']:.2f} GB")
                    print(f"  Device: {gpu_usage.get('name', 'Unknown GPU')}")
                else:
                    print("GPU USAGE:")
                    print("  No GPU detected or PyTorch not available")

                print()
                print("="*80)

                # Performance analysis
                print("\nPERFORMANCE ANALYSIS:")

                if cpu_usage < 50:
                    print("  ⚠ CPU underutilized - Consider increasing num_workers")
                elif cpu_usage > 95:
                    print("  ⚠ CPU bottleneck - Reduce num_workers or simplify augmentation")
                else:
                    print("  ✓ CPU utilization is good")

                if ram_usage['percent'] > 90:
                    print("  ⚠ RAM critical - Reduce batch_size or prefetch_factor")
                elif ram_usage['percent'] > 75:
                    print("  ⚠ RAM high - Monitor closely")
                else:
                    print("  ✓ RAM utilization is good")

                if gpu_usage:
                    if gpu_usage['percent'] < 70:
                        print("  ⚠ GPU underutilized - Data loading may be bottleneck")
                        print("    → Increase num_workers")
                        print("    → Enable pin_memory=True")
                        print("    → Increase prefetch_factor")
                    elif gpu_usage['percent'] > 98:
                        print("  ✓ GPU fully utilized - Excellent!")
                    else:
                        print("  ✓ GPU utilization is good")

                    if gpu_usage['memory_percent'] > 95:
                        print("  ⚠ GPU memory critical - Reduce batch_size")
                    elif gpu_usage['memory_percent'] < 50:
                        print("  ⚠ GPU memory underutilized - Increase batch_size or use mixed precision")

                print("="*80)

                # Efficiency score
                if gpu_usage:
                    efficiency = (cpu_usage * 0.3 + ram_usage['percent'] * 0.2 + gpu_usage['percent'] * 0.5) / 100
                else:
                    efficiency = (cpu_usage * 0.6 + ram_usage['percent'] * 0.4) / 100

                print(f"\nOverall Efficiency: {efficiency*100:.1f}%")

                if efficiency > 0.85:
                    print("Status: Excellent resource utilization!")
                elif efficiency > 0.70:
                    print("Status: Good, but room for improvement")
                elif efficiency > 0.50:
                    print("Status: Moderate utilization")
                else:
                    print("Status: Poor utilization - Check configuration")

                # Log to CSV
                if self.csv_writer:
                    self.csv_writer.writerow([
                        datetime.now().isoformat(),
                        cpu_usage,
                        ram_usage['percent'],
                        ram_usage['used_gb'],
                        gpu_usage['percent'] if gpu_usage else 0,
                        gpu_usage['memory_used_gb'] if gpu_usage else 0,
                        gpu_usage['memory_total_gb'] if gpu_usage else 0
                    ])
                    self.csv_file.flush()

                time.sleep(self.interval)

        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
            if self.csv_file:
                self.csv_file.close()
                print(f"Log saved to: {self.log_file}")

    def __del__(self):
        if self.csv_file:
            self.csv_file.close()


def main():
    parser = argparse.ArgumentParser(description='Monitor CPU, GPU, and RAM usage')
    parser.add_argument('--log', type=str, default=None,
                        help='Log file path (CSV format)')
    parser.add_argument('--interval', type=float, default=1.0,
                        help='Update interval in seconds')
    args = parser.parse_args()

    monitor = ResourceMonitor(log_file=args.log, interval=args.interval)
    monitor.monitor_loop()


if __name__ == '__main__':
    main()