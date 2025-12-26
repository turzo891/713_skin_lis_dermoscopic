#!/usr/bin/env python3
"""
Real-Time Graphical System Monitor

Displays live updating graphs of CPU, GPU, and RAM usage during training.
Updates every second with smooth animations.

Usage:
    python3 monitor_with_graphs.py

    or with custom settings:
    python3 monitor_with_graphs.py --interval 0.5 --history 60
"""

import sys
import time
import argparse
from collections import deque
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

try:
    import psutil
except ImportError:
    print("Installing psutil...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
    import psutil

try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False


class LiveMonitor:
    """Real-time system monitor with live graphs."""

    def __init__(self, history_length=60, update_interval=1000):
        """
        Args:
            history_length: Number of data points to keep (seconds)
            update_interval: Update interval in milliseconds
        """
        self.history_length = history_length
        self.update_interval = update_interval

        # Data storage (deques for efficient append/pop)
        self.times = deque(maxlen=history_length)
        self.cpu_data = deque(maxlen=history_length)
        self.ram_data = deque(maxlen=history_length)
        self.gpu_util_data = deque(maxlen=history_length)
        self.gpu_mem_data = deque(maxlen=history_length)

        # Initialize with zeros
        for _ in range(history_length):
            self.times.append(0)
            self.cpu_data.append(0)
            self.ram_data.append(0)
            self.gpu_util_data.append(0)
            self.gpu_mem_data.append(0)

        self.start_time = time.time()

        # Create figure and subplots
        self.fig = plt.figure(figsize=(14, 10))
        self.fig.suptitle('Real-Time System Monitor', fontsize=16, fontweight='bold')

        gs = GridSpec(3, 2, figure=self.fig, hspace=0.3, wspace=0.3)

        # Subplots
        self.ax_cpu = self.fig.add_subplot(gs[0, 0])
        self.ax_ram = self.fig.add_subplot(gs[0, 1])
        self.ax_gpu_util = self.fig.add_subplot(gs[1, 0])
        self.ax_gpu_mem = self.fig.add_subplot(gs[1, 1])
        self.ax_summary = self.fig.add_subplot(gs[2, :])

        # Initialize plots
        self.line_cpu, = self.ax_cpu.plot([], [], 'b-', linewidth=2, label='CPU Usage')
        self.line_ram, = self.ax_ram.plot([], [], 'g-', linewidth=2, label='RAM Usage')
        self.line_gpu_util, = self.ax_gpu_util.plot([], [], 'r-', linewidth=2, label='GPU Compute')
        self.line_gpu_mem, = self.ax_gpu_mem.plot([], [], 'm-', linewidth=2, label='GPU Memory')

        # Configure CPU plot
        self.ax_cpu.set_ylim(0, 100)
        self.ax_cpu.set_xlim(0, history_length)
        self.ax_cpu.set_xlabel('Time (seconds ago)')
        self.ax_cpu.set_ylabel('CPU Usage (%)')
        self.ax_cpu.set_title('CPU Utilization', fontweight='bold')
        self.ax_cpu.grid(True, alpha=0.3)
        self.ax_cpu.legend()
        self.ax_cpu.fill_between([], [], 0, alpha=0.3, color='blue')

        # Configure RAM plot
        self.ax_ram.set_ylim(0, 100)
        self.ax_ram.set_xlim(0, history_length)
        self.ax_ram.set_xlabel('Time (seconds ago)')
        self.ax_ram.set_ylabel('RAM Usage (%)')
        self.ax_ram.set_title('RAM Utilization', fontweight='bold')
        self.ax_ram.grid(True, alpha=0.3)
        self.ax_ram.legend()

        # Configure GPU compute plot
        self.ax_gpu_util.set_ylim(0, 100)
        self.ax_gpu_util.set_xlim(0, history_length)
        self.ax_gpu_util.set_xlabel('Time (seconds ago)')
        self.ax_gpu_util.set_ylabel('GPU Usage (%)')
        self.ax_gpu_util.set_title('GPU Compute Utilization', fontweight='bold')
        self.ax_gpu_util.grid(True, alpha=0.3)
        self.ax_gpu_util.legend()

        # Configure GPU memory plot
        self.ax_gpu_mem.set_ylim(0, 100)
        self.ax_gpu_mem.set_xlim(0, history_length)
        self.ax_gpu_mem.set_xlabel('Time (seconds ago)')
        self.ax_gpu_mem.set_ylabel('GPU Memory (%)')
        self.ax_gpu_mem.set_title('GPU Memory Utilization', fontweight='bold')
        self.ax_gpu_mem.grid(True, alpha=0.3)
        self.ax_gpu_mem.legend()

        # Configure summary plot (text display)
        self.ax_summary.axis('off')
        self.summary_text = self.ax_summary.text(
            0.5, 0.5, '',
            ha='center', va='center',
            fontsize=12, fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

    def get_system_stats(self):
        """Get current system statistics."""
        # CPU
        cpu_percent = psutil.cpu_percent(interval=0.1)

        # RAM
        memory = psutil.virtual_memory()
        ram_percent = memory.percent
        ram_used_gb = memory.used / (1024**3)
        ram_total_gb = memory.total / (1024**3)

        # GPU
        if TORCH_AVAILABLE:
            try:
                gpu_util = torch.cuda.utilization()
                gpu_mem_used = torch.cuda.memory_allocated() / (1024**3)
                gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                gpu_mem_percent = (gpu_mem_used / gpu_mem_total) * 100
            except:
                gpu_util = 0
                gpu_mem_percent = 0
                gpu_mem_used = 0
                gpu_mem_total = 0
        else:
            gpu_util = 0
            gpu_mem_percent = 0
            gpu_mem_used = 0
            gpu_mem_total = 0

        return {
            'cpu_percent': cpu_percent,
            'ram_percent': ram_percent,
            'ram_used_gb': ram_used_gb,
            'ram_total_gb': ram_total_gb,
            'gpu_util': gpu_util,
            'gpu_mem_percent': gpu_mem_percent,
            'gpu_mem_used_gb': gpu_mem_used,
            'gpu_mem_total_gb': gpu_mem_total
        }

    def update(self, frame):
        """Update function called by animation."""
        # Get current stats
        stats = self.get_system_stats()

        # Update time
        elapsed = time.time() - self.start_time
        self.times.append(elapsed)

        # Update data
        self.cpu_data.append(stats['cpu_percent'])
        self.ram_data.append(stats['ram_percent'])
        self.gpu_util_data.append(stats['gpu_util'])
        self.gpu_mem_data.append(stats['gpu_mem_percent'])

        # Calculate x-axis (time ago)
        current_time = elapsed
        x_data = [current_time - t for t in reversed(self.times)]

        # Update CPU plot
        self.line_cpu.set_data(x_data, list(reversed(self.cpu_data)))

        # Update fill
        self.ax_cpu.collections.clear()
        self.ax_cpu.fill_between(
            x_data,
            list(reversed(self.cpu_data)),
            0,
            alpha=0.3,
            color='blue'
        )

        # Update RAM plot
        self.line_ram.set_data(x_data, list(reversed(self.ram_data)))
        self.ax_ram.collections.clear()
        self.ax_ram.fill_between(
            x_data,
            list(reversed(self.ram_data)),
            0,
            alpha=0.3,
            color='green'
        )

        # Update GPU compute plot
        self.line_gpu_util.set_data(x_data, list(reversed(self.gpu_util_data)))
        self.ax_gpu_util.collections.clear()
        self.ax_gpu_util.fill_between(
            x_data,
            list(reversed(self.gpu_util_data)),
            0,
            alpha=0.3,
            color='red'
        )

        # Update GPU memory plot
        self.line_gpu_mem.set_data(x_data, list(reversed(self.gpu_mem_data)))
        self.ax_gpu_mem.collections.clear()
        self.ax_gpu_mem.fill_between(
            x_data,
            list(reversed(self.gpu_mem_data)),
            0,
            alpha=0.3,
            color='magenta'
        )

        # Calculate averages
        cpu_avg = np.mean(self.cpu_data)
        ram_avg = np.mean(self.ram_data)
        gpu_util_avg = np.mean(self.gpu_util_data)
        gpu_mem_avg = np.mean(self.gpu_mem_data)

        # Update summary text
        summary = f"""

                         CURRENT SYSTEM STATUS                                

                                                                              
  CPU:        {stats['cpu_percent']:5.1f}%  (avg: {cpu_avg:5.1f}%)                                       
  RAM:        {stats['ram_percent']:5.1f}%  ({stats['ram_used_gb']:5.2f} GB / {stats['ram_total_gb']:5.2f} GB)  (avg: {ram_avg:5.1f}%)      
  GPU Compute: {stats['gpu_util']:5.1f}%  (avg: {gpu_util_avg:5.1f}%)                                    
  GPU Memory:  {stats['gpu_mem_percent']:5.1f}%  ({stats['gpu_mem_used_gb']:5.2f} GB / {stats['gpu_mem_total_gb']:5.2f} GB)  (avg: {gpu_mem_avg:5.1f}%)   
                                                                              
  Time:       {datetime.now().strftime('%H:%M:%S')}                                                    
  Uptime:     {int(elapsed // 60)}m {int(elapsed % 60)}s                                                
                                                                              

        """

        # Add performance analysis
        if gpu_util_avg > 90:
            summary += "\n GPU is fully utilized - Excellent!"
        elif gpu_util_avg > 70:
            summary += "\n GPU utilization is good"
        elif gpu_util_avg > 40:
            summary += "\n GPU underutilized - Increase batch_size or num_workers"
        else:
            summary += "\n GPU barely used - Check if training is running"

        if cpu_avg > 90:
            summary += "\n CPU bottleneck - Reduce num_workers"
        elif cpu_avg > 60:
            summary += "\n CPU utilization is good"

        if ram_avg > 90:
            summary += "\n RAM critical - Reduce batch_size or prefetch_factor"
        elif ram_avg > 70:
            summary += "\n RAM high - Monitor closely"

        self.summary_text.set_text(summary)

        return self.line_cpu, self.line_ram, self.line_gpu_util, self.line_gpu_mem, self.summary_text

    def start(self):
        """Start the live monitoring."""
        print("Starting real-time graphical monitor...")
        print("Close the window to stop monitoring.")

        # Create animation
        ani = animation.FuncAnimation(
            self.fig,
            self.update,
            interval=self.update_interval,
            blit=False,
            cache_frame_data=False
        )

        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Real-time graphical system monitor'
    )
    parser.add_argument(
        '--interval',
        type=float,
        default=1.0,
        help='Update interval in seconds (default: 1.0)'
    )
    parser.add_argument(
        '--history',
        type=int,
        default=60,
        help='History length in seconds (default: 60)'
    )

    args = parser.parse_args()

    # Create and start monitor
    monitor = LiveMonitor(
        history_length=args.history,
        update_interval=int(args.interval * 1000)
    )

    monitor.start()


if __name__ == '__main__':
    main()
