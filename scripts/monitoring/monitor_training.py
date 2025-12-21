#!/usr/bin/env python3
"""
Real-time training monitor with live plots.
Usage: python3 monitor_training.py [log_file_path]
"""

import re
import time
import sys
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime

def parse_log_file(filepath):
    """Parse training log file for metrics."""
    epochs = defaultdict(list)

    try:
        with open(filepath, 'r') as f:
            content = f.read()

        # Find epoch summaries
        epoch_pattern = r'Epoch (\d+)/\d+.*?Train Loss: ([\d.]+), Train Acc: ([\d.]+).*?Val Loss: ([\d.]+), Val Acc: ([\d.]+)'
        matches = re.finditer(epoch_pattern, content, re.DOTALL)

        for match in matches:
            epoch_num = int(match.group(1))
            epochs['epoch'].append(epoch_num)
            epochs['train_loss'].append(float(match.group(2)))
            epochs['train_acc'].append(float(match.group(3)))
            epochs['val_loss'].append(float(match.group(4)))
            epochs['val_acc'].append(float(match.group(5)))

    except FileNotFoundError:
        pass

    return epochs

def create_plot(filepath, output_dir='training_plots'):
    """Create and save training plots."""
    data = parse_log_file(filepath)

    if not data['epoch']:
        return False

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    epochs = data['epoch']

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.tight_layout(pad=4.0)

    # Plot Loss
    axes[0].plot(epochs, data['train_loss'], 'b-o', label='Train Loss', linewidth=2, markersize=6)
    axes[0].plot(epochs, data['val_loss'], 'r-s', label='Val Loss', linewidth=2, markersize=6)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Plot Accuracy
    axes[1].plot(epochs, data['train_acc'], 'b-o', label='Train Acc', linewidth=2, markersize=6)
    axes[1].plot(epochs, data['val_acc'], 'r-s', label='Val Acc', linewidth=2, markersize=6)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(loc='lower right', fontsize=10)
    axes[1].grid(True, alpha=0.3)

    # Add current epoch info
    current_epoch = epochs[-1]
    current_val_acc = data['val_acc'][-1]
    current_val_loss = data['val_loss'][-1]

    fig.suptitle(f'EfficientNet Training on ISIC2019 | Epoch {current_epoch}/50 | Val Acc: {current_val_acc:.4f} | Val Loss: {current_val_loss:.4f}',
                 fontsize=14, fontweight='bold')

    # Save plot
    output_path = os.path.join(output_dir, 'training_progress.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return True

def print_summary(filepath):
    """Print current training summary."""
    data = parse_log_file(filepath)

    if not data['epoch']:
        print("No training data found yet...")
        return

    epochs = data['epoch']
    current_epoch = epochs[-1]
    current_train_loss = data['train_loss'][-1]
    current_train_acc = data['train_acc'][-1]
    current_val_loss = data['val_loss'][-1]
    current_val_acc = data['val_acc'][-1]

    # Find best validation accuracy
    best_val_acc = max(data['val_acc'])
    best_epoch = epochs[data['val_acc'].index(best_val_acc)]

    print(f"\n{'='*70}")
    print(f"Training Progress - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    print(f"Current Epoch: {current_epoch}/50")
    print(f"Train Loss: {current_train_loss:.4f} | Train Acc: {current_train_acc:.4f}")
    print(f"Val Loss:   {current_val_loss:.4f} | Val Acc:   {current_val_acc:.4f}")
    print(f"Best Val Acc: {best_val_acc:.4f} (Epoch {best_epoch})")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        log_file = '/tmp/claude/-home-spoof-adv-pat/tasks/b261336.output'

    print(f"Monitoring training from: {log_file}")
    print("Saving plots to: training_plots/training_progress.png")
    print("Press Ctrl+C to exit.\n")

    last_epoch = 0
    try:
        while True:
            # Check if log file exists
            if not os.path.exists(log_file):
                print(f"Waiting for log file: {log_file}")
                time.sleep(5)
                continue

            # Parse data and check for new epochs
            data = parse_log_file(log_file)
            if data['epoch'] and data['epoch'][-1] > last_epoch:
                last_epoch = data['epoch'][-1]
                print_summary(log_file)
                create_plot(log_file)

            # Wait before next check
            time.sleep(10)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        # Create final plot
        if os.path.exists(log_file):
            print("Creating final plot...")
            create_plot(log_file)
            print_summary(log_file)
        print("Done!")
