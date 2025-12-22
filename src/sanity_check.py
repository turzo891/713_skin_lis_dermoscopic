#!/usr/bin/env python3
"""
Comprehensive Sanity Check for Skin Cancer Classification Project

Run this script to verify that your system is properly configured and ready for training.

Usage:
    python3 sanity_check.py

    or

    python3 sanity_check.py --quick  # Skip model testing (faster)
"""

import sys
import os
import argparse
from pathlib import Path
import subprocess
import importlib.util

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    """Print a formatted header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}")
    print(f"{text.center(80)}")
    print(f"{'='*80}{Colors.ENDC}\n")

def print_success(text):
    """Print a success message."""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")

def print_warning(text):
    """Print a warning message."""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")

def print_error(text):
    """Print an error message."""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")

def print_info(text):
    """Print an info message."""
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")

def check_python_version():
    """Check if Python version is compatible."""
    print_header("1. PYTHON VERSION CHECK")

    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    print(f"Python version: {version_str}")

    if version.major == 3 and version.minor >= 8:
        print_success(f"Python {version_str} is compatible")
        return True
    else:
        print_error(f"Python {version_str} is NOT compatible. Requires Python 3.8+")
        return False

def check_dependencies():
    """Check if all required Python packages are installed."""
    print_header("2. PYTHON DEPENDENCIES CHECK")

    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'timm': 'timm (PyTorch Image Models)',
        'albumentations': 'Albumentations',
        'sklearn': 'scikit-learn',
        'pandas': 'pandas',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'PIL': 'Pillow',
        'cv2': 'OpenCV'
    }

    missing_packages = []
    installed_versions = {}

    for package, name in required_packages.items():
        try:
            if package == 'sklearn':
                import sklearn
                module = sklearn
            elif package == 'PIL':
                import PIL
                module = PIL
            elif package == 'cv2':
                import cv2
                module = cv2
            else:
                module = __import__(package)

            version = getattr(module, '__version__', 'unknown')
            installed_versions[name] = version
            print_success(f"{name:30} {version}")
        except ImportError:
            missing_packages.append(name)
            print_error(f"{name:30} NOT INSTALLED")

    if missing_packages:
        print_error(f"\nMissing packages: {', '.join(missing_packages)}")
        print_info("Install with: pip install -r requirements.txt")
        return False
    else:
        print_success("\nAll dependencies installed successfully")
        return True

def check_cuda():
    """Check CUDA availability and GPU information."""
    print_header("3. CUDA AND GPU CHECK")

    try:
        import torch

        cuda_available = torch.cuda.is_available()

        if cuda_available:
            cuda_version = torch.version.cuda
            num_gpus = torch.cuda.device_count()

            print_success(f"CUDA is available")
            print(f"  CUDA version: {cuda_version}")
            print(f"  Number of GPUs: {num_gpus}")

            for i in range(num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"  GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")

            return True
        else:
            print_warning("CUDA is NOT available - Training will be very slow on CPU")
            print_info("For GPU support, install CUDA toolkit and GPU-enabled PyTorch")
            return False
    except Exception as e:
        print_error(f"Error checking CUDA: {str(e)}")
        return False

def check_directory_structure():
    """Check if project directory structure is correct."""
    print_header("4. DIRECTORY STRUCTURE CHECK")

    required_dirs = {
        'src': 'Source code',
        'scripts': 'Executable scripts',
        'scripts/training': 'Training scripts',
        'scripts/data': 'Data processing scripts',
        'scripts/testing': 'Unit tests',
        'scripts/monitoring': 'Monitoring scripts',
        'configs': 'Configuration files',
        'data': 'Datasets',
        'models': 'Saved models',
        'results': 'Results and outputs',
        'logs': 'Training logs',
        'notebooks': 'Jupyter notebooks'
    }

    missing_dirs = []

    for dir_path, description in required_dirs.items():
        if os.path.isdir(dir_path):
            print_success(f"{dir_path:30} {description}")
        else:
            missing_dirs.append(dir_path)
            print_error(f"{dir_path:30} NOT FOUND")

    if missing_dirs:
        print_error(f"\nMissing directories: {', '.join(missing_dirs)}")
        print_info("Create with: mkdir -p " + " ".join(missing_dirs))
        return False
    else:
        print_success("\nAll required directories exist")
        return True

def check_dataset():
    """Check if ISIC2019 dataset is properly downloaded."""
    print_header("5. DATASET CHECK")

    dataset_path = Path("data/ISIC2019")
    image_dir = dataset_path / "ISIC_2019_Training_Input" / "ISIC_2019_Training_Input"
    csv_gt = dataset_path / "ISIC_2019_Training_GroundTruth.csv"
    csv_meta = dataset_path / "ISIC_2019_Training_Metadata.csv"

    # Check if dataset directory exists
    if not dataset_path.exists():
        print_error("Dataset directory not found: data/ISIC2019")
        print_info("Download with: cd scripts/data && python3 download_isic_alternative.py")
        return False

    # Check CSV files
    csv_ok = True
    if csv_gt.exists():
        print_success(f"Ground truth CSV found: {csv_gt}")
    else:
        print_error(f"Ground truth CSV not found: {csv_gt}")
        csv_ok = False

    if csv_meta.exists():
        print_success(f"Metadata CSV found: {csv_meta}")
    else:
        print_error(f"Metadata CSV not found: {csv_meta}")
        csv_ok = False

    # Check images
    if image_dir.exists():
        image_files = list(image_dir.glob("*.jpg"))
        num_images = len(image_files)

        if num_images > 0:
            print_success(f"Image directory found: {image_dir}")
            print(f"  Number of images: {num_images:,}")

            if num_images == 25331:
                print_success(f"  Complete dataset (expected 25,331 images)")
            elif num_images < 25331:
                print_warning(f"  Incomplete dataset (expected 25,331, found {num_images:,})")
            else:
                print_warning(f"  More images than expected (expected 25,331, found {num_images:,})")
        else:
            print_error(f"Image directory is empty: {image_dir}")
            return False
    else:
        print_error(f"Image directory not found: {image_dir}")
        print_info("Download with: cd scripts/data && python3 download_isic_alternative.py")
        return False

    # Analyze dataset if CSVs exist
    if csv_ok and csv_gt.exists():
        try:
            import pandas as pd

            labels_df = pd.read_csv(csv_gt)
            print(f"\n  Ground truth shape: {labels_df.shape}")

            # Count samples per class
            class_cols = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
            print("\n  Class distribution:")
            for col in class_cols:
                if col in labels_df.columns:
                    count = int(labels_df[col].sum())
                    pct = count / len(labels_df) * 100
                    print(f"    {col:6} {count:>6,} ({pct:>5.2f}%)")

            # Check for class imbalance
            counts = [int(labels_df[col].sum()) for col in class_cols if col in labels_df.columns]
            if counts:
                max_count = max(counts)
                min_count = min(counts)
                imbalance_ratio = max_count / min_count

                if imbalance_ratio > 10:
                    print_warning(f"\n  Severe class imbalance: {imbalance_ratio:.1f}:1 ratio")
                    print_info("  Use weighted loss: --weighted_loss flag during training")
                else:
                    print_success(f"\n  Class imbalance ratio: {imbalance_ratio:.1f}:1")

        except Exception as e:
            print_warning(f"Could not analyze dataset: {str(e)}")

    return csv_ok and image_dir.exists() and num_images > 0

def check_models():
    """Check if model definitions are working."""
    print_header("6. MODEL ARCHITECTURE CHECK")

    try:
        sys.path.insert(0, 'src')
        from models import get_model
        import torch

        models_to_test = {
            'resnet50': 'ResNet50',
            'efficientnet': 'EfficientNet-B4',
            'densenet': 'DenseNet201',
            'vit': 'Vision Transformer',
            'swin': 'Swin Transformer'
        }

        all_passed = True

        for model_name, display_name in models_to_test.items():
            try:
                # Create model
                model = get_model(model_name, num_classes=8, pretrained=False)

                # Test forward pass
                dummy_input = torch.randn(2, 3, 224, 224)
                with torch.no_grad():
                    output = model(dummy_input)

                # Check output shape
                if output.shape == (2, 8):
                    print_success(f"{display_name:25} OK (output: {output.shape})")
                else:
                    print_error(f"{display_name:25} FAILED (wrong output shape: {output.shape})")
                    all_passed = False

            except Exception as e:
                print_error(f"{display_name:25} FAILED ({str(e)[:50]})")
                all_passed = False

        return all_passed

    except Exception as e:
        print_error(f"Could not load models: {str(e)}")
        return False

def check_data_loader():
    """Check if data loader is working."""
    print_header("7. DATA LOADER CHECK")

    try:
        sys.path.insert(0, 'src')
        from data_loader import create_data_loaders

        # Check if dataset exists first
        if not os.path.exists('data/ISIC2019/ISIC_2019_Training_GroundTruth.csv'):
            print_warning("Dataset not found - skipping data loader test")
            print_info("Download dataset first: cd scripts/data && python3 download_isic_alternative.py")
            return False

        print("Creating data loaders (this may take a minute)...")

        train_loader, val_loader, test_loader, label_encoder = create_data_loaders(
            data_dir='data/ISIC2019/ISIC_2019_Training_Input/ISIC_2019_Training_Input',
            csv_path='data/ISIC2019/ISIC_2019_Training_GroundTruth.csv',
            batch_size=16,
            image_size=224,
            num_workers=0,
            val_split=0.15,
            test_split=0.15
        )

        print_success(f"Data loaders created successfully")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Validation batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        print(f"  Number of classes: {len(label_encoder)}")

        # Test loading one batch
        print("\nTesting batch loading...")
        batch_images, batch_labels = next(iter(train_loader))
        print_success(f"Successfully loaded a batch")
        print(f"  Batch images shape: {batch_images.shape}")
        print(f"  Batch labels shape: {batch_labels.shape}")

        return True

    except Exception as e:
        print_error(f"Data loader test failed: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

def check_scripts():
    """Check if training scripts are executable."""
    print_header("8. TRAINING SCRIPTS CHECK")

    scripts_to_check = {
        'scripts/training/train_single_model.py': 'Single model training',
        'scripts/training/train_with_logging.py': 'Training with logging',
        'scripts/training/train_kfold_cv.py': 'K-fold cross-validation',
        'scripts/data/exploratory_data_analysis.py': 'Exploratory data analysis',
        'scripts/data/advanced_visualizations.py': 'Advanced visualizations',
        'scripts/data/validate_dataset.py': 'Dataset validation'
    }

    all_ok = True

    for script_path, description in scripts_to_check.items():
        if os.path.isfile(script_path):
            # Check if file is executable or readable
            if os.access(script_path, os.R_OK):
                print_success(f"{description:30} {script_path}")
            else:
                print_warning(f"{description:30} {script_path} (not executable)")
        else:
            print_error(f"{description:30} NOT FOUND: {script_path}")
            all_ok = False

    return all_ok

def check_config_files():
    """Check if configuration files exist."""
    print_header("9. CONFIGURATION FILES CHECK")

    config_files = {
        'requirements.txt': 'Python dependencies',
        'README.md': 'Project documentation',
        'USER_MANUAL.md': 'User manual',
        'SANITY_CHECK_REPORT.md': 'Sanity check report'
    }

    all_ok = True

    for file_path, description in config_files.items():
        if os.path.isfile(file_path):
            file_size = os.path.getsize(file_path)
            print_success(f"{description:30} {file_path} ({file_size:,} bytes)")
        else:
            print_warning(f"{description:30} NOT FOUND: {file_path}")
            if file_path == 'requirements.txt':
                all_ok = False

    return all_ok

def generate_summary(results):
    """Generate and print summary of all checks."""
    print_header("SANITY CHECK SUMMARY")

    total_checks = len(results)
    passed_checks = sum(results.values())
    failed_checks = total_checks - passed_checks

    print(f"Total checks: {total_checks}")
    print_success(f"Passed: {passed_checks}")

    if failed_checks > 0:
        print_error(f"Failed: {failed_checks}")

    print("\nDetailed Results:")
    for check_name, passed in results.items():
        status = Colors.OKGREEN + "PASSED" + Colors.ENDC if passed else Colors.FAIL + "FAILED" + Colors.ENDC
        print(f"  {check_name:40} {status}")

    print(f"\n{Colors.BOLD}Overall Status:{Colors.ENDC} ", end="")
    if failed_checks == 0:
        print_success("READY FOR TRAINING")
        print("\nYou can start training with:")
        print(f"{Colors.OKCYAN}  cd scripts/training")
        print(f"  python3 train_single_model.py --model resnet50 --epochs 50{Colors.ENDC}")
    elif passed_checks >= total_checks - 2:
        print_warning("MOSTLY READY (some optional checks failed)")
        print("\nYou can proceed with caution, but review the failed checks above.")
    else:
        print_error("NOT READY FOR TRAINING")
        print("\nPlease fix the failed checks before training.")
        print("\nCommon fixes:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Download dataset: cd scripts/data && python3 download_isic_alternative.py")
        print("  3. Check GPU drivers: nvidia-smi")

    return failed_checks == 0

def main():
    """Main function to run all sanity checks."""
    parser = argparse.ArgumentParser(description='Comprehensive sanity check for skin cancer classification project')
    parser.add_argument('--quick', action='store_true', help='Skip model testing (faster)')
    parser.add_argument('--skip-data', action='store_true', help='Skip dataset check')
    args = parser.parse_args()

    print(f"{Colors.HEADER}{Colors.BOLD}")
    print("="*80)
    print("SKIN CANCER CLASSIFICATION PROJECT - SANITY CHECK".center(80))
    print("="*80)
    print(f"{Colors.ENDC}")

    results = {}

    # Run all checks
    results['Python Version'] = check_python_version()
    results['Dependencies'] = check_dependencies()
    results['CUDA/GPU'] = check_cuda()
    results['Directory Structure'] = check_directory_structure()

    if not args.skip_data:
        results['Dataset'] = check_dataset()

    if not args.quick:
        results['Models'] = check_models()
        if not args.skip_data:
            results['Data Loader'] = check_data_loader()

    results['Scripts'] = check_scripts()
    results['Config Files'] = check_config_files()

    # Generate summary
    all_passed = generate_summary(results)

    # Save report to file
    report_path = "sanity_check_results.txt"
    try:
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("SANITY CHECK RESULTS\n")
            f.write("="*80 + "\n\n")
            for check_name, passed in results.items():
                status = "PASSED" if passed else "FAILED"
                f.write(f"{check_name:40} {status}\n")
            f.write("\n" + "="*80 + "\n")
            if all_passed:
                f.write("Overall Status: READY FOR TRAINING\n")
            else:
                f.write("Overall Status: NOT READY (see failed checks above)\n")

        print(f"\n{Colors.OKBLUE}Report saved to: {report_path}{Colors.ENDC}")
    except Exception as e:
        print_warning(f"Could not save report: {str(e)}")

    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)

if __name__ == '__main__':
    main()
