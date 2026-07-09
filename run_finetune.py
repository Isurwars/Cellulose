# @file run_finetune.py
# @copyright Copyright © 2026 Isaías Rodríguez (isurwars@gmail.com)
# @par License
# SPDX-License-Identifier: AGPL-3.0-only

"""
run_finetune.py — Wrapper script to auto-detect hardware and run finetuning with optimized settings.
"""

import argparse
import atexit
import logging
import os
import shutil
import subprocess
import sys
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_system_ram_gb() -> float:
    """Return total system RAM in GB by parsing /proc/meminfo on Linux."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    return float(parts[1]) / (1024.0 * 1024.0)
    except Exception:
        pass
    return 16.0  # Fallback to standard laptop size if detection fails


def get_gpu_vram_gb(device_id: int) -> float:
    """Return GPU VRAM in GB for the specified device ID."""
    if not torch.cuda.is_available():
        return 0.0
    try:
        vram_bytes = torch.cuda.get_device_properties(device_id).total_memory
        return vram_bytes / (1024.0 ** 3)
    except Exception as e:
        logging.warning(f"Could not retrieve GPU VRAM: {e}")
        return 0.0


def main() -> None:
    # 1. Parse arguments we want to auto-scale or manage
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data_path", default="cellulose.db", type=str)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--accumulation_steps", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--device_id", type=int, default=0)

    args, unknown_args = parser.parse_known_args()

    # Detect hardware capabilities
    ram_gb = get_system_ram_gb()
    cpu_cores = os.cpu_count() or 4
    vram_gb = get_gpu_vram_gb(args.device_id)

    logging.info("=" * 60)
    logging.info("HARDWARE DETECTION SUMMARY")
    logging.info(f"  CPU Cores: {cpu_cores}")
    logging.info(f"  System RAM: {ram_gb:.1f} GB")
    if torch.cuda.is_available():
        logging.info(f"  GPU VRAM: {vram_gb:.1f} GB (Device {args.device_id})")
    else:
        logging.info("  GPU: None / Not Available (CPU mode)")
    logging.info("=" * 60)

    # 2. Dynamic parameter auto-scaling logic
    # Maintain target effective batch size of 32 (batch_size * accumulation_steps = 32)
    auto_batch_size = args.batch_size
    auto_accumulation_steps = args.accumulation_steps
    auto_num_workers = args.num_workers

    if auto_batch_size is None or auto_accumulation_steps is None:
        if vram_gb >= 20.0:
            # High-end Workstation GPU
            batch = 16
            accum = 2
        elif vram_gb >= 10.0:
            # Mid-range GPU
            batch = 8
            accum = 4
        elif torch.cuda.is_available():
            # Laptop GPU (e.g. RTX 4060 Laptop 8GB VRAM)
            batch = 2
            accum = 16
        else:
            # CPU only
            batch = 4
            accum = 8

        if auto_batch_size is None:
            auto_batch_size = batch
        if auto_accumulation_steps is None:
            auto_accumulation_steps = accum

    if auto_num_workers is None:
        # Determine num_workers based on CPU cores and System RAM
        if ram_gb > 250.0 and cpu_cores >= 32:
            # Workstation: High-parallelism database read
            auto_num_workers = 16
        else:
            # Laptop: Balance resource contention
            auto_num_workers = 4

    logging.info("AUTO-TUNED TRAINING PARAMETERS")
    logging.info(f"  Batch Size: {auto_batch_size} (User override: {args.batch_size})")
    logging.info(f"  Accumulation Steps: {auto_accumulation_steps} (User override: {args.accumulation_steps})")
    logging.info(f"  Num Workers: {auto_num_workers} (User override: {args.num_workers})")
    logging.info(f"  Effective Batch Size: {auto_batch_size * auto_accumulation_steps}")
    logging.info("=" * 60)

    # 3. RAM-backed database caching (/dev/shm)
    shm_data_path = None
    original_data_path = args.data_path

    if os.path.exists(original_data_path) and os.path.isfile(original_data_path):
        if os.path.exists("/dev/shm") and os.access("/dev/shm", os.W_OK):
            db_filename = os.path.basename(original_data_path)
            shm_target = os.path.join("/dev/shm", f"cellulose_shm_{db_filename}")

            logging.info(f"Caching database to RAM: copying '{original_data_path}' to '{shm_target}'...")
            try:
                shutil.copy2(original_data_path, shm_target)
                shm_data_path = shm_target
                logging.info("Database cache successfully created in /dev/shm.")

                # Register cleaning handler at exit
                def cleanup() -> None:
                    if os.path.exists(shm_target):
                        logging.info(f"Cleaning up RAM database cache: '{shm_target}'")
                        try:
                            os.remove(shm_target)
                        except Exception as ce:
                            logging.error(f"Error deleting cached database: {ce}")

                atexit.register(cleanup)
            except Exception as e:
                logging.error(f"Failed to copy database to /dev/shm: {e}. Falling back to '{original_data_path}'")
        else:
            logging.warning("/dev/shm is not writable or does not exist. Using original database path.")
    else:
        logging.warning(f"Data path '{original_data_path}' not found. Passing argument as is.")

    # 4. Construct final arguments
    cmd_args = [
        "--data_path", shm_data_path if shm_data_path else original_data_path,
        "--batch_size", str(auto_batch_size),
        "--accumulation_steps", str(auto_accumulation_steps),
        "--num_workers", str(auto_num_workers),
        "--device_id", str(args.device_id),
    ]

    # Combine with remaining command line arguments
    final_args = cmd_args + unknown_args

    # Configure PyTorch memory allocator options
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Launch PyTorch training
    target_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_electronic.py")
    logging.info(f"Launching training process: {sys.executable} {target_script} {' '.join(final_args)}")
    try:
        subprocess.run([sys.executable, target_script] + final_args, check=True)
    except KeyboardInterrupt:
        logging.info("Training interrupted. Exiting.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        logging.error(f"Training script failed with exit code {e.returncode}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
