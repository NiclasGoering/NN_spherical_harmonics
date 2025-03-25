import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import os
import sys
import glob
import numpy as np
import json
import yaml
import traceback
import math
from datetime import datetime
from functools import partial
import torch.cuda.amp as amp
from typing import List, Dict, Tuple, Any, Optional
import time
import gzip
import io

# Import your model and helper functions
from helpers.FFNN import DeepNN
from helpers.utils import save_dataset, save_results, save_model

# Ensure prints flush immediately
print = partial(print, flush=True)

# PERFORMANCE CONFIGURATION
# These settings are configurable for different experiment sizes
TINY_THRESHOLD = 1000     # n_train < TINY_THRESHOLD for tiny experiments
SMALL_THRESHOLD = 10000   # TINY_THRESHOLD <= n_train < SMALL_THRESHOLD for small experiments
MEDIUM_THRESHOLD = 100000 # SMALL_THRESHOLD <= n_train < MEDIUM_THRESHOLD for medium experiments
LARGE_THRESHOLD = 2000000 # MEDIUM_THRESHOLD <= n_train < LARGE_THRESHOLD for large experiments
# n_train >= LARGE_THRESHOLD for huge experiments

# Reduce max parallel for high-dimensional datasets
# We'll dynamically adjust these based on dimension in worker_process
MAX_PARALLEL_TINY = 8     # For tiny experiments (reduced from 16)
MAX_PARALLEL_SMALL = 4    # For small experiments (reduced from 8)
MAX_PARALLEL_MEDIUM = 4   # For medium experiments (reduced from 8)
MAX_PARALLEL_LARGE = 2    # For large experiments (reduced from 4)
MAX_PARALLEL_HUGE = 1     # For huge experiments (reduced from 2)

BATCH_SIZE_TINY = 1024    # Batch size for tiny experiments
BATCH_SIZE_SMALL = 4096   # Batch size for small experiments
BATCH_SIZE_MEDIUM = 8192  # Batch size for medium experiments
BATCH_SIZE_LARGE = 65536  # Batch size for large experiments
BATCH_SIZE_HUGE = 131072  # Batch size for huge experiments (>1M samples)

ORIG_BATCH_SIZE = 32768   # Original batch size reference for LR scaling

BATCH_POWER = 0.15        # Power factor for batch size LR scaling

# Set maximum evaluation samples to control evaluation time
MAX_EVAL_SAMPLES = 20000  # Maximum number of samples to use for evaluation

def load_yaml_config(config_path):
    """Load and return the configuration from a YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def extract_info_from_path(path):
    """Extract parameters from path."""
    path_lower = path.lower()
    path_upper = path.upper()  # Add this line to also check uppercase
    dist_type = None

    # Check for abbreviated identifiers first (NE, NL, NP)
    if 'NE' in path_upper:
        dist_type = "NE"
    elif 'NL' in path_upper:
        dist_type = "NL"
    elif 'NP' in path_upper:
        dist_type = "NP"
    elif 'SU' in path_upper:
        dist_type = "SU"
    # Fall back to original checks
    elif 'poly' in path_lower:
        dist_type = "NP"
    elif 'lin' in path_lower:
        dist_type = "NL"
    elif 'exp' in path_lower:
        dist_type = "NE"
    elif 'sphere' in path_lower:
        dist_type = "SU"
    else:
        dist_type = "NP"  # Default to "NP"

    basename = os.path.basename(path)
    parts = basename.split('_')

    info = {
        "distribution_type": dist_type
    }

    # Extract order number
    order_num = None
    for i, part in enumerate(parts):
        if part == "O" and i < len(parts) - 1 and parts[i+1].isdigit():
            order_num = parts[i+1]
            break
        if part.startswith('O_') and part[2:].isdigit():
            order_num = part[2:]
            break
    
    if order_num is not None:
        info['order_num'] = order_num

    # Rest of the function remains the same
    for i, part in enumerate(parts):
        if part.startswith('d') and part[1:].isdigit():
            info['input_dim'] = int(part[1:])
        if part.startswith('H') and part[1:].isdigit():
            info['hidden_size'] = int(part[1:])
        if part.startswith('D') and part[1:].isdigit():
            info['depth'] = int(part[1:])
        if part.startswith('a') and i < len(parts) - 1:
            try:
                info['alpha'] = float(part[1:])
            except ValueError:
                pass
    
    return info

def generate_unique_id(config):
    """Generate a unique identifier for this configuration."""
    ds_name = config['ds_name']
    ds_directory = config.get('ds_directory', '')
    base_path = os.path.basename(ds_directory) if ds_directory else ''
    
    # Get order number from the dataset parameters
    order_num = config.get('order_num', "1")
    
    align_suffix = "_align" if config['alignment'] else ""
    
    unique_id = (
        f"{ds_name}_O{order_num}"  # Use O1, O2, etc. format
        f"_h{config['hidden_size']}"
        f"_d{config['depth']}"
        f"_n{config['n_train']}"
        f"_lr{config['lr']}"
        f"_mode{config['mode']}"
        f"_exp{config['experiment_num']}"
        f"{align_suffix}"
    )
    
    return unique_id

def find_dataset_files(directory):
    """Find X and y dataset files in the given directory - simplified approach."""
    # Check for X and y files first (following the new data structure)
    x_files = glob.glob(os.path.join(directory, "dataset_X_*.pt.gz"))
    y_files = glob.glob(os.path.join(directory, "dataset_y_*.pt.gz"))
    
    if x_files and y_files:
        return {'x': x_files[0], 'y': y_files[0]}
    
    # Fall back to combined files if split files not found
    combined_files = glob.glob(os.path.join(directory, "dataset_*.pt.gz"))
    if not combined_files:
        combined_files = glob.glob(os.path.join(directory, "dataset_*.pt"))
    
    if combined_files:
        return {'combined': combined_files[0]}
    
    return None

def load_dataset_info(directory):
    """Load dataset info - simplified approach."""
    if not os.path.isdir(directory):
        return None
    
    # Find dataset files
    dataset_files = find_dataset_files(directory)
    if not dataset_files:
        return None
    
    # Extract parameters
    params = extract_info_from_path(directory)
    
    # Extract dataset name from directory
    ds_name = os.path.basename(directory)
    
    # Calculate size of dataset (simple approximation)
    file_size_mb = 0
    if 'combined' in dataset_files:
        file_size_mb = os.path.getsize(dataset_files['combined']) / (1024 * 1024)
    elif 'x' in dataset_files and 'y' in dataset_files:
        x_size_mb = os.path.getsize(dataset_files['x']) / (1024 * 1024)
        y_size_mb = os.path.getsize(dataset_files['y']) / (1024 * 1024)
        file_size_mb = x_size_mb + y_size_mb
    
    return {
        "files": dataset_files,
        "name": ds_name,
        "params": params,
        "directory": directory,
        "size_mb": file_size_mb
    }

def load_dataset_directly(dataset_files, device):
    """
    Efficient dataset loading - optimized from the old version but compatible with new data structure.
    """
    data = {'X': None, 'y': None}
    
    try:
        if 'combined' in dataset_files:
            # Load combined file
            file_path = dataset_files['combined']
            print(f"Loading combined dataset from: {file_path}")
            
            if file_path.endswith('.pt.gz'):
                # Handle gzipped files
                with gzip.open(file_path, 'rb') as f:
                    loaded_data = torch.load(f, map_location='cpu')
            else:
                # Regular .pt files
                loaded_data = torch.load(file_path, map_location='cpu')
            
            # Check if it has X and y keys
            if isinstance(loaded_data, dict) and 'X' in loaded_data and 'y' in loaded_data:
                data['X'] = loaded_data['X'].to(device, non_blocking=True) if loaded_data['X'] is not None else None
                data['y'] = loaded_data['y'].to(device, non_blocking=True) if loaded_data['y'] is not None else None
        else:
            # Load separate X and y files - more direct approach
            if 'x' in dataset_files:
                print(f"Loading X data from: {dataset_files['x']}")
                if dataset_files['x'].endswith('.pt.gz'):
                    with gzip.open(dataset_files['x'], 'rb') as f:
                        x_data = torch.load(f, map_location='cpu')
                else:
                    x_data = torch.load(dataset_files['x'], map_location='cpu')
                
                # Simplified handling - use X directly if it's a tensor
                if isinstance(x_data, dict) and 'X' in x_data:
                    data['X'] = x_data['X'].to(device, non_blocking=True) if x_data['X'] is not None else None
                else:
                    data['X'] = x_data.to(device, non_blocking=True)
            
            if 'y' in dataset_files:
                print(f"Loading y data from: {dataset_files['y']}")
                if dataset_files['y'].endswith('.pt.gz'):
                    with gzip.open(dataset_files['y'], 'rb') as f:
                        y_data = torch.load(f, map_location='cpu')
                else:
                    y_data = torch.load(dataset_files['y'], map_location='cpu')
                
                # Simplified handling - use y directly if it's a tensor
                if isinstance(y_data, dict) and 'y' in y_data:
                    data['y'] = y_data['y'].to(device, non_blocking=True) if y_data['y'] is not None else None
                else:
                    data['y'] = y_data.to(device, non_blocking=True)
        
        # Validation check
        if data['X'] is None or data['y'] is None:
            print("WARNING: X or y data is None after loading!")
        else:
            print(f"Successfully loaded dataset - X shape: {data['X'].shape}, y shape: {data['y'].shape}")
    
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        traceback.print_exc()
    
    return data

def generate_all_combinations(config):
    """Generate all parameter combinations."""
    base_cfg = config["base_config"]
    sweeps = config["sweeps"]
    
    all_combinations = []
    
    for sweep_name, sweep_info in sweeps.items():
        # Handle both directory-based and explicit path-based configs
        dataset_paths = sweep_info.get("dataset_paths", [])
        dataset_dirs = sweep_info.get("dataset_dir", [])
        sweep_params = sweep_info.get("parameters", {})
        
        # Handle directory-based approach if specified
        if dataset_dirs:
            expanded_paths = []
            for base_dir in dataset_dirs:
                print(f"Scanning directory: {base_dir}")
                try:
                    # Get all subdirectories that might contain datasets
                    subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) 
                              if os.path.isdir(os.path.join(base_dir, d))]
                    print(f"Found {len(subdirs)} potential dataset directories")
                    expanded_paths.extend(subdirs)
                except Exception as e:
                    print(f"Error scanning directory {base_dir}: {str(e)}")
            
            # Add discovered paths to dataset_paths
            dataset_paths.extend(expanded_paths)
        
        dataset_infos = []
        for ds_path in dataset_paths:
            ds_info = load_dataset_info(ds_path)
            if ds_info:
                dataset_infos.append(ds_info)
        
        print(f"Loaded {len(dataset_infos)} valid datasets for sweep {sweep_name}")
        
        for ds_info in dataset_infos:
            ds_params = ds_info['params']
            input_dim = ds_params.get('input_dim')
            if not input_dim:
                # Try to extract from directory name
                input_dim_match = None
                for part in ds_info['name'].split('_'):
                    if part.startswith('d') and part[1:].isdigit():
                        input_dim = int(part[1:])
                        break
                if not input_dim:
                    print(f"Warning: Could not determine input_dim for {ds_info['directory']}, skipping")
                    continue
            
            # Pass the order number to the combinations
            order_num = ds_params.get('order_num')
            
            for n_train in sweep_params.get("n_train", [1024]):
                for lr in sweep_params.get("learning_rates", [0.001]):
                    for hidden_size in sweep_params.get("hidden_sizes", [256]):
                        for depth in sweep_params.get("depths", [1]):
                            for mode in sweep_params.get("modes", ["standard"]):
                                for alignment in sweep_params.get("alignment", [False]):
                                    for exp_num in range(1, base_cfg.get("num_experiments", 1) + 1):
                                        combo = {
                                            'dataset_files': ds_info['files'],
                                            'ds_directory': ds_info['directory'],
                                            'ds_name': ds_info['name'],
                                            'hidden_size': hidden_size,
                                            'depth': depth,
                                            'input_dim': input_dim,
                                            'n_train': n_train,
                                            'lr': lr,
                                            'mode': mode,
                                            'gamma': ds_params.get('gamma', 1.0),
                                            'experiment_num': exp_num,
                                            'base_width': sweep_params.get('base_width', 10),
                                            'alignment': alignment,
                                            'sweep_name': sweep_name,
                                            'alpha': ds_params.get('alpha', 1.0),
                                            'size_mb': ds_info.get('size_mb', 0),
                                            'distribution_type': ds_params.get('distribution_type', '')
                                        }
                                        # Add order_num if available
                                        if order_num:
                                            combo['order_num'] = order_num
                                        all_combinations.append(combo)
    
    return all_combinations

def worker_process(gpu_id, num_gpus, all_combinations, config, full_results_dir, timestamp, checkpoint_log_path, completed_configs):
    """
    MEMORY-EFFICIENT IMPLEMENTATION: Balances parallelism with memory constraints.
    Adapts parallelism based on dataset dimensions and size.
    """
    try:
        start_time = time.time()
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
        
        # Set memory management configuration to avoid fragmentation
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        # Enable H100 optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
            torch.backends.cuda.enable_mem_efficient_sdp = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
        torch.cuda.empty_cache()
        
        # Report memory status
        if hasattr(torch.cuda, 'memory_reserved'):
            reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
            allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
            print(f"[GPU {gpu_id}] Initial memory: Reserved: {reserved:.2f} GB, Allocated: {allocated:.2f} GB")
        
        print(f"[GPU {gpu_id}] Worker started on device {device}")
        
        # Filter combinations for this GPU using modulo assignment
        worker_combinations = [combo for i, combo in enumerate(all_combinations) if i % num_gpus == gpu_id]
        print(f"[GPU {gpu_id}] Assigned {len(worker_combinations)} configurations")
        
        # Remove already completed configurations
        worker_combinations = [combo for combo in worker_combinations 
                              if generate_unique_id(combo) not in completed_configs]
        
        if not worker_combinations:
            print(f"[GPU {gpu_id}] All configurations already completed")
            return
            
        print(f"[GPU {gpu_id}] Processing {len(worker_combinations)} incomplete configurations")
        
        # Group by dataset directory
        by_dataset = {}
        for combo in worker_combinations:
            ds_dir = combo['ds_directory']
            if ds_dir not in by_dataset:
                by_dataset[ds_dir] = []
            by_dataset[ds_dir].append(combo)
        
        # Process each dataset group
        completed_count = 0
        total_to_process = len(worker_combinations)
        
        for ds_dir, dataset_combos in by_dataset.items():
            # Skip empty dataset groups
            if not dataset_combos:
                continue
            
            # Force memory cleanup before loading new dataset
            torch.cuda.empty_cache()
            
            print(f"[GPU {gpu_id}] Loading dataset from: {ds_dir}")
            
            try:
                # Load dataset once for all experiments
                dataset_files = dataset_combos[0]['dataset_files']
                data = load_dataset_directly(dataset_files, device)
                
                X_full = data['X']
                y_full = data['y']
                
                # Get data dimensions for dynamic parallelism adjustments
                input_dim = X_full.shape[1]
                dataset_size = X_full.shape[0]
                
                print(f"[GPU {gpu_id}] Loaded dataset - X shape: {X_full.shape}, y shape: {y_full.shape}")
                
                # Dynamically adjust parallelism based on data dimensions
                # High-dimensional data needs less parallelism to avoid OOM
                parallel_scale = 1.0
                if input_dim >= 64:
                    parallel_scale = 0.25  # Reduce parallelism by 75% for d>=64
                elif input_dim >= 32:
                    parallel_scale = 0.5   # Reduce parallelism by 50% for 32<=d<64
                elif input_dim >= 16:
                    parallel_scale = 0.75  # Reduce parallelism by 25% for 16<=d<32
                
                # Apply scaling to max parallel values
                tiny_parallel = max(1, int(MAX_PARALLEL_TINY * parallel_scale))
                small_parallel = max(1, int(MAX_PARALLEL_SMALL * parallel_scale))
                medium_parallel = max(1, int(MAX_PARALLEL_MEDIUM * parallel_scale))
                large_parallel = max(1, int(MAX_PARALLEL_LARGE * parallel_scale))
                huge_parallel = max(1, int(MAX_PARALLEL_HUGE * parallel_scale))
                
                print(f"[GPU {gpu_id}] Adjusted parallelism for dim={input_dim}: " 
                      f"tiny={tiny_parallel}, small={small_parallel}, medium={medium_parallel}, "
                      f"large={large_parallel}, huge={huge_parallel}")
                
                # Group experiments by size for optimal batching using configurable thresholds
                tiny_exps = [c for c in dataset_combos if c['n_train'] < TINY_THRESHOLD]
                small_exps = [c for c in dataset_combos if TINY_THRESHOLD <= c['n_train'] < SMALL_THRESHOLD]
                medium_exps = [c for c in dataset_combos if SMALL_THRESHOLD <= c['n_train'] < MEDIUM_THRESHOLD]
                large_exps = [c for c in dataset_combos if MEDIUM_THRESHOLD <= c['n_train'] < LARGE_THRESHOLD]
                huge_exps = [c for c in dataset_combos if c['n_train'] >= LARGE_THRESHOLD]
                
                # --- Process tiny experiments in parallel batches ---
                if tiny_exps:
                    for i in range(0, len(tiny_exps), tiny_parallel):
                        batch = tiny_exps[i:i+tiny_parallel]
                        print(f"[GPU {gpu_id}] Processing batch of {len(batch)}/{len(tiny_exps)} tiny experiments")
                        
                        # Adjust batch size for high-dimensional data
                        actual_batch_size = BATCH_SIZE_TINY
                        if input_dim >= 64:
                            actual_batch_size = min(actual_batch_size, 512)
                        elif input_dim >= 32:
                            actual_batch_size = min(actual_batch_size, 768)
                            
                        n_completed = fast_parallel_training(
                            batch, device, X_full, y_full, config["base_config"],
                            actual_batch_size, 10, config["base_config"]["epochs"],
                            full_results_dir, timestamp, gpu_id, checkpoint_log_path, completed_configs, BATCH_POWER
                        )
                        completed_count += n_completed
                        print(f"[GPU {gpu_id}] Progress: {completed_count}/{total_to_process} ({completed_count/total_to_process:.1%})")
                        torch.cuda.empty_cache()  # Clean up after each batch
                
                # --- Process small experiments in parallel batches ---
                if small_exps:
                    for i in range(0, len(small_exps), small_parallel):
                        batch = small_exps[i:i+small_parallel]
                        print(f"[GPU {gpu_id}] Processing batch of {len(batch)}/{len(small_exps)} small experiments")
                        
                        # Adjust batch size for high-dimensional data
                        actual_batch_size = BATCH_SIZE_SMALL
                        if input_dim >= 64:
                            actual_batch_size = min(actual_batch_size, 2048)
                        elif input_dim >= 32:
                            actual_batch_size = min(actual_batch_size, 3072)
                            
                        n_completed = fast_parallel_training(
                            batch, device, X_full, y_full, config["base_config"],
                            actual_batch_size, 20, config["base_config"]["epochs"],
                            full_results_dir, timestamp, gpu_id, checkpoint_log_path, completed_configs, BATCH_POWER
                        )
                        completed_count += n_completed
                        print(f"[GPU {gpu_id}] Progress: {completed_count}/{total_to_process} ({completed_count/total_to_process:.1%})")
                        torch.cuda.empty_cache()  # Clean up after each batch
                
                # --- Process medium experiments in smaller parallel batches ---
                if medium_exps:
                    for i in range(0, len(medium_exps), medium_parallel):
                        batch = medium_exps[i:i+medium_parallel]
                        print(f"[GPU {gpu_id}] Processing batch of {len(batch)}/{len(medium_exps)} medium experiments")
                        
                        # Adjust batch size for high-dimensional data
                        actual_batch_size = BATCH_SIZE_MEDIUM
                        if input_dim >= 64:
                            actual_batch_size = min(actual_batch_size, 4096)
                        elif input_dim >= 32:
                            actual_batch_size = min(actual_batch_size, 6144)
                            
                        n_completed = fast_parallel_training(
                            batch, device, X_full, y_full, config["base_config"],
                            actual_batch_size, 30, config["base_config"]["epochs"],
                            full_results_dir, timestamp, gpu_id, checkpoint_log_path, completed_configs, BATCH_POWER
                        )
                        completed_count += n_completed
                        print(f"[GPU {gpu_id}] Progress: {completed_count}/{total_to_process} ({completed_count/total_to_process:.1%})")
                        torch.cuda.empty_cache()  # Clean up after each batch
                
                # --- Process large experiments with larger batch sizes and less parallelism ---
                if large_exps:
                    for i in range(0, len(large_exps), large_parallel):
                        batch = large_exps[i:i+large_parallel]
                        print(f"[GPU {gpu_id}] Processing batch of {len(batch)}/{len(large_exps)} large experiments")
                        
                        # Adjust batch size for high-dimensional data
                        actual_batch_size = BATCH_SIZE_LARGE
                        if input_dim >= 64:
                            actual_batch_size = min(actual_batch_size, 16384)
                        elif input_dim >= 32:
                            actual_batch_size = min(actual_batch_size, 32768)
                            
                        n_completed = fast_parallel_training(
                            batch, device, X_full, y_full, config["base_config"],
                            actual_batch_size, 40, config["base_config"]["epochs"], 
                            full_results_dir, timestamp, gpu_id, checkpoint_log_path, completed_configs, BATCH_POWER
                        )
                        completed_count += n_completed
                        print(f"[GPU {gpu_id}] Progress: {completed_count}/{total_to_process} ({completed_count/total_to_process:.1%})")
                        torch.cuda.empty_cache()  # Clean up after each batch
                
                # --- Process huge experiments individually with maximum batch size ---
                if huge_exps:
                    for i in range(0, len(huge_exps), huge_parallel):
                        batch = huge_exps[i:i+huge_parallel]
                        print(f"[GPU {gpu_id}] Processing batch of {len(batch)}/{len(huge_exps)} huge experiments")
                        
                        # Adjust batch size for high-dimensional data
                        actual_batch_size = BATCH_SIZE_HUGE
                        if input_dim >= 64:
                            actual_batch_size = min(actual_batch_size, 32768)
                        elif input_dim >= 32:
                            actual_batch_size = min(actual_batch_size, 65536)
                            
                        n_completed = fast_parallel_training(
                            batch, device, X_full, y_full, config["base_config"],
                            actual_batch_size, 50, config["base_config"]["epochs"], 
                            full_results_dir, timestamp, gpu_id, checkpoint_log_path, completed_configs, BATCH_POWER
                        )
                        completed_count += n_completed
                        print(f"[GPU {gpu_id}] Progress: {completed_count}/{total_to_process} ({completed_count/total_to_process:.1%})")
                        torch.cuda.empty_cache()  # Clean up after each batch
                
            except Exception as e:
                print(f"[GPU {gpu_id}] ERROR processing dataset {ds_dir}: {str(e)}")
                traceback.print_exc()
                
                # More aggressive cleanup on error
                torch.cuda.empty_cache()
                continue
            
            # Clear memory after processing a dataset
            del X_full, y_full, data
            torch.cuda.empty_cache()
            
            # Report memory status after dataset
            if hasattr(torch.cuda, 'memory_reserved'):
                reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
                allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
                print(f"[GPU {gpu_id}] Memory after dataset: Reserved: {reserved:.2f} GB, Allocated: {allocated:.2f} GB")
        
        elapsed_time = time.time() - start_time
        print(f"[GPU {gpu_id}] Completed all experiments in {elapsed_time:.2f} seconds")
        print(f"[GPU {gpu_id}] Average time per experiment: {elapsed_time/max(1, completed_count):.2f} seconds")
        
    except Exception as e:
        print(f"[GPU {gpu_id}] Fatal error: {str(e)}")
        traceback.print_exc()

def fast_parallel_training(config_batch, device, X_full, y_full, base_config, 
                          batch_size, eval_interval, max_epochs,
                          full_results_dir, timestamp, gpu_id, checkpoint_log_path, completed_configs, BATCH_POWER):
    """
    Ultra-fast parallel training of multiple models on a single GPU with improved memory management.
    Returns the number of successfully completed experiments.
    """
    # Set memory management configuration to reduce fragmentation
    if hasattr(torch.cuda, 'memory_stats'):
        torch.cuda.empty_cache()
        # Try to enable expandable segments for better memory management
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Adjust MAX_PARALLEL values based on input dimension to prevent OOM
    # Lower parallelism for higher dimensions
    d = config_batch[0]['input_dim']
    n_train = config_batch[0]['n_train']
    config_count = len(config_batch)
    
    # Scale down batch size if dimensions are very high (d ≥ 32)
    if d >= 64:
        batch_size = min(batch_size, 32768)  # Maximum batch size for very high dimensions
    elif d >= 32:
        batch_size = min(batch_size, 49152)  # Slightly larger batch size for high dimensions
        
    # Batch size for evaluation (smaller than training to save memory)
    eval_batch_size = min(16384, batch_size)
    
    print(f"[GPU {gpu_id}] Using batch_size={batch_size}, eval_batch_size={eval_batch_size} for d={d}")
    
    # Get test data more efficiently
    n_test = base_config['n_test']
    fixed_seed = abs(hash(config_batch[0]['ds_directory'])) % (2**32)
    generator = torch.Generator(device=device)
    generator.manual_seed(fixed_seed)
    
    # Generate indices first to avoid creating full tensor copies
    indices = torch.randperm(len(X_full), device=device, generator=generator)
    test_indices = indices[:n_test]
    train_master_indices = indices[n_test:]
    
    X_test = X_full[test_indices]
    y_test = y_full[test_indices]
    
    # Setup for parallel training
    models = []
    optimizers = []
    schedulers = [] 
    train_data = []
    config_items = []
    unique_ids = []
    early_stop_flags = []
    
    # Initialize all models
    print(f"[GPU {gpu_id}] Initializing {len(config_batch)} models...")
    for config_item in config_batch:
        unique_id = generate_unique_id(config_item)
        
        if unique_id in completed_configs:
            continue
            
        # Sample training data efficiently
        n_train = config_item['n_train']
        sample_seed = hash(f"sample_{n_train}_{config_item['ds_name']}_{config_item['experiment_num']}")
        torch.manual_seed(sample_seed)
        
        if n_train < len(train_master_indices):
            train_indices = train_master_indices[torch.randperm(len(train_master_indices), device=device)[:n_train]]
            # Store indices rather than full tensors
            train_indices_list = train_indices.tolist()
            # Store indices reference instead of creating tensor slices
            train_data_ref = (train_indices, len(train_indices))
        else:
            # Store reference to all training indices
            train_data_ref = (train_master_indices, len(train_master_indices))
            train_indices_list = train_master_indices.tolist()
        
        # Save dataset if requested, but do this without creating extra copies in GPU memory
        if base_config.get('save_dataset', False):
            dataset_dir = os.path.join(full_results_dir, "datasets")
            os.makedirs(dataset_dir, exist_ok=True)
            dataset_path = os.path.join(dataset_dir, f"dataset_{unique_id}.pt")
            
            # Process in batches to avoid OOM
            with torch.no_grad():
                # Create empty tensors on CPU
                X_train_cpu = torch.zeros((len(train_indices_list), X_full.shape[1]), dtype=X_full.dtype)
                y_train_cpu = torch.zeros((len(train_indices_list), y_full.shape[1]), dtype=y_full.dtype)
                
                # Copy data in batches
                batch_size_copy = 10000  # Smaller batch size for copying
                for i in range(0, len(train_indices_list), batch_size_copy):
                    end_idx = min(i + batch_size_copy, len(train_indices_list))
                    batch_indices = train_indices_list[i:end_idx]
                    X_train_cpu[i:end_idx] = X_full[batch_indices].cpu()
                    y_train_cpu[i:end_idx] = y_full[batch_indices].cpu()
                
                # Save dataset
                dataset = {
                    'X': X_train_cpu, 
                    'y': y_train_cpu, 
                    'X_test': X_test.cpu(), 
                    'y_test': y_test.cpu()
                }
                try:
                    save_dataset(dataset, dataset_path)
                except Exception as e:
                    print(f"[GPU {gpu_id}] Warning: Could not save dataset: {e}")
                
                # Clean up
                del X_train_cpu, y_train_cpu, dataset
        
        # Initialize model
        model_seed = hash(f"model_{config_item['ds_name']}_{timestamp}_{gpu_id}_{config_item['experiment_num']}")
        torch.manual_seed(model_seed)
        
        input_dim = config_item['input_dim']
        hidden_size = config_item['hidden_size']
        depth = config_item['depth']
        
        model = DeepNN(
            input_dim, 
            hidden_size, 
            depth, 
            mode=config_item['mode'], 
            alignment=config_item['alignment'],
            base_width=config_item.get('base_width', 10),
            gamma=config_item.get('gamma', 1.0)
        ).to(device)

        # LR scaling with batch size (keeping the existing scaling mechanism)
        batch_size_ratio = batch_size / ORIG_BATCH_SIZE
        actual_batch_power = 1/4 if batch_size > 32768 else BATCH_POWER
        base_lr = config_item["lr"]
        
        # Apply batch size scaling to learning rate
        scaled_lr = base_lr * (batch_size_ratio ** actual_batch_power)
        
        # Apply input dimension scaling to learning rate to prevent instability
        dim_scale = 1.0 / (1.0 + 0.01 * min(input_dim, 64))
        scaled_lr = scaled_lr * dim_scale
        
        print(f"[GPU {gpu_id}] Model {unique_id}: Base LR: {base_lr}, Scaled LR: {scaled_lr} (batch_factor: {batch_size_ratio**actual_batch_power:.4f}, dim_factor: {dim_scale:.4f})")
        
        weight_decay = float(base_config["weight_decay"])
        
        # Use AdamW for better convergence
        optimizer = optim.Adam(
            model.parameters(), 
            lr=scaled_lr, 
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Use a more robust scheduler that can handle error spikes
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,  # Halve the LR when plateauing
            patience=5,   # Wait 5 evaluation periods before reducing
            min_lr=scaled_lr * 0.01,  # Don't go below 1% of initial LR
            verbose=False
        )
        
        # Store everything
        models.append(model)
        optimizers.append(optimizer)
        schedulers.append(scheduler)
        train_data.append(train_data_ref)  # Store reference, not actual tensors
        config_items.append(config_item)
        unique_ids.append(unique_id)
        early_stop_flags.append(False)
    
    if not models:  # All experiments were already completed
        return 0
    
    # Mixed precision for efficiency
    amp_dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    scaler = torch.amp.GradScaler(enabled=device.type == 'cuda')
    
    # Error history tracking
    train_errors = [[] for _ in range(len(models))]
    test_errors = [[] for _ in range(len(models))]
    epoch_numbers = [[] for _ in range(len(models))]
    
    # Warmup period
    warmup_epochs = max(5, int(max_epochs * 0.1))
    
    # ============== Improved Evaluation Function ==============
    # Only evaluate on a subset of train data (max 20,000 samples) for faster evaluation
    def evaluate_model(model, train_indices, n_samples, eval_batch_size, max_eval_samples=MAX_EVAL_SAMPLES):
        """
        Evaluate model on a subset of training data (at most max_eval_samples).
        This significantly speeds up evaluation for large datasets.
        """
        model.eval()
        
        # Use subset for large datasets to speed up evaluation
        if n_samples > max_eval_samples:
            sample_indices = torch.randperm(n_samples, device=device)[:max_eval_samples]
            eval_indices = train_indices[sample_indices]
            actual_samples = max_eval_samples
            print(f"[GPU {gpu_id}] Evaluating on {max_eval_samples}/{n_samples} training samples (subset)")
        else:
            eval_indices = train_indices
            actual_samples = n_samples
        
        train_error_sum = 0.0
        train_count = 0
        
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=device.type == 'cuda'):
                # Process in batches for evaluation
                for start_idx in range(0, actual_samples, eval_batch_size):
                    end_idx = min(start_idx + eval_batch_size, actual_samples)
                    batch_indices = eval_indices[start_idx:end_idx]
                    batch_X = X_full[batch_indices]
                    batch_y = y_full[batch_indices]
                    
                    batch_output = model(batch_X)
                    batch_error = ((batch_output - batch_y) ** 2).sum().item()
                    train_error_sum += batch_error
                    train_count += batch_indices.size(0)
                    
                    # Clean up immediately
                    del batch_X, batch_y, batch_output, batch_error
        
        if train_count > 0:
            return train_error_sum / train_count
        else:
            return float('inf')
    
    # Track initial errors - using memory-efficient batched evaluation
    print(f"[GPU {gpu_id}] Computing initial errors...")
    for i, model in enumerate(models):
        if early_stop_flags[i]:
            continue
        
        # Get training indices
        train_indices, n_samples = train_data[i]
        
        # Use the new evaluation function
        train_error = evaluate_model(model, train_indices, n_samples, eval_batch_size)
        
        # Process test data in batches
        test_error_sum = 0.0
        test_count = 0
        
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=device.type == 'cuda'):
                for start_idx in range(0, len(X_test), eval_batch_size):
                    end_idx = min(start_idx + eval_batch_size, len(X_test))
                    batch_X_test = X_test[start_idx:end_idx]
                    batch_y_test = y_test[start_idx:end_idx]
                    
                    batch_output = model(batch_X_test)
                    batch_error = ((batch_output - batch_y_test) ** 2).sum().item()
                    test_error_sum += batch_error
                    test_count += batch_X_test.size(0)
                    
                    del batch_X_test, batch_y_test, batch_output, batch_error
        
        # Compute mean test error
        if test_count > 0:
            test_error = test_error_sum / test_count
        else:
            test_error = float('inf')
        
        # Store errors
        train_errors[i].append(train_error)
        test_errors[i].append(test_error)
        epoch_numbers[i].append(0)
        
        # Save initial model if requested
        if base_config.get('save_model', False):
            try:
                initial_model_dir = os.path.join(full_results_dir, "initial_models")
                os.makedirs(initial_model_dir, exist_ok=True)
                initial_model_path = os.path.join(initial_model_dir, f"initial_model_{unique_ids[i]}.pt")
                save_model(model, initial_model_path)
            except Exception as e:
                print(f"[GPU {gpu_id}] Warning: Could not save initial model: {e}")
    
    # Early stopping parameters
    early_stop_threshold = 1e-5
    early_stop_patience = 5
    best_errors = [float('inf') for _ in range(len(models))]
    patience_counters = [0 for _ in range(len(models))]
    
    print(f"[GPU {gpu_id}] Starting training for {len(models)} models...")
    # Training loop with periodic memory cleanup
    for epoch in range(max_epochs):
        # Check if all models have early stopped
        if all(early_stop_flags):
            break
        
        # Periodically clear cache
        if epoch % 5 == 0:
            torch.cuda.empty_cache()
            
        # Train each model with one batch
        for i, model in enumerate(models):
            if early_stop_flags[i]:
                continue
                
            model.train()
            optimizer = optimizers[i]
            train_indices, n_samples = train_data[i]
            
            # Sample random batch
            if n_samples <= batch_size:
                batch_indices = train_indices
            else:
                # Generate random indices for batch
                rand_idx = torch.randint(0, n_samples, (batch_size,), device=device)
                batch_indices = train_indices[rand_idx]
            
            batch_X = X_full[batch_indices]
            batch_y = y_full[batch_indices]
            
            # One training step with gradient clipping to prevent spikes
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=device.type == 'cuda'):
                output = model(batch_X)
                loss = torch.mean((output - batch_y) ** 2)
            
            # Scale the loss and compute gradients
            scaler.scale(loss).backward()
            
            # Unscale the gradients for clipping
            scaler.unscale_(optimizer)
            
            # Clip gradients to prevent error spikes
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update parameters
            scaler.step(optimizer)
            scaler.update()
            
            # Step scheduler after warmup
            if epoch >= warmup_epochs:
                # Note: ReduceLROnPlateau scheduler is updated during evaluation
                pass
            
            # Clean up batch tensors to prevent memory buildup
            del batch_X, batch_y, output, loss, batch_indices
        
        # Evaluate periodically using batched evaluation
        if (epoch + 1) % eval_interval == 0 or epoch == max_epochs - 1:
            for i, model in enumerate(models):
                if early_stop_flags[i]:
                    continue
                    
                train_indices, n_samples = train_data[i]
                
                # Use the new evaluation function
                train_error = evaluate_model(model, train_indices, n_samples, eval_batch_size)
                
                # Calculate test error
                test_error_sum = 0.0
                test_count = 0
                
                with torch.no_grad():
                    with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=device.type == 'cuda'):
                        for start_idx in range(0, len(X_test), eval_batch_size):
                            end_idx = min(start_idx + eval_batch_size, len(X_test))
                            batch_X_test = X_test[start_idx:end_idx]
                            batch_y_test = y_test[start_idx:end_idx]
                            
                            batch_output = model(batch_X_test)
                            batch_error = ((batch_output - batch_y_test) ** 2).sum().item()
                            test_error_sum += batch_error
                            test_count += batch_X_test.size(0)
                            
                            del batch_X_test, batch_y_test, batch_output, batch_error
                
                test_error = test_error_sum / test_count if test_count > 0 else float('inf')
                
                # Store errors
                train_errors[i].append(train_error)
                test_errors[i].append(test_error)
                epoch_numbers[i].append(epoch + 1)
                
                # Update scheduler based on train error
                schedulers[i].step(train_error)
                
                # Early stopping logic
                if train_error < best_errors[i]:
                    best_errors[i] = train_error
                    patience_counters[i] = 0
                else:
                    patience_counters[i] += 1
                
                if train_error < early_stop_threshold or patience_counters[i] >= early_stop_patience:
                    early_stop_flags[i] = True
            
            # Force garbage collection after evaluation
            torch.cuda.empty_cache()
    
    # Add fine tuning phase with memory optimizations
    fine_tuning_epochs = base_config.get("fine_tuning_epochs", 500)
    print(f"[GPU {gpu_id}] Starting fine-tuning phase...")
    
    # Do fine-tuning phase for all models
    for i, model in enumerate(models):
        if early_stop_flags[i] and best_errors[i] > early_stop_threshold:
            train_indices, n_samples = train_data[i]
            optimizer = optimizers[i]
            
            # Use much lower learning rate for fine-tuning
            fine_tuning_lr = optimizer.param_groups[0]['lr'] * 0.1  # 10x smaller
            for param_group in optimizer.param_groups:
                param_group['lr'] = fine_tuning_lr
            
            print(f"[GPU {gpu_id}] Fine-tuning model {unique_ids[i]} with LR={fine_tuning_lr:.8f}")
            
            # Use cosine decay with warm restarts for fine-tuning
            fine_tuning_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=100,  # Restart every 100 epochs
                T_mult=2,  # Double period after each restart
                eta_min=fine_tuning_lr * 0.1  # Don't go below 10% of fine-tuning LR
            )
            
            # Track best state during fine-tuning
            best_ft_error = float('inf')
            best_ft_state = None
            
            # Fine-tuning loop with reduced memory usage
            for ft_epoch in range(fine_tuning_epochs):
                model.train()
                
                # Sample batch
                if n_samples <= batch_size:
                    batch_indices = train_indices
                else:
                    rand_idx = torch.randint(0, n_samples, (batch_size,), device=device)
                    batch_indices = train_indices[rand_idx]
                
                batch_X = X_full[batch_indices]
                batch_y = y_full[batch_indices]
                
                # One training step with gradient clipping
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=device.type == 'cuda'):
                    output = model(batch_X)
                    loss = torch.mean((output - batch_y) ** 2)
                
                loss.backward()
                
                # Clip gradients to prevent spikes
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Stricter clipping for fine-tuning
                
                optimizer.step()
                fine_tuning_scheduler.step()
                
                # Clean up
                del batch_X, batch_y, output, loss, batch_indices
                
                # Evaluate periodically during fine-tuning
                if ft_epoch % 50 == 0 or ft_epoch == fine_tuning_epochs - 1:
                    train_error = evaluate_model(model, train_indices, n_samples, eval_batch_size)
                    
                    # Save best state if improved
                    if train_error < best_ft_error:
                        best_ft_error = train_error
                        best_ft_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    
                    # Calculate test error only at the end to save computation
                    if ft_epoch == fine_tuning_epochs - 1:
                        # Restore best state first
                        if best_ft_state is not None:
                            model.load_state_dict(best_ft_state)
                            model.to(device)  # Ensure it's on the right device
                            
                        with torch.no_grad():
                            with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=device.type == 'cuda'):
                                # Full evaluation after fine-tuning
                                train_error = evaluate_model(model, train_indices, n_samples, eval_batch_size)
                                
                                # Calculate test error
                                test_error_sum = 0.0
                                test_count = 0
                                
                                for start_idx in range(0, len(X_test), eval_batch_size):
                                    end_idx = min(start_idx + eval_batch_size, len(X_test))
                                    batch_X_test = X_test[start_idx:end_idx]
                                    batch_y_test = y_test[start_idx:end_idx]
                                    
                                    batch_output = model(batch_X_test)
                                    batch_error = ((batch_output - batch_y_test) ** 2).sum().item()
                                    test_error_sum += batch_error
                                    test_count += batch_X_test.size(0)
                                    
                                    del batch_X_test, batch_y_test, batch_output, batch_error
                                
                                test_error = test_error_sum / test_count if test_count > 0 else float('inf')
                                
                                # Store errors
                                train_errors[i].append(train_error)
                                test_errors[i].append(test_error)
                                epoch_numbers[i].append(max_epochs + ft_epoch + 1)
                
                # Periodic cleanup during fine-tuning
                if ft_epoch % 50 == 0:
                    torch.cuda.empty_cache()
            
            # Clean up the best state if it was saved
            del best_ft_state
    
    # Save results
    completed_count = 0
    print(f"[GPU {gpu_id}] Saving results for {len(models)} models...")
    
    for i in range(len(models)):
        model = models[i]
        config_item = config_items[i]
        train_indices, n_samples = train_data[i]
        unique_id = unique_ids[i]
        
        # Get final error from history
        final_train_error = train_errors[i][-1]
        final_test_error = test_errors[i][-1]
        
        # No need to re-evaluate since we already have the errors
        
        # Add final epoch if not already added
        if epoch_numbers[i][-1] != max_epochs + fine_tuning_epochs:
            train_errors[i].append(final_train_error)
            test_errors[i].append(final_test_error)
            epoch_numbers[i].append(max_epochs + fine_tuning_epochs)
        
        # Get the final learning rate
        final_lr = optimizers[i].param_groups[0]['lr']
        
        # Create result object
        result = {
            'dataset_name': config_item['ds_name'],
            'dataset_directory': config_item['ds_directory'],
            'hidden_size': config_item['hidden_size'],
            'depth': config_item['depth'],
            'input_dim': config_item['input_dim'],
            'base_width': config_item.get('base_width', 10),
            'n_train': config_item['n_train'],
            'learning_rate': config_item['lr'],
            'mode': config_item['mode'],
            'alignment': config_item['alignment'],
            'gamma': config_item.get('gamma', 1.0),
            'alpha': config_item.get('alpha', 1.0),
            'distribution_type': config_item.get('distribution_type', ''),
            'test_error': final_test_error,
            'initial_train_error': train_errors[i][0],
            'final_train_error': final_train_error,
            'error_history': {
                'train_errors': train_errors[i],
                'test_errors': test_errors[i],
                'epochs': epoch_numbers[i],
                'early_stopped': early_stop_flags[i],
                'stopped_epoch': epoch_numbers[i][-1],
                'best_error': best_errors[i],
                'final_lr': final_lr
            },
            'worker_gpu': gpu_id,
            'model_seed': hash(f"model_{config_item['ds_name']}_{timestamp}_{gpu_id}_{config_item['experiment_num']}"),
            'experiment_num': config_item['experiment_num'],
            'sweep_name': config_item['sweep_name'],
            'parallel_trained': True,
            'batch_size': batch_size,
            'scaled_lr': scaled_lr,
            'batch_size_ratio': batch_size_ratio,
            'batch_power': actual_batch_power
        }
        
        # Add order_num if available
        if 'order_num' in config_item:
            result['order_num'] = config_item['order_num']
        
        # Save results to file
        try:
            results_file_path = os.path.join(full_results_dir, f"results_{timestamp}_gpu{gpu_id}.jsonl")
            with open(results_file_path, "a") as f:
                f.write(json.dumps(result) + "\n")
                f.flush()  # Ensure immediate write to disk
            
            # Save final model if requested
            if base_config.get('save_model', False):
                final_model_dir = os.path.join(full_results_dir, "final_models")
                os.makedirs(final_model_dir, exist_ok=True)
                final_model_path = os.path.join(final_model_dir, f"final_model_{unique_id}.pt")
                save_model(model, final_model_path)
            
            # Mark as completed
            with open(checkpoint_log_path, "a") as cp_f:
                cp_f.write(unique_id + "\n")
                cp_f.flush()  # Ensure immediate write
                
            completed_configs.add(unique_id)
            completed_count += 1
        except Exception as e:
            print(f"[GPU {gpu_id}] Error saving results for {unique_id}: {e}")
    
    # Final cleanup
    del models, optimizers, schedulers, train_data
    torch.cuda.empty_cache()
    
    return completed_count

def main():
    try:
        start_time = time.time()
        print(f"Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        
        if len(sys.argv) < 2:
            print("Usage: python main.py <config_file.yaml>")
            sys.exit(1)
        
        config_path = sys.argv[1]
        print(f"Loading config from: {config_path}")
        
        config = load_yaml_config(config_path)
        
        # Extract Base Config
        base_cfg = config["base_config"]
        base_results_dir = base_cfg["base_results_dir"]
        restart_checkpoint = base_cfg.get("restart_checkpoint")
        
        # Don't automatically change epochs - use what's in the config file
        # Just make sure fine_tuning_epochs is set
        if "fine_tuning_epochs" not in base_cfg:
            base_cfg["fine_tuning_epochs"] = 500
        
        # Create experiment name
        sweep_names = list(config["sweeps"].keys())
        experiment_name = f"{'_'.join(sweep_names)}_exp_{datetime.now().strftime('%Y%m%d')}"
        
        # Set up Results Directory
        full_results_dir = os.path.join(base_results_dir, experiment_name)
        os.makedirs(full_results_dir, exist_ok=True)
        
        # Set up Checkpointing
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_log_path = os.path.join(full_results_dir, f"checkpoint_{timestamp}.txt")
        
        # Handle restart logic
        if restart_checkpoint is not None:
            checkpoint_log_path = restart_checkpoint
            with open(checkpoint_log_path, "r") as f:
                completed_configs = set(line.strip() for line in f if line.strip())
            timestamp = os.path.basename(restart_checkpoint).replace("checkpoint_", "").replace(".txt", "")
            print(f"Restarting from checkpoint with {len(completed_configs)} completed configurations")
        else:
            if os.path.exists(checkpoint_log_path):
                with open(checkpoint_log_path, "r") as f:
                    completed_configs = set(line.strip() for line in f if line.strip())
                print(f"Using existing checkpoint with {len(completed_configs)} completed configs")
            else:
                completed_configs = set()
                print(f"Starting new run")
            
            # Save hyperparameters
            hyperparams_path = os.path.join(full_results_dir, f"hyperparameters_{timestamp}.yaml")
            with open(hyperparams_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
        
        # Generate all combinations
        print("Generating parameter combinations...")
        all_combinations = generate_all_combinations(config)
        print(f"Generated {len(all_combinations)} combinations")
        
        # Filter out completed configurations
        remaining = [c for c in all_combinations if generate_unique_id(c) not in completed_configs]
        print(f"Remaining configurations to process: {len(remaining)}/{len(all_combinations)}")
        
        if not remaining:
            print("All configurations already completed!")
            return
        
        # Get number of available GPUs
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            print("No GPUs available. Running on CPU.")
            num_gpus = 1
        
        print(f"Using {num_gpus} GPU(s)")
        
        # Set multiprocessing start method
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            print("spawn method already set")
        
        # Launch one process per GPU
        mp.spawn(
            worker_process,
            args=(num_gpus, all_combinations, config, full_results_dir, timestamp, checkpoint_log_path, completed_configs),
            nprocs=num_gpus,
            join=True
        )
        
        total_time = time.time() - start_time
        print(f"All processes completed in {total_time:.2f} seconds")
        print(f"Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()