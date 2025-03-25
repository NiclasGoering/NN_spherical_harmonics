#!/usr/bin/env python3
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
#PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# PERFORMANCE CONFIGURATION
# These settings are configurable for different experiment sizes
TINY_THRESHOLD = 1000     # n_train < TINY_THRESHOLD for tiny experiments
SMALL_THRESHOLD = 10000   # TINY_THRESHOLD <= n_train < SMALL_THRESHOLD for small experiments
MEDIUM_THRESHOLD = 100000 # SMALL_THRESHOLD <= n_train < MEDIUM_THRESHOLD for medium experiments
LARGE_THRESHOLD = 2000000 # MEDIUM_THRESHOLD <= n_train < LARGE_THRESHOLD for large experiments
# n_train >= LARGE_THRESHOLD for huge experiments

MAX_PARALLEL_TINY = 16    # For tiny experiments
MAX_PARALLEL_SMALL = 8    # For small experiments
MAX_PARALLEL_MEDIUM = 8   # For medium experiments
MAX_PARALLEL_LARGE = 1    # For large experiments
MAX_PARALLEL_HUGE = 1     # For huge experiments

BATCH_SIZE_TINY = 1024    # Batch size for tiny experiments
BATCH_SIZE_SMALL = 4096   # Batch size for small experiments
BATCH_SIZE_MEDIUM = 8192  # Batch size for medium experiments
BATCH_SIZE_LARGE = 65536  # Batch size for large experiments
BATCH_SIZE_HUGE = 131072  # Batch size for huge experiments (>1M samples)

ORIG_BATCH_SIZE = 32768   # Original batch size reference for LR scaling

BATCH_POWER = 0.5

# New constant for evaluation subset size
EVAL_SUBSET_SIZE = 20000  # Maximum number of points to use for evaluation

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
    ds_path = config.get('ds_directory', '')
    base_path = os.path.basename(ds_path) if ds_path else ''
    
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
    """Find X and y dataset files in the given directory."""
    # Look for the X and y files
    x_files = glob.glob(os.path.join(directory, "dataset_X_*.pt.gz"))
    y_files = glob.glob(os.path.join(directory, "dataset_y_*.pt.gz"))
    
    # If no separate X and y files, look for combined files
    if not (x_files and y_files):
        combined_files = glob.glob(os.path.join(directory, "dataset_*.pt.gz"))
        if not combined_files:
            combined_files = glob.glob(os.path.join(directory, "dataset_*.pt"))
        
        if combined_files:
            return {'combined': combined_files[0]}
    
    if x_files and y_files:
        return {'x': x_files[0], 'y': y_files[0]}
    
    return None

def load_dataset_info(directory):
    """Load dataset info."""
    if not os.path.isdir(directory):
        return None
    
    # Find dataset files
    dataset_files = find_dataset_files(directory)
    if not dataset_files:
        return None
    
    # Extract parameters
    params = extract_info_from_path(directory)
    
    # Check if we found the files
    if not dataset_files:
        return None
    
    # Set parameters based on the path info
    dist_type = params.get("distribution_type", "")
    input_dim = params.get("input_dim", "")
    
    # Extract dataset name from directory
    ds_name = os.path.basename(directory)
    
    # Calculate size of dataset
    if 'combined' in dataset_files:
        file_size_mb = os.path.getsize(dataset_files['combined']) / (1024 * 1024)
    else:
        x_size_mb = os.path.getsize(dataset_files['x']) / (1024 * 1024) if 'x' in dataset_files else 0
        y_size_mb = os.path.getsize(dataset_files['y']) / (1024 * 1024) if 'y' in dataset_files else 0
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
    Load dataset directly to GPU, handling both combined and separate X/y files.
    
    Args:
        dataset_files: Dictionary with 'combined' key or 'x' and 'y' keys
        device: Device to load data to
        
    Returns:
        Dictionary with 'X' and 'y' tensors
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
                if loaded_data['X'] is not None:
                    data['X'] = loaded_data['X'].to(device, non_blocking=True)
                if loaded_data['y'] is not None:
                    data['y'] = loaded_data['y'].to(device, non_blocking=True)
        else:
            # Load separate X and y files
            # Load X file
            x_path = dataset_files['x']
            print(f"Loading X data from: {x_path}")
            
            if x_path.endswith('.pt.gz'):
                with gzip.open(x_path, 'rb') as f:
                    x_data = torch.load(f, map_location='cpu')
            else:
                x_data = torch.load(x_path, map_location='cpu')
            
            # Check if it's a dictionary with 'X' key or a direct tensor
            if isinstance(x_data, dict) and 'X' in x_data and x_data['X'] is not None:
                data['X'] = x_data['X'].to(device, non_blocking=True)
            else:
                data['X'] = x_data.to(device, non_blocking=True)
            
            # Load y file
            y_path = dataset_files['y']
            print(f"Loading y data from: {y_path}")
            
            if y_path.endswith('.pt.gz'):
                with gzip.open(y_path, 'rb') as f:
                    y_data = torch.load(f, map_location='cpu')
            else:
                y_data = torch.load(y_path, map_location='cpu')
            
            # Check if it's a dictionary with 'y' key or a direct tensor
            if isinstance(y_data, dict) and 'y' in y_data and y_data['y'] is not None:
                data['y'] = y_data['y'].to(device, non_blocking=True)
            else:
                data['y'] = y_data.to(device, non_blocking=True)
    
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        traceback.print_exc()
    
    # Final check to ensure both X and y are loaded
    if data['X'] is None or data['y'] is None:
        raise ValueError("Failed to load X or y data from files")
    
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
    FASTEST IMPLEMENTATION: Aggressively parallel worker process.
    Runs maximum number of experiments in parallel based on size.
    """
    try:
        start_time = time.time()
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
        
        # Enable H100 optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
            torch.backends.cuda.enable_mem_efficient_sdp = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
        torch.cuda.empty_cache()
        
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
                
            print(f"[GPU {gpu_id}] Loading dataset from: {ds_dir}")
            
            try:
                # Load dataset once for all experiments
                dataset_files = dataset_combos[0]['dataset_files']
                data = load_dataset_directly(dataset_files, device)
                
                X_full = data['X']
                y_full = data['y']
                
                print(f"[GPU {gpu_id}] Loaded dataset - X shape: {X_full.shape}, y shape: {y_full.shape}")
                
                # Group experiments by size for optimal batching using configurable thresholds
                tiny_exps = [c for c in dataset_combos if c['n_train'] < TINY_THRESHOLD]
                small_exps = [c for c in dataset_combos if TINY_THRESHOLD <= c['n_train'] < SMALL_THRESHOLD]
                medium_exps = [c for c in dataset_combos if SMALL_THRESHOLD <= c['n_train'] < MEDIUM_THRESHOLD]
                large_exps = [c for c in dataset_combos if MEDIUM_THRESHOLD <= c['n_train'] < LARGE_THRESHOLD]
                huge_exps = [c for c in dataset_combos if c['n_train'] >= LARGE_THRESHOLD]
                
                # --- Process tiny experiments in large parallel batches ---
                if tiny_exps:
                    for i in range(0, len(tiny_exps), MAX_PARALLEL_TINY):
                        batch = tiny_exps[i:i+MAX_PARALLEL_TINY]
                        print(f"[GPU {gpu_id}] Processing batch of {len(batch)}/{len(tiny_exps)} tiny experiments")
                        n_completed = fast_parallel_training(
                            batch, device, X_full, y_full, config["base_config"],
                            BATCH_SIZE_TINY, 10, config["base_config"]["epochs"],  # Use full epochs
                            full_results_dir, timestamp, gpu_id, checkpoint_log_path, completed_configs, BATCH_POWER
                        )
                        completed_count += n_completed
                        print(f"[GPU {gpu_id}] Progress: {completed_count}/{total_to_process} ({completed_count/total_to_process:.1%})")
                
                # --- Process small experiments in parallel batches ---
                if small_exps:
                    for i in range(0, len(small_exps), MAX_PARALLEL_SMALL):
                        batch = small_exps[i:i+MAX_PARALLEL_SMALL]
                        print(f"[GPU {gpu_id}] Processing batch of {len(batch)}/{len(small_exps)} small experiments")
                        n_completed = fast_parallel_training(
                            batch, device, X_full, y_full, config["base_config"],
                            BATCH_SIZE_SMALL, 20, config["base_config"]["epochs"],  # Use full epochs
                            full_results_dir, timestamp, gpu_id, checkpoint_log_path, completed_configs, BATCH_POWER
                        )
                        completed_count += n_completed
                        print(f"[GPU {gpu_id}] Progress: {completed_count}/{total_to_process} ({completed_count/total_to_process:.1%})")
                
                # --- Process medium experiments in smaller parallel batches ---
                if medium_exps:
                    for i in range(0, len(medium_exps), MAX_PARALLEL_MEDIUM):
                        batch = medium_exps[i:i+MAX_PARALLEL_MEDIUM]
                        print(f"[GPU {gpu_id}] Processing batch of {len(batch)}/{len(medium_exps)} medium experiments")
                        n_completed = fast_parallel_training(
                            batch, device, X_full, y_full, config["base_config"],
                            BATCH_SIZE_MEDIUM, 30, config["base_config"]["epochs"],  # Use full epochs
                            full_results_dir, timestamp, gpu_id, checkpoint_log_path, completed_configs, BATCH_POWER
                        )
                        completed_count += n_completed
                        print(f"[GPU {gpu_id}] Progress: {completed_count}/{total_to_process} ({completed_count/total_to_process:.1%})")
                
                # --- Process large experiments with larger batch sizes and less parallelism ---
                if large_exps:
                    for i in range(0, len(large_exps), MAX_PARALLEL_LARGE):
                        batch = large_exps[i:i+MAX_PARALLEL_LARGE]
                        print(f"[GPU {gpu_id}] Processing batch of {len(batch)}/{len(large_exps)} large experiments")
                        n_completed = fast_parallel_training(
                            batch, device, X_full, y_full, config["base_config"],
                            BATCH_SIZE_LARGE, 40, config["base_config"]["epochs"], 
                            full_results_dir, timestamp, gpu_id, checkpoint_log_path, completed_configs, BATCH_POWER
                        )
                        completed_count += n_completed
                        print(f"[GPU {gpu_id}] Progress: {completed_count}/{total_to_process} ({completed_count/total_to_process:.1%})")
                
                # --- Process huge experiments individually with maximum batch size ---
                if huge_exps:
                    for i in range(0, len(huge_exps), MAX_PARALLEL_HUGE):
                        batch = huge_exps[i:i+MAX_PARALLEL_HUGE]
                        print(f"[GPU {gpu_id}] Processing batch of {len(batch)}/{len(huge_exps)} huge experiments")
                        n_completed = fast_parallel_training(
                            batch, device, X_full, y_full, config["base_config"],
                            BATCH_SIZE_HUGE, 50, config["base_config"]["epochs"], 
                            full_results_dir, timestamp, gpu_id, checkpoint_log_path, completed_configs, BATCH_POWER
                        )
                        completed_count += n_completed
                        print(f"[GPU {gpu_id}] Progress: {completed_count}/{total_to_process} ({completed_count/total_to_process:.1%})")
                
            except Exception as e:
                print(f"[GPU {gpu_id}] ERROR processing dataset {ds_dir}: {str(e)}")
                traceback.print_exc()
                continue
            
            # Clear memory after processing a dataset
            del X_full, y_full, data
            torch.cuda.empty_cache()
        
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
    Ultra-fast parallel training of multiple models on a single GPU with improved learning rate handling.
    Uses a subset of training data for evaluation to speed up the process.
    Returns the number of successfully completed experiments.
    """
    # Get test data
    n_test = base_config['n_test']
    fixed_seed = abs(hash(str(config_batch[0]['ds_directory']))) % (2**32)
    generator = torch.Generator(device=device)
    generator.manual_seed(fixed_seed)
    indices = torch.randperm(len(X_full), device=device, generator=generator)
    test_indices = indices[:n_test]
    train_master_indices = indices[n_test:]
    X_test = X_full[test_indices]
    y_test = y_full[test_indices]
    
    # Setup for parallel training
    models = []
    optimizers = []
    schedulers = []  # New: track schedulers
    train_data = []
    eval_data = []  # Track eval data separately
    config_items = []
    unique_ids = []
    early_stop_flags = []
    
    # Initialize all models
    for config_item in config_batch:
        unique_id = generate_unique_id(config_item)
        
        if unique_id in completed_configs:
            continue
            
        # Sample training data
        n_train = config_item['n_train']
        sample_seed = hash(f"sample_{n_train}_{config_item['ds_name']}_{config_item['experiment_num']}")
        torch.manual_seed(sample_seed)
        
        if n_train < len(train_master_indices):
            train_indices = train_master_indices[torch.randperm(len(train_master_indices), device=device)[:n_train]]
            X_train = X_full[train_indices]
            y_train = y_full[train_indices]
        else:
            X_train = X_full[train_master_indices]
            y_train = y_full[train_master_indices]
        
        # Create evaluation subset (limited to EVAL_SUBSET_SIZE points)
        # Use a separate seed to ensure consistency
        eval_seed = hash(f"eval_{n_train}_{config_item['ds_name']}_{config_item['experiment_num']}")
        generator = torch.Generator(device=device)
        generator.manual_seed(eval_seed)
        
        # If train set is smaller than EVAL_SUBSET_SIZE, use all of it
        if len(X_train) <= EVAL_SUBSET_SIZE:
            X_eval = X_train
            y_eval = y_train
        else:
            # Otherwise, sample EVAL_SUBSET_SIZE random points
            eval_indices = torch.randperm(len(X_train), device=device, generator=generator)[:EVAL_SUBSET_SIZE]
            X_eval = X_train[eval_indices]
            y_eval = y_train[eval_indices]
        
        # Save dataset if requested
        if base_config.get('save_dataset', False):
            dataset_dir = os.path.join(full_results_dir, "datasets")
            os.makedirs(dataset_dir, exist_ok=True)
            dataset_path = os.path.join(dataset_dir, f"dataset_{unique_id}.pt")
            dataset = {'X': X_train.cpu(), 'y': y_train.cpu(), 'X_test': X_test.cpu(), 'y_test': y_test.cpu()}
            save_dataset(dataset, dataset_path)
        
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
        
        # Create optimizer with improved learning rate strategy
        # Use more conservative scaling for larger batch sizes
        batch_size_ratio = batch_size / ORIG_BATCH_SIZE
        
        # Adjust BATCH_POWER based on batch size - more conservative for larger batches
        actual_batch_power = 1/4 if batch_size > 32768 else BATCH_POWER
        
        # Use the learning rate from config file
        base_lr = config_item["lr"]
        scaled_lr = base_lr * (batch_size_ratio ** actual_batch_power)
        weight_decay = float(base_config["weight_decay"])
        # Add weight decay for regularization
        optimizer = optim.Adam(model.parameters(), lr=scaled_lr, weight_decay=weight_decay)
        
        # Use CosineAnnealingLR instead of ReduceLROnPlateau
        # This provides a smooth decline in learning rate without rapid collapse
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_epochs,  # Full cycle length matches total epochs
            eta_min=scaled_lr * 0.05  # Don't let LR get too close to zero
        )
        
        # Store everything
        models.append(model)
        optimizers.append(optimizer)
        schedulers.append(scheduler)  # Store scheduler
        train_data.append((X_train, y_train))
        eval_data.append((X_eval, y_eval))  # Store eval data separately
        config_items.append(config_item)
        unique_ids.append(unique_id)
        early_stop_flags.append(False)
    
    if not models:  # All experiments were already completed
        return 0
    
    # Parallel training with BF16 mixed precision
    amp_dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    scaler = torch.amp.GradScaler(enabled=device.type == 'cuda')
    
    # Error history tracking
    train_errors = [[] for _ in range(len(models))]
    test_errors = [[] for _ in range(len(models))]
    epoch_numbers = [[] for _ in range(len(models))]
    
    # Implement a warmup period for the first ~10% of training
    warmup_epochs = max(5, int(max_epochs * 0.1))
    
    # Track initial errors
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=device.type == 'cuda'):
            for i, model in enumerate(models):
                if early_stop_flags[i]:
                    continue
                    
                # Use the evaluation subset for initial error calculation
                X_eval, y_eval = eval_data[i]
                model.eval()
                
                eval_output = model(X_eval)
                test_output = model(X_test)
                
                train_error = torch.mean((eval_output - y_eval) ** 2).item()
                test_error = torch.mean((test_output - y_test) ** 2).item()
                
                train_errors[i].append(train_error)
                test_errors[i].append(test_error)
                epoch_numbers[i].append(0)
                
                # Save initial model if requested
                if base_config.get('save_model', False):
                    initial_model_dir = os.path.join(full_results_dir, "initial_models")
                    os.makedirs(initial_model_dir, exist_ok=True)
                    initial_model_path = os.path.join(initial_model_dir, f"initial_model_{unique_ids[i]}.pt")
                    save_model(model, initial_model_path)
    
    # More reasonable early stopping threshold - not too aggressive
    early_stop_threshold = 1e-5  # Increased from 1e-7
    early_stop_patience = 5  # Number of evaluations without improvement before stopping
    best_errors = [float('inf') for _ in range(len(models))]
    patience_counters = [0 for _ in range(len(models))]
    
    # Fast parallel training loop
    for epoch in range(max_epochs):
        # Check if all models have early stopped
        if all(early_stop_flags):
            break
            
        # Train each model with one batch
        for i, model in enumerate(models):
            if early_stop_flags[i]:
                continue
                
            model.train()
            optimizer = optimizers[i]
            X_train, y_train = train_data[i]
            
            # Sample random batch
            if len(X_train) <= batch_size:
                batch_X, batch_y = X_train, y_train
            else:
                batch_indices = torch.randperm(len(X_train), device=device)[:batch_size]
                batch_X = X_train[batch_indices]
                batch_y = y_train[batch_indices]
            
            # One training step
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=device.type == 'cuda'):
                output = model(batch_X)
                loss = torch.mean((output - batch_y) ** 2)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Step the scheduler every epoch (for CosineAnnealingLR)
            # Skip during warmup period
            if epoch >= warmup_epochs:
                schedulers[i].step()
        
        # Evaluate periodically
        if (epoch + 1) % eval_interval == 0 or epoch == max_epochs - 1:
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=device.type == 'cuda'):
                    for i, model in enumerate(models):
                        if early_stop_flags[i]:
                            continue
                            
                        # Use the evaluation subset for periodic evaluations
                        X_eval, y_eval = eval_data[i]
                        model.eval()
                        
                        eval_output = model(X_eval)
                        test_output = model(X_test)
                        
                        train_error = torch.mean((eval_output - y_eval) ** 2).item()
                        test_error = torch.mean((test_output - y_test) ** 2).item()
                        
                        train_errors[i].append(train_error)
                        test_errors[i].append(test_error)
                        epoch_numbers[i].append(epoch + 1)
                        
                        # Improved early stopping logic with patience
                        if train_error < best_errors[i]:
                            best_errors[i] = train_error
                            patience_counters[i] = 0
                        else:
                            patience_counters[i] += 1
                        
                        # Check for early stopping with patience
                        if train_error < early_stop_threshold or patience_counters[i] >= early_stop_patience:
                            early_stop_flags[i] = True
    
    # Add fine tuning phase - always run full fine-tuning
    fine_tuning_epochs = base_config.get("fine_tuning_epochs", 500)
    
    # Do fine-tuning phase for all models
    for i, model in enumerate(models):
        if early_stop_flags[i] and best_errors[i] > early_stop_threshold:  # Only do fine-tuning if not already converged
            X_train, y_train = train_data[i]
            X_eval, y_eval = eval_data[i]  # Use evaluation subset during fine-tuning too
            optimizer = optimizers[i]
            
            # Reset learning rate for fine-tuning to a smaller value
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] / 10  # Much lower LR for fine-tuning
            
            # Fine-tuning loop
            for ft_epoch in range(fine_tuning_epochs):
                model.train()
                
                # Sample batch
                if len(X_train) <= batch_size:
                    batch_X, batch_y = X_train, y_train
                else:
                    batch_indices = torch.randperm(len(X_train), device=device)[:batch_size]
                    batch_X = X_train[batch_indices]
                    batch_y = y_train[batch_indices]
                
                # One training step
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=device.type == 'cuda'):
                    output = model(batch_X)
                    loss = torch.mean((output - batch_y) ** 2)
                
                loss.backward()
                optimizer.step()
                
                # Evaluate at the end
                if ft_epoch == fine_tuning_epochs - 1:
                    with torch.no_grad():
                        with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=device.type == 'cuda'):
                            model.eval()
                            eval_output = model(X_eval)
                            test_output = model(X_test)
                            
                            train_error = torch.mean((eval_output - y_eval) ** 2).item()
                            test_error = torch.mean((test_output - y_test) ** 2).item()
                            
                            train_errors[i].append(train_error)
                            test_errors[i].append(test_error)
                            epoch_numbers[i].append(max_epochs + ft_epoch + 1)
    
    # Save results
    completed_count = 0
    for i in range(len(models)):
        model = models[i]
        config_item = config_items[i]
        X_train, y_train = train_data[i]
        unique_id = unique_ids[i]
        
        # Final evaluation - use FULL training set for final error calculation
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=device.type == 'cuda'):
                model.eval()
                
                # Calculate error on the full training set for the final result
                train_output = model(X_train)
                test_output = model(X_test)
                
                final_train_error = torch.mean((train_output - y_train) ** 2).item()
                final_test_error = torch.mean((test_output - y_test) ** 2).item()
        
        # Add final epoch if not already added
        if epoch_numbers[i][-1] != max_epochs + fine_tuning_epochs:
            train_errors[i].append(final_train_error)
            test_errors[i].append(final_test_error)
            epoch_numbers[i].append(max_epochs + fine_tuning_epochs)
        
        # Get the final learning rate
        final_lr = optimizers[i].param_groups[0]['lr']
        
        # Base result
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
                'final_lr': final_lr,  # Include final learning rate in results
                'eval_subset_size': min(EVAL_SUBSET_SIZE, len(X_train))  # Include eval subset size in results
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
        
        # Save results
        results_file_path = os.path.join(full_results_dir, f"results_{timestamp}_gpu{gpu_id}.jsonl")
        with open(results_file_path, "a") as f:
            f.write(json.dumps(result) + "\n")
            f.flush()
        
        # Save final model if requested
        if base_config.get('save_model', False):
            final_model_dir = os.path.join(full_results_dir, "final_models")
            os.makedirs(final_model_dir, exist_ok=True)
            final_model_path = os.path.join(final_model_dir, f"final_model_{unique_id}.pt")
            save_model(model, final_model_path)
        
        # Mark as completed
        with open(checkpoint_log_path, "a") as cp_f:
            cp_f.write(unique_id + "\n")
        completed_configs.add(unique_id)
        completed_count += 1
    
    return completed_count

def main():
    try:
        start_time = time.time()
        print(f"Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Using evaluation subset size: {EVAL_SUBSET_SIZE}")
        
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