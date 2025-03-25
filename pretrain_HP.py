import numpy as np
import torch
import os
from datetime import datetime
import matplotlib.pyplot as plt
from mpi4py import MPI
import gzip
import io
import json
import time
import math
from itertools import combinations_with_replacement

def save_dataset_compressed(X, y, filepath, rank, max_retries=3):
    """
    Save full dataset with compression, with verification and retries.
    """
    # Create directory if it doesn't exist
    directory = os.path.dirname(filepath)
    try:
        os.makedirs(directory, exist_ok=True)
        print(f"Process {rank}: Directory verified: {directory}")
    except Exception as e:
        print(f"Process {rank}: ERROR creating directory {directory}: {e}")
        return False
    
    # Try multiple times to save
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Process {rank}: Compressing dataset (attempt {attempt}/{max_retries})...")
            
            # Convert to float32 to save space
            if X is not None:
                X_f32 = X.detach().cpu().to(torch.float32)
            else:
                X_f32 = None
                
            if y is not None:
                y_f32 = y.detach().cpu().to(torch.float32)
            else:
                y_f32 = None
            
            # Create a buffer to compress the data
            buffer = io.BytesIO()
            torch.save({'X': X_f32, 'y': y_f32}, buffer)
            compressed_data = gzip.compress(buffer.getvalue(), compresslevel=9)
            
            # Save compressed data
            with open(filepath, 'wb') as f:
                f.write(compressed_data)
                f.flush()  # Ensure data is written to disk
                os.fsync(f.fileno())  # Force OS to flush file buffers
            
            # Verify the file exists with non-zero size
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                file_size_mb = file_size / (1024 * 1024)
                
                if file_size > 0:
                    print(f"Process {rank}: VERIFIED: Saved compressed dataset ({file_size_mb:.2f} MB) to {filepath}")
                    return True
                else:
                    print(f"Process {rank}: WARNING: File has zero size, retrying...")
            else:
                print(f"Process {rank}: WARNING: File was not created, retrying...")
                
        except Exception as e:
            print(f"Process {rank}: ERROR during save attempt {attempt}: {e}")
            
        # Only retry if not the last attempt
        if attempt < max_retries:
            print(f"Process {rank}: Waiting before retry...")
            time.sleep(2)  # Wait before retrying
        else:
            print(f"Process {rank}: FAILED: Could not save dataset after {max_retries} attempts")
            return False
    
    return False

def save_results(results, save_dir, name, max_retries=3):
    """Save results with verification and retries."""
    # Create directory if it doesn't exist
    try:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Directory verified: {save_dir}")
    except Exception as e:
        print(f"ERROR creating directory {save_dir}: {e}")
        return False
    
    # Create filename
    filepath = os.path.join(save_dir, f"results_{name}.json")
    
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Saving results (attempt {attempt}/{max_retries})...")
            
            # Use temporary file approach for safety
            temp_filepath = filepath + ".tmp"
            
            # Convert to JSON format with pretty print
            with open(temp_filepath, 'w') as f:
                json.dump(results, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            
            # Verify temp file exists with content
            if os.path.exists(temp_filepath) and os.path.getsize(temp_filepath) > 0:
                # Verify JSON is valid by reading it back
                with open(temp_filepath, 'r') as f:
                    # Just try to load it to verify integrity
                    json.load(f)
                
                # If we get here, JSON is valid, so rename to final file
                if os.path.exists(filepath):
                    os.remove(filepath)
                os.rename(temp_filepath, filepath)
                
                print(f"VERIFIED: Saved results to {filepath}")
                return True
            else:
                print(f"WARNING: Results file was not created properly")
                
        except Exception as e:
            print(f"ERROR during results save attempt {attempt}: {e}")
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
            
        # Only retry if not the last attempt
        if attempt < max_retries:
            print(f"Waiting before retry...")
            time.sleep(2)
        else:
            print(f"FAILED: Could not save results after {max_retries} attempts")
            return False
    
    return False

def generate_data(distribution_type, train_size, d, device, r=0.5):
    """
    Generate data from specified distribution using float32 instead of float64.
    """
    if distribution_type == 'normal':
        # Standard normal distribution with float32
        X = torch.randn(train_size, d, device=device, dtype=torch.float32)
        
    elif distribution_type == 'uniform':
        # Uniform distribution in [-1, 1] with float32
        X = 2 * torch.rand(train_size, d, device=device, dtype=torch.float32) - 1
    
    elif distribution_type == 'sphere_uniform':
        # Uniform distribution on the unit (d-1)-sphere embedded in dimension d
        # First sample from normal distribution
        X = torch.randn(train_size, d, device=device, dtype=torch.float32)
        # Normalize to unit length to get uniform distribution on sphere
        X = X / torch.norm(X, dim=1, keepdim=True)
        
    elif distribution_type == 'spiked_normal':
        # Spiked normal with float32
        theta = torch.randn(d, device=device, dtype=torch.float32)
        theta = theta / torch.norm(theta)
        X = torch.randn(train_size, d, device=device, dtype=torch.float32)
        spike_scale = d**r
        Z = torch.randn(train_size, 1, device=device, dtype=torch.float32) * torch.sqrt(torch.tensor(spike_scale, dtype=torch.float32))
        X = X + Z * theta
    else:
        raise ValueError(f"Unknown distribution type: {distribution_type}")
    
    return X

def count_hermite_polynomials(dim, degree):
    """
    Count number of multivariate Hermite polynomials of degree k in dimension d.
    This is equivalent to the number of ways to place degree indistinguishable 
    objects into dim distinguishable bins (stars and bars problem).
    """
    from math import comb
    # Formula: (n+d-1) choose (d-1)
    return comb(degree + dim - 1, dim - 1)

def sample_random_multiindex(dim, degree):
    """Sample a single random multiindex with the given total degree."""
    # Initialize with zeros
    multiindex = [0] * dim
    
    # Randomly distribute degree among dimensions
    for _ in range(degree):
        bin_idx = np.random.randint(0, dim)
        multiindex[bin_idx] += 1
    
    return tuple(multiindex)

def generate_multiindices_efficient(dim, degree, num_indices=None):
    """
    Efficiently generate random multiindices without creating all combinations first.
    
    Args:
        dim (int): Dimension
        degree (int): Degree of polynomial
        num_indices (int, optional): Number of indices to randomly select
        
    Returns:
        list: List of multiindices as tuples
    """
    # Calculate total number of possible multiindices
    total_possible = count_hermite_polynomials(dim, degree)
    
    # If num_indices not specified or greater than total possible, set to total
    num_to_generate = min(num_indices or total_possible, total_possible)
    
    print(f"Generating {num_to_generate} multiindices for dimension {dim}, degree {degree} (from {total_possible} possible)")
    
    # If total possible is small enough, generate all of them using original method
    if total_possible < 10000:
        # Original method for small cases
        all_indices = list(combinations_with_replacement(range(dim), degree))
        all_multiindices = []
        for idx in all_indices:
            multiindex = [0] * dim
            for i in idx:
                multiindex[i] += 1
            all_multiindices.append(tuple(multiindex))
        
        # Remove duplicates
        all_multiindices = list(set(all_multiindices))
        
        # Random sample if needed
        if num_indices is not None and num_indices < len(all_multiindices):
            indices = np.random.choice(len(all_multiindices), num_indices, replace=False)
            return [all_multiindices[i] for i in indices]
        return all_multiindices
    
    # For large spaces, directly sample
    multiindices = set()
    max_attempts = min(num_to_generate * 10, 1000000)  # Reasonable limit on attempts
    attempts = 0
    
    while len(multiindices) < num_to_generate and attempts < max_attempts:
        # Generate a random multiindex of the given degree
        multiindex = sample_random_multiindex(dim, degree)
        multiindices.add(multiindex)
        attempts += 1
        
        # Progress reporting for large cases
        if attempts % 10000 == 0:
            print(f"  Progress: {len(multiindices)}/{num_to_generate} multiindices generated after {attempts} attempts")
    
    if len(multiindices) < num_to_generate:
        print(f"Warning: Could only generate {len(multiindices)} unique multiindices after {attempts} attempts")
    
    return list(multiindices)

def hermite_polynomial_torch(n, x):
    """
    Memory-efficient implementation of probabilist's Hermite polynomial for GPU.
    For H_n(x) where H_0(x) = 1, H_1(x) = x, H_2(x) = x²-1, etc.
    """
    if n == 0:
        return torch.ones_like(x)
    elif n == 1:
        return x
    else:
        # Use recurrence relation with lower memory footprint
        h_n_minus_2 = torch.ones_like(x)  # H_0(x)
        h_n_minus_1 = x                   # H_1(x)
        
        for i in range(2, n + 1):
            # Replace h_n_minus_2 in-place to save memory
            temp = h_n_minus_2.clone()
            h_n_minus_2 = h_n_minus_1
            h_n_minus_1 = x * h_n_minus_2 - (i - 1) * temp
            del temp  # Explicitly delete to free memory
            
            # Reduced frequency of memory clearing
            if i % 5 == 0:
                torch.cuda.empty_cache()
        
        return h_n_minus_1

class HermitePolynomials:
    """
    Implementation of Hermite polynomials on a single GPU.
    Optimized for MPI where each process uses one GPU.
    """
    def __init__(self, k_ranges, belongs_to_rkhs, d, device, num_k=None):
        # Store basic parameters
        self.d = d
        self.k_ranges = k_ranges
        self.belongs_to_rkhs = belongs_to_rkhs
        self.device = device
        
        # Find max degree needed
        self.max_degree = max([k_max for k_min, k_max in k_ranges])
        
        # Collect degrees to include
        self.degrees_to_include = []
        for k_min, k_max in k_ranges:
            for degree in range(k_min, k_max + 1):
                # Skip odd degrees > 1 if belongs_to_rkhs is True
                if belongs_to_rkhs and degree > 1 and degree % 2 == 1:
                    continue
                if degree not in self.degrees_to_include:
                    self.degrees_to_include.append(degree)
        
        # Generate multiindex-degree pairs
        self.multiindex_pairs = []
        
        for degree in self.degrees_to_include:
            # Count total available polynomials for this degree
            total_polynomials = count_hermite_polynomials(d, degree)
            
            # Determine how many to use
            num_to_use = min(num_k if num_k is not None else total_polynomials, total_polynomials)
            
            # Generate random multiindices efficiently
            multiindices = generate_multiindices_efficient(d, degree, num_to_use)
            
            # Add to our list with the degree
            for m_idx in multiindices:
                self.multiindex_pairs.append((degree, m_idx))
        
        # Count total polynomials
        self.total_polynomials = len(self.multiindex_pairs)
        
        # Equal weight for each polynomial
        self.coefficient = 1.0 / self.total_polynomials
        
        print(f"Created HermitePolynomials with {self.total_polynomials} polynomials on {device}")
        
        # Initialize the SingleGPUHermite
        self.model = SingleGPUHermite(self.multiindex_pairs, d, device, self.coefficient)
    
    def __call__(self, x):
        """
        Compute the sum of all Hermite polynomials.
        
        Args:
            x: Input tensor of shape [batch_size, d]
            
        Returns:
            Sum of all polynomials with equal weights
        """
        # Make sure input is on the right device
        if x.device != self.device:
            x = x.to(self.device)
            
        # Process using the single GPU model
        result = self.model(x)
        return result


class SingleGPUHermite:
    """
    Handles computing a subset of Hermite polynomials on a single GPU.
    Optimized for H100 GPUs with large memory.
    """
    def __init__(self, multiindex_pairs, d, device, coefficient):
        self.device = device
        self.d = d
        self.multiindex_pairs = multiindex_pairs
        self.coefficient = coefficient
        
        # Process large chunks for H100s with 95GB memory
        self.polynomials_chunk_size = 20000  # Increased for H100s
        
        # Extract degrees and multiindices
        self.degrees = [k for k, _ in self.multiindex_pairs]
        self.multiindices = [m_idx for _, m_idx in self.multiindex_pairs]
        
        # Precompute normalization factors - for Hermite polynomials under Gaussian measure
        self.normalization_factors = []
        
        for degree, m_idx in self.multiindex_pairs:
            # For multivariate Hermite polynomials, normalization is product of univariate norms
            norm = 1.0
            for power in m_idx:
                if power > 0:
                    # Normalization for univariate Hermite polynomial H_n(x) is sqrt(n!)
                    norm *= math.sqrt(math.factorial(power))
            
            self.normalization_factors.append(1.0 / norm)
        
        # Convert normalization factors to tensor once for efficiency
        self.norm_tensor = torch.tensor(self.normalization_factors, device=device, dtype=torch.float32)
    
    def __call__(self, x):
        """Compute sum of all Hermite polynomials for this GPU's subset, optimized for H100 GPUs"""
        # Initialize sum
        result = torch.zeros(x.shape[0], device=self.device)
        
        # Process in large chunks for H100 GPUs
        num_polynomials = len(self.multiindex_pairs)
        num_chunks = (num_polynomials + self.polynomials_chunk_size - 1) // self.polynomials_chunk_size
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * self.polynomials_chunk_size
            end_idx = min((chunk_idx + 1) * self.polynomials_chunk_size, num_polynomials)
            
            # Get subset of polynomials for this chunk
            chunk_multiindex_pairs = self.multiindex_pairs[start_idx:end_idx]
            chunk_norms = self.norm_tensor[start_idx:end_idx]
            
            # Process each multiindex Hermite polynomial
            for i, (degree, multiindex) in enumerate(chunk_multiindex_pairs):
                # Initialize with ones
                poly_value = torch.ones(x.shape[0], device=self.device)
                
                # Compute the multivariate Hermite polynomial as product of univariate ones
                for dim_idx, power in enumerate(multiindex):
                    if power > 0:  # Skip if power is 0
                        # Extract this dimension's values
                        x_dim = x[:, dim_idx]
                        
                        # Compute univariate Hermite polynomial
                        h_value = hermite_polynomial_torch(power, x_dim)
                        
                        # Multiply to get multivariate polynomial
                        poly_value = poly_value * h_value
                
                # Apply normalization
                poly_value = poly_value * chunk_norms[i]
                
                # Add to result with equal weight
                result += poly_value * self.coefficient
            
            # Only clear cache at the end of each chunk for H100s
            torch.cuda.empty_cache()
        
        return result

def create_target_function(k_ranges, belongs_to_rkhs, d, device, num_k=None):
    """
    Create a target function that computes Hermite polynomials.
    
    Args:
        k_ranges: List of tuples (k_min, k_max) specifying k ranges
        belongs_to_rkhs: Whether to exclude odd k > 1
        d: Input dimension
        device: GPU device to use
        num_k: Number of random polynomials to select per degree
    """
    # Create the polynomial computer for a single GPU
    function_computer = HermitePolynomials(k_ranges, belongs_to_rkhs, d, device, num_k)
    
    print(f"Creating target function with {function_computer.total_polynomials} Hermite polynomials")
    print(f"k ranges: {k_ranges}, belongs_to_rkhs: {belongs_to_rkhs}")
    print(f"Degrees included: {function_computer.degrees_to_include}")
    print(f"Coefficient for each polynomial: {function_computer.coefficient}")
    
    # Store information about the target function
    function_computer.k_ranges = k_ranges
    
    return function_computer

def format_k_ranges(k_ranges):
    """Format k_ranges for filename or display"""
    return '_'.join([f"{k_min}-{k_max}" for k_min, k_max in k_ranges])

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Get total available GPUs
    num_gpus = torch.cuda.device_count()
    
    # CRITICAL CHANGE: Assign specific GPU to this MPI process
    # Use modulo to handle cases where num_processes > num_gpus
    assigned_gpu = rank % num_gpus
    
    # Set default device for this process to the assigned GPU
    torch.cuda.set_device(assigned_gpu)
    device = torch.device(f'cuda:{assigned_gpu}')
    
    if rank == 0:
        print(f"Number of available GPUs: {num_gpus}")
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} with {props.total_memory / 1e9:.1f} GB memory")
    
    print(f"Process {rank}/{size} assigned to GPU {assigned_gpu}, device = {device}")
    comm.Barrier()  # Synchronize for clean output
    
    # --- Hyperparameters ---
    # Define distributions to explore - normal is most appropriate for Hermite
    distributions = ['normal']
    
    # Dimensions to explore
    dimensions = [2,4,8,16,32,64]
    
    # Increase training samples for H100s (adjust as needed)
    train_size = 5000000  # 5M samples
    
    # Hermite polynomials parameters
    k_ranges_list = [
        [(2, 5)],  # Degrees 2-5
        [(5, 10)], # Degrees 5-10
        [(10, 15)],
        [(15, 20)],
        [(20, 25)],
        [(25, 30)], # Degrees 25-30
    ]
    
    # Number of random polynomials to select per degree
    num_k_values = [100]  # Take up to 100 random polynomials per degree
    
    belongs_to_rkhs_values = [True]
    
    # Spiked normal hyperparameter values
    r_values = [0.8]  # Exponent for d^r in spiked normal
    
    # Number of experiments
    num_experiments = 1
    
    # Base directory for saving data
    data_base_dir = "/scratch/goring/HP_2403"
    
    # Calculate total number of combinations
    total_combinations = []
    
    for dist in distributions:
        for dim in dimensions:
            for k_ranges in k_ranges_list:
                # Check if the maximum degree is reasonable for this dimension
                max_degree = max([k_max for k_min, k_max in k_ranges])
                # Increase max degree limit for H100s
                if max_degree > 100:  # Increased from 20 to 30
                    continue
                    
                for belongs_to_rkhs in belongs_to_rkhs_values:
                    for num_k in num_k_values:
                        for exp_num in range(1, num_experiments + 1):
                            if dist == 'spiked_normal':
                                for r_val in r_values:
                                    total_combinations.append((dist, dim, k_ranges, belongs_to_rkhs, num_k, r_val, exp_num))
                            else:
                                total_combinations.append((dist, dim, k_ranges, belongs_to_rkhs, num_k, None, exp_num))
    
    # Distribute across MPI processes
    num_combinations = len(total_combinations)
    combinations_per_process = (num_combinations + size - 1) // size
    start_idx = rank * combinations_per_process
    end_idx = min((rank + 1) * combinations_per_process, num_combinations)
    
    # Process only the combinations assigned to this rank
    my_combinations = total_combinations[start_idx:end_idx]
    
    if rank == 0:
        print(f"\nTotal combinations: {num_combinations}")
        print(f"Number of MPI processes: {size}")
        print(f"Data base directory: {data_base_dir}")
    
    # Display this process's assignment
    print(f"Process {rank}: Processing {len(my_combinations)} combinations from {start_idx} to {end_idx-1}")
    comm.Barrier()  # Synchronize for clean output
    
    # Process each combination assigned to this rank
    for idx, (dist_type, d, k_ranges, belongs_to_rkhs, num_k, r_value, exp_num) in enumerate(my_combinations):
        # Create compact abbreviations for naming
        dist_abbr = {'normal': 'N', 'uniform': 'U', 'sphere_uniform': 'SU', 'spiked_normal': 'SN'}[dist_type]
        rkhs_abbr = 'RKHS' if belongs_to_rkhs else 'NoRKHS'
        k_ranges_str = format_k_ranges(k_ranges)
        
        # Create a compact name for this combination
        if dist_type == 'spiked_normal':
            run_name = f"HP_{dist_abbr}_r{r_value}_d{d}_k{k_ranges_str}_{rkhs_abbr}_numk{num_k}"
        else:
            run_name = f"HP_{dist_abbr}_d{d}_k{k_ranges_str}_{rkhs_abbr}_numk{num_k}"
            
        print(f"\nProcess {rank} starting combination {idx+1}/{len(my_combinations)}: {run_name} (Exp {exp_num})")
        
        # Set seed based on experiment number and rank for uniqueness
        seed = 42 + exp_num + rank * 100
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        try:
            # Force initialization of this GPU
            _ = torch.randn(1000, 1000, device=device)
            torch.cuda.synchronize(device)
            
            # Generate data on this process's assigned GPU
            print(f"Process {rank} generating data on GPU {assigned_gpu}")
            X = generate_data(dist_type, train_size, d, device, 
                            r=r_value if r_value is not None else 0.5)
            
            # Create target function using only this process's GPU
            print(f"Process {rank} creating target function on GPU {assigned_gpu}")
            target_function = create_target_function(k_ranges, belongs_to_rkhs, d, device, num_k)
            
            # Process data - now much simpler with just one GPU per process
            print(f"Process {rank} computing target function with {X.shape[0]} samples")
            
            # Use larger batch size for H100
            batch_size = 300000  # Can be much larger with dedicated GPU
            num_batches = (X.shape[0] + batch_size - 1) // batch_size
            
            y_list = []
            for i in range(num_batches):
                start_idx_batch = i * batch_size
                end_idx_batch = min((i + 1) * batch_size, X.shape[0])
                
                print(f"Process {rank}: Processing batch {i+1}/{num_batches} ({start_idx_batch}-{end_idx_batch})")
                
                with torch.no_grad():
                    y_batch = target_function(X[start_idx_batch:end_idx_batch]).unsqueeze(1)
                    y_list.append(y_batch)
            
            # Concatenate results
            y = torch.cat(y_list, dim=0)
            del y_list  # Free memory
            
            # Rest of the saving logic...
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if dist_type == 'spiked_normal':
                smart_name = f"HP_{dist_abbr}_r{r_value}_d{d}_k{k_ranges_str}_{rkhs_abbr}_numk{num_k}_{exp_num}"
            else:
                smart_name = f"HP_{dist_abbr}_d{d}_k{k_ranges_str}_{rkhs_abbr}_numk{num_k}_{exp_num}"
            
            data_subdir = f"HP_{smart_name}_{timestamp}"
            data_save_dir = os.path.join(data_base_dir, data_subdir)
            
            try:
                os.makedirs(data_save_dir, exist_ok=True)
                
                # Save results
                results = {
                    'hyperparameters': {
                        'distribution_type': dist_type,
                        'input_dim': d,
                        'k_ranges': k_ranges,
                        'belongs_to_rkhs': belongs_to_rkhs,
                        'num_k': num_k,
                        'train_size': train_size,
                        'r_value': r_value if dist_type == 'spiked_normal' else None,
                        'experiment_number': exp_num,
                        'random_seed': seed,
                        'mpi_rank': rank,
                        'mpi_size': size,
                        'gpu_device': assigned_gpu
                    },
                    'target_function': {
                        'total_polynomials': target_function.total_polynomials,
                        'coefficient': target_function.coefficient,
                        'multiindex_pairs': [
                            {
                                'degree': int(degree),
                                'multiindex': list(map(int, m_idx))
                            }
                            for degree, m_idx in target_function.multiindex_pairs
                        ],
                        'degrees': target_function.degrees_to_include
                    }
                }
                
                save_results(results, data_save_dir, smart_name)
                
                # Save dataset - each process saves its own results
                x_path = os.path.join(data_save_dir, f"dataset_X_{smart_name}.pt.gz")
                y_path = os.path.join(data_save_dir, f"dataset_y_{smart_name}.pt.gz")
                
                save_dataset_compressed(X, None, x_path, rank)
                save_dataset_compressed(None, y, y_path, rank)
                
                print(f"Process {rank}: Successfully saved data for {run_name}")
                
            except Exception as e:
                print(f"Process {rank}: ERROR during save: {e}")
            
            # Clean up to save memory before next combination
            del X, y, target_function
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Process {rank}: ERROR processing {run_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Final synchronization
    comm.Barrier()
    if rank == 0:
        print("\nAll processes completed their assigned work.")

if __name__ == "__main__":
    main()