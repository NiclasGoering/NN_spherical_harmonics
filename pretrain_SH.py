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

def normalize_input(X):
    """Normalize input to lie on the sphere."""
    norms = torch.norm(X, dim=1, keepdim=True)
    return X / norms

def gegenbauer_polynomial_torch(n, alpha, x):
    """
    Memory-efficient implementation of Gegenbauer polynomial for GPU.
    """
    if n == 0:
        return torch.ones_like(x)
    elif n == 1:
        return 2.0 * alpha * x
    else:
        # Use recurrence relation with lower memory footprint
        C_n_minus_2 = torch.ones_like(x)  # C_0^alpha(x)
        C_n_minus_1 = 2.0 * alpha * x     # C_1^alpha(x)
        
        for i in range(2, n + 1):
            # Replace C_n_minus_2 in-place to save memory
            temp = C_n_minus_2.clone()
            C_n_minus_2 = C_n_minus_1
            C_n_minus_1 = ((2.0 * (i + alpha - 1.0) * x * C_n_minus_2) - 
                          ((i + 2.0 * alpha - 2.0) * temp)) / float(i)
            del temp  # Explicitly delete to free memory
            
            # Reduced frequency of memory clearing
            if i % 5 == 0:
                torch.cuda.empty_cache()
        
        return C_n_minus_1

def count_harmonics_in_dimension(dimension, degree):
    """
    Count the number of linearly independent hyperspherical harmonics of degree k in dimension d.
    """
    from math import comb
    
    if degree == 0:
        return 1
    
    # Formula for multiplicity of harmonics
    return comb(degree + dimension - 1, degree) - comb(degree + dimension - 3, degree - 2)

def factorial(n):
    """Iterative factorial function that works for large n"""
    if n <= 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def gamma_torch(x):
    """
    PyTorch approximation of gamma function
    """
    # Use cases for integers, half-integers, and special values
    if isinstance(x, (int, float)):
        if x == 1:
            return 1.0
        elif x == 0.5:
            return np.sqrt(np.pi)
        # For integers, use factorial
        elif x > 0 and x == int(x):
            return factorial(int(x) - 1)
        # For half-integers
        elif x > 0 and x - 0.5 == int(x - 0.5):
            n = int(x - 0.5)
            return factorial_product(2*n - 1) * np.sqrt(np.pi) / (2**(2*n - 1))
    
    # Default fallback to scipy's gamma
    from scipy import special
    return float(special.gamma(x))

def factorial_product(n):
    """Helper for factorial calculation via products"""
    if n <= 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

class HypersphericalHarmonics:
    """
    Implementation of hyperspherical harmonics on a single GPU.
    Optimized for MPI where each process uses one GPU.
    """
    def __init__(self, k_ranges, belongs_to_rkhs, p, device):
        # Store basic parameters
        self.p = p
        self.alpha = (p - 2) / 2
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
        
        # Generate all (k,j) pairs
        self.k_j_pairs = []
        for degree in self.degrees_to_include:
            n_harmonics = count_harmonics_in_dimension(p, degree)
            for j in range(n_harmonics):
                self.k_j_pairs.append((degree, j))
        
        # Count total harmonics
        self.total_harmonics = len(self.k_j_pairs)
        
        # Equal weight for each harmonic
        self.coefficient = 1.0 / self.total_harmonics
        
        print(f"Created HypersphericalHarmonics with {self.total_harmonics} harmonics on {device}")
        
        # Initialize the SingleGPUHarmonics
        self.model = SingleGPUHarmonics(self.k_j_pairs, p, device, self.coefficient)
    
    def __call__(self, x):
        """
        Compute the sum of all spherical harmonics.
        
        Args:
            x: Input tensor of shape [batch_size, p]
            
        Returns:
            Sum of all harmonics with equal weights
        """
        # Make sure input is on the right device
        if x.device != self.device:
            x = x.to(self.device)
            
        # Process using the single GPU model
        result = self.model(x)
        return result


class SingleGPUHarmonics:
    """
    Handles computing a subset of spherical harmonics on a single GPU.
    Optimized for H100 GPUs with large memory.
    """
    def __init__(self, k_j_pairs, p, device, coefficient):
        self.device = device
        self.p = p
        self.alpha = (p - 2) / 2
        self.k_j_pairs = k_j_pairs
        self.coefficient = coefficient
        
        # Process large chunks for H100s with 95GB memory
        self.harmonics_chunk_size = 20000  # Increased for H100s
        
        # Extract degrees and harmonic indices
        self.degrees = [k for k, _ in self.k_j_pairs]
        self.j_indices = [j for _, j in self.k_j_pairs]
        
        # Precompute projection vectors for each harmonic
        self.projection_vectors = []
        
        for degree, j in self.k_j_pairs:
            if degree == 0:
                # For degree 0, use a dummy vector
                self.projection_vectors.append(torch.zeros(p, device=device))
                continue
                
            # Generate a deterministic vector for this (degree, j) pair
            np.random.seed(degree * 10000 + j)
                
            if j == 0:
                # For j=0, use the canonical direction (1,0,0,...)
                v = np.zeros(p)
                v[0] = 1.0
            elif j < p:
                # Use standard basis vectors for smaller indices
                v = np.zeros(p)
                v[j % p] = 1.0
            else:
                # For larger indices, create diverse projections
                v = np.sin(np.arange(p) * (j % 10 + 1)) + np.cos(np.arange(p) * (j % 7 + 1))
                v = v / np.linalg.norm(v)
                
            # Convert to tensor and store
            self.projection_vectors.append(torch.tensor(v, device=device, dtype=torch.float32))
        
        # Compute normalization factors
        self.normalization_factors = []
        
        for i, (degree, j) in enumerate(self.k_j_pairs):
            if degree == 0:
                # Simple normalization for constant harmonics
                norm = 1.0 / np.sqrt(gamma_torch(p/2) * 2 * np.pi**(p/2))
            elif j == 0:
                # Exact normalization for zonal harmonics (handle carefully for high dimensions)
                try:
                    norm = np.sqrt(
                        (2*degree + p - 2) * gamma_torch(degree + p - 2) / 
                        (2 * np.pi**(p/2) * factorial(degree) * gamma_torch(p/2) * gamma_torch(self.alpha+1))
                    )
                except (OverflowError, ValueError):
                    # Fallback for large numbers
                    print(f"Warning: Using approximate normalization for degree {degree}, j={j}")
                    norm = 1.0 / np.sqrt(count_harmonics_in_dimension(p, degree))
            else:
                # Approximate normalization for non-zonal
                norm = 1.0 / np.sqrt(count_harmonics_in_dimension(p, degree))
                
            self.normalization_factors.append(norm)
        
        # Convert normalization factors to tensor once for efficiency
        self.norm_tensor = torch.tensor(self.normalization_factors, device=device, dtype=torch.float32)
    
    def __call__(self, x):
        """Compute sum of all harmonics for this GPU's subset, optimized for H100 GPUs"""
        # Normalize input to lie on the sphere
        x_norm = normalize_input(x)
        
        # Initialize sum
        result = torch.zeros(x_norm.shape[0], device=self.device)
        
        # Process in large chunks for H100 GPUs
        num_harmonics = len(self.k_j_pairs)
        num_chunks = (num_harmonics + self.harmonics_chunk_size - 1) // self.harmonics_chunk_size
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * self.harmonics_chunk_size
            end_idx = min((chunk_idx + 1) * self.harmonics_chunk_size, num_harmonics)
            
            # Get subset of harmonics for this chunk
            chunk_k_j_pairs = self.k_j_pairs[start_idx:end_idx]
            chunk_projections = self.projection_vectors[start_idx:end_idx]
            chunk_norms = self.norm_tensor[start_idx:end_idx]
            
            # Stack projection vectors
            projections_matrix = torch.stack(chunk_projections)
            
            # Compute all projections at once
            all_projections = torch.matmul(x_norm, projections_matrix.T)
            
            # Group by degree for vectorized processing
            degrees_in_chunk = {}
            for i, (degree, _) in enumerate(chunk_k_j_pairs):
                if degree not in degrees_in_chunk:
                    degrees_in_chunk[degree] = []
                degrees_in_chunk[degree].append((i, degree))
            
            # Process all harmonics of the same degree together for better parallelism
            for degree, indices_list in degrees_in_chunk.items():
                if degree == 0:
                    # Handle constant harmonics (degree 0) together
                    for idx_in_chunk, _ in indices_list:
                        idx_in_norm_tensor = idx_in_chunk
                        harmonic_value = torch.ones(x_norm.shape[0], device=self.device) * chunk_norms[idx_in_norm_tensor]
                        result += harmonic_value * self.coefficient
                else:
                    # Process each degree group
                    for idx_in_chunk, _ in indices_list:
                        # Apply Gegenbauer polynomial
                        poly_value = gegenbauer_polynomial_torch(degree, self.alpha, all_projections[:, idx_in_chunk])
                        harmonic_value = poly_value * chunk_norms[idx_in_chunk]
                        
                        # Add to result with equal weight
                        result += harmonic_value * self.coefficient
            
            # Less aggressive memory cleanup
            del all_projections, projections_matrix
            
            # Only clear cache at the end of each chunk for H100s
            torch.cuda.empty_cache()
        
        return result

def create_target_function(k_ranges, belongs_to_rkhs, p, device):
    """
    Create a target function that computes hyperspherical harmonics.
    
    Args:
        k_ranges: List of tuples (k_min, k_max) specifying k ranges
        belongs_to_rkhs: Whether to exclude odd k > 1
        p: Input dimension
        device: GPU device to use
    """
    # Create the harmonic computer for a single GPU
    function_computer = HypersphericalHarmonics(k_ranges, belongs_to_rkhs, p, device)
    
    print(f"Creating target function with {function_computer.total_harmonics} hyperspherical harmonics")
    print(f"k ranges: {k_ranges}, belongs_to_rkhs: {belongs_to_rkhs}")
    print(f"Degrees included: {function_computer.degrees_to_include}")
    print(f"Coefficient for each harmonic: {function_computer.coefficient}")
    
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
    # Define distributions to explore
    distributions = ['sphere_uniform']
    
    # Dimensions to explore
    dimensions = [20,2, 4, 8, 10, 15, 25, 30]
    
    # Increase training samples for H100s (adjust as needed)
    train_size = 2000000  # 5M samples
    
    # Spherical harmonics parameters
    k_ranges_list = [
        [(2, 5)],       # Only k values 0-5
        [(5, 10)],
        [(10, 15)],
             # Only k values 20-25
    ]
    
    belongs_to_rkhs_values = [True]
    
    # Spiked normal hyperparameter values
    r_values = [0.8]  # Exponent for d^r in spiked normal
    
    # Number of experiments
    num_experiments = 1
    
    # Base directory for saving data
    data_base_dir = "/scratch/goring/SH_2303"
    
    # Calculate total number of combinations
    total_combinations = []
    
    for dist in distributions:
        for dim in dimensions:
            for k_ranges in k_ranges_list:
                # Check if the maximum degree is reasonable for this dimension
                max_degree = max([k_max for k_min, k_max in k_ranges])
                # Increase max degree limit for H100s
                if max_degree > 30:  # Increased from 20 to 30
                    continue
                    
                for belongs_to_rkhs in belongs_to_rkhs_values:
                    for exp_num in range(1, num_experiments + 1):
                        if dist == 'spiked_normal':
                            for r_val in r_values:
                                total_combinations.append((dist, dim, k_ranges, belongs_to_rkhs, r_val, exp_num))
                        else:
                            total_combinations.append((dist, dim, k_ranges, belongs_to_rkhs, None, exp_num))
    
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
    for idx, (dist_type, d, k_ranges, belongs_to_rkhs, r_value, exp_num) in enumerate(my_combinations):
        # Create compact abbreviations for naming
        dist_abbr = {'normal': 'N', 'uniform': 'U', 'sphere_uniform': 'SU', 'spiked_normal': 'SN'}[dist_type]
        rkhs_abbr = 'RKHS' if belongs_to_rkhs else 'NoRKHS'
        k_ranges_str = format_k_ranges(k_ranges)
        
        # Create a compact name for this combination
        if dist_type == 'spiked_normal':
            run_name = f"{dist_abbr}_r{r_value}_d{d}_k{k_ranges_str}_{rkhs_abbr}"
        else:
            run_name = f"{dist_abbr}_d{d}_k{k_ranges_str}_{rkhs_abbr}"
            
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
            target_function = create_target_function(k_ranges, belongs_to_rkhs, d, device)
            
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
                smart_name = f"{dist_abbr}_r{r_value}_d{d}_k{k_ranges_str}_{rkhs_abbr}_{exp_num}"
            else:
                smart_name = f"{dist_abbr}_d{d}_k{k_ranges_str}_{rkhs_abbr}_{exp_num}"
            
            data_subdir = f"SH_{smart_name}_{timestamp}"
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
                        'train_size': train_size,
                        'r_value': r_value if dist_type == 'spiked_normal' else None,
                        'experiment_number': exp_num,
                        'random_seed': seed,
                        'mpi_rank': rank,
                        'mpi_size': size,
                        'gpu_device': assigned_gpu
                    },
                    'target_function': {
                        'total_harmonics': target_function.total_harmonics,
                        'coefficient': target_function.coefficient,
                        'k_j_pairs': [(int(k), int(j)) for k, j in target_function.k_j_pairs],
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