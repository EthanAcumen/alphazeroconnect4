import torch
import time

# Define devices
device_cpu = torch.device("cpu")
device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device_gpu == device_cpu:
    print("CUDA not available. Running on CPU only.")
else:
    size = 1000  # Adjust matrix size as needed
    iterations = 100

    # Create random matrices
    a_cpu = torch.randn(size, size, device=device_cpu)
    b_cpu = torch.randn(size, size, device=device_cpu)
    a_gpu = torch.randn(size, size, device=device_gpu)
    b_gpu = torch.randn(size, size, device=device_gpu)

    # CPU timing
    start_cpu = time.time()
    for _ in range(iterations):
        c_cpu = torch.mm(a_cpu, b_cpu)
    end_cpu = time.time()
    cpu_time = end_cpu - start_cpu

    # GPU timing: synchronize before and after timing
    torch.cuda.synchronize()  # Ensure GPU is ready
    start_gpu = time.time()
    for _ in range(iterations):
        c_gpu = torch.mm(a_gpu, b_gpu)
    torch.cuda.synchronize()  # Wait for GPU computations to finish
    end_gpu = time.time()
    gpu_time = end_gpu - start_gpu

    print(f"CPU time: {cpu_time:.4f} seconds")
    print(f"GPU time: {gpu_time:.4f} seconds")
