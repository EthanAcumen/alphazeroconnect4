import torch

# Check if CUDA is available
print("CUDA Available:", torch.cuda.is_available())

# Get number of available GPUs
print("GPU Count:", torch.cuda.device_count())

# Get the name of the first GPU
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("CUDA Version:", torch.version.cuda)
    print("cuDNN Version:", torch.backends.cudnn.version())


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Create a tensor
tensor = torch.tensor([1.0, 2.0, 3.0], device=device)

# Perform an operation
tensor = tensor * 2
print("Tensor on GPU:", tensor)

import time

# Matrix size
size = 10000

# CPU computation
cpu_tensor1 = torch.randn(size, size)
cpu_tensor2 = torch.randn(size, size)

start = time.time()
cpu_result = torch.matmul(cpu_tensor1, cpu_tensor2)
end = time.time()
print(f"CPU Time: {end - start:.4f} seconds")

# GPU computation
device = torch.device("cuda")
gpu_tensor1 = torch.randn(size, size, device=device)
gpu_tensor2 = torch.randn(size, size, device=device)

torch.cuda.synchronize()  # Ensure previous ops are done
start = time.time()
gpu_result = torch.matmul(gpu_tensor1, gpu_tensor2)
torch.cuda.synchronize()  # Ensure computation finishes before timing
end = time.time()
print(f"GPU Time: {end - start:.4f} seconds")
