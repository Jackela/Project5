import torch
import sys

print(f"--- PyTorch and System Information ---")
print(f"PyTorch Version: {torch.version}")
print(f"Python Version: {sys.version}")
print("-" * 35)

is_cuda_available = torch.cuda.is_available()
print(f"Is CUDA available? -> {is_cuda_available}")
print("-" * 35)

if is_cuda_available:
    print(f"CUDA Version (built-in with PyTorch): {torch.version.cuda}")
    device_count = torch.cuda.device_count()
    print(f"Number of available GPUs: {device_count}")
    if device_count > 0:
        current_device_id = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device_id)
        print(f"Current GPU ID: {current_device_id}")
        print(f"Current GPU Name: {gpu_name}")
    else:
        print("No GPUs found even though CUDA is available.")
else:
    print("CUDA is not available. PyTorch is running in CPU-only mode.")
print("-" * 35)