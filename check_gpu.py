import torch

# 檢查PyTorch是否能使用GPU
def check_gpu_usage():
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Current GPU Memory Allocated: {torch.cuda.memory_allocated(0) / (1024**3):.2f} GB")
        print(f"Current GPU Memory Cached: {torch.cuda.memory_reserved(0) / (1024**3):.2f} GB")
    else:
        print("CUDA is not available. GPU is not being used.")

if __name__ == "__main__":
    check_gpu_usage()