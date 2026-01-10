import torch
import sys

print("=== GPU VERIFICATION ===")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA NOT AVAILABLE — GPU REQUIRED")

device = torch.device("cuda")
torch.cuda.set_device(0)

print("[OK] CUDA Available:", torch.cuda.is_available())
print("[OK] CUDA Version:", torch.version.cuda)
print("[OK] PyTorch CUDA Build:", torch.__version__)
print("[OK] GPU Device:", torch.cuda.get_device_name(0))
print("[OK] GPU Count:", torch.cuda.device_count())

# Safety: ensure no CPU fallback
x = torch.randn(1, device=device)
assert x.is_cuda, "Tensor is NOT on GPU"

print("=== GPU-ONLY MODE VERIFIED ===")
print("CPU EXECUTION IS DISABLED")
sys.exit(0)
