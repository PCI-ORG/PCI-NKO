import sys
import torch

def get_device():
    """Determine which device to use (MPS, CUDA, or CPU)."""
    print(f"System platform: {sys.platform}")
    
    if sys.platform == 'darwin':  # Check if the operating system is macOS
        print("Checking for Apple MPS...")
        if torch.backends.mps.is_available():
            print("MPS is available.")
            return torch.device("mps")
        else:
            print("MPS is NOT available.")
    
    print("Checking for CUDA...")
    if torch.cuda.is_available():
        cuda_devices = os.getenv("CUDA_VISIBLE_DEVICES", "0")
        print(f"CUDA is available. Using GPU: {cuda_devices}")
        return torch.device("cuda")
    
    print("Neither MPS nor CUDA is available. Using CPU.")
    return torch.device("cpu")

if __name__ == "__main__":
    device = get_device()
    print(f"Selected device: {device}")