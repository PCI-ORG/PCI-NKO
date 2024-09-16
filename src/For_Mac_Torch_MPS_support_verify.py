import torch
import sys

def verify_mps_support():
    if sys.platform != 'darwin':
        print("MPS is only supported on macOS.")
        return
    
    if torch.backends.mps.is_available():
        print("Apple MPS (Metal Performance Shaders) is available.")
    else:
        print("Apple MPS (Metal Performance Shaders) is NOT available.")

if __name__ == "__main__":
    verify_mps_support()