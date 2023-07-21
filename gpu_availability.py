import subprocess
import torch

def get_available_gpus():
    """
    Get a list of available GPUs.

    Returns:
        available_gpus (list): A list of GPU indices that are available for computation.
    """
    try:
        num_gpus = torch.cuda.device_count()
        available_gpus = []
        for gpu_id in range(num_gpus):
            if torch.cuda.is_available() and is_gpu_available(gpu_id):
                available_gpus.append(gpu_id)
        return available_gpus
    except Exception as e:
        print(f"Error checking available GPUs: {e}")
        return []

def is_gpu_available(gpu_id):
    """
    Check if a specific GPU is available for computation.

    Args:
        gpu_id (int): The GPU index to check.

    Returns:
        is_available (bool): True if the GPU is available for computation, False otherwise.
    """
    try:
        command = f"nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits --id={gpu_id}"
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        output, _ = process.communicate()
        is_available = len(output.strip()) == 0
        return is_available
    except Exception as e:
        print(f"Error checking GPU {gpu_id} availability: {e}")
        return False

# Example usage:
if __name__ == "__main__":
    available_gpus = get_available_gpus()
    print("Available GPUs:", available_gpus)
