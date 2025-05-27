import subprocess
import argparse
from pathlib import Path

def get_dataset_path(dataset_name, dataset_type="nerf_synthetic"):
    """Get the full path to the dataset."""
    dataset_path = Path("data") / dataset_type / dataset_name
    
    if not dataset_path.exists():
        raise ValueError(f"Dataset '{dataset_name}' not found in {dataset_type}. "
                       f"Please ensure the dataset is placed in {dataset_path}")
    
    return dataset_path

def main():
    parser = argparse.ArgumentParser(description="Train NeRF model on specified dataset")
    parser.add_argument("dataset_name", type=str, help="Name of the dataset to train on")
    parser.add_argument("--dataset-type", type=str, default="nerf_synthetic",
                      choices=["nerf_synthetic", "tanks_and_temple", "custom"],
                      help="Type of dataset (default: nerf_synthetic)")
    parser.add_argument("--config", type=str, default="configs/nerf.yaml",
                      help="Path to config file (default: configs/nerf.yaml)")
    
    args = parser.parse_args()
    
    try:
        # Get dataset path
        dataset_path = get_dataset_path(args.dataset_name, args.dataset_type)
        
        # Construct and run command
        command = [
            "python",
            "train_per_scene.py",
            args.config,
            f"defaults.expname={args.dataset_name}",
            f"dataset.datadir={dataset_path}"
        ]
        
        print(f"Running: {' '.join(command)}")
        subprocess.run(command, check=True)
        
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error code {e.returncode}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())