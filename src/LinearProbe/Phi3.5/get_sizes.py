import torch
import os

# Define the base directory path
base_path = "/aiau001_scratch/logan/phi/line_features"

# List of tensor files
tensor_files = [
    "gt_1_image_495_thickness_4_resolution_384_phi3_before_projection.pt",
    "gt_2_image_998_thickness_4_resolution_1152_phi3_before_projection.pt",
    "gt_2_image_998_thickness_4_resolution_768_phi3_before_projection.pt"
]

# Load each tensor and print its shape
for file_name in tensor_files:
    file_path = os.path.join(base_path, file_name)
    try:
        tensor = torch.load(file_path)
        print(f"\nFile: {file_name}")
        print(f"Shape: {tensor.shape}")
    except Exception as e:
        print(f"\nError loading {file_name}: {str(e)}")
