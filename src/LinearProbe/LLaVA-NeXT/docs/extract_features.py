from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

from PIL import Image
import torch
import os
from tqdm import tqdm

# Load model
pretrained = "lmms-lab/llava-onevision-qwen2-0.5b-si"
model_name = "llava_qwen"
device = "cuda" if torch.cuda.is_available() else "cpu"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map)

model.eval()


# Function to process a single image
def process_single_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

    # Generate dummy input for the model
    dummy_input = DEFAULT_IMAGE_TOKEN + "\nDummy text"
    input_ids = tokenizer_image_token(dummy_input, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

    # Include image_sizes
    image_sizes = [image.size]

    # Run the model to trigger feature extraction
    try:
        _ = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            max_new_tokens=1,
        )
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None, None

    # Retrieve features
    vision_features = model.get_last_vision_features()
    projected_features = model.get_last_projected_features()

    return vision_features, projected_features


# Folder containing images
image_folder = "/mnt/nvme-storage/nsrc/BlindTestExtEval/MoreDistsToTrain"

# Output folder for features
output_folder = "/mnt/nvme-storage/nsrc/BlindTestExtEval/touching_circles_images_features_llava-onevision-qwen2-0.5b-si/"
os.makedirs(output_folder, exist_ok=True)

# Process all images in the folder
for image_file in tqdm(os.listdir(image_folder)):
    if image_file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")):
        image_path = os.path.join(image_folder, image_file)

        vision_features, projected_features = process_single_image(image_path)

        if vision_features is None or projected_features is None:
            print(f"Skipping {image_file} due to processing error.")
            continue

        # Save features
        base_name = os.path.splitext(image_file)[0]

        torch.save(vision_features, os.path.join(output_folder, f"{base_name}_vision_features.pt"))
        torch.save(projected_features, os.path.join(output_folder, f"{base_name}_projected_features.pt"))

print("Processing complete. Features saved in:", output_folder)

# Print dimensions of the last processed image (as an example)
if vision_features is not None:
    print("Vision features shape:", vision_features.shape)
if projected_features is not None:
    print("Projected features shape:", projected_features.shape)
