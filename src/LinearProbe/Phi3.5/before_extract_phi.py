from modeling_phi3_v import Phi3ImageEmbedding, Phi3VModel, Phi3VForCausalLM, Phi3VConfig
from transformers import CLIPProcessor, CLIPModel, AutoModelForCausalLM, AutoProcessor
import torch
from PIL import Image
import os
from tqdm import tqdm

# ---------------------------- Check CUDA Availability ----------------------------
if not torch.cuda.is_available():
    print("Error: CUDA is not available. Please check your system's GPU or CUDA installation.")
    exit(1)

DEVICE = "cuda"

# ---------------------------- Model and Processor Setup ----------------------------
print("Loading Phi-3 model and processor...")
PRETRAINED_MODEL_NAME = "microsoft/Phi-3.5-vision-instruct"
phi3_processor = AutoProcessor.from_pretrained(
    PRETRAINED_MODEL_NAME,
    trust_remote_code=True
)
#phi3_model = AutoModelForCausalLM.from_pretrained(
#    PRETRAINED_MODEL_NAME,
#    trust_remote_code=True,
#   torch_dtype=torch.float16
#).to(DEVICE)
config = Phi3VConfig.from_pretrained(PRETRAINED_MODEL_NAME)
phi3_model = Phi3VForCausalLM.from_pretrained(
    PRETRAINED_MODEL_NAME,
    config=config,
    torch_dtype=torch.float16
).to(DEVICE)
phi3_model.eval()
print("Phi-3 model loaded successfully.")

# ---------------------------- Define Image Processing Function ----------------------------
def process_image(image, phi3_model, phi3_processor, device):
    """
    Processes a single image to extract Phi-3 pre-projection embeddings.
    
    Args:
        image (PIL.Image): The image to process.
        phi3_model (AutoModelForCausalLM): The pre-loaded Phi-3 model.
        phi3_processor (AutoProcessor): The pre-loaded Phi-3 processor.
        device (str): The device to perform computations on.
        
    Returns:
        torch.Tensor: The pre-projection embeddings
    """
    try:
        # Process image with Phi-3 processor - including special image token
        phi3_inputs = phi3_processor(images=image, text="<|image_1|>", return_tensors="pt").to(device)
        
        # Access the image embedding module
        img_embed_module = phi3_model.model.vision_embed_tokens
        
        with torch.no_grad():
            _ = img_embed_module(
                phi3_inputs["input_ids"],
		pixel_values=phi3_inputs["pixel_values"],
		image_sizes=phi3_inputs["image_sizes"]
            )

		# Now you can access the intermediate concatenated features
            concatenated_features = img_embed_module.last_concatenated
            projected_features = img_embed_module.projected
            features = [concatenated_features.cpu(), projected_features.cpu()]
            #return concatenated_features.cpu()
            return features

            # Get image features through CLIP
            pixel_values = phi3_inputs['pixel_values']
            print(f"Flattened pixel_values shape: {pixel_values.shape}")
            img_features = img_embed_module.get_img_features(pixel_values)
            #img_features = img_embed_module.get_img_features(phi3_inputs['pixel_values'])
            
            # Reshape features for HD transform
            num_images = 1  # Since we're processing one image at a time
            img_features = img_features.reshape(num_images, -1, img_embed_module.image_dim_out)
            
            # Get image sizes (assuming 336x336 per crop for simplicity)
            image_sizes = [(336, 336)]
            
            # Apply HD transform to get pre-projection features
            global_features = img_features[:, 0]
            global_features_hd = img_embed_module.reshape_hd_patches_2x2merge(global_features, 1, 1)
            global_features_hd_newline = img_embed_module.add_image_newline(global_features_hd)
            
            all_embeddings = []
            for i, img_size in enumerate(image_sizes):
                h_crop, w_crop = 1, 1  # Assuming single crop for simplicity
                num_crops = h_crop * w_crop
                
                sub_features = img_features[i, 1:1+num_crops]
                sub_features_hd = img_embed_module.reshape_hd_patches_2x2merge(sub_features, h_crop, w_crop)
                sub_features_hd_newline = img_embed_module.add_image_newline(sub_features_hd)
                
                # Combine sub-features, separator, and global features
                all_embeddings.extend([
                    sub_features_hd_newline.squeeze(0),
                    img_embed_module.glb_GN.squeeze(0),
                    global_features_hd_newline[i]
                ])
            
            # Concatenate all features to get the pre-projection embeddings
            pre_projection_features = torch.cat(all_embeddings, dim=0)

            return pre_projection_features.cpu()
            
    except Exception as e:
        print(f"Processing error details: {str(e)}")
        raise e

# ---------------------------- Define Directories ----------------------------
#image_folder = "/home/u2/ldb0046/VLMFeatures/newDataset"  # Replace with your actual image folder path
image_folder = "/home/u2/ldb0046/VLMFeatures/Phi3.5"  # Replace with your actual image folder path
#output_folder = "/aiau001_scratch/logan/phi/circle_features"  # Replace with your desired output path
output_folder = "/home/u2/ldb0046/VLMFeatures/Phi3.5"  # Replace with your desired output path
os.makedirs(output_folder, exist_ok=True)

# ---------------------------- Process All Images ----------------------------
print("Starting image processing...")
for image_file in tqdm(os.listdir(image_folder), desc="Processing Images"):
    if image_file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")):
        image_path = os.path.join(image_folder, image_file)
        base_name = os.path.splitext(image_file)[0]
        print(image_path)
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_file}: {str(e)}")
            continue

        try:
            # Extract pre-projection embeddings
            features = process_image(
                image,
                phi3_model,
                phi3_processor,
                DEVICE
            )
        except Exception as e:
            print(f"Error processing image {image_file}: {str(e)}")
            continue

        # Save pre-projection features
        save_path = os.path.join(output_folder, f"{base_name}_phi3_before_projection.pt")
        torch.save(features[0], save_path)

        save_path = os.path.join(output_folder, f"{base_name}_phi3_after_projection.pt")
        torch.save(features[1], save_path)

print("Processing complete. Features saved in:", output_folder)

if 'features' in locals() and features is not None:
    print("Pre-projection features shape:", features[0].shape)
    print("After-projection features shape:", features[1].shape)
