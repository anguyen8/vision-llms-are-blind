from transformers import CLIPProcessor, CLIPModel, AutoModel, AutoProcessor
import torch
from PIL import Image

import os
from tqdm import tqdm
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import requests

# ---------------------------- Load CLIP Model and Processor ----------------------------

# Load pre-trained CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Set models to evaluation mode
clip_model.eval()

# Move CLIP model to GPU if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = clip_model.to(DEVICE)

PRETRAINED_MODEL_NAME = "microsoft/Phi-3.5-vision-instruct"
phi3_processor = AutoProcessor.from_pretrained(
    PRETRAINED_MODEL_NAME,
    trust_remote_code=True
)
phi3_model = AutoModelForCausalLM.from_pretrained(
    PRETRAINED_MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(DEVICE)

# Set Phi-3 model to evaluation mode
phi3_model.eval()

# Move Phi-3 model to GPU if available
phi3_model = phi3_model.to(DEVICE)

# ---------------------------- Prepare Your Data ----------------------------

# List of image file paths
image_paths = ["/home/u2/ldb0046/VLMFeatures/Phi3.5/gt_1_image_976_thickness_2_resolution_384.png"]  # Replace with your image paths

# Load images using PIL
images = [Image.open(image_path).convert("RGB") for image_path in image_paths]

# Process images using CLIP processor
clip_inputs = clip_processor(images=images, return_tensors="pt").to(DEVICE)
with torch.no_grad():
#    clip_embeddings = clip_model.get_image_features(**clip_inputs)
    vision_outputs = clip_model.vision_model(
        pixel_values=clip_inputs["pixel_values"],
        output_hidden_states=True,
        return_dict=True
    )
    
    # Get the last hidden state (contains spatial information)
    clip_embeddings  = vision_outputs.last_hidden_state  # Shape: (batch_size, num_patches, hidden_size)


# Normalize CLIP embeddings if needed
# clip_embeddings = clip_embeddings / clip_embeddings.norm(p=2, dim=-1, keepdim=True)

# ---------------------------- Generate Embeddings Using Phi-3 ----------------------------

# Process images using Phi-3 processor
user_prompt = '<|user|>\n'
assistant_prompt = '<|assistant|>\n'
prompt_suffix = "<|end|>\n"
prompt = f"{user_prompt}<|image_1|>\nWhat is in this image?{prompt_suffix}{assistant_prompt}"
        
phi3_inputs = phi3_processor(text=prompt, images=images, return_tensors="pt", padding=True).to(DEVICE)

# Generate embeddings using Phi-3
with torch.no_grad():
    phi3_embeddings = phi3_model.model.vision_embed_tokens.get_img_features(phi3_inputs['pixel_values'].flatten(0, 1))
print(clip_processor.feature_extractor.size)

# Example: Comparing embeddings
for idx, image_path in enumerate(image_paths):
     print(f"Image: {image_path}")
     print(f"CLIP Embedding Vector: {clip_embeddings[idx]}")
#     print(f"Phi-3 Embedding Vector: {phi3_embeddings[idx]}")
     print("-" * 50)

# compare shapes
print(clip_embeddings.shape)
#print(last_hidden_state.shape)
print(phi3_embeddings.shape)
