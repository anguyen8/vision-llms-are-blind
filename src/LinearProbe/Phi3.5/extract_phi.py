# ---------------------------- Import Necessary Libraries ----------------------------
from transformers import CLIPProcessor, CLIPModel, AutoModelForCausalLM, AutoProcessor
import torch
from PIL import Image
import os
from tqdm import tqdm

# ---------------------------- Check CUDA Availability ----------------------------
if not torch.cuda.is_available():
    print("Error: CUDA is not available. Please check your system's GPU or CUDA installation.")
    exit(1)  # Exit with error code 1 to indicate an issue

DEVICE = "cuda"

# ---------------------------- CLIP Model and Processor Setup ----------------------------
print("Loading CLIP model and processor...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()
clip_model.to(DEVICE)
print("CLIP model loaded successfully.")

# ---------------------------- Phi-3 Model and Processor Setup ----------------------------
print("Loading Phi-3 model and processor...")
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
phi3_model.eval()
print("Phi-3 model loaded successfully.")

# ---------------------------- Define Image Processing Function ----------------------------
def process_image(image, clip_model, clip_processor, phi3_model, phi3_processor, device):
    """
    Processes a single image to extract CLIP and Phi-3 embeddings.

    Args:
        image (PIL.Image): The image to process.
        clip_model (CLIPModel): The pre-loaded CLIP model.
        clip_processor (CLIPProcessor): The pre-loaded CLIP processor.
        phi3_model (AutoModelForCausalLM): The pre-loaded Phi-3 model.
        phi3_processor (AutoProcessor): The pre-loaded Phi-3 processor.
        device (str): The device to perform computations on.

    Returns:
        dict: A dictionary containing CLIP and Phi-3 embeddings.
    """
    # ---------------------------- CLIP Embedding Extraction ----------------------------
    clip_inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        #clip_embeddings = clip_model.get_image_features(**clip_inputs)
        vision_outputs = clip_model.vision_model(
            pixel_values=clip_inputs["pixel_values"],
            output_hidden_states=True,
            return_dict=True
        )

	# Get the last hidden state (contains spatial information)
        clip_embeddings  = vision_outputs.last_hidden_state  # Shape: (batch_size, num_patches, hidden_size)


    user_prompt = '<|user|>\n'
    assistant_prompt = '<|assistant|>\n'
    prompt_suffix = "<|end|>\n"
    prompt = f"{user_prompt}<|image_1|>\nWhat is in this image?{prompt_suffix}{assistant_prompt}"
            
    #phi3_inputs = phi3_processor(text=prompt, images=image, return_tensors="pt", padding=True).to(DEVICE)

    # Generate embeddings using Phi-3
    #with torch.no_grad():
       # phi3_embeddings = phi3_model.model.vision_embed_tokens.get_img_features(phi3_inputs['pixel_values'].flatten(0, 1))

    return {
        "clip_embeddings": clip_embeddings.cpu(),
        "phi3_embeddings": phi3_embeddings.cpu(),
    }

# ---------------------------- Define Directories ----------------------------
# Folder containing images
image_folder = "./min_line_dataset"  # Replace with your actual image folder path

# Output folder for features
output_folder = "/aiau001_scratch/logan/min_line_features"  # Replace with your desired output path
os.makedirs(output_folder, exist_ok=True)

# ---------------------------- Process All Images ----------------------------
print("Starting image processing...")
for image_file in tqdm(os.listdir(image_folder), desc="Processing Images"):
    if image_file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")):
        image_path = os.path.join(image_folder, image_file)
        base_name = os.path.splitext(image_file)[0]

        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_file}: {str(e)}")
            continue

        try:
            # Extract embeddings
            features = process_image(
                image,
                clip_model,
                clip_processor,
                phi3_model,
                phi3_processor,
                DEVICE
            )
        except Exception as e:
            print(f"Error processing image {image_file}: {str(e)}")
            continue

        # Save CLIP embeddings
        clip_save_path = os.path.join(output_folder, f"{base_name}_clip_embeddings.pt")
        torch.save(features["clip_embeddings"], clip_save_path)

        # Save Phi-3 embeddings
        phi3_save_path = os.path.join(output_folder, f"{base_name}_phi3_embeddings.pt")
        torch.save(features["phi3_embeddings"], phi3_save_path)

print("Processing complete. Features saved in:", output_folder)

if 'features' in locals() and features is not None:
    print("CLIP embeddings shape:", features["clip_embeddings"].shape)
    print("Phi-3 embeddings shape:", features["phi3_embeddings"].shape)
