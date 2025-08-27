import sys
sys.path.insert(0, '/home/u2/ldb0046/VLMFeatures/Phi3.5')

from processing_phi3_v import Phi3VImageProcessor, Phi3VProcessor
from modeling_phi3_v import Phi3VForCausalLM
from transformers import AutoTokenizer
from PIL import Image

# Initialize components manually
image_processor = Phi3VImageProcessor(num_crops=1)  # Adjust parameters as needed
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-3-vision")
processor = Phi3VProcessor(image_processor=image_processor, tokenizer=tokenizer)

# Load model
model = Phi3VForCausalLM.from_pretrained("microsoft/phi-3-vision")

# Test with image
image = Image.open("/home/u2/ldb0046/VLMFeatures/Phi3.5/gt_1_image_976_thickness_2_resolution_384.png")
text = "Describe this image:"
inputs = processor(text=text, images=image, return_tensors="pt")
