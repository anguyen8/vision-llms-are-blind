from PIL import Image
import requests
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor
model_path = "./"

kwargs = {}
kwargs['torch_dtype'] = torch.bfloat16

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype="auto", _attn_implementation='flash_attention_2').cuda()

user_prompt = '<|user|>\n'
assistant_prompt = '<|assistant|>\n'
prompt_suffix = "<|end|>\n"


# single-image prompt
prompt = f"{user_prompt}<|image_1|>\nWhat is shown in this image?{prompt_suffix}{assistant_prompt}"
url = "https://icons.iconarchive.com/icons/iconarchive/fish-illustration/256/Small-3-Tiny-Fish-icon.png"
# print(f">>> Prompt\n{prompt}")
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")
generate_ids = model.generate(**inputs, 
                              max_new_tokens=1000,
                              eos_token_id=processor.tokenizer.eos_token_id,
                              )
generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
response = processor.batch_decode(generate_ids, 
                                  skip_special_tokens=True, 
                                  clean_up_tokenization_spaces=False)[0]


#################################################### EXAMPLE 1 ####################################################
# single-image prompt
prompt = f"{user_prompt}<|image_1|>\nWhat is shown in this image?{prompt_suffix}{assistant_prompt}"
url = "https://www.ilankelman.org/stopsigns/australia.jpg"
# print(f">>> Prompt\n{prompt}")
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")
generate_ids = model.generate(**inputs, 
                              max_new_tokens=1000,
                              eos_token_id=processor.tokenizer.eos_token_id,
                              )
generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
response = processor.batch_decode(generate_ids, 
                                  skip_special_tokens=True, 
                                  clean_up_tokenization_spaces=False)[0]


#################################################### EXAMPLE 2 ####################################################
# chat template
chat = [
    {"role": "user", "content": "<|image_1|>\nWhat is shown in this image?"},
    {"role": "assistant", "content": "The image depicts a street scene with a prominent red stop sign in the foreground. The background showcases a building with traditional Chinese architecture, characterized by its red roof and ornate decorations. There are also several statues of lions, which are common in Chinese culture, positioned in front of the building. The street is lined with various shops and businesses, and there's a car passing by."},
    {"role": "user", "content": "What is so special about this image"}
]
url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image = Image.open(requests.get(url, stream=True).raw)
prompt = processor.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
# need to remove last <|endoftext|> if it is there, which is used for training, not inference. For training, make sure to add <|endoftext|> in the end.
if prompt.endswith("<|endoftext|>"):
    prompt = prompt.rstrip("<|endoftext|>")

# print(f">>> Prompt\n{prompt}")

inputs = processor(prompt, [image], return_tensors="pt").to("cuda:0")
generate_ids = model.generate(**inputs, 
                              max_new_tokens=1000,
                              eos_token_id=processor.tokenizer.eos_token_id,
                              )
generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]



############################# to markdown #############################
# single-image prompt
prompt = f"{user_prompt}<|image_1|>\nCan you convert the table to markdown format?{prompt_suffix}{assistant_prompt}"
url = "https://support.content.office.net/en-us/media/3dd2b79b-9160-403d-9967-af893d17b580.png"
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")

# print(f">>> Prompt\n{prompt}")
generate_ids = model.generate(**inputs, 
                              max_new_tokens=1000,
                              eos_token_id=processor.tokenizer.eos_token_id,
                              )
generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
response = processor.batch_decode(generate_ids, 
                                  skip_special_tokens=False, 
                                  clean_up_tokenization_spaces=False)[0]

