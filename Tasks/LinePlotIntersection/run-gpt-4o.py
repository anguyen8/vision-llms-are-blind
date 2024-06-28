import json
import os
from tqdm import tqdm
import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(description="Generate batch tasks for GPT-4o model.")
parser.add_argument(
    "--config_json",
    type=str,
    required=True,
    help="Path to the configuration JSON file.",
)
parser.add_argument(
    "--url_suffix", type=str, required=True, help="URL suffix for image paths."
)
parser.add_argument(
    "--output_file",
    type=str,
    required=True,
    help="Output file name for the batch data.",
)

args = parser.parse_args()

# Load JSON data
with open(args.config_json, "r") as file:
    data = json.load(file)

url_suffix = args.url_suffix

tasks = []

prompt = "How many times do the blue and red line plots cross each other?"
prompt = "How many times do the blue and red lines intersect?"

# Iterate over each image configuration
for image_name, image_info in tqdm(data.items()):
    image_url = f"{url_suffix}/{image_name}"
    task = {
        "custom_id": f"uid__{image_name}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o",
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        },
                    ],
                },
            ],
        },
    }

    tasks.append(task)

# Write tasks to output file
with open(args.output_file, "w") as file:
    for obj in tasks:
        file.write(json.dumps(obj) + "\n")
