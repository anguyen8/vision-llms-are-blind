import json
from openai import OpenAI
import pandas as pd
from IPython.display import Image, display
from datasets import load_dataset
from tqdm import tqdm
import argparse
import os

# Set up argument parsing
parser = argparse.ArgumentParser(description="Generate batch tasks for GPT-4o model.")
parser.add_argument("--config_json", type=str, required=True, help="Path to the configuration JSON file.")
parser.add_argument("--url_suffix", type=str, required=True, help="URL suffix for image paths.")
parser.add_argument("--output_file", type=str, required=True, help="Output file name for the batch data.")

args = parser.parse_args()

df = pd.read_json(args.config_json)

url_suffix = args.url_suffix

tasks = []

prompt = """Which letter is being circled?"""

for index, row in tqdm(df.iterrows()):
    image_url = f"{url_suffix}/{os.path.basename(row['image_path'])}"
    task = {
        "custom_id": f"uid__{row['image_path']}",
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

with open(args.output_file, "w") as file:
    for obj in tasks:
        file.write(json.dumps(obj) + "\n")