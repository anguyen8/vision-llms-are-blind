import os
import base64
from anthropic import Anthropic
import time
from tqdm import tqdm

# Configure API
client = Anthropic()
MODEL_NAME = "claude-3-sonnet-20240229"

# Set directories
parent_folder = "/Volumes/ThirdHeart/Github-Y/vision-llms-are-blind/Tasks/LinePlotIntersection/images"

# Get image files recursively
image_files = []
for root, dirs, files in os.walk(parent_folder):
    for file in files:
        if file.lower().endswith((".png")):
            image_files.append(os.path.join(root, file))
image_files = sorted(image_files)

print(f"Found {len(image_files)} image files.")

# Rate limiting setup
rate_limit = 60  # Max number of requests per 3 minutes
request_count = 0
start_time = time.time()

for image_path in tqdm(image_files, desc="Processing Images"):
    result_file_path = os.path.splitext(image_path)[0] + f"-{MODEL_NAME}-output.md"

    # Skip if results already exist
    if os.path.exists(result_file_path):
        print(f"Skipping {image_path}, already processed.")
        continue

    try:
        # Read and encode the image
        with open(image_path, "rb") as image_file:
            binary_data = image_file.read()
            base_64_encoded_data = base64.b64encode(binary_data)
            base64_string = base_64_encoded_data.decode("utf-8")

        # Prepare the message
        message_list = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64_string,
                        },
                    },
                    {
                        "type": "text",
                        "text": "How many line intersections are there in the image?",
                    },
                ],
            }
        ]

        # Send the message to Claude
        response = client.messages.create(
            model=MODEL_NAME, max_tokens=2048, messages=message_list
        )

        # Save the response
        with open(result_file_path, "w") as file:
            file.write(response.content[0].text)

    except Exception as e:
        print(
            f"An error occurred with {image_path}: {str(e)}. Retrying after a 5-minute wait."
        )
        time.sleep(300)  # Wait for 5 minutes
        continue

    # Manage rate limiting
    request_count += 1
    if request_count >= rate_limit:
        elapsed_time = time.time() - start_time
        if elapsed_time < 180:
            sleep_time = 180 - elapsed_time
            print(f"Rate limit reached. Sleeping for {sleep_time} seconds.")
            time.sleep(sleep_time)
        request_count = 0
        start_time = time.time()
