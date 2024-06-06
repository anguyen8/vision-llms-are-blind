import google.generativeai as genai
import time
import os
from tqdm import tqdm
from google.api_core.exceptions import ResourceExhausted, GoogleAPIError

# Configure API
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Set directories
parent_folder = (
    "/Volumes/ThirdHeart/Github-Y/vision-llms-are-blind/Tasks/NestedSquares/images"
)

# Get video files recursively
video_files = []
for root, dirs, files in os.walk(parent_folder):
    for file in files:
        if file.endswith(".png"):
            video_files.append(os.path.join(root, file))
video_files = sorted(video_files)

print(f"Found {len(video_files)} video files.")

# shuffle the videos
import random

random.shuffle(video_files)

# Rate limiting setup
rate_limit = 10  # Max number of requests per 3 minutes
request_count = 0
start_time = time.time()

for image_path in tqdm(video_files, desc="Processing Videos"):
    # Create result path in the same directory as the video file with a different filename
    result_file_path = os.path.splitext(image_path)[0] + "-gemini-output.md"

    # Skip if results already exist
    if os.path.exists(result_file_path):
        print(f"Skipping {image_path}, already processed.")
        continue

    try:
        print("Uploading file...")
        video_file = genai.upload_file(path=image_path)
        print(f"Completed upload: {video_file.uri}")

        # Wait for processing to complete
        while video_file.state.name == "PROCESSING":
            print("Waiting for video to be processed.")
            time.sleep(10)
            video_file = genai.get_file(video_file.name)

        # Handle failed processing
        if video_file.state.name == "FAILED":
            print(
                f"Processing failed for {video_file.uri}. Skipping to next video after a 5-minute wait."
            )
            time.sleep(300)  # Wait for 5 minutes
            continue

        print(f"Video processing complete: {video_file.uri}")

        # Set prompt for the model
        prompt = "Count total number of squares in the image."

        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro-latest",
            safety_settings=safety_settings,
            generation_config=generation_config,
        )

        print("Making LLM inference request...")
        response = model.generate_content(
            [prompt, video_file], request_options={"timeout": 600}
        )

        # Check if the response is valid
        if not response.parts:
            print(
                "No content was generated. Check the `response.prompt_feedback` for more details."
            )
            continue

        print(response.text)

        # Save the response
        with open(result_file_path, "w") as file:
            file.write(response.text)

        # Clean up by deleting the uploaded file
        genai.delete_file(video_file.name)
        print(f"Deleted file {video_file.uri}")

    except ResourceExhausted as e:
        print(f"API quota exceeded: {str(e)}. Waiting for 3 minutes before retrying.")
        time.sleep(120)  # Wait for 2 minutes
        continue
    except GoogleAPIError as e:
        print(f"Google API error occurred: {str(e)}. Retrying after a 5-minute wait.")
        time.sleep(120)  # Wait for 2 minutes
        continue
    except Exception as e:
        print(
            f"An unexpected error occurred with {image_path}: {str(e)}. Retrying after a 5-minute wait."
        )
        time.sleep(120)  # Wait for 2 minutes
        continue
    # Manage rate limiting
    request_count += 1
    if request_count >= rate_limit:
        elapsed_time = time.time() - start_time
        if elapsed_time < 120:
            sleep_time = 120 - elapsed_time
            print(f"Rate limit reached. Sleeping for {sleep_time} seconds.")
            time.sleep(sleep_time)
        request_count = 0
        start_time = time.time()
