import requests
import time
import os
import ijson

OLLAMA_HOST = os.getenv(key="OLLAMA_HOST")
import requests
import re

def create_image(prompt_text, seed=2960155758, api_url=f"http://{OLLAMA_HOST}:9000/render"):
    """
    Creates an image by sending a request to the Easy Diffusion API with the given prompt.
    
    Parameters:
        prompt_text (str): The text prompt for the image generation.
        seed (int): Seed for image generation to ensure reproducibility.
        api_url (str): The API URL to send the request to.
    
    Returns:
        bytes: Image data in bytes, or None if there was an error.
    """
    
    headers = {}
    
    payload = {
        "prompt": prompt_text,
        "seed": seed,
        "used_random_seed": True,
        "negative_prompt": "",
        "num_outputs": 1,
        "num_inference_steps": 25,
        "guidance_scale": 7.5,
        "width": 512,
        "height": 512,
        "vram_usage_level": "balanced",
        "sampler_name": "euler_a",
        "use_stable_diffusion_model": "sd-v1-4",
        "clip_skip": False,
        "use_vae_model": "",
        "stream_progress_updates": True,
        "stream_image_progress": False,
        "show_only_filtered_image": True,
        "block_nsfw": False,
        "output_format": "jpeg",
        "output_quality": 75,
        "output_lossless": False,
        "metadata_output_format": "none",
        "original_prompt": prompt_text,
        "active_tags": [],
        "inactive_tags": [],
        "enable_vae_tiling": True,
        "scheduler_name": "simple",
        "session_id": "1731339067668"
    }
    
    # Send the request
    try:
        response = requests.post(api_url, headers=headers, json=payload, verify=False)
        response.raise_for_status()
        stream=response.json()['stream']
        # If successful, return the response content (image in bytes)
        return stream
    
    except requests.RequestException as e:
        print("Error generating image:", e)
        return None
    
def poll_image_status(url):
    for _ in range(10):
        response = requests.get(f"http://{OLLAMA_HOST}:9000/{url}")
        if response.status_code == 200:
            if 'base64' in response.text:
                pattern = r'data:image/jpeg;base64[^"]*"'
                matches = re.findall(pattern, string=response.text)
                return matches[0]
        time.sleep(5)
    return None
   
# Example usage
# prompt = "a photograph of an astronaut riding a horse"
# image_data = create_image(prompt)
# if image_data:
#     with open("generated_image.jpeg", "wb") as f:
#         f.write(image_data)
#     print("Image saved as 'generated_image.jpeg'")
# else:
#     print("Failed to generate image.")
