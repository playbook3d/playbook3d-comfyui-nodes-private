import os
import time
import tempfile
import requests
import torch
import numpy as np
from PIL import Image
import io

import folder_paths
from lumaai import LumaAI

def get_jwt_from_playbook_key(api_key: str) -> str:
    """Get JWT access token from Playbook API key"""
    url = f"https://accounts.playbook3d.com/token-wrapper/get-tokens/{api_key}"
    response = requests.get(url)
    
    if response.status_code != 200:
        raise ValueError(f"Failed to get JWT token. Status: {response.status_code}")
        
    data = response.json()
    return data.get('access_token')

def upload_image_to_s3(image_tensor, api_key, run_id=""):
    """Upload image tensor to Playbook S3 and return the signed URL"""
    try:
        # Get JWT token
        jwt_token = get_jwt_from_playbook_key(api_key)
        node_id = "photonnode"
        
        print(f"Debug - Got JWT token successfully")
        
        # Convert tensor to PNG bytes
        if image_tensor.dim() == 4 and image_tensor.shape[0] == 1:
            image_tensor = image_tensor.squeeze(0)
        
        arr = (image_tensor.cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
        img = Image.fromarray(arr)
        
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Prepare multipart form data
        files = {
            'file': ('image.png', img_byte_arr, 'image/png')
        }
        
        # Use run_id in path if provided
        upload_path = f"{run_id}/{node_id}" if run_id else node_id
        url = f"https://accounts.playbook3d.com/upload-assets/{upload_path}"
        
        headers = {
            "Authorization": f"Bearer {jwt_token}"
        }
        
        print(f"Debug - Uploading to URL: {url}")
        response = requests.post(url, files=files, headers=headers)
        print(f"Debug - Upload response status: {response.status_code}")
        
        if response.status_code != 201:
            raise ValueError(f"Failed to upload image. Status: {response.status_code}\nResponse: {response.text}")
        
        # Extract and return the S3 URL from response
        result = response.json()
        return result['url']
        
    except Exception as e:
        print(f"Debug - Exception type: {type(e)}")
        print(f"Debug - Exception details: {str(e)}")
        raise

def download_image_to_temp(image_url):
    """Download the image at image_url to a local temporary file (.jpg)."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_img:
        r = requests.get(image_url, stream=True)
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                temp_img.write(chunk)
        return temp_img.name

def image_to_tensor(path):
    """Load an image from disk, convert to RGB float32 tensor [H, W, C]."""
    img_pil = Image.open(path).convert("RGB")
    arr = np.array(img_pil, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr)
    return tensor

class Playbook_PhotonText2Image:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "luma_api_key": ("STRING", {"multiline": False}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "aspect_ratio": ("STRING", {"default": "1:1"}),
                "save": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "filename": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "Playbook 3D"

    def validate_aspect_ratio(self, aspect_ratio):
        try:
            w, h = map(int, aspect_ratio.split(":"))
            if w <= 0 or h <= 0:
                raise ValueError("Aspect ratio values must be positive.")
            return True
        except ValueError:
            raise ValueError("Invalid aspect_ratio format. Must be 'W:H' with positive integers.")

    def run(self, luma_api_key, prompt, aspect_ratio, save, filename=""):
        if not prompt:
            raise ValueError("Prompt is required")
        self.validate_aspect_ratio(aspect_ratio)

        client = LumaAI(auth_token=luma_api_key)

        print(f"Debug: Creating Photon image from prompt: {prompt}")
        generation = client.generations.image.create(
            prompt=prompt,
            aspect_ratio=aspect_ratio,
        )
        gen_id = generation.id

        while True:
            g = client.generations.get(id=gen_id)
            if g.state == "completed":
                break
            if g.state == "failed":
                raise ValueError(f"Generation failed: {g.failure_reason}")
            time.sleep(2)

        image_url = g.assets.image
        temp_path = download_image_to_temp(image_url)

        if save:
            out_dir = folder_paths.get_output_directory()
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            final_name = filename or gen_id
            final_path = os.path.join(out_dir, f"{final_name}.jpg")
            os.rename(temp_path, final_path)
            temp_path = final_path
            print(f"Debug: Saved Photon image to {final_path}")

        image_tensor = image_to_tensor(temp_path)
        if image_tensor is None:
            raise ValueError("Error: Could not create image tensor from Photon output.")
        image_tensor = image_tensor.unsqueeze(0)

        return (image_tensor,)

class Playbook_PhotonModifyImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "luma_api_key": ("STRING", {"multiline": False}),
                "api_key": ("STRING", {"multiline": False}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "modify_image": ("IMAGE", {"forceInput": True}),
                "save": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "filename": ("STRING", {"default": ""}),
                "run_id": ("STRING", {"default": ""})
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "Playbook 3D"

    def run(self, luma_api_key, api_key, prompt, modify_image, save, filename="", run_id=""):
        if not prompt:
            raise ValueError("Prompt is required")

        modify_image_url = upload_image_to_s3(modify_image, api_key, run_id)
        print(f"Debug - Image uploaded to S3: {modify_image_url}")

        client = LumaAI(auth_token=luma_api_key)

        print(f"Debug: Modifying image with Photon, prompt: {prompt}")
        generation = client.generations.image.create(
            prompt=prompt,
            modify_image_ref={"url": modify_image_url},
        )
        gen_id = generation.id

        while True:
            g = client.generations.get(id=gen_id)
            if g.state == "completed":
                break
            if g.state == "failed":
                raise ValueError(f"Modification failed: {g.failure_reason}")
            time.sleep(2)

        image_url = g.assets.image
        temp_path = download_image_to_temp(image_url)

        if save:
            out_dir = folder_paths.get_output_directory()
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            final_name = filename or gen_id
            final_path = os.path.join(out_dir, f"{final_name}.jpg")
            os.rename(temp_path, final_path)
            temp_path = final_path
            print(f"Debug: Saved modified Photon image to {final_path}")

        image_tensor = image_to_tensor(temp_path)
        if image_tensor is None:
            raise ValueError("Error: Could not create image tensor after modification.")
        image_tensor = image_tensor.unsqueeze(0)

        return (image_tensor,)

class Playbook_PhotonPreviewImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_url": ("STRING", {"forceInput": True})
            }
        }

    FUNCTION = "run"
    CATEGORY = "Playbook 3D"
    RETURN_TYPES = ()
    OUTPUT_NODE = True

    def run(self, image_url):
        return {"ui": {"image_url": [image_url]}}