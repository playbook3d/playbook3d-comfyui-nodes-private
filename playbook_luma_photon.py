import os
import time
import tempfile
import requests
import torch
import numpy as np
from PIL import Image

import folder_paths
from lumaai import LumaAI


def download_image_to_temp(image_url):
    """
    Download the image at image_url to a local temporary file (.jpg).
    Returns the temp file path.
    """
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_img:
        r = requests.get(image_url, stream=True)
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                temp_img.write(chunk)
        return temp_img.name


def image_to_tensor(path):
    """
    Load an image from disk, convert to RGB float32 tensor [H, W, C].
    """
    img_pil = Image.open(path).convert("RGB")
    arr = np.array(img_pil, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr)
    return tensor


class Playbook_PhotonText2Image:
    """
    Generate a single image from a text prompt using the Luma Photon API.
    Returns a 4D PyTorch tensor (1, H, W, 3) to ComfyUI as an IMAGE output.
    """
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
        """Validate the aspect ratio 'W:H' format."""
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

        # Use the Luma API Key directly
        client = LumaAI(auth_token=luma_api_key)

        print(f"Debug: Creating Photon image from prompt: {prompt}")
        # Create the image generation
        # If your environment doesn’t accept "model=...", omit or adapt that parameter.
        generation = client.generations.image.create(
            prompt=prompt,
            aspect_ratio=aspect_ratio,
        )
        gen_id = generation.id

        # Poll until done
        while True:
            g = client.generations.get(id=gen_id)
            if g.state == "completed":
                break
            if g.state == "failed":
                raise ValueError(f"Generation failed: {g.failure_reason}")
            time.sleep(2)

        # Download final image
        image_url = g.assets.image
        temp_path = download_image_to_temp(image_url)

        # Optionally save to your output directory
        if save:
            out_dir = folder_paths.get_output_directory()
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            final_name = filename or gen_id
            final_path = os.path.join(out_dir, f"{final_name}.jpg")
            os.rename(temp_path, final_path)
            temp_path = final_path
            print(f"Debug: Saved Photon image to {final_path}")

        # Convert to a PyTorch tensor, then add a leading batch dimension
        image_tensor = image_to_tensor(temp_path)
        if image_tensor is None:
            raise ValueError("Error: Could not create image tensor from Photon output.")
        # (H, W, 3) -> (1, H, W, 3) so ComfyUI sees correct color
        image_tensor = image_tensor.unsqueeze(0)

        return (image_tensor,)


class Playbook_PhotonModifyImage:
    """
    Modify an existing image via Photon by providing a reference image plus a prompt.
    Returns the modified image as a 4D tensor (1, H, W, 3).
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "luma_api_key": ("STRING", {"multiline": False}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "modify_image_url": ("STRING", {"forceInput": True}),
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

    def run(self, luma_api_key, prompt, modify_image_url, save, filename=""):
        if not prompt:
            raise ValueError("Prompt is required")
        if not modify_image_url:
            raise ValueError("Must provide a URL for the image to modify")

        # Directly pass user Luma Key
        client = LumaAI(auth_token=luma_api_key)

        print(f"Debug: Modifying image with Photon, prompt: {prompt}")
        generation = client.generations.image.create(
            prompt=prompt,
            modify_image_ref={"url": modify_image_url},
        )
        gen_id = generation.id

        # Poll
        while True:
            g = client.generations.get(id=gen_id)
            if g.state == "completed":
                break
            if g.state == "failed":
                raise ValueError(f"Modification failed: {g.failure_reason}")
            time.sleep(2)

        # Download result
        image_url = g.assets.image
        temp_path = download_image_to_temp(image_url)

        # Optionally save
        if save:
            out_dir = folder_paths.get_output_directory()
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            final_name = filename or gen_id
            final_path = os.path.join(out_dir, f"{final_name}.jpg")
            os.rename(temp_path, final_path)
            temp_path = final_path
            print(f"Debug: Saved modified Photon image to {final_path}")

        # Convert to tensor
        image_tensor = image_to_tensor(temp_path)
        if image_tensor is None:
            raise ValueError("Error: Could not create image tensor after modification.")
        # Expand dims for ComfyUI color
        image_tensor = image_tensor.unsqueeze(0)

        return (image_tensor,)


class Playbook_PhotonPreviewImage:
    """
    Preview a remote image URL in the ComfyUI interface (no returned IMAGE tensor).
    """
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
        # Tells ComfyUI to show this image in the UI panel
        return {"ui": {"image_url": [image_url]}}
