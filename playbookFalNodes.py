import os
import requests
import time
import torch
import numpy as np
import tempfile
from PIL import Image
from fal_client import submit, upload_file
from dotenv import load_dotenv


load_dotenv()



def upload_image(image):
    try:
        # Convert to numpy
        if isinstance(image, torch.Tensor):
            image_np = image.cpu().numpy()
        else:
            image_np = np.array(image)

        if image_np.ndim == 4:
            image_np = image_np.squeeze(0)
        if image_np.ndim == 2:
            image_np = np.stack([image_np] * 3, axis=-1)
        elif image_np.shape[0] == 3:
            image_np = np.transpose(image_np, (1, 2, 0))

        if image_np.dtype in [np.float32, np.float64]:
            image_np = (image_np * 255).astype(np.uint8)

        pil_image = Image.fromarray(image_np)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            pil_image.save(temp_file, format="PNG")
            temp_file_path = temp_file.name

        image_url = upload_file(temp_file_path)
        return image_url
    except Exception as e:
        print(f"Error uploading image: {str(e)}")
        return None
    finally:
        if 'temp_file_path' in locals():
            os.unlink(temp_file_path)


class Playbook_FalClient:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "playbook_api_key": ("STRING", {"multiline": False}),
            },
        }

    RETURN_TYPES = ("FALCLIENT",)
    RETURN_NAMES = ("client",)
    FUNCTION = "run"
    CATEGORY = "Playbook Fal"

    def run(self, playbook_api_key):
        base_url = "https://accounts.playbookengine.com"

        # Get user token using the Playbook API key
        jwt_request = requests.get(f"{base_url}/token-wrapper/get-tokens/{playbook_api_key}")

        if jwt_request.status_code != 200:
            raise ValueError("Failed to retrieve user token. Check your Playbook API key.")

        try:
            user_token = jwt_request.json()["access_token"]
        except Exception as e:
            print(f"Error parsing user token: {e}")
            raise ValueError("Invalid response when retrieving user token.")

        headers = {"Authorization": f"Bearer {user_token}"}

        # Retrieve the Fal API key
        # fal_key_request = requests.get(f"{base_url}/api/get-fal-api-key", headers=headers)

        # if fal_key_request.status_code == 200:
        #     fal_api_key = fal_key_request.json().get("fal_api_key")
        #     if not fal_api_key:
        #         raise ValueError("Failed to retrieve Fal API key from endpoint.")
        # else:
        #     raise ValueError(f"Failed to retrieve Fal API key, status code {fal_key_request.status_code}")

        fal_api_key = os.getenv('FAL_API_KEY')
        if not fal_api_key:
            raise ValueError("FAL_API_KEY not found in environment variables")

        # Returns the Fal client object. since there is no official fal api client, let's return a dict containing the fal_api_key.
        client = {"fal_api_key": fal_api_key}
        return (client,)


class Playbook_MiniMaxHailuo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "client": ("FALCLIENT", {"forceInput": True}),
                "model_choice": (["minimax", "hailuo"], {"default": "minimax"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "mode": (["text-to-video", "image-to-video"], {"default": "text-to-video"}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_url", "message")
    FUNCTION = "run"
    CATEGORY = "Playbook Fal"

    def run(self, client, model_choice, prompt, mode, image=None):
        fal_api_key = client.get("fal_api_key")
        if not fal_api_key:
            raise ValueError("Fal API key not found in client object.")

        os.environ["FAL_KEY"] = fal_api_key

        if model_choice == "minimax":
            base_endpoint = "fal-ai/minimax-video"
        else:
            base_endpoint = "fal-ai/hailuo-video"

        arguments = {"prompt": prompt}

        if mode == "image-to-video":
            if image is None:
                return ("", "Error: Image required for image-to-video mode")
            image_url = upload_image(image)
            if not image_url:
                return ("", "Error: Unable to upload image.")
            arguments["image_url"] = image_url
            endpoint = f"{base_endpoint}/image-to-video"
        else:
            endpoint = base_endpoint

        try:
            handler = submit(endpoint, arguments=arguments)
            result = handler.get()
            video_url = result["video"]["url"]
            print(f"Video generated successfully: {video_url}")
            return (video_url, "Success")
        except Exception as e:
            print(f"Error generating video: {str(e)}")
            return ("", "Error: Unable to generate video.")


class Playbook_Kling:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "client": ("FALCLIENT", {"forceInput": True}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "duration": (["5", "10"], {"default": "5"}),
                "aspect_ratio": (["16:9", "9:16", "1:1"], {"default": "16:9"}),
                "mode": (["text-to-video", "image-to-video"], {"default": "text-to-video"})
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_url",)
    FUNCTION = "run"
    CATEGORY = "Playbook Fal"

    def run(self, client, prompt, duration, aspect_ratio, mode, image=None):
        fal_api_key = client.get("fal_api_key")
        if not fal_api_key:
            raise ValueError("Fal API key not found in client object.")
        
        os.environ["FAL_KEY"] = fal_api_key
        
        arguments = {
            "prompt": prompt,
            "duration": duration,
            "aspect_ratio": aspect_ratio
        }

        if mode == "image-to-video":
            if image is None:
                print("Error: Image required for image-to-video mode.")
                return ("",)
            image_url = upload_image(image)
            if not image_url:
                print("Error: Unable to upload image.")
                return ("",)
            arguments["image_url"] = image_url
            endpoint = "fal-ai/kling-video/v1/standard/image-to-video"
        else:
            endpoint = "fal-ai/kling-video/v1/standard/text-to-video"

        try:
            print("Submitting video generation request to Fal API...")
            handler = submit(endpoint, arguments=arguments)
            result = handler.get()
            video_url = result["video"]["url"]
            print(f"Video generated successfully: {video_url}")
            return (video_url,)
        except Exception as e:
            print(f"Error generating video: {str(e)}")
            return ("",)
