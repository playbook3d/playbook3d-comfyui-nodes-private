import os
import requests
import time
import torch
import numpy as np
import tempfile
import cv2
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


def get_fal_api_key(api_key):
    base_url = "https://dev-accounts.playbook3d.com"
    
    # 1. Retrieve user token
    jwt_request = requests.get(f"{base_url}/token-wrapper/get-tokens/{api_key}")
    if not jwt_request or jwt_request.status_code != 200:
        raise ValueError("Invalid response. Check your Playbook API key.")

    user_token = jwt_request.json().get("access_token")
    if not user_token:
        raise ValueError("No access_token in response. Check your Playbook API key.")

    base_url = "https://dev-api.playbook3d.com"

    secrets_url = f"{base_url}/get-secrets"
    headers = {"Authorization": f"Bearer {user_token}"}
    secrets_request = requests.get(secrets_url, headers=headers)
    print("Secrets Response:", secrets_request.text)

    if secrets_request.status_code != 200:
        raise ValueError(
            f"Failed to retrieve secrets from {secrets_url}. "
            f"Status Code: {secrets_request.status_code}"
        )

    print("Secrets Response:", secrets_request.text)

    secrets_json = secrets_request.json()
    fal_api_key = secrets_json.get("FAL_API_KEY")
    if not fal_api_key:
        raise ValueError("FAL_API_KEY not found in secrets response.")

    return fal_api_key


def video_to_frames(video_url):
    frames = []
    try:
        # Download video
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
            response = requests.get(video_url, stream=True)
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    temp_video.write(chunk)
            temp_video_path = temp_video.name

        # OpenCV to read frames
        cap = cv2.VideoCapture(temp_video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR (OpenCV) -> RGB (PIL)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            frames.append(pil_img)
        cap.release()
    except Exception as e:
        print(f"Error converting video to frames: {e}")
    finally:
        if 'temp_video_path' in locals():
            try:
                os.unlink(temp_video_path)
            except:
                pass
    return frames


class Playbook_MiniMaxHailuo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False}),
                "model_choice": (["minimax", "hailuo"], {"default": "minimax"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "mode": (["text-to-video", "image-to-video"], {"default": "text-to-video"}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "Playbook Fal"

    def run(self, api_key, model_choice, prompt, mode, image=None):
        fal_api_key = get_fal_api_key(api_key)
        os.environ["FAL_KEY"] = fal_api_key

        if model_choice == "minimax":
            base_endpoint = "fal-ai/minimax-video"
        else:
            base_endpoint = "fal-ai/hailuo-video"

        arguments = {"prompt": prompt}

        if mode == "image-to-video":
            if image is None:
                return ([], "Error: Image required for image-to-video mode")
            image_url = upload_image(image)
            if not image_url:
                return ([], "Error: Unable to upload image.")
            arguments["image_url"] = image_url
            endpoint = f"{base_endpoint}/image-to-video"
        else:
            endpoint = base_endpoint

        try:
            handler = submit(endpoint, arguments=arguments)
            result = handler.get()
            video_url = result["video"]["url"]
            print(f"Video generated successfully: {video_url}")

            # Convert video to frames
            images = video_to_frames(video_url)
            if not images:
                return ([], "Error: Failed to extract frames from video.")

            return (images, "Success")

        except Exception as e:
            print(f"Error generating video: {str(e)}")
            return ([], "Error: Unable to generate video.")


class Playbook_Kling:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "duration": (["5", "10"], {"default": "5"}),
                "aspect_ratio": (["16:9", "9:16", "1:1"], {"default": "16:9"}),
                "mode": (["text-to-video", "image-to-video"], {"default": "text-to-video"})
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "Playbook Fal"

    def run(self, api_key, prompt, duration, aspect_ratio, mode, image=None):
        fal_api_key = get_fal_api_key(api_key)
        os.environ["FAL_KEY"] = fal_api_key

        arguments = {
            "prompt": prompt,
            "duration": duration,
            "aspect_ratio": aspect_ratio
        }

        if mode == "image-to-video":
            if image is None:
                print("Error: Image required for image-to-video mode.")
                return ([], "Error: Image required for image-to-video mode.")
            image_url = upload_image(image)
            if not image_url:
                print("Error: Unable to upload image.")
                return ([], "Error: Unable to upload image.")
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

            # Convert video to frames
            images = video_to_frames(video_url)
            if not images:
                return ([], "Error: Failed to extract frames from video.")

            return (images, "Success")

        except Exception as e:
            print(f"Error generating video: {str(e)}")
            return ([], "Error: Unable to generate video.")