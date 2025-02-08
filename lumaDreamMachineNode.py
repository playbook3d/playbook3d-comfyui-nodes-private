import os
import time
import cv2
import requests
import tempfile
import torch
import numpy as np
from PIL import Image
from lumaai import LumaAI
import folder_paths
import io
from PIL import Image as PILImage

def download_video_to_temp(video_url):
    """
    Download the .mp4 from video_url to a local temp file and return that file's path.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
        r = requests.get(video_url, stream=True)
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                temp_video.write(chunk)
        return temp_video.name


def video_to_images(path):
    """
    Convert the .mp4 at path into a 4D PyTorch tensor of frames: [num_frames, height, width, 3].
    """
    images = []
    cap = cv2.VideoCapture(path)
    print("Debug: Starting video to frames conversion")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        frame_tensor = torch.from_numpy(frame)
        images.append(frame_tensor)
        
    cap.release()
    
    if not images:
        return None

    frames = torch.stack(images)
    print(f"Debug: Final tensor shape: {frames.shape}")
    print(f"Debug: Final tensor dtype: {frames.dtype}")
    
    return frames


def image_to_temp_url(image_tensor):
    """
    Placeholder: Takes a ComfyUI 'IMAGE' PyTorch tensor, writes it to disk as PNG, returns a 'file://' URL.
    In a real environment, you'd upload to S3 or an image host and return a real https:// URL.
    """
    import numpy as np
    from PIL import Image as PILImage

    # Ensure we have [H, W, C] shape if there's a batch dim [1, H, W, C].
    if image_tensor.dim() == 4 and image_tensor.shape[0] == 1:
        image_tensor = image_tensor.squeeze(0)

    # Clip to [0, 1], scale to [0..255], convert to uint8
    arr = (image_tensor.cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    PILImage.fromarray(arr).save(temp_file.name)
    temp_file.close()

    return f"file://{temp_file.name}"

def validate_luma_api_key(api_key):
    """
    Validates that a Luma API key is properly formatted.
    Returns True if valid, raises ValueError if invalid.
    """
    if not api_key:
        raise ValueError("Luma API key is required")
    if not isinstance(api_key, str):
        raise ValueError("Luma API key must be a string")
    if not api_key.startswith("luma"):
        raise ValueError("Invalid Luma API key format. API key must start with 'luma'")
    return True

class Playbook_LumaAIClient:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "luma_api_key": ("STRING", {"multiline": False})
            }
        }

    RETURN_TYPES = ("LUMACLIENT",)
    RETURN_NAMES = ("client",)
    FUNCTION = "run"
    CATEGORY = "Playbook 3D"

    def run(self, luma_api_key):
        """
        Create a LumaAI client directly with the user-provided Luma key.
        """
        client = LumaAI(auth_token=luma_api_key)
        return (client,)


class Playbook_Text2Video:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "luma_api_key": ("STRING", {"multiline": False}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "loop": ("BOOLEAN", {"default": False}),
                "aspect_ratio": ("STRING", {"default": "16:9", "multiline": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "Playbook 3D"

    def validate_aspect_ratio(self, aspect_ratio):
        """Validate the aspect ratio string format, e.g. '16:9'."""
        try:
            width, height = map(int, aspect_ratio.split(':'))
            if width <= 0 or height <= 0:
                raise ValueError("Aspect ratio values must be positive numbers")
            return True
        except ValueError:
            raise ValueError(
                "Invalid aspect ratio format. Must be 'W:H' (e.g., '16:9') with positive integers."
            )

    def run(self, luma_api_key, prompt, loop, aspect_ratio):
        """
        Generate a video from a text prompt, return frames as a 4D tensor.
        """
        if not prompt:
            raise ValueError("Prompt is required")
        
        validate_luma_api_key(luma_api_key)
        
        self.validate_aspect_ratio(aspect_ratio)
        client = LumaAI(auth_token=luma_api_key)

        print(f"Debug: Creating generation with prompt: {prompt}")
        g = client.generations.create(prompt=prompt, loop=loop, aspect_ratio=aspect_ratio)
        gen_id = g.id
        
        print(f"Debug: Waiting for generation {gen_id}")
        while True:
            g = client.generations.get(id=gen_id)
            if g.state == "completed":
                break
            if g.state == "failed":
                raise ValueError(f"Generation failed: {g.failure_reason}")
            time.sleep(3)

        print("Debug: Generation completed")
        video_url = g.assets.video
        temp_path = download_video_to_temp(video_url)

        # Convert .mp4 to frames
        print("Debug: Converting video to images")
        images = video_to_images(temp_path)
        if images is None:
            raise ValueError("Error: No frames extracted.")
            
        return (images,)


def get_jwt_from_playbook_key(api_key: str) -> str:
        """Get JWT access token from Playbook API key"""
        url = f"https://accounts.playbook3d.com/token-wrapper/get-tokens/{api_key}"
        response = requests.get(url)
        
        if response.status_code != 200:
            raise ValueError(f"Failed to get JWT token. Status: {response.status_code}")
            
        data = response.json()
        return data.get('access_token')
        

def get_run_id() -> str:
    """Get a unique run ID from Playbook API"""
    url = "https://api.playbook3d.com/get_run_id"
    response = requests.get(url)
    
    if response.status_code != 200:
        raise ValueError(f"Failed to get run ID. Status: {response.status_code}")
        
    data = response.json()
    return data.get('run_id')


class Playbook_Image2Video:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "luma_api_key": ("STRING", {"multiline": False}),
                "api_key": ("STRING", {"multiline": False}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "loop": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "init_image": ("IMAGE", {"forceInput": True}),
                "final_image": ("IMAGE", {"forceInput": True}),
                "run_id": ("STRING", {"default": ""})  # Added this
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "Playbook 3D"


    def upload_image_to_s3(self, image_tensor, api_key, run_id, node_id):
        """
        Upload image tensor to Playbook S3 and return the signed URL.
        Uses provided run_id if available, otherwise fetches one from the API.
        """
        try:
            # First get JWT using playbook key
            jwt_token = get_jwt_from_playbook_key(api_key)
            print(f"Debug - Got JWT token successfully")
            
            # Get run_id from API if not provided
            if not run_id:
                url = "https://api.playbook3d.com/get_run_id"
                response = requests.get(url)
                if response.status_code != 200:
                    raise ValueError(f"Failed to get run ID. Status: {response.status_code}")
                run_id = response.json().get('run_id')
                print(f"Debug - Using fallback run ID: {run_id}")
            else:
                print(f"Debug - Using provided run ID: {run_id}")
                
            # Convert tensor to PNG bytes
            if image_tensor.dim() == 4 and image_tensor.shape[0] == 1:
                image_tensor = image_tensor.squeeze(0)
            
            arr = (image_tensor.cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
            img = PILImage.fromarray(arr)
            
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            # Prepare multipart form data
            files = {
                'file': ('image.png', img_byte_arr, 'image/png')
            }
            
            # Use run_id in path (either provided or fetched)
            upload_path = f"{run_id}/{node_id}"
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
    
    

    def run(self, luma_api_key, api_key, prompt, loop, init_image=None, final_image=None, run_id=""):
        """
        Generate a video from one or two ComfyUI images plus a text prompt. 
        Returns frames as a 4D tensor.
        """
        if init_image is None and final_image is None:
            raise ValueError("At least one image is required (init or final).")
        
        validate_luma_api_key(luma_api_key)

        node_id = "dreammachinenode"
        print(f"Debug - Using node ID: {node_id}")

        client = LumaAI(auth_token=luma_api_key)

        # Upload images to S3 and get signed URLs
        keyframes = {}
        if init_image is not None:
            init_url = self.upload_image_to_s3(init_image, api_key, run_id, node_id)
            keyframes["frame0"] = {"type": "image", "url": init_url}
        if final_image is not None:
            final_url = self.upload_image_to_s3(final_image, api_key, run_id, node_id)
            keyframes["frame1"] = {"type": "image", "url": final_url}

        g = client.generations.create(prompt=prompt, loop=loop, keyframes=keyframes)
        gen_id = g.id

        while True:
            g = client.generations.get(id=gen_id)
            if g.state == "completed":
                break
            if g.state == "failed":
                raise ValueError(f"Generation failed: {g.failure_reason}")
            time.sleep(3)

        video_url = g.assets.video
        temp_path = download_video_to_temp(video_url)

        images = video_to_images(temp_path)
        if images is None:
            raise ValueError("Error: No frames extracted.")

        return (images,)

class Playbook_InterpolateGenerations:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "luma_api_key": ("STRING", {"multiline": False}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "generation_id_1": ("STRING", {"default": "", "forceInput": True}),
                "generation_id_2": ("STRING", {"default": "", "forceInput": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "Playbook 3D"

    def run(self, luma_api_key, prompt, generation_id_1, generation_id_2):
        """
        Generate a video by interpolating between two existing generation IDs.
        Returns frames as a 4D tensor.
        """
        if not generation_id_1 or not generation_id_2:
            raise ValueError("Both generation IDs are required")

        client = LumaAI(auth_token=luma_api_key)

        kf = {
            "frame0": {"type": "generation", "id": generation_id_1},
            "frame1": {"type": "generation", "id": generation_id_2},
        }

        g = client.generations.create(prompt=prompt, keyframes=kf)
        gen_id = g.id

        while True:
            g = client.generations.get(id=gen_id)
            if g.state == "completed":
                break
            if g.state == "failed":
                raise ValueError(f"Generation failed: {g.failure_reason}")
            time.sleep(3)

        video_url = g.assets.video
        temp_path = download_video_to_temp(video_url)

        images = video_to_images(temp_path)
        if images is None:
            raise ValueError("Error: No frames extracted.")

        return (images,)


class Playbook_ExtendGeneration:
    """
    Accept init_image and final_image as 'IMAGE' inputs, or generation IDs. 
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "luma_api_key": ("STRING", {"multiline": False}),
                "api_key": ("STRING", {"multiline": False}),  # Need to add this too for S3 upload
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "init_image": ("IMAGE", {"forceInput": True}),
                "final_image": ("IMAGE", {"forceInput": True}),
                "init_generation_id": ("STRING", {"default": "", "forceInput": True}),
                "final_generation_id": ("STRING", {"default": "", "forceInput": True}),
                "run_id": ("STRING", {"default": ""})  # Added this
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "Playbook 3D"

    def upload_image_to_s3(self, image_tensor, api_key, run_id, node_id):
        """
        Upload image tensor to Playbook S3 and return the signed URL.
        Uses provided run_id if available, otherwise fetches one from the API.
        """
        try:
            # First get JWT using playbook key
            jwt_token = get_jwt_from_playbook_key(api_key)
            print(f"Debug - Got JWT token successfully")
            
            # Get run_id from API if not provided
            if not run_id:
                url = "https://api.playbook3d.com/get_run_id"
                response = requests.get(url)
                if response.status_code != 200:
                    raise ValueError(f"Failed to get run ID. Status: {response.status_code}")
                run_id = response.json().get('run_id')
                print(f"Debug - Using fallback run ID: {run_id}")
            else:
                print(f"Debug - Using provided run ID: {run_id}")
                
            # Convert tensor to PNG bytes
            if image_tensor.dim() == 4 and image_tensor.shape[0] == 1:
                image_tensor = image_tensor.squeeze(0)
            
            arr = (image_tensor.cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
            img = PILImage.fromarray(arr)
            
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            # Prepare multipart form data
            files = {
                'file': ('image.png', img_byte_arr, 'image/png')
            }
            
            # Use run_id in path (either provided or fetched)
            upload_path = f"{run_id}/{node_id}"
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

    def run(
        self,
        luma_api_key,
        api_key,
        prompt,
        init_image=None,
        final_image=None,
        init_generation_id="",
        final_generation_id="",
        run_id=""
    ):
        """
        Extend from an init image/generation to a final image/generation. Returns frames as 4D tensor.
        """
        # Check presence of required inputs differently
        has_init = init_image is not None or bool(init_generation_id)
        has_final = final_image is not None or bool(final_generation_id)
        
        if not (has_init or has_final):
            raise ValueError("You must provide at least one side of extension (init/final).")
        
        # Check for conflicts
        if init_image is not None and init_generation_id:
            raise ValueError("Cannot provide both an init image and an init generation ID.")
        if final_image is not None and final_generation_id:
            raise ValueError("Cannot provide both a final image and a final generation ID.")

        node_id = "dreammachinenode"
        client = LumaAI(auth_token=luma_api_key)

        kf = {}
        if init_image is not None:
            init_url = self.upload_image_to_s3(init_image, api_key, run_id, node_id)
            kf["frame0"] = {"type": "image", "url": init_url}
        elif init_generation_id:
            kf["frame0"] = {"type": "generation", "id": init_generation_id}

        if final_image is not None:
            final_url = self.upload_image_to_s3(final_image, api_key, run_id, node_id)
            kf["frame1"] = {"type": "image", "url": final_url}
        elif final_generation_id:
            kf["frame1"] = {"type": "generation", "id": final_generation_id}

        g = client.generations.create(prompt=prompt, keyframes=kf)
        gen_id = g.id

        while True:
            g = client.generations.get(id=gen_id)
            if g.state == "completed":
                break
            if g.state == "failed":
                raise ValueError(f"Generation failed: {g.failure_reason}")
            time.sleep(3)

        video_url = g.assets.video
        temp_path = download_video_to_temp(video_url)

        images = video_to_images(temp_path)
        if images is None:
            raise ValueError("Error: No frames extracted.")

        return (images,)


class Playbook_PreviewVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_url": ("STRING", {"forceInput": True})
            }
        }

    FUNCTION = "run"
    CATEGORY = "Playbook 3D"
    RETURN_TYPES = ()
    OUTPUT_NODE = True

    def run(self, video_url):
        """
        Return a UI payload to preview the video in ComfyUI's interface.
        """
        return {"ui": {"video_url": [video_url]}}
