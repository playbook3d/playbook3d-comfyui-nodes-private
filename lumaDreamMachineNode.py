import os
import time
import cv2
import requests
import tempfile
from PIL import Image
from lumaai import LumaAI
import folder_paths
import torch
import numpy as np

def download_video_to_temp(video_url):
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
        r = requests.get(video_url, stream=True)
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                temp_video.write(chunk)
        return temp_video.name

def video_to_images(path):
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


class Playbook_LumaAIClient:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False})
            }
        }

    RETURN_TYPES = ("LUMACLIENT",)
    RETURN_NAMES = ("client",)
    FUNCTION = "run"
    CATEGORY = "Playbook 3D"

    def run(self, api_key):
        # Directly use the user-provided Luma key
        client = LumaAI(auth_token=api_key)
        return (client,)


class Playbook_Text2Video:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "loop": ("BOOLEAN", {"default": False}),
                "aspect_ratio": ("STRING", {"default": "16:9", "multiline": False}),
                "save": ("BOOLEAN", {"default": True}),
            },
            "optional": {"filename": ("STRING", {"default": ""})},
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
            raise ValueError("Invalid aspect ratio format. Must be two positive numbers separated by ':' (e.g., '16:9')")

    def run(self, api_key, prompt, loop, aspect_ratio, save, filename):
        if not prompt:
            raise ValueError("Prompt is required")
        
        self.validate_aspect_ratio(aspect_ratio)
        client = LumaAI(auth_token=api_key)

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
        
        if save:
            out_dir = folder_paths.get_output_directory()
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            name = filename or gen_id
            final_path = os.path.join(out_dir, f"{name}.mp4")
            os.rename(temp_path, final_path)
            temp_path = final_path
            print(f"Debug: Saved video to {final_path}")

        print("Debug: Converting video to images")
        images = video_to_images(temp_path)
        if images is None:
            raise ValueError("Error: No frames extracted.")
            
        return (images,)


class Playbook_Image2Video:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "loop": ("BOOLEAN", {"default": False}),
                "save": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "init_image_url": ("STRING", {"default": "", "forceInput": True}),
                "final_image_url": ("STRING", {"default": "", "forceInput": True}),
                "filename": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "Playbook 3D"

    def run(self, api_key, prompt, loop, save, init_image_url="", final_image_url="", filename=""):
        if not init_image_url and not final_image_url:
            raise ValueError("At least one image URL is required")

        client = LumaAI(auth_token=api_key)

        keyframes = {}
        if init_image_url:
            keyframes["frame0"] = {"type": "image", "url": init_image_url}
        if final_image_url:
            keyframes["frame1"] = {"type": "image", "url": final_image_url}

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
        
        if save:
            out_dir = folder_paths.get_output_directory()
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            name = filename or gen_id
            final_path = os.path.join(out_dir, f"{name}.mp4")
            os.rename(temp_path, final_path)
            temp_path = final_path

        images = video_to_images(temp_path)
        if images is None:
            raise ValueError("Error: No frames extracted.")

        return (images,)


class Playbook_InterpolateGenerations:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "save": ("BOOLEAN", {"default": True}),
                "generation_id_1": ("STRING", {"default": "", "forceInput": True}),
                "generation_id_2": ("STRING", {"default": "", "forceInput": True}),
            },
            "optional": {"filename": ("STRING", {"default": ""})},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "Playbook 3D"

    def run(self, api_key, prompt, save, generation_id_1, generation_id_2, filename=""):
        if not generation_id_1 or not generation_id_2:
            raise ValueError("Both generation IDs are required")

        client = LumaAI(auth_token=api_key)

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
        
        if save:
            out_dir = folder_paths.get_output_directory()
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            name = filename or gen_id
            final_path = os.path.join(out_dir, f"{name}.mp4")
            os.rename(temp_path, final_path)
            temp_path = final_path

        images = video_to_images(temp_path)
        if images is None:
            raise ValueError("Error: No frames extracted.")

        return (images,)


class Playbook_ExtendGeneration:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "save": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "init_image_url": ("STRING", {"default": "", "forceInput": True}),
                "final_image_url": ("STRING", {"default": "", "forceInput": True}),
                "init_generation_id": ("STRING", {"default": "", "forceInput": True}),
                "final_generation_id": ("STRING", {"default": "", "forceInput": True}),
                "filename": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "Playbook 3D"

    def run(
        self,
        api_key,
        prompt,
        save,
        init_image_url="",
        final_image_url="",
        init_generation_id="",
        final_generation_id="",
        filename="",
    ):
        if not init_generation_id and not final_generation_id:
            raise ValueError("You must provide at least one generation id")
        if init_image_url and init_generation_id:
            raise ValueError("Cannot provide both an init image and an init generation")
        if final_image_url and final_generation_id:
            raise ValueError("Cannot provide both a final image and a final generation")

        client = LumaAI(auth_token=api_key)

        kf = {}
        if init_image_url:
            kf["frame0"] = {"type": "image", "url": init_image_url}
        if final_image_url:
            kf["frame1"] = {"type": "image", "url": final_image_url}
        if init_generation_id:
            kf["frame0"] = {"type": "generation", "id": init_generation_id}
        if final_generation_id:
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
        
        if save:
            out_dir = folder_paths.get_output_directory()
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            name = filename or gen_id
            final_path = os.path.join(out_dir, f"{name}.mp4")
            os.rename(temp_path, final_path)
            temp_path = final_path

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
        return {"ui": {"video_url": [video_url]}}
