import os
import time
import tempfile
import folder_paths
from lumaai import LumaAI
from .lumaDreamMachineNode import (
    download_video_to_temp,
    video_to_images,
)
import torch

def image_to_temp_url(image_tensor):
    """
    Placeholder: Writes the 'IMAGE' (4D or 3D PyTorch tensor) to a temporary file,
    returns a "file://" URL or local web server path.

    In a real solution, you'd upload to a hosting service or a local dev server
    that can serve the file at a public URL. For now, we just return "file://....".
    """
    if image_tensor.dim() == 4:
        # If shape is [1, H, W, C], remove batch dimension
        image_tensor = image_tensor.squeeze(0)
    # image_tensor now shape [H, W, C] in float32, range [0..1]
    # Convert to temp PNG
    import numpy as np
    from PIL import Image

    # Move to CPU if needed
    if not image_tensor.device.type == "cpu":
        image_tensor = image_tensor.cpu()

    arr = (image_tensor.numpy().clip(0, 1) * 255).astype("uint8")
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    Image.fromarray(arr).save(temp_file.name, format="PNG")
    temp_file.close()

    return f"file://{temp_file.name}"


class Playbook_Ray2Text2Video:
    """
    Create a new video from a text prompt using the Ray 2 model.
    Returns frames as a 4D tensor for ComfyUI.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "luma_api_key": ("STRING", {"multiline": False}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "loop": ("BOOLEAN", {"default": False}),
                "aspect_ratio": (
                    [
                        "16:9",
                        "9:16",
                        "1:1",
                        "4:3",
                        "3:4",
                        "21:9",
                    ],
                ),
                "duration": (
                    [
                        "5s",
                        "9s",
                    ],
                ),
                "resolution": (
                    [
                        "540p",
                        "720p",
                    ],
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "Playbook 3D"

    def validate_aspect_ratio(self, aspect_ratio):
        """Validate the aspect ratio string format (e.g. '16:9')."""
        try:
            width, height = map(int, aspect_ratio.split(':'))
            if width <= 0 or height <= 0:
                raise ValueError("Aspect ratio values must be positive integers.")
            return True
        except ValueError:
            raise ValueError("Invalid aspect ratio. Must be 'W:H' with positive integers.")

    def run(
        self,
        luma_api_key,
        prompt,
        loop,
        aspect_ratio,
        duration,
        resolution,
    ):
        if not prompt:
            raise ValueError("Prompt is required.")

        self.validate_aspect_ratio(aspect_ratio)
        client = LumaAI(auth_token=luma_api_key)

        print(f"Debug: Creating Ray2 video from text prompt: {prompt}")
        generation = client.generations.create(
            prompt=prompt,
            model="ray-2",
            loop=loop,
            aspect_ratio=aspect_ratio,
            duration=duration,
            resolution=resolution,
        )
        gen_id = generation.id

        print(f"Debug: Waiting for Ray2 generation {gen_id}")
        while True:
            g = client.generations.get(id=gen_id)
            if g.state == "completed":
                break
            if g.state == "failed":
                raise ValueError(f"Generation failed: {g.failure_reason}")
            time.sleep(3)

        # Download .mp4
        video_url = g.assets.video
        temp_path = download_video_to_temp(video_url)

        print("Debug: Converting video to images")
        images = video_to_images(temp_path)
        if images is None:
            raise ValueError("Error: No frames extracted.")
        print(f"Debug: Final tensor shape: {images.shape}")
        print(f"Debug: Final tensor dtype: {images.dtype}")

        return (images,)


class Playbook_Ray2Image2Video:
    """
    Create a new video from an init image to a final image + a prompt, using Ray2.
    Returns frames as a 4D tensor. 'init_image' and 'final_image' are ComfyUI IMAGES.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "luma_api_key": ("STRING", {"multiline": False}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "loop": ("BOOLEAN", {"default": False}),
                "duration": (
                    [
                        "5s",
                        "9s",
                    ],
                ),
                "resolution": (
                    [
                        "540p",
                        "720p",
                    ],
                ),
            },
            "optional": {
                "init_image": ("IMAGE", {"forceInput": True}),
                "final_image": ("IMAGE", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "Playbook 3D"

    def run(
        self,
        luma_api_key,
        prompt,
        loop,
        duration,
        resolution,
        init_image=None,
        final_image=None,
    ):
        # At least one image is required
        if init_image is None and final_image is None:
            raise ValueError("At least one image is required (init or final).")

        client = LumaAI(auth_token=luma_api_key)

        # Convert ComfyUI images to URLs for Luma
        keyframes = {}
        if init_image is not None:
            init_url = image_to_temp_url(init_image)
            keyframes["frame0"] = {"type": "image", "url": init_url}
        if final_image is not None:
            final_url = image_to_temp_url(final_image)
            keyframes["frame1"] = {"type": "image", "url": final_url}

        print(f"Debug: Creating Ray2 video from images + prompt: {prompt}")
        g = client.generations.create(
            prompt=prompt,
            model="ray-2",
            loop=loop,
            keyframes=keyframes,
            duration=duration,
            resolution=resolution,
        )
        gen_id = g.id

        print(f"Debug: Waiting for Ray2 generation {gen_id}")
        while True:
            g = client.generations.get(id=gen_id)
            if g.state == "completed":
                break
            if g.state == "failed":
                raise ValueError(f"Generation failed: {g.failure_reason}")
            time.sleep(3)

        video_url = g.assets.video
        temp_path = download_video_to_temp(video_url)

        print("Debug: Converting video to images")
        images = video_to_images(temp_path)
        if images is None:
            raise ValueError("Error: No frames extracted.")
        print(f"Debug: Final tensor shape: {images.shape}")
        print(f"Debug: Final tensor dtype: {images.dtype}")

        return (images,)


class Playbook_Ray2InterpolateGenerations:
    """
    Create a new Ray2 video by interpolating between two existing Luma generation IDs
    plus a text prompt. Returns frames as a 4D tensor.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "luma_api_key": ("STRING", {"multiline": False}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "duration": (
                    [
                        "5s",
                        "9s",
                    ],
                ),
                "resolution": (
                    [
                        "540p",
                        "720p",
                    ],
                ),
                "generation_id_1": ("STRING", {"default": "", "forceInput": True}),
                "generation_id_2": ("STRING", {"default": "", "forceInput": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "Playbook 3D"

    def run(
        self,
        luma_api_key,
        prompt,
        duration,
        resolution,
        generation_id_1,
        generation_id_2,
    ):
        if not generation_id_1 or not generation_id_2:
            raise ValueError("Both generation_id_1 and generation_id_2 are required.")

        client = LumaAI(auth_token=luma_api_key)

        kf = {
            "frame0": {"type": "generation", "id": generation_id_1},
            "frame1": {"type": "generation", "id": generation_id_2},
        }

        print("Debug: Creating Ray2 interpolation video")
        g = client.generations.create(
            prompt=prompt,
            model="ray-2",
            keyframes=kf,
            duration=duration,
            resolution=resolution,
        )
        gen_id = g.id

        print(f"Debug: Waiting for Ray2 interpolation {gen_id}")
        while True:
            g = client.generations.get(id=gen_id)
            if g.state == "completed":
                break
            if g.state == "failed":
                raise ValueError(f"Generation failed: {g.failure_reason}")
            time.sleep(3)

        video_url = g.assets.video
        temp_path = download_video_to_temp(video_url)

        print("Debug: Converting video to images")
        images = video_to_images(temp_path)
        if images is None:
            raise ValueError("Error: No frames extracted.")
        print(f"Debug: Final tensor shape: {images.shape}")
        print(f"Debug: Final tensor dtype: {images.dtype}")

        return (images,)


class Playbook_Ray2ExtendGeneration:
    """
    Extend or bridge from an image/generation to another image/generation with Ray2
    plus a text prompt. Returns frames as a 4D tensor.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "luma_api_key": ("STRING", {"multiline": False}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "duration": (
                    [
                        "5s",
                        "9s",
                    ],
                ),
                "resolution": (
                    [
                        "540p",
                        "720p",
                    ],
                ),
            },
            "optional": {
                # If you want to extend from or to an existing generation
                "init_generation_id": ("STRING", {"default": "", "forceInput": True}),
                "final_generation_id": ("STRING", {"default": "", "forceInput": True}),
                # If you want to extend from or to an IMAGE
                "init_image": ("IMAGE", {"forceInput": True}),
                "final_image": ("IMAGE", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "Playbook 3D"

    def run(
        self,
        luma_api_key,
        prompt,
        duration,
        resolution,
        init_generation_id="",
        final_generation_id="",
        init_image=None,
        final_image=None,
    ):
        # Must have at least one generation_id or one image
        # but can't supply both for the same "slot" (init or final).
        if not init_generation_id and not init_image and not final_generation_id and not final_image:
            raise ValueError("You must provide at least one init or final reference (image or generation).")

        if init_generation_id and init_image:
            raise ValueError("Cannot provide both an init image and init generation ID.")
        if final_generation_id and final_image:
            raise ValueError("Cannot provide both a final image and a final generation ID.")

        client = LumaAI(auth_token=luma_api_key)

        kf = {}
        # init
        if init_generation_id:
            kf["frame0"] = {"type": "generation", "id": init_generation_id}
        elif init_image is not None:
            init_url = image_to_temp_url(init_image)
            kf["frame0"] = {"type": "image", "url": init_url}
        # final
        if final_generation_id:
            kf["frame1"] = {"type": "generation", "id": final_generation_id}
        elif final_image is not None:
            final_url = image_to_temp_url(final_image)
            kf["frame1"] = {"type": "image", "url": final_url}

        print("Debug: Creating Ray2 extension video")
        g = client.generations.create(
            prompt=prompt,
            model="ray-2",
            keyframes=kf,
            duration=duration,
            resolution=resolution,
        )
        gen_id = g.id

        print(f"Debug: Waiting for Ray2 extension {gen_id}")
        while True:
            g = client.generations.get(id=gen_id)
            if g.state == "completed":
                break
            if g.state == "failed":
                raise ValueError(f"Generation failed: {g.failure_reason}")
            time.sleep(3)

        video_url = g.assets.video
        temp_path = download_video_to_temp(video_url)

        print("Debug: Converting video to images")
        images = video_to_images(temp_path)
        if images is None:
            raise ValueError("Error: No frames extracted.")
        print(f"Debug: Final tensor shape: {images.shape}")
        print(f"Debug: Final tensor dtype: {images.dtype}")

        return (images,)


class Playbook_Ray2PreviewVideo:
    """
    Same as the existing Playbook_PreviewVideo, but named for Ray2 usage.
    Exposes the video in ComfyUI's UI without returning frames.
    """
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
        # Tells ComfyUI to show this video in the UI panel
        return {"ui": {"video_url": [video_url]}}