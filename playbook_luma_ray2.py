import os
import time
import folder_paths
from lumaai import LumaAI
from .lumaDreamMachineNode import (
    download_video_to_temp,
    video_to_images,
)


class Playbook_Ray2Text2Video:
    """
    Create a new video from a text prompt using the Ray 2 model.
    Returns frames as a 4D tensor for ComfyUI.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "loop": ("BOOLEAN", {"default": False}),
                "aspect_ratio": ("STRING", {"default": "16:9"}),
                "duration": ("STRING", {"default": "5s"}),
                "resolution": ("STRING", {"default": "540p"}),
                "save": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "filename": ("STRING", {"default": ""}),
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
        api_key,
        prompt,
        loop,
        aspect_ratio,
        duration,
        resolution,
        save,
        filename="",
    ):
        if not prompt:
            raise ValueError("Prompt is required.")

        # Validate aspect ratio
        self.validate_aspect_ratio(aspect_ratio)

        # Create Luma client directly with the user-provided key
        client = LumaAI(auth_token=api_key)

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

        # Poll until complete
        print(f"Debug: Waiting for Ray2 generation {gen_id}")
        while True:
            g = client.generations.get(id=gen_id)
            if g.state == "completed":
                break
            if g.state == "failed":
                raise ValueError(f"Generation failed: {g.failure_reason}")
            time.sleep(3)

        # Download video
        video_url = g.assets.video
        temp_path = download_video_to_temp(video_url)

        # Optionally save
        if save:
            out_dir = folder_paths.get_output_directory()
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            name = filename or gen_id
            final_path = os.path.join(out_dir, f"{name}.mp4")
            os.rename(temp_path, final_path)
            temp_path = final_path
            print(f"Debug: Saved Ray2 video to {final_path}")

        print("Debug: Converting video to images")
        images = video_to_images(temp_path)
        if images is None:
            raise ValueError("Error: No frames extracted.")
        # Optional debug prints for shape & dtype:
        print(f"Debug: Final tensor shape: {images.shape}")
        print(f"Debug: Final tensor dtype: {images.dtype}")

        return (images,)


class Playbook_Ray2Image2Video:
    """
    Create a new video from images (init & final) + a text prompt using the Ray 2 model.
    Returns frames as a 4D tensor.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "loop": ("BOOLEAN", {"default": False}),
                "duration": ("STRING", {"default": "5s"}),
                "resolution": ("STRING", {"default": "540p"}),
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

    def run(
        self,
        api_key,
        prompt,
        loop,
        duration,
        resolution,
        save,
        init_image_url="",
        final_image_url="",
        filename="",
    ):
        if not init_image_url and not final_image_url:
            raise ValueError("At least one image URL (init or final) is required.")

        # Create Luma client
        client = LumaAI(auth_token=api_key)

        # Build keyframes
        keyframes = {}
        if init_image_url:
            keyframes["frame0"] = {"type": "image", "url": init_image_url}
        if final_image_url:
            keyframes["frame1"] = {"type": "image", "url": final_image_url}

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

        # Poll
        print(f"Debug: Waiting for Ray2 generation {gen_id}")
        while True:
            g = client.generations.get(id=gen_id)
            if g.state == "completed":
                break
            if g.state == "failed":
                raise ValueError(f"Generation failed: {g.failure_reason}")
            time.sleep(3)

        # Download video
        video_url = g.assets.video
        temp_path = download_video_to_temp(video_url)

        # Optionally save
        if save:
            out_dir = folder_paths.get_output_directory()
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            name = filename or gen_id
            final_path = os.path.join(out_dir, f"{name}.mp4")
            os.rename(temp_path, final_path)
            temp_path = final_path
            print(f"Debug: Saved Ray2 Image2Video to {final_path}")

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
                "api_key": ("STRING", {"multiline": False}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "duration": ("STRING", {"default": "5s"}),
                "resolution": ("STRING", {"default": "540p"}),
                "save": ("BOOLEAN", {"default": True}),
                "generation_id_1": ("STRING", {"default": "", "forceInput": True}),
                "generation_id_2": ("STRING", {"default": "", "forceInput": True}),
            },
            "optional": {
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
        duration,
        resolution,
        save,
        generation_id_1,
        generation_id_2,
        filename="",
    ):
        if not generation_id_1 or not generation_id_2:
            raise ValueError("Both generation_id_1 and generation_id_2 are required.")

        # Create Luma client
        client = LumaAI(auth_token=api_key)

        # Keyframes referencing previous generations
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

        # Poll
        print(f"Debug: Waiting for Ray2 interpolation {gen_id}")
        while True:
            g = client.generations.get(id=gen_id)
            if g.state == "completed":
                break
            if g.state == "failed":
                raise ValueError(f"Generation failed: {g.failure_reason}")
            time.sleep(3)

        # Download
        video_url = g.assets.video
        temp_path = download_video_to_temp(video_url)

        # Optionally save
        if save:
            out_dir = folder_paths.get_output_directory()
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            name = filename or gen_id
            final_path = os.path.join(out_dir, f"{name}.mp4")
            os.rename(temp_path, final_path)
            temp_path = final_path
            print(f"Debug: Saved Ray2 interpolation video to {final_path}")

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
                "api_key": ("STRING", {"multiline": False}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "duration": ("STRING", {"default": "5s"}),
                "resolution": ("STRING", {"default": "540p"}),
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
        duration,
        resolution,
        save,
        init_image_url="",
        final_image_url="",
        init_generation_id="",
        final_generation_id="",
        filename="",
    ):
        if not init_generation_id and not final_generation_id:
            raise ValueError("You must provide at least one generation ID (init or final).")
        if init_image_url and init_generation_id:
            raise ValueError("Cannot provide both an init image and an init generation ID.")
        if final_image_url and final_generation_id:
            raise ValueError("Cannot provide both a final image and a final generation ID.")

        # Create Luma client
        client = LumaAI(auth_token=api_key)

        # Build keyframes
        kf = {}
        if init_image_url:
            kf["frame0"] = {"type": "image", "url": init_image_url}
        if final_image_url:
            kf["frame1"] = {"type": "image", "url": final_image_url}
        if init_generation_id:
            kf["frame0"] = {"type": "generation", "id": init_generation_id}
        if final_generation_id:
            kf["frame1"] = {"type": "generation", "id": final_generation_id}

        print("Debug: Creating Ray2 extension video")
        g = client.generations.create(
            prompt=prompt,
            model="ray-2",
            keyframes=kf,
            duration=duration,
            resolution=resolution,
        )
        gen_id = g.id

        # Poll
        print(f"Debug: Waiting for Ray2 extension {gen_id}")
        while True:
            g = client.generations.get(id=gen_id)
            if g.state == "completed":
                break
            if g.state == "failed":
                raise ValueError(f"Generation failed: {g.failure_reason}")
            time.sleep(3)

        # Download
        video_url = g.assets.video
        temp_path = download_video_to_temp(video_url)

        # Optionally save
        if save:
            out_dir = folder_paths.get_output_directory()
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            name = filename or gen_id
            final_path = os.path.join(out_dir, f"{name}.mp4")
            os.rename(temp_path, final_path)
            temp_path = final_path
            print(f"Debug: Saved Ray2 extension video to {final_path}")

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
