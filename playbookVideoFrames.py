# import os
# import time
# from PIL import Image

# class Playbook_DownloadFrames:
#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             "required": {
#                 # We expect a list of frames (images) from the previous node
#                 "frames": ("LIST", {}),
#             },
#         }

#     RETURN_TYPES = ("STRING",)
#     RETURN_NAMES = ("message",)
#     FUNCTION = "run"
#     CATEGORY = "Playbook Fal"

#     def run(self, frames):
#         """
#         Saves frames to the local ./output_frames folder, 
#         one image per file with a timestamp-based name.
#         """
#         output_dir = "output_frames"
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)

#         if not frames:
#             return ("No frames to save.",)

#         # Save each frame with a unique filename
#         for idx, frame in enumerate(frames):
#             if not isinstance(frame, Image.Image):
#                 print(f"Skipping item {idx}: not a valid PIL Image")
#                 continue
#             timestamp = int(time.time())
#             frame_path = os.path.join(output_dir, f"frame_{timestamp}_{idx}.png")
#             frame.save(frame_path, format="PNG")

#         return (f"Saved {len(frames)} frames to '{output_dir}'",)

import os
import time
from PIL import Image
import numpy as np

class Playbook_DownloadFrames:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Must be a list of frames (PIL Images in memory)
                "frames": ("LIST", {}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("message",)
    FUNCTION = "run"
    CATEGORY = "Playbook Fal"

    def run(self, frames):
        """
        Saves frames (PIL Images) to the local ./output_frames folder
        as PNGs, one file per frame.
        """
        output_dir = "output_frames"
        os.makedirs(output_dir, exist_ok=True)

        if not frames:
            return ("No frames received; nothing to save.",)

        saved_count = 0
        for idx, frame in enumerate(frames):
            if not isinstance(frame, Image.Image):
                print(f"Skipping item {idx}: Not a valid PIL Image object.")
                continue

            timestamp = int(time.time())
            filepath = os.path.join(output_dir, f"frame_{timestamp}_{idx}.png")
            frame.save(filepath, format="PNG")
            saved_count += 1

        message = f"Saved {saved_count} PNG frame(s) to '{output_dir}'."
        print(message)
        return (message,)



class Playbook_ExtractFirstFrame:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("LIST", {}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("first_frame",)
    FUNCTION = "run"
    CATEGORY = "Playbook Fal"

    def run(self, frames):
        # If no frames exist, return a dummy black image
        if not frames or not isinstance(frames, list):
            black = Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))
            return (black,)

        # Return the first frame from the list
        first_frame = frames[0]
        return (first_frame,)

