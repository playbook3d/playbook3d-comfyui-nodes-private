import os
import cv2
import torch
import numpy as np

class LoadVideoFromFile:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"multiline": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("frames", "frame_count")
    FUNCTION = "load_video"
    CATEGORY = "Utils"

    def load_video(self, file_path):
        if not os.path.exists(file_path):
            raise ValueError("File does not exist: " + file_path)

        cap = cv2.VideoCapture(file_path)
        frames = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
            frames.append(frame_tensor)
            frame_count += 1

        cap.release()
        return (torch.stack(frames), frame_count)
