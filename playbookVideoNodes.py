import os

class PlaybookVideoPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_info",)
    FUNCTION = "preview_video"
    CATEGORY = "Playbook/Video"
    OUTPUT_NODE = True

    def preview_video(self, video_path):
        if not video_path or not os.path.exists(video_path):
            return ("Video file not found",)
            
        try:
            file_size = os.path.getsize(video_path) / (1024 * 1024)  # Size in MB
            video_info = f"Video saved at: {video_path}\nSize: {file_size:.2f} MB"
            return (video_info,)
        except Exception as e:
            return (f"Error getting video info: {str(e)}",)