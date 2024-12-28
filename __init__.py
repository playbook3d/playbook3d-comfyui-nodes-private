from .playbookFalNodes import (
    Playbook_FalClient,
    Playbook_MiniMaxHailuo,
    Playbook_Kling,
)
from .playbookVideoPreview import LoadVideoFromFile

NODE_CLASS_MAPPINGS = {
    "Playbook FalClient": Playbook_FalClient,
    "Playbook MiniMaxHailuo": Playbook_MiniMaxHailuo,
    "Playbook Kling": Playbook_Kling,
    "Playbook Video Preview": LoadVideoFromFile
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Playbook FalClient": "Playbook Fal Client",
    "Playbook MiniMaxHailuo": "Playbook MiniMax/Hailuo Video",
    "Playbook Kling": "Playbook Kling Video",
    "Playbook Video Preview": "Video Preview"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']