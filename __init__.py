from .playbookFalNodes import (
    Playbook_FalClient,
    Playbook_MiniMaxHailuo,
    Playbook_Kling,
    Playbook_Haiper,
)
from .playbookVideoPreview import LoadVideoFromFile

NODE_CLASS_MAPPINGS = {
    "Playbook FalClient": Playbook_FalClient,
    "Playbook MiniMaxHailuo": Playbook_MiniMaxHailuo,
    "Playbook Kling": Playbook_Kling,
    "Playbook Haiper": Playbook_Haiper,
    "Load Video From File": LoadVideoFromFile
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Playbook FalClient": "Playbook Fal Client",
    "Playbook MiniMaxHailuo": "Playbook MiniMax/Hailuo Video",
    "Playbook Kling": "Playbook Kling Video",
    "Playbook Haiper": "Playbook Haiper Video",
    "Load Video From File": "Load Video From File"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
