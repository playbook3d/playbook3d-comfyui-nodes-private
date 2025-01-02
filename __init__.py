# from .playbookFalNodes import (
#     Playbook_FalClient,
#     Playbook_MiniMaxHailuo,
#     Playbook_Kling,
# )
# from .playbookVideoNodes import PlaybookVideoPreview

# NODE_CLASS_MAPPINGS = {
#     "Playbook FalClient": Playbook_FalClient,
#     "Playbook MiniMaxHailuo": Playbook_MiniMaxHailuo,
#     "Playbook Kling": Playbook_Kling,
#     "Playbook Video Preview": PlaybookVideoPreview,

# }

# NODE_DISPLAY_NAME_MAPPINGS = {
#     "Playbook FalClient": "Playbook Fal Client",
#     "Playbook MiniMaxHailuo": "Playbook MiniMax/Hailuo Video",
#     "Playbook Kling": "Playbook Kling Video",
#     "Playbook Video Preview": "Playbook Video Preview",
# }

# __all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

from .playbookFalNodes import (
    Playbook_MiniMaxHailuo,
    Playbook_Kling,
)
from .playbookVideoNodes import PlaybookVideoPreview

NODE_CLASS_MAPPINGS = {
    "Playbook MiniMaxHailuo": Playbook_MiniMaxHailuo,
    "Playbook Kling": Playbook_Kling,
    "Playbook Video Preview": PlaybookVideoPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Playbook MiniMaxHailuo": "Playbook MiniMax/Hailuo Video",
    "Playbook Kling": "Playbook Kling Video",
    "Playbook Video Preview": "Playbook Video Preview",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
