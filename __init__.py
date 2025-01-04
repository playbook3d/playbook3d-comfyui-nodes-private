from .playbookFalNodes import (
    Playbook_MiniMaxHailuo,
    Playbook_Kling,
)

from .playbookVideoNodes import PlaybookVideoPreview

from .lumaDreamMachineNode import (
    Playbook_LumaAIClient,
    Playbook_Text2Video,
    Playbook_Image2Video,
    Playbook_InterpolateGenerations,
    Playbook_ExtendGeneration,
    Playbook_PreviewVideo,
)

NODE_CLASS_MAPPINGS = {
    "Playbook MiniMaxHailuo": Playbook_MiniMaxHailuo,
    "Playbook Kling": Playbook_Kling,
    "Playbook Video Preview": PlaybookVideoPreview,
    "Playbook LumaAIClient": Playbook_LumaAIClient,
    "Playbook Text2Video": Playbook_Text2Video,
    "Playbook Image2Video": Playbook_Image2Video,
    "Playbook InterpolateGenerations": Playbook_InterpolateGenerations,
    "Playbook ExtendGeneration": Playbook_ExtendGeneration,
    "Playbook PreviewVideo": Playbook_PreviewVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Playbook MiniMaxHailuo": "Playbook MiniMax/Hailuo Video",
    "Playbook Kling": "Playbook Kling Video",
    "Playbook Video Preview": "Playbook Video Preview",
    "Playbook LumaAIClient": "Playbook LumaAI Client",
    "Playbook Text2Video": "Playbook Luma Video",
    "Playbook Image2Video": "Playbook Image to Video",
    "Playbook InterpolateGenerations": "Playbook Interpolate Generations",
    "Playbook ExtendGeneration": "Playbook Extend Generation",
    "Playbook PreviewVideo": "Playbook LumaAI Preview Video",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']