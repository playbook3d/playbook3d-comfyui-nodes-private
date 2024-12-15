from .lumaDreamMachineNode import (
    Playbook_LumaAIClient,
    Playbook_Text2Video,
    Playbook_Image2Video,
    Playbook_InterpolateGenerations,
    Playbook_ExtendGeneration,
    Playbook_PreviewVideo,
)


NODE_CLASS_MAPPINGS = {
    "Playbook LumaAIClient": Playbook_LumaAIClient,
    "Playbook Text2Video": Playbook_Text2Video,
    "Playbook Image2Video": Playbook_Image2Video,
    "Playbook InterpolateGenerations": Playbook_InterpolateGenerations,
    "Playbook ExtendGeneration": Playbook_ExtendGeneration,
    "Playbook PreviewVideo": Playbook_PreviewVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Playbook LumaAIClient": "Playbook LumaAI Client",
    "Playbook Text2Video": "Playbook Text to Video",
    "Playbook Image2Video": "Playbook Image to Video",
    "Playbook InterpolateGenerations": "Playbook Interpolate Generations",
    "Playbook ExtendGeneration": "Playbook Extend Generation",
    "Playbook PreviewVideo": "Playbook LumaAI Preview Video",
}


__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']