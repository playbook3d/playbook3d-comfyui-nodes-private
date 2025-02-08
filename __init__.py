from .lumaDreamMachineNode import (
    Playbook_LumaAIClient,
    Playbook_Text2Video,
    Playbook_Image2Video,
    Playbook_ExtendGeneration,
)

from .playbook_luma_ray2 import (
    Playbook_Ray2Text2Video,
)

from .playbook_luma_photon import (
    Playbook_PhotonText2Image,
    Playbook_PhotonModifyImage,
    
)

NODE_CLASS_MAPPINGS = {
    "Playbook LumaAIClient": Playbook_LumaAIClient,
    "Playbook Text2Video": Playbook_Text2Video,
    "Playbook Image2Video": Playbook_Image2Video,
    "Playbook ExtendGeneration": Playbook_ExtendGeneration,
    "Playbook Ray2Text2Video": Playbook_Ray2Text2Video,
    "Playbook PhotonText2Image": Playbook_PhotonText2Image,
    "Playbook PhotonModifyImage": Playbook_PhotonModifyImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Playbook LumaAIClient": "Playbook LumaAI Client",
    "Playbook Text2Video": "Playbook Dream Machine Text to Video",
    "Playbook Image2Video": "Playbook Dream Machine Image to Video",
    "Playbook ExtendGeneration": "Playbook Dream Machine Extend Generation",
    "Playbook Ray2Text2Video": "Playbook Ray2 Text to Video",
    "Playbook PhotonText2Image": "Playbook Photon Image",
    "Playbook PhotonModifyImage": "Playbook Photon Modify Image",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']