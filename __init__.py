from .lumaDreamMachineNode import (
    Playbook_LumaAIClient,
    Playbook_Text2Video,
    Playbook_Image2Video,
    Playbook_InterpolateGenerations,
    Playbook_ExtendGeneration,
    Playbook_PreviewVideo,
)

from .playbook_luma_ray2 import (
    Playbook_Ray2Text2Video,
    Playbook_Ray2Image2Video,
    Playbook_Ray2InterpolateGenerations,
    Playbook_Ray2ExtendGeneration,
    Playbook_Ray2PreviewVideo,
)

from .playbook_luma_photon import (
    Playbook_PhotonText2Image,
    Playbook_PhotonModifyImage,
    Playbook_PhotonPreviewImage,
)

NODE_CLASS_MAPPINGS = {
    "Playbook LumaAIClient": Playbook_LumaAIClient,
    "Playbook Text2Video": Playbook_Text2Video,
    "Playbook Image2Video": Playbook_Image2Video,
    "Playbook InterpolateGenerations": Playbook_InterpolateGenerations,
    "Playbook ExtendGeneration": Playbook_ExtendGeneration,
    "Playbook PreviewVideo": Playbook_PreviewVideo,
    "Playbook Ray2Text2Video": Playbook_Ray2Text2Video,
    "Playbook Ray2Image2Video": Playbook_Ray2Image2Video,
    "Playbook Ray2InterpolateGenerations": Playbook_Ray2InterpolateGenerations,
    "Playbook Ray2ExtendGeneration": Playbook_Ray2ExtendGeneration,
    "Playbook Ray2PreviewVideo": Playbook_Ray2PreviewVideo,
    "Playbook PhotonText2Image": Playbook_PhotonText2Image,
    "Playbook PhotonModifyImage": Playbook_PhotonModifyImage,
    "Playbook PhotonPreviewImage": Playbook_PhotonPreviewImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Playbook LumaAIClient": "Playbook LumaAI Client",
    "Playbook Text2Video": "Playbook Dream Machine Text to Video",
    "Playbook Image2Video": "Playbook Dream Machine Image to Video",
    "Playbook InterpolateGenerations": "Playbook Dream Machine Interpolate Generations",
    "Playbook ExtendGeneration": "Playbook Dream Machine Extend Generation",
    "Playbook PreviewVideo": "Playbook Dream Machine Preview Video",
    "Playbook Ray2Text2Video": "Playbook Ray2 Text to Video)",
    "Playbook Ray2Image2Video": "Playbook Ray2 Image to Video",
    "Playbook Ray2InterpolateGenerations": "Playbook Ray2 Interpolate",
    "Playbook Ray2ExtendGeneration": "Playbook Ray2 Extend",
    "Playbook Ray2PreviewVideo": "Playbook Ray2 Preview",
    "Playbook PhotonText2Image": "Playbook Photon Image",
    "Playbook PhotonModifyImage": "Playbook Photon Modify Image",
    "Playbook PhotonPreviewImage": "Playbook Photon Image Preview",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']