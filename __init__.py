from .playbookFalNodes import (
    Playbook_MiniMaxHailuo,
    Playbook_Kling,
)

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
    "Playbook MiniMaxHailuo": Playbook_MiniMaxHailuo,
    "Playbook Kling": Playbook_Kling,
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
    "Playbook MiniMaxHailuo": "Playbook MiniMax/Hailuo Video",
    "Playbook Kling": "Playbook Kling Video",
    "Playbook LumaAIClient": "Playbook LumaAI Client",
    "Playbook Text2Video": "Playbook Luma Video",
    "Playbook Image2Video": "Playbook Image to Video",
    "Playbook InterpolateGenerations": "Playbook Interpolate Generations",
    "Playbook ExtendGeneration": "Playbook Extend Generation",
    "Playbook PreviewVideo": "Playbook LumaAI Preview Video",
    "Playbook Ray2Text2Video": "Playbook Ray2 Video (Text)",
    "Playbook Ray2Image2Video": "Playbook Ray2 Video (Image)",
    "Playbook Ray2InterpolateGenerations": "Playbook Ray2 Interpolate",
    "Playbook Ray2ExtendGeneration": "Playbook Ray2 Extend",
    "Playbook Ray2PreviewVideo": "Playbook Ray2 Preview",
    "Playbook PhotonText2Image": "Playbook Photon Image",
    "Playbook PhotonModifyImage": "Playbook Photon Modify Image",
    "Playbook PhotonPreviewImage": "Playbook Photon Preview",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']