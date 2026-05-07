from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .nodes import RikanPromptRelayEncodeTimeline, RikanPromptRelayLoraGate, RikanPromptRelayPowerLoraGate
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override


class PromptRelay(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            PromptRelayEncodeTimeline,
            PromptRelayLoraGate,
            PromptRelayPowerLoraGate,
        ]


async def comfy_entrypoint() -> PromptRelay:
    return PromptRelay()

NODE_CLASS_MAPPINGS = {
    "RikanPromptRelayEncodeTimeline": RikanPromptRelayEncodeTimeline,
    "RikanPromptRelayLoraGate": RikanPromptRelayLoraGate,
    "RikanPromptRelayPowerLoraGate": RikanPromptRelayPowerLoraGate, 
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RikanPromptRelayEncodeTimeline": "Rikan Prompt Relay Encode (Timeline)",
    "RikanPromptRelayLoraGate": "Rikan Prompt Relay LoRA Gate",
    "RikanPromptRelayPowerLoraGate": "Rikan Prompt Relay Power Lora Gate",
}


WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
