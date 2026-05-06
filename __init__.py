from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .nodes import PromptRelayEncodeTimeline, PromptRelayLoraGate
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override


class PromptRelay(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            PromptRelayEncodeTimeline,
            PromptRelayLoraGate  # <--- Tambahkan di sini
        ]


async def comfy_entrypoint() -> PromptRelay:
    return PromptRelay()

NODE_CLASS_MAPPINGS = {
    "PromptRelayEncodeTimeline": PromptRelayEncodeTimeline,
    "PromptRelayLoraGate": PromptRelayLoraGate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptRelayEncodeTimeline": "Prompt Relay Encode (Timeline)",
    "PromptRelayLoraGate": "Prompt Relay LoRA Gate",
}


WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
