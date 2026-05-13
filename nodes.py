import logging
import math
import types
import torch
import torch.nn as nn
import numpy as np
import base64
import os
import time
import io as pyio
from PIL import Image
import comfy.utils
import folder_paths
import node_helpers
import comfy.model_management
import comfy.ldm.modules.attention
import json
from comfy_api.latest import io, ComfyExtension
from typing_extensions import override

log = logging.getLogger(__name__)

# ==============================================================================
# 1. CORE UTILS & MATH
# ==============================================================================

def build_temporal_cost(q_token_idx, Lq, Lk, device, dtype, tokens_per_frame):
    offset = torch.zeros(Lq, Lk, device=device, dtype=dtype)
    query_frames = torch.arange(Lq, device=device, dtype=torch.long) // tokens_per_frame
    for seg in q_token_idx:
        local = seg["local_token_idx"].to(device=device)
        d = (query_frames.float()[:, None] - seg["midpoint"]).abs()
        strength = seg.get("strength", 1.0)
        cost = strength * (torch.relu(d - seg["window"]) ** 2) / (2 * seg["sigma"] ** 2)
        offset[:, local] = cost.to(offset.dtype)
    return offset

def build_temporal_cost_scaled(q_token_idx, Lq, Lk, device, dtype, latent_frames):
    offset = torch.zeros(Lq, Lk, device=device, dtype=dtype)
    query_frames = torch.arange(Lq, device=device, dtype=torch.float32) * latent_frames / Lq
    for seg in q_token_idx:
        local = seg["local_token_idx"].to(device=device)
        d = (query_frames[:, None] - seg["midpoint"]).abs()
        sigma_a = seg.get("sigma_audio", seg["sigma"])
        window_a = seg.get("window_audio", seg["window"])
        strength_a = seg.get("strength_audio", 1.0)
        cost = strength_a * (torch.relu(d - window_a) ** 2) / (2 * sigma_a ** 2)
        offset[:, local] = cost.to(offset.dtype)
    return offset

def create_mask_fn(q_token_idx, fallback_tokens_per_frame, latent_frames):
    cache = {}
    max_token_idx = max(int(seg["local_token_idx"].max().item()) for seg in q_token_idx) + 1 if q_token_idx else 0
    def mask_fn(q, k, transformer_options):
        Lq, Lk = q.shape[1], k.shape[1]
        if Lq == Lk: return None
        cond_or_uncond = transformer_options.get("cond_or_uncond", [])
        if 1 in cond_or_uncond and 0 not in cond_or_uncond: return None
        grid_sizes = transformer_options.get("grid_sizes", None)
        video_tpf = int(grid_sizes[1]) * int(grid_sizes[2]) if grid_sizes is not None else fallback_tokens_per_frame
        video_lq = latent_frames * video_tpf
        if Lk == video_lq or Lk < max_token_idx: return None
        mode = "video" if Lq == video_lq else "scaled"
        key = (Lq, Lk, mode, q.device)
        if key not in cache:
            if mode == "video":
                cost = build_temporal_cost(q_token_idx, Lq, Lk, q.device, q.dtype, video_tpf)
            else:
                cost = build_temporal_cost_scaled(q_token_idx, Lq, Lk, q.device, q.dtype, latent_frames)
            cache[key] = -cost
            
            nonzero = (cost > 0).sum().item()
            total = Lq * Lk
            log.info(f"[RikanPromptRelay] Built penalty matrix ({mode}): Lq={Lq}, Lk={Lk}, nonzero={nonzero}/{total}")
            
        return cache[key].to(q.dtype)
    return mask_fn

def build_segments(token_ranges, segment_lengths, epsilon=1e-3):
    sigma = 1.0 / math.log(1.0 / epsilon) if 0 < epsilon < 1 else 0.1448
    q_token_idx = []
    frame_cursor = 0
    for (tok_start, tok_end), L in zip(token_ranges, segment_lengths):
        if L <= 0:
            frame_cursor += L
            continue
        midpoint = (2 * frame_cursor + L) // 2
        base_window = max(L // 2 - 2, 0)
        q_token_idx.append({
            "local_token_idx": torch.arange(tok_start, tok_end),
            "midpoint": midpoint,
            "window": float(base_window),
            "sigma": sigma,
            "strength": 1.0,
            "window_audio": float(base_window),
            "sigma_audio": sigma,
            "strength_audio": 1.0,
        })
        frame_cursor += L
    return q_token_idx

def get_raw_tokenizer(clip):
    tokenizer_wrapper = clip.tokenizer
    for attr_name in dir(tokenizer_wrapper):
        if attr_name.startswith("_"): continue
        inner = getattr(tokenizer_wrapper, attr_name, None)
        if inner is not None and hasattr(inner, "tokenizer"):
            return inner.tokenizer
    raise RuntimeError("Could not find raw tokenizer.")

def map_token_indices(raw_tokenizer, global_prompt, local_prompts):
    prefixed_locals = [" " + lp for lp in local_prompts]
    has_eos = getattr(raw_tokenizer, "add_eos", False)
    eos_adj = 1 if has_eos else 0
    prev_len = len(raw_tokenizer(global_prompt)["input_ids"]) - eos_adj
    token_ranges = []
    built = global_prompt
    for plp in prefixed_locals:
        built += plp
        cur_len = len(raw_tokenizer(built)["input_ids"]) - eos_adj
        token_ranges.append((prev_len, cur_len))
        prev_len = cur_len
    return global_prompt + "".join(prefixed_locals), token_ranges

def distribute_segment_lengths(num_segments, latent_frames, specified_lengths=None):
    if specified_lengths:
        lengths = specified_lengths
    else:
        step = -(-latent_frames // num_segments)
        lengths = [step] * num_segments
    effective, cursor = [], 0
    for L in lengths:
        end = min(cursor + L, latent_frames)
        effective.append(max(end - cursor, 0))
        cursor = end
    return effective

def _convert_to_latent_lengths(pixel_lengths, temporal_stride, latent_frames):
    if not pixel_lengths: return []
    total_pixel = sum(pixel_lengths)
    if total_pixel <= 0: return [1] * len(pixel_lengths)
    target_total = min(latent_frames, max(1, round(total_pixel / temporal_stride)))
    if target_total >= latent_frames - 1: target_total = latent_frames
    exact = [p * target_total / total_pixel for p in pixel_lengths]
    result = [int(e) for e in exact]
    diff = target_total - sum(result)
    if diff > 0:
        order = sorted(range(len(exact)), key=lambda i: -(exact[i] - int(exact[i])))
        for k in range(diff): result[order[k % len(order)]] += 1
    return result

# ==============================================================================
# 2. MODEL PATCHING & LORA GATE HELPERS
# ==============================================================================

def detect_model_type(model):
    diff_model = model.model.diffusion_model
    if hasattr(diff_model, "patch_size") and not hasattr(diff_model, "patchifier"):
        return "wan", tuple(diff_model.patch_size), 4
    if hasattr(diff_model, "patchifier"):
        return "ltx", (1, 1, 1), int(diff_model.vae_scale_factors[0])
    raise ValueError("Unsupported model type.")

class _CrossAttnPatch:
    def __init__(self, impl, mask_fn):
        self.impl, self.mask_fn = impl, mask_fn
    def __get__(self, obj, objtype=None):
        def wrapped(self_module, *args, **kwargs):
            return self.impl(self_module, self.mask_fn, *args, **kwargs)
        return types.MethodType(wrapped, obj)

def apply_patches(model_clone, arch, mask_fn):
    diffusion_model = model_clone.get_model_object("diffusion_model")
    if arch == "wan":
        from comfy.ldm.wan.model import WanI2VCrossAttention
        for idx, block in enumerate(diffusion_model.blocks):
            impl = _wan_i2v_forward if isinstance(block.cross_attn, WanI2VCrossAttention) else _wan_t2v_forward
            model_clone.add_object_patch(f"diffusion_model.blocks.{idx}.cross_attn.forward", _CrossAttnPatch(impl, mask_fn).__get__(block.cross_attn))
    elif arch == "ltx":
        for idx, block in enumerate(diffusion_model.transformer_blocks):
            for attr in ("attn2", "audio_attn2"):
                module = getattr(block, attr, None)
                if module: model_clone.add_object_patch(f"diffusion_model.transformer_blocks.{idx}.{attr}.forward", _CrossAttnPatch(_ltx_forward, mask_fn).__get__(module))

def _wan_t2v_forward(self, mask_fn, x, context, transformer_options={}, **kwargs):
    q, k, v = self.norm_q(self.q(x)), self.norm_k(self.k(context)), self.v(context)
    mask = mask_fn(q, k, transformer_options)
    return self.o(comfy.ldm.modules.attention.attention_pytorch(q, k, v, self.num_heads, mask=mask, transformer_options=transformer_options) if mask is not None else comfy.ldm.modules.attention.optimized_attention(q, k, v, self.num_heads, transformer_options=transformer_options))

def _wan_i2v_forward(self, mask_fn, x, context, context_img_len, transformer_options={}, **kwargs):
    q = self.norm_q(self.q(x))
    img_x = comfy.ldm.modules.attention.optimized_attention(q, self.norm_k_img(self.k_img(context[:, :context_img_len])), self.v_img(context[:, :context_img_len]), self.num_heads, transformer_options=transformer_options)
    mask = mask_fn(q, self.norm_k(self.k(context[:, context_img_len:])), transformer_options)
    attn_txt = comfy.ldm.modules.attention.attention_pytorch(q, self.norm_k(self.k(context[:, context_img_len:])), self.v(context[:, context_img_len:]), self.num_heads, mask=mask, transformer_options=transformer_options) if mask is not None else comfy.ldm.modules.attention.optimized_attention(q, self.norm_k(self.k(context[:, context_img_len:])), self.v(context[:, context_img_len:]), self.num_heads, transformer_options=transformer_options)
    return self.o(attn_txt + img_x)

def _ltx_forward(self, mask_fn, x, context=None, mask=None, pe=None, k_pe=None, transformer_options={}):
    from comfy.ldm.lightricks.model import apply_rotary_emb
    is_self = context is None
    c = x if is_self else context
    q, k, v = self.q_norm(self.to_q(x)), self.k_norm(self.to_k(c)), self.to_v(c)
    if pe is not None: q, k = apply_rotary_emb(q, pe), apply_rotary_emb(k, pe if k_pe is None else k_pe)
    if not is_self:
        tm = mask_fn(q, k, transformer_options)
        mask = tm if mask is None else mask + tm
    out = comfy.ldm.modules.attention.optimized_attention(q, k, v, self.heads, mask=mask, transformer_options=transformer_options)
    if self.to_gate_logits: out = out * (2.0 * torch.sigmoid(self.to_gate_logits(x))).unsqueeze(-1)
    return self.to_out(out)

# ── LORA GATE INTERNAL CLASSES ───────────────────────────────────────────────

class _TemporalGatedLoRA:
    def __init__(self, original, lora_down, lora_up, scale, strength, gate_weights, tokens_per_frame):
        self.original = original
        self._lora_down = lora_down.cpu()
        self._lora_up = lora_up.cpu()
        self.scale = scale
        self.strength = strength
        self._gate_weights = gate_weights.cpu()
        self.tokens_per_frame = tokens_per_frame

    def forward(self, x, *args, **kwargs):
        base = self.original(x, *args, **kwargs)
        lora_down = self._lora_down.to(device=x.device, dtype=x.dtype)
        lora_up = self._lora_up.to(device=x.device, dtype=x.dtype)

        delta = (x @ lora_down.T) @ lora_up.T
        delta = delta * (self.scale * self.strength)

        if x.dim() < 2: return base + delta

        seq_len = x.shape[-2]
        n_frames = len(self._gate_weights)
        n_video_tokens = n_frames * self.tokens_per_frame

        if seq_len >= n_video_tokens and n_video_tokens > 0:
            gate_weights = self._gate_weights.to(device=x.device, dtype=x.dtype)
            frame_idx = torch.arange(n_video_tokens, device=x.device) // self.tokens_per_frame
            frame_idx = frame_idx.clamp(0, n_frames - 1)
            video_gates = gate_weights[frame_idx]
            
            if seq_len > n_video_tokens:
                extra = torch.zeros(seq_len - n_video_tokens, device=x.device, dtype=x.dtype)
                token_gates = torch.cat([video_gates, extra])
            else:
                token_gates = video_gates
            
            view_shape = (1,) * (x.dim() - 2) + (seq_len, 1)
            delta = delta * token_gates.view(view_shape)

        return base + delta

def _build_gate_weights(seg, latent_frames):
    midpoint, window, sigma = float(seg["midpoint"]), float(seg["window"]), float(seg["sigma"])
    weights = []
    for frame in range(latent_frames):
        d = abs(frame - midpoint)
        cost = (max(d - window, 0.0) ** 2) / (2.0 * sigma ** 2)
        weights.append(math.exp(-cost))
    return torch.tensor(weights, dtype=torch.float32)

def _get_module(diffusion_model, local_path):
    m = diffusion_model
    for part in local_path.split("."): m = getattr(m, part)
    return m

def _apply_gated_lora(model_clone, lora_path, segment, strength, tokens_per_frame, latent_frames):
    try:
        try: import comfy.loras as _lora_mod
        except ImportError: import comfy.lora as _lora_mod
    except ImportError:
        log.error("[Rikan MultiLoraGate] comfy.loras is not available.")
        return

    lora_file = comfy.utils.load_torch_file(lora_path, safe_load=True)
    key_map = {}
    try:
        _lora_mod.model_lora_keys_unet(model_clone.model, key_map)
    except Exception as e:
        log.error("[Rikan MultiLoraGate] Failed to create LoRA key map: %s", e)
        return

    try:
        patches = _lora_mod.load_lora(lora_file, key_map)
    except Exception as e:
        log.error("[Rikan MultiLoraGate] Failed to parse LoRA file: %s", e)
        return

    gate_weights = _build_gate_weights(segment, latent_frames)
    diffusion_model = model_clone.get_model_object("diffusion_model")
    
    applied = 0
    skipped_not_weight = 0
    skipped_not_diffusion = 0
    skipped_bad_data = 0
    skipped_not_linear = 0
    _logged_patch_sample = False

    for param_key, patch_data in patches.items():
        if not _logged_patch_sample:
            log.info("[Rikan MultiLoraGate] patch_data sample — key=%s attrs=%s", param_key, [a for a in dir(patch_data) if not a.startswith("_")])
            log.info("[Rikan MultiLoraGate] patch_data.weights=%s multiplier=%s", repr(patch_data.weights), repr(patch_data.multiplier))
            _logged_patch_sample = True
            
        if not param_key.endswith(".weight"):
            skipped_not_weight += 1
            continue
            
        module_path = param_key[:-7]
        if not module_path.startswith("diffusion_model."):
            skipped_not_diffusion += 1
            continue
        
        local_path = module_path[len("diffusion_model."):]
        
        try:
            lora_up = patch_data.weights[0]
            lora_down = patch_data.weights[1]
            alpha = patch_data.multiplier
        except (TypeError, IndexError, AttributeError):
            skipped_bad_data += 1
            continue
        
        if lora_up is None or lora_down is None:
            skipped_bad_data += 1
            continue
            
        if lora_down.dim() != 2 or lora_up.dim() != 2:
            skipped_not_linear += 1
            continue
        
        rank = lora_down.shape[0]
        scale = float(alpha) / rank if alpha is not None else 1.0
        
        target = model_clone.object_patches.get(module_path)
        if target is None:
            try:
                target = _get_module(diffusion_model, local_path)
            except AttributeError:
                skipped_not_diffusion += 1
                continue

        wrapped = _TemporalGatedLoRA(target.forward, lora_down, lora_up, scale, strength, gate_weights, tokens_per_frame)
        model_clone.add_object_patch(module_path + ".forward", wrapped.forward)
        applied += 1

    log.info(
        "[Rikan MultiLoraGate] Segment midpoint=%.1f window=%.1f — applied %d, not_weight=%d, not_diffusion=%d, bad_data=%d, not_linear=%d",
        float(segment["midpoint"]), float(segment["window"]), applied,
        skipped_not_weight, skipped_not_diffusion, skipped_bad_data, skipped_not_linear,
    )

# ==============================================================================
# 3. NODES
# ==============================================================================

class RikanPromptRelayEncodeTimeline(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RikanPromptRelayEncodeTimeline",
            display_name="Rikan Prompt Relay Encode (Timeline)",
            category="Rikannodes",
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Latent.Input("latent"),
                io.String.Input("global_prompt", multiline=True, default=""),
                io.Int.Input("max_frames", default=129, min=1, max=10000),
                io.String.Input("timeline_data", default=""),
                io.String.Input("local_prompts", multiline=True, default=""),
                io.String.Input("segment_lengths", default=""),
                io.Float.Input("epsilon", default=1e-3, min=1e-6, max=0.99, step=1e-4),
                io.Float.Input("fps", default=24.0, min=0.1, max=240.0, optional=True),
                io.Float.Input("seconds", default=5.375, min=0.1, max=1000.0, step=0.1, optional=True),
                io.Combo.Input("time_units", options=["frames", "seconds"], default="frames", optional=True),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Conditioning.Output(display_name="positive"),
                io.Float.Output(display_name="fps_out"),
                io.Int.Output(display_name="max_frames_out"),
            ],
        )

    @classmethod
    def execute(cls, model, clip, latent, global_prompt, max_frames, timeline_data, local_prompts, segment_lengths, epsilon, fps=24.0, seconds=5.375, time_units="frames") -> io.NodeOutput:
        locals_list = [p.strip() for p in local_prompts.split("|") if p.strip()]
        if not locals_list: raise ValueError("At least one local prompt is required.")

        arch, patch_size, temporal_stride = detect_model_type(model)
        samples = latent["samples"]
        latent_frames = samples.shape[2]
        tokens_per_frame = (samples.shape[3] // patch_size[1]) * (samples.shape[4] // patch_size[2])

        parsed_lengths = None
        if segment_lengths.strip():
            pixel_lengths = [int(x.strip()) for x in segment_lengths.split(",") if x.strip()]
            parsed_lengths = _convert_to_latent_lengths(pixel_lengths, temporal_stride, latent_frames)

        full_prompt, token_ranges = map_token_indices(get_raw_tokenizer(clip), global_prompt, locals_list)
        
        log.info("[RikanPromptRelay] Global: tokens [0:%d] (%d tokens)", token_ranges[0][0], token_ranges[0][0])
        for i, (s, e) in enumerate(token_ranges):
            log.info("[RikanPromptRelay] Segment %d: tokens [%d:%d] (%d tokens)", i, s, e, e - s)

        conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(full_prompt))
        effective_lengths = distribute_segment_lengths(len(locals_list), latent_frames, parsed_lengths)

        log.info(
            "[RikanPromptRelay] Latent: %d frames, %d tokens/frame, segments: %s",
            latent_frames, tokens_per_frame, effective_lengths,
        )

        q_token_idx = build_segments(token_ranges, effective_lengths, epsilon)
        mask_fn = create_mask_fn(q_token_idx, tokens_per_frame, latent_frames)

        patched = model.clone()
        apply_patches(patched, arch, mask_fn)

        for key in ["rikan_pr_segments", "pr_segments"]: patched.model_options[key] = q_token_idx
        for key in ["rikan_pr_latent_frames", "pr_latent_frames"]: patched.model_options[key] = latent_frames
        for key in ["rikan_pr_tokens_per_frame", "pr_tokens_per_frame"]: patched.model_options[key] = tokens_per_frame

        print(f"\n[Timeline] SUCCESS: Injected {len(q_token_idx)} segments.\n")
        return io.NodeOutput(patched, conditioning, float(fps), int(max_frames))

class RikanPromptRelayMultiLoraGate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RikanPromptRelayMultiLoraGate",
            display_name="Rikan Prompt Relay Multi LoRA Gate",
            category="Rikannodes",
            inputs=[
                io.Model.Input("model"),
                io.String.Input("lora_data", default="[]", multiline=True),
            ],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, model, lora_data) -> io.NodeOutput:
        segments = model.model_options.get("rikan_pr_segments")
        latent_frames = model.model_options.get("rikan_pr_latent_frames")
        tokens_per_frame = model.model_options.get("rikan_pr_tokens_per_frame")

        if segments is None:
            log.warning("[Rikan MultiLoraGate] Prompt Relay metadata not found. Ensure it is connected to the Timeline node.")
            return io.NodeOutput(model)

        try:
            loras_to_apply = json.loads(lora_data)
        except Exception as e:
            log.error(f"[Rikan MultiLoraGate] Failed to process JSON data: {e}")
            return io.NodeOutput(model)

        patched = model.clone()
        total_segments = len(segments)

        for lora in loras_to_apply:
            if not lora.get("enable", False) or lora.get("name") == "None":
                continue

            lora_name = lora.get("name")
            seg_idx = int(lora.get("segment", 0)) 
            strength = float(lora.get("modelStr", 1.0))

            if seg_idx < 0 or seg_idx >= total_segments:
                log.warning(f"[Rikan MultiLoraGate] SKIP: LoRA '{lora_name}' attempts to use Segment {seg_idx}, "
                            f"but the Timeline only has {total_segments} segment(s) (Available indices 0 to {total_segments-1}).")
                continue

            lora_path = folder_paths.get_full_path("loras", lora_name)
            if lora_path is None:
                log.warning(f"[Rikan MultiLoraGate] LoRA file not found: {lora_name}")
                continue

            _apply_gated_lora(patched, lora_path, segments[seg_idx], strength, tokens_per_frame, latent_frames)

            patched.model_options["rikan_pr_segments"] = segments
            patched.model_options["rikan_pr_latent_frames"] = latent_frames
            patched.model_options["rikan_pr_tokens_per_frame"] = tokens_per_frame

        return io.NodeOutput(patched)


class RikanI2VPainterTiledVAE(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="rikan-i2vpainter-tiled-vae",
            display_name="Rikan I2V Painter (Tiled VAE)",
            category="Rikannodes",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=832, min=16, max=4096, step=16),
                io.Int.Input("height", default=480, min=16, max=4096, step=16),
                io.Int.Input("length", default=81, min=1, max=4096, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.Int.Input("tile_size", default=512, min=64, max=4096, step=64),
                io.Int.Input("overlap", default=64, min=0, max=4096, step=32),
                io.Int.Input("temporal_size", default=64, min=8, max=4096, step=4),
                io.Int.Input("temporal_overlap", default=24, min=4, max=4096, step=4),
                io.Float.Input("motion_amplitude", default=1.3, min=1.0, max=2.0, step=0.05),
                io.Boolean.Input("color_protect", default=True),
                io.Float.Input("correct_strength", default=0.01, min=0.0, max=0.3, step=0.01),
                io.ClipVisionOutput.Input("clip_vision", optional=True),
                io.Image.Input("start_image", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="high_positive"),
                io.Conditioning.Output(display_name="high_negative"),
                io.Conditioning.Output(display_name="low_positive"),
                io.Conditioning.Output(display_name="low_negative"),
                io.Latent.Output(display_name="latent"),
            ]
        )

    @classmethod
    def execute(cls, positive, negative, vae, width, height, length, batch_size,
                tile_size=512, overlap=64, temporal_size=64, temporal_overlap=8,
                motion_amplitude=1.3, color_protect=True, correct_strength=0.01, 
                start_image=None, clip_vision=None) -> io.NodeOutput:
        
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], 
                           device=comfy.model_management.intermediate_device())
        
        positive_original = positive
        negative_original = negative
        
        if start_image is not None:
            start_image = start_image[:1]
            start_image = comfy.utils.common_upscale(start_image.movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            image = torch.ones((length, height, width, 3), device=start_image.device, dtype=start_image.dtype) * 0.5
            image[0] = start_image[0]
            
            concat_latent_image = vae.encode_tiled(image, tile_x=tile_size, tile_y=tile_size, overlap=overlap, tile_t=temporal_size, overlap_t=temporal_overlap)
            mask = torch.ones((1, 1, latent.shape[2], concat_latent_image.shape[-2], concat_latent_image.shape[-1]), device=start_image.device)
            mask[:, :, 0] = 0.0
            
            concat_latent_image_original = concat_latent_image.clone()
            
            if motion_amplitude > 1.0:
                base_latent = concat_latent_image[:, :, 0:1]
                gray_latent = concat_latent_image[:, :, 1:]
                
                diff = gray_latent - base_latent
                diff_mean = diff.mean(dim=(1, 3, 4), keepdim=True)
                diff_centered = diff - diff_mean
                
                scaled_latent = base_latent + diff_centered * motion_amplitude + diff_mean
                scaled_latent = torch.clamp(scaled_latent, -6, 6)
                concat_latent_image = torch.cat([base_latent, scaled_latent], dim=2)
                
                post_enhanced = concat_latent_image.clone()
                
                if color_protect and correct_strength > 0:
                    orig_mean = concat_latent_image_original.mean(dim=(2, 3, 4))
                    enhanced_mean = post_enhanced.mean(dim=(2, 3, 4))
                    
                    mean_drift = torch.abs(enhanced_mean - orig_mean) / (torch.abs(orig_mean) + 1e-6)
                    problem_channels = mean_drift > 0.18
                    
                    if problem_channels.any():
                        drift_amount = enhanced_mean - orig_mean
                        correction = drift_amount * problem_channels.float() * correct_strength * 0.03
                        
                        for b in range(batch_size):
                            for c in range(16):
                                if correction[b, c].abs() > 0:
                                    post_enhanced[b, c] = torch.where(
                                        post_enhanced[b, c] > 0,
                                        post_enhanced[b, c] - correction[b, c],
                                        post_enhanced[b, c]
                                    )
                    
                    orig_brightness = concat_latent_image_original.mean()
                    enhanced_brightness = post_enhanced.mean()
                    
                    if enhanced_brightness < orig_brightness * 0.92:
                        brightness_boost = min(orig_brightness / (enhanced_brightness + 1e-6), 1.05)
                        post_enhanced = torch.where(
                            post_enhanced < 0.5,
                            post_enhanced * brightness_boost,
                            post_enhanced
                        )
                    
                    concat_latent_image = torch.clamp(post_enhanced, -6, 6)

            positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask})
            negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask})
            
            positive_original = node_helpers.conditioning_set_values(positive_original, {"concat_latent_image": concat_latent_image_original, "concat_mask": mask})
            negative_original = node_helpers.conditioning_set_values(negative_original, {"concat_latent_image": concat_latent_image_original, "concat_mask": mask})
            
            ref_latent = vae.encode_tiled(start_image[:,:,:,:3], tile_x=tile_size, tile_y=tile_size, overlap=overlap, tile_t=temporal_size, overlap_t=temporal_overlap)
            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [ref_latent]}, append=True)
            negative = node_helpers.conditioning_set_values(negative, {"reference_latents": [torch.zeros_like(ref_latent)]}, append=True)
            
            positive_original = node_helpers.conditioning_set_values(positive_original, {"reference_latents": [ref_latent]}, append=True)
            negative_original = node_helpers.conditioning_set_values(negative_original, {"reference_latents": [torch.zeros_like(ref_latent)]}, append=True)

        if clip_vision is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision})
            
            positive_original = node_helpers.conditioning_set_values(positive_original, {"clip_vision_output": clip_vision})
            negative_original = node_helpers.conditioning_set_values(negative_original, {"clip_vision_output": clip_vision})

        out_latent = {"samples": latent}
        return io.NodeOutput(positive, negative, positive_original, negative_original, out_latent)


class RikanI2VPainter(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="rikan-i2vpainter",
            display_name="Rikan I2V Painter",
            category="Rikannodes",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=832, min=16, max=4096, step=16),
                io.Int.Input("height", default=480, min=16, max=4096, step=16),
                io.Int.Input("length", default=81, min=1, max=4096, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.Float.Input("motion_amplitude", default=1.3, min=1.0, max=2.0, step=0.05),
                io.Boolean.Input("color_protect", default=True),
                io.Float.Input("correct_strength", default=0.01, min=0.0, max=0.3, step=0.01),
                io.ClipVisionOutput.Input("clip_vision", optional=True),
                io.Image.Input("start_image", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="high_positive"),
                io.Conditioning.Output(display_name="high_negative"),
                io.Conditioning.Output(display_name="low_positive"),
                io.Conditioning.Output(display_name="low_negative"),
                io.Latent.Output(display_name="latent"),
            ]
        )

    @classmethod
    def execute(cls, positive, negative, vae, width, height, length, batch_size,
                motion_amplitude=1.3, color_protect=True, correct_strength=0.01, 
                start_image=None, clip_vision=None) -> io.NodeOutput:
        
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], 
                           device=comfy.model_management.intermediate_device())
        
        positive_original = positive
        negative_original = negative
        
        if start_image is not None:
            start_image = start_image[:1]
            start_image = comfy.utils.common_upscale(start_image.movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            image = torch.ones((length, height, width, 3), device=start_image.device, dtype=start_image.dtype) * 0.5
            image[0] = start_image[0]
            
            concat_latent_image = vae.encode(image[:, :, :, :3])
            mask = torch.ones((1, 1, latent.shape[2], concat_latent_image.shape[-2], concat_latent_image.shape[-1]), device=start_image.device)
            mask[:, :, 0] = 0.0
            
            concat_latent_image_original = concat_latent_image.clone()
            
            if motion_amplitude > 1.0:
                base_latent = concat_latent_image[:, :, 0:1]
                gray_latent = concat_latent_image[:, :, 1:]
                
                diff = gray_latent - base_latent
                diff_mean = diff.mean(dim=(1, 3, 4), keepdim=True)
                diff_centered = diff - diff_mean
                
                scaled_latent = base_latent + diff_centered * motion_amplitude + diff_mean
                scaled_latent = torch.clamp(scaled_latent, -6, 6)
                concat_latent_image = torch.cat([base_latent, scaled_latent], dim=2)
                
                post_enhanced = concat_latent_image.clone()
                
                if color_protect and correct_strength > 0:
                    orig_mean = concat_latent_image_original.mean(dim=(2, 3, 4))
                    enhanced_mean = post_enhanced.mean(dim=(2, 3, 4))
                    
                    mean_drift = torch.abs(enhanced_mean - orig_mean) / (torch.abs(orig_mean) + 1e-6)
                    problem_channels = mean_drift > 0.18
                    
                    if problem_channels.any():
                        drift_amount = enhanced_mean - orig_mean
                        correction = drift_amount * problem_channels.float() * correct_strength * 0.03
                        
                        for b in range(batch_size):
                            for c in range(16):
                                if correction[b, c].abs() > 0:
                                    post_enhanced[b, c] = torch.where(
                                        post_enhanced[b, c] > 0,
                                        post_enhanced[b, c] - correction[b, c],
                                        post_enhanced[b, c]
                                    )
                    
                    orig_brightness = concat_latent_image_original.mean()
                    enhanced_brightness = post_enhanced.mean()
                    
                    if enhanced_brightness < orig_brightness * 0.92:
                        brightness_boost = min(orig_brightness / (enhanced_brightness + 1e-6), 1.05)
                        post_enhanced = torch.where(
                            post_enhanced < 0.5,
                            post_enhanced * brightness_boost,
                            post_enhanced
                        )
                    
                    concat_latent_image = torch.clamp(post_enhanced, -6, 6)

            positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask})
            negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask})
            
            positive_original = node_helpers.conditioning_set_values(positive_original, {"concat_latent_image": concat_latent_image_original, "concat_mask": mask})
            negative_original = node_helpers.conditioning_set_values(negative_original, {"concat_latent_image": concat_latent_image_original, "concat_mask": mask})
            
            ref_latent = vae.encode(start_image[:,:,:,:3])
            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [ref_latent]}, append=True)
            negative = node_helpers.conditioning_set_values(negative, {"reference_latents": [torch.zeros_like(ref_latent)]}, append=True)
            
            positive_original = node_helpers.conditioning_set_values(positive_original, {"reference_latents": [ref_latent]}, append=True)
            negative_original = node_helpers.conditioning_set_values(negative_original, {"reference_latents": [torch.zeros_like(ref_latent)]}, append=True)

        if clip_vision is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision})
            
            positive_original = node_helpers.conditioning_set_values(positive_original, {"clip_vision_output": clip_vision})
            negative_original = node_helpers.conditioning_set_values(negative_original, {"clip_vision_output": clip_vision})

        out_latent = {"samples": latent}
        return io.NodeOutput(positive, negative, positive_original, negative_original, out_latent)


class RikanHiddenBase64ImageLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base64_data": ("STRING", {"multiline": True, "default": ""}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "raw_base64")
    FUNCTION = "execute"
    CATEGORY = "Rikannodes"

    def execute(self, base64_data):
        if not base64_data or base64_data.strip() == "":
            return (torch.zeros((1, 64, 64, 3)), "")
        
        try:
            # Membersihkan header jika ada (misal: data:image/png;base64,...)
            if "," in base64_data:
                base64_data = base64_data.split(",")[1]
            
            img_bytes = base64.b64decode(base64_data)
            img = Image.open(pyio.BytesIO(img_bytes)).convert("RGB")
            
            # Konversi ke tensor ComfyUI
            img_np = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np)[None,]
            
            return (img_tensor, base64_data)
        except Exception as e:
            print(f"[RikanLoader] Error: {e}")
            return (torch.zeros((1, 64, 64, 3)), "")

class RikanHiddenBase64ImageSaver:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("base64_data",)
    FUNCTION = "execute"
    CATEGORY = "Rikannodes"
    OUTPUT_NODE = True # Wajib agar node tetap jalan tanpa kabel output

    def execute(self, image):
        if image is None: 
            return {"ui": {"base64_data": [""]}, "result": ("",)}
            
        try:
            # Ambil frame pertama
            img_tensor = image[0]
            img_np = (img_tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            
            # Encode ke Base64 di memori (Tanpa simpan file)
            buffered = pyio.BytesIO()
            img_pil.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            # Kirim ke UI (JavaScript) dan ke Output kabel
            return {"ui": {"base64_data": [img_base64]}, "result": (img_base64,)}
        except Exception as e:
            print(f"[RikanSaver] Error: {e}")
            return {"ui": {"base64_data": [""]}, "result": ("",)}

# ── NEW: WAN SPATIO-TEMPORAL TILED VAE DECODE ───────────────────────────────

class RikanWanSpatioTemporalTiledVAEDecode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RikanWanSpatioTemporalTiledVAEDecode",
            display_name="Rikan Wan Spatio-Temporal Tiled VAE Decode",
            category="Rikannodes",
            inputs=[
                io.Vae.Input("vae"),
                io.Latent.Input("latents"),
                io.Int.Input("spatial_tiles", default=4, min=1, max=8),
                io.Int.Input("spatial_overlap", default=4, min=0, max=8),
                io.Int.Input("temporal_tile_length", default=16, min=2, max=1000),
                io.Int.Input("temporal_overlap", default=4, min=0, max=8),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
            ]
        )

    @classmethod
    def execute(cls, vae, latents, spatial_tiles=2, spatial_overlap=1, temporal_tile_length=16, temporal_overlap=1) -> io.NodeOutput:
        samples = latents["samples"]
        batch, channels, frames, height, width = samples.shape

        # Menganalisis faktor skala secara dinamis (mengadaptasi berbagai jenis model)
        dummy_latent = torch.zeros((1, channels, 2, 2, 2), device=samples.device, dtype=samples.dtype)
        dummy_out = vae.decode(dummy_latent)
        
        # PERBAIKAN: Membaca shape dari belakang agar aman dari perbedaan 4D vs 5D tensor
        f_out = dummy_out.shape[-4]
        h_out = dummy_out.shape[-3]
        w_out = dummy_out.shape[-2]

        time_scale_factor = (f_out - 1) // (2 - 1)
        height_scale_factor = h_out // 2
        width_scale_factor = w_out // 2

        image_frames = 1 + (frames - 1) * time_scale_factor
        output_height = height * height_scale_factor
        output_width = width * width_scale_factor

        target_device = samples.device
        target_dtype = torch.float32

        def compute_chunk_boundaries(chunk_start, temp_len, temp_overlap, total_frames):
            if chunk_start == 0:
                return chunk_start, min(chunk_start + temp_len, total_frames)
            overlap_start = max(1, chunk_start - temp_overlap - 1)
            extra_frames = chunk_start - overlap_start
            chunk_end = min(chunk_start + temp_len - extra_frames, total_frames)
            return overlap_start, chunk_end

        def decode_spatial(temporal_chunk):
            chunk_frames = temporal_chunk.shape[2]
            chunk_out_frames = 1 + (chunk_frames - 1) * time_scale_factor

            base_tile_height = (height + (spatial_tiles - 1) * spatial_overlap) // spatial_tiles
            base_tile_width = (width + (spatial_tiles - 1) * spatial_overlap) // spatial_tiles

            out_chunk = torch.zeros((batch, chunk_out_frames, output_height, output_width, 3), device=target_device, dtype=target_dtype)
            out_weights = torch.zeros((batch, chunk_out_frames, output_height, output_width, 1), device=target_device, dtype=target_dtype)

            for v in range(spatial_tiles):
                for h in range(spatial_tiles):
                    h_start = h * (base_tile_width - spatial_overlap)
                    v_start = v * (base_tile_height - spatial_overlap)
                    h_end = min(h_start + base_tile_width, width) if h < spatial_tiles - 1 else width
                    v_end = min(v_start + base_tile_height, height) if v < spatial_tiles - 1 else height

                    tile = temporal_chunk[:, :, :, v_start:v_end, h_start:h_end]
                    
                    log.info(f"[Rikan VAE] Decoding spatial tile ({v},{h}) size: {tile.shape}")
                    decoded_tile = vae.decode(tile).to(target_device, target_dtype)
                    
                    # PERBAIKAN: Memastikan formatnya selalu 5D [Batch, Frame, Height, Width, Channels]
                    if len(decoded_tile.shape) == 4:
                        decoded_tile = decoded_tile.view(batch, chunk_out_frames, decoded_tile.shape[-3], decoded_tile.shape[-2], decoded_tile.shape[-1])
                    
                    out_h_start = v_start * height_scale_factor
                    out_h_end = v_end * height_scale_factor
                    out_w_start = h_start * width_scale_factor
                    out_w_end = h_end * width_scale_factor

                    tile_weights = torch.ones((batch, chunk_out_frames, out_h_end - out_h_start, out_w_end - out_w_start, 1), device=target_device, dtype=target_dtype)

                    overlap_out_h = spatial_overlap * height_scale_factor
                    overlap_out_w = spatial_overlap * width_scale_factor

                    if h > 0:
                        h_blend = torch.linspace(0, 1, overlap_out_w, device=target_device, dtype=target_dtype)
                        tile_weights[:, :, :, :overlap_out_w, :] *= h_blend.view(1, 1, 1, -1, 1)
                    if h < spatial_tiles - 1:
                        h_blend = torch.linspace(1, 0, overlap_out_w, device=target_device, dtype=target_dtype)
                        tile_weights[:, :, :, -overlap_out_w:, :] *= h_blend.view(1, 1, 1, -1, 1)
                    if v > 0:
                        v_blend = torch.linspace(0, 1, overlap_out_h, device=target_device, dtype=target_dtype)
                        tile_weights[:, :, :overlap_out_h, :, :] *= v_blend.view(1, 1, -1, 1, 1)
                    if v < spatial_tiles - 1:
                        v_blend = torch.linspace(1, 0, overlap_out_h, device=target_device, dtype=target_dtype)
                        tile_weights[:, :, -overlap_out_h:, :, :] *= v_blend.view(1, 1, -1, 1, 1)

                    out_chunk[:, :, out_h_start:out_h_end, out_w_start:out_w_end, :] += decoded_tile * tile_weights
                    out_weights[:, :, out_h_start:out_h_end, out_w_start:out_w_end, :] += tile_weights

            out_chunk /= out_weights + 1e-8
            return out_chunk

        output = torch.empty((batch, image_frames, output_height, output_width, 3), device=target_device, dtype=target_dtype)

        chunk_start = 0
        while chunk_start < frames:
            overlap_start, chunk_end = compute_chunk_boundaries(chunk_start, temporal_tile_length, temporal_overlap, frames)

            temp_tile = samples[:, :, overlap_start:chunk_end]
            
            log.info(f"[Rikan VAE] Processing temporal chunk {overlap_start}:{chunk_end} (length: {chunk_end - overlap_start})")
            decoded_chunk = decode_spatial(temp_tile)

            if chunk_start == 0:
                output[:, :decoded_chunk.shape[1]] = decoded_chunk
            else:
                decoded_chunk = decoded_chunk[:, 1:]
                out_t_start = 1 + overlap_start * time_scale_factor
                out_t_end = out_t_start + decoded_chunk.shape[1]

                overlap_frames = temporal_overlap * time_scale_factor
                frame_weights = torch.linspace(0, 1, overlap_frames + 2, device=target_device, dtype=target_dtype)[1:-1]
                tile_weights = frame_weights.view(1, -1, 1, 1, 1)

                after_overlap = out_t_start + overlap_frames

                overlap_out = decoded_chunk[:, :overlap_frames]
                output[:, out_t_start:after_overlap] *= (1 - tile_weights)
                output[:, out_t_start:after_overlap] += (tile_weights * overlap_out)

                output[:, after_overlap:out_t_end] = decoded_chunk[:, overlap_frames:]

            chunk_start = chunk_end

        output = output.view(batch * image_frames, output_height, output_width, 3)

        if target_device.type == "cuda":
            torch.cuda.empty_cache()

        return io.NodeOutput(output)

from comfy_api.latest import io
import torch
import comfy.utils

class RikanQwenCustomImageSize(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RikanQwenCustomImageSize",
            display_name="Rikan Qwen Custom Image Size",
            category="Rikannodes",
            inputs=[
                io.Image.Input("pixels"),
                io.Vae.Input("vae"),
                io.String.Input("text", multiline=True, default=""),
                
                # Menambahkan opsi Zoom Out
                io.Combo.Input("resize_mode", options=["Crop to Fit", "Generative Fill", "Zoom Out"], default="Crop to Fit"),
                io.Combo.Input("prompt_theme", options=["Technical Precision", "Artistic Continuity"], default="Technical Precision"),
                
                io.Combo.Input("resolution_preset", options=["Custom", "512p (SD 1.5)", "720p (HD)", "1080p (FHD)", "1024p (SDXL)"], default="720p (HD)"),
                io.Combo.Input("orientation", options=["Horizontal (Landscape)", "Vertical (Portrait)", "Square (1:1)"], default="Horizontal (Landscape)"),
                io.Int.Input("custom_width", default=1280, min=16, max=8192, step=8),
                io.Int.Input("custom_height", default=720, min=16, max=8192, step=8),
                io.Int.Input("batch_size", default=1, min=1, max=4096)
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
                io.String.Output(display_name="combined_prompt")
            ]
        )

    @classmethod
    def execute(cls, pixels, vae, text, resize_mode, prompt_theme, resolution_preset, orientation, custom_width, custom_height, batch_size) -> io.NodeOutput:
        # Menentukan resolusi kanvas target
        if resolution_preset == "Custom":
            w, h = custom_width, custom_height
        else:
            if resolution_preset == "512p (SD 1.5)": base_long, base_short = 768, 512
            elif resolution_preset == "720p (HD)": base_long, base_short = 1280, 720
            elif resolution_preset == "1080p (FHD)": base_long, base_short = 1920, 1080
            elif resolution_preset == "1024p (SDXL)": base_long, base_short = 1344, 768
            
            if orientation == "Horizontal (Landscape)": w, h = base_long, base_short
            elif orientation == "Vertical (Portrait)": w, h = base_short, base_long
            else: w, h = base_short, base_short

        w, h = (w // 8) * 8, (h // 8) * 8
        pixels_batch = pixels[0:1].repeat(batch_size, 1, 1, 1)
        B, orig_H, orig_W, C = pixels_batch.shape

        if resize_mode == "Crop to Fit":
            pixels_processed = comfy.utils.common_upscale(pixels_batch.movedim(-1, 1), w, h, "bilinear", "center").movedim(1, -1)
            t = vae.encode(pixels_processed[:,:,:,:3])
            return io.NodeOutput({"samples": t}, text)
        
        else:
            # Mode Generative Fill atau Zoom Out
            base_scale = min(w / orig_W, h / orig_H)
            
            if resize_mode == "Zoom Out":
                # Mengecilkan gambar asli menjadi 70% dari ukuran maksimalnya agar menyisakan ruang kosong di sekelilingnya
                final_scale = base_scale * 0.70
            else:
                # Mode Generative Fill biasa: menyentuh tepi kanvas (fit to edge)
                final_scale = base_scale
            
            new_W, new_H = round(orig_W * final_scale), round(orig_H * final_scale)
            resized = comfy.utils.common_upscale(pixels_batch.movedim(-1, 1), new_W, new_H, "bilinear", "disabled").movedim(1, -1)
            
            canvas = torch.zeros((batch_size, h, w, C), device=pixels_batch.device, dtype=pixels_batch.dtype)
            canvas.fill_(0.5) 
            
            y_off, x_off = (h - new_H) // 2, (w - new_W) // 2
            canvas[:, y_off:y_off+new_H, x_off:x_off+new_W, :] = resized
            t = vae.encode(canvas[:,:,:,:3])
            
            # --- Pembentukan Prompt Otomatis ---
            if resize_mode == "Zoom Out":
                action_desc = "perform a wide-angle zoom out"
            else:
                action_desc = "zoom out to match orientation"

            if prompt_theme == "Technical Precision":
                addon = f"Seamlessly {action_desc} while maintaining the central original image as an untouched anchor. Generate the top and bottom areas by extending existing textures and lighting, make it natural like it always the result if the original image was zoomed out. Zero distortion of original details."
            else: # Artistic Continuity
                addon = f"Seamlessly {action_desc} while perform a context-aware generative fill on the top and bottom empty spaces. Match the existing color grade, depth of field, and environmental theme perfectly. Ensure the new areas are a natural aesthetic continuation of the original image, make it natural like it always the result if the original image was zoomed out."
            
            final_prompt = f"{text}\n{addon}" if text.strip() != "" else addon
            return io.NodeOutput({"samples": t}, final_prompt)


# ==============================================================================
# MAPPINGS & REGISTRATION
# ==============================================================================

class RikannodesExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            RikanPromptRelayEncodeTimeline,
            RikanPromptRelayMultiLoraGate,
            RikanI2VPainterTiledVAE,
            RikanI2VPainter,
            RikanHiddenBase64ImageLoader,
            RikanHiddenBase64ImageSaver,
            RikanWanSpatioTemporalTiledVAEDecode,
            RikanQwenCustomImageSize
        ]

async def comfy_entrypoint() -> RikannodesExtension:
    return RikannodesExtension()

NODE_CLASS_MAPPINGS = {
    "RikanPromptRelayEncodeTimeline": RikanPromptRelayEncodeTimeline,
    "RikanPromptRelayMultiLoraGate": RikanPromptRelayMultiLoraGate,
    "rikan-i2vpainter-tiled-vae": RikanI2VPainterTiledVAE,
    "rikan-i2vpainter": RikanI2VPainter,
    "RikanHiddenBase64ImageLoader": RikanHiddenBase64ImageLoader,
    "RikanHiddenBase64ImageSaver": RikanHiddenBase64ImageSaver, 
    "RikanWanSpatioTemporalTiledVAEDecode": RikanWanSpatioTemporalTiledVAEDecode,
    "RikanQwenCustomImageSize": RikanQwenCustomImageSize
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RikanPromptRelayEncodeTimeline": "Rikan Prompt Relay Encode (Timeline)",
    "RikanPromptRelayMultiLoraGate": "Rikan Prompt Relay Multi LoRA Gate",
    "rikan-i2vpainter-tiled-vae": "Rikan I2V Painter (Tiled VAE)",
    "rikan-i2vpainter": "Rikan I2V Painter",
    "RikanHiddenBase64ImageLoader": "Rikan Hidden Base64 Image Loader",
    "RikanHiddenBase64ImageSaver": "Rikan Hidden Base64 Image Saver",
    "RikanWanSpatioTemporalTiledVAEDecode": "Rikan Wan Spatio-Temporal Tiled VAE Decode",
    "RikanQwenCustomImageSize": "Rikan Qwen Custom Image Size"
}
