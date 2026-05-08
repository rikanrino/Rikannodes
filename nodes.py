import logging
import math
import types
import torch
import torch.nn as nn
import comfy.utils
import folder_paths
import comfy.ldm.modules.attention
from comfy_api.latest import io

log = logging.getLogger(__name__)

# ==============================================================================
# 1. CORE UTILS & MATH (Dibutuhkan oleh Timeline Node)
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
# 2. MODEL PATCHING
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

# Patch implementation functions (Simplified)
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

# ==============================================================================
# 3. MAIN NODE: RIkan Prompt Relay Encode (Timeline)
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
    def execute(cls, model, clip, latent, global_prompt, max_frames, timeline_data, local_prompts, segment_lengths, epsilon, fps=24.0, time_units="frames") -> io.NodeOutput:
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
        conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(full_prompt))
        effective_lengths = distribute_segment_lengths(len(locals_list), latent_frames, parsed_lengths)

        q_token_idx = build_segments(token_ranges, effective_lengths, epsilon)
        mask_fn = create_mask_fn(q_token_idx, tokens_per_frame, latent_frames)

        patched = model.clone()
        apply_patches(patched, arch, mask_fn)

        # Simpan metadata untuk kompatibilitas
        for key in ["rikan_pr_segments", "pr_segments"]: patched.model_options[key] = q_token_idx
        for key in ["rikan_pr_latent_frames", "pr_latent_frames"]: patched.model_options[key] = latent_frames
        for key in ["rikan_pr_tokens_per_frame", "pr_tokens_per_frame"]: patched.model_options[key] = tokens_per_frame

        print(f"\n[Timeline] SUCCESS: Injected {len(q_token_idx)} segments.\n")
        return io.NodeOutput(patched, conditioning, float(fps), int(max_frames))

# ==============================================================================
# MAPPINGS
# ==============================================================================

NODE_CLASS_MAPPINGS = { "RikanPromptRelayEncodeTimeline": RikanPromptRelayEncodeTimeline }
NODE_DISPLAY_NAME_MAPPINGS = { "RikanPromptRelayEncodeTimeline": "Rikan Prompt Relay Encode (Timeline)" }
