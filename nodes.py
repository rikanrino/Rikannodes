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
# 1. CORE MATH & TOKENIZATION (Berasal dari prompt_relay.py)
# ==============================================================================

def build_temporal_cost(q_token_idx, Lq, Lk, device, dtype, tokens_per_frame):
    """Gaussian penalty matrix [Lq, Lk] for video cross-attention (integer frame indexing)."""
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
    """Penalty matrix for queries that don't map to integer frames (e.g. LTXAV audio tokens)."""
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
    """Closure: mask_fn(q, k, transformer_options) -> additive mask or None."""
    cache = {}
    max_token_idx = max(int(seg["local_token_idx"].max().item()) for seg in q_token_idx) + 1 if q_token_idx else 0

    def mask_fn(q, k, transformer_options):
        Lq, Lk = q.shape[1], k.shape[1]

        if Lq == Lk:
            return None

        cond_or_uncond = transformer_options.get("cond_or_uncond", [])
        if 1 in cond_or_uncond and 0 not in cond_or_uncond:
            return None

        grid_sizes = transformer_options.get("grid_sizes", None)
        video_tpf = int(grid_sizes[1]) * int(grid_sizes[2]) if grid_sizes is not None else fallback_tokens_per_frame
        video_lq = latent_frames * video_tpf

        if Lk == video_lq or Lk < max_token_idx:
            return None

        mode = "video" if Lq == video_lq else "scaled"
        key = (Lq, Lk, mode, q.device)
        
        if key not in cache:
            if mode == "video":
                cost = build_temporal_cost(q_token_idx, Lq, Lk, q.device, q.dtype, video_tpf)
            else:
                cost = build_temporal_cost_scaled(q_token_idx, Lq, Lk, q.device, q.dtype, latent_frames)
            
            log.info(
                "[PromptRelay] Built penalty matrix (%s): Lq=%d, Lk=%d, nonzero=%d/%d",
                mode, Lq, Lk, (cost > 0).sum().item(), cost.numel(),
            )
            cache[key] = -cost

        return cache[key].to(q.dtype)

    return mask_fn


def build_segments(token_ranges, segment_lengths, epsilon=1e-3, relay_options=None):
    sigma = 1.0 / math.log(1.0 / epsilon) if 0 < epsilon < 1 else 0.1448

    opts = relay_options or {}
    v_strength = opts.get("video_strength", 1.0)
    v_window_scale = opts.get("video_window_scale", 1.0)
    a_epsilon = opts.get("audio_epsilon")
    a_strength = opts.get("audio_strength", 1.0)
    a_window_scale = opts.get("audio_window_scale", 1.0)

    if a_epsilon is not None and 0 < a_epsilon < 1:
        sigma_audio = 1.0 / math.log(1.0 / a_epsilon)
    else:
        sigma_audio = sigma

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
            "window": max(base_window * v_window_scale, 0.0),
            "sigma": sigma,
            "strength": v_strength,
            "window_audio": max(base_window * a_window_scale, 0.0),
            "sigma_audio": sigma_audio,
            "strength_audio": a_strength,
        })
        frame_cursor += L

    return q_token_idx


def get_raw_tokenizer(clip):
    tokenizer_wrapper = clip.tokenizer
    for attr_name in dir(tokenizer_wrapper):
        if attr_name.startswith("_"):
            continue
        inner = getattr(tokenizer_wrapper, attr_name, None)
        if inner is not None and hasattr(inner, "tokenizer"):
            return inner.tokenizer

    raise RuntimeError("Could not find raw tokenizer on CLIP object.")


def map_token_indices(raw_tokenizer, global_prompt, local_prompts):
    prefixed_locals = [" " + lp for lp in local_prompts]
    full_prompt = global_prompt + "".join(prefixed_locals)
    has_eos = getattr(raw_tokenizer, "add_eos", False)
    eos_adj = 1 if has_eos else 0

    prev_len = len(raw_tokenizer(global_prompt)["input_ids"]) - eos_adj
    token_ranges = []
    built = global_prompt

    for plp in prefixed_locals:
        built += plp
        cur_len = len(raw_tokenizer(built)["input_ids"]) - eos_adj
        if cur_len <= prev_len:
            raise ValueError(f"Local prompt produced no tokens: '{plp.strip()}'")
        token_ranges.append((prev_len, cur_len))
        prev_len = cur_len

    return full_prompt, token_ranges


def distribute_segment_lengths(num_segments, latent_frames, specified_lengths=None):
    if specified_lengths:
        if len(specified_lengths) != num_segments:
            raise ValueError("Number of segment_lengths must match number of local prompts")
        lengths = specified_lengths
    else:
        step = -(-latent_frames // num_segments)
        lengths = [step] * num_segments

    effective = []
    cursor = 0
    for L in lengths:
        end = min(cursor + L, latent_frames)
        effective.append(max(end - cursor, 0))
        cursor = end
    return effective


# ==============================================================================
# 2. MODEL PATCHING (Berasal dari patches.py)
# ==============================================================================

def _masked_attention(q, k, v, heads, mask, transformer_options={}, **kwargs):
    return comfy.ldm.modules.attention.attention_pytorch(
        q, k, v, heads, mask=mask,
        _inside_attn_wrapper=True,
        transformer_options=transformer_options,
        **kwargs,
    )


def _wan_t2v_forward(self, mask_fn, x, context, transformer_options={}, **kwargs):
    q = self.norm_q(self.q(x))
    k = self.norm_k(self.k(context))
    v = self.v(context)

    mask = mask_fn(q, k, transformer_options)
    if mask is not None:
        x = _masked_attention(q, k, v, heads=self.num_heads, mask=mask,
                              transformer_options=transformer_options)
    else:
        x = comfy.ldm.modules.attention.optimized_attention(
            q, k, v, heads=self.num_heads, transformer_options=transformer_options,
        )
    return self.o(x)


def _wan_i2v_forward(self, mask_fn, x, context, context_img_len, transformer_options={}, **kwargs):
    context_img = context[:, :context_img_len]
    context_text = context[:, context_img_len:]

    q = self.norm_q(self.q(x))

    k_img = self.norm_k_img(self.k_img(context_img))
    v_img = self.v_img(context_img)
    img_x = comfy.ldm.modules.attention.optimized_attention(
        q, k_img, v_img, heads=self.num_heads, transformer_options=transformer_options,
    )

    k = self.norm_k(self.k(context_text))
    v = self.v(context_text)

    mask = mask_fn(q, k, transformer_options)
    if mask is not None:
        x = _masked_attention(q, k, v, heads=self.num_heads, mask=mask,
                              transformer_options=transformer_options)
    else:
        x = comfy.ldm.modules.attention.optimized_attention(
            q, k, v, heads=self.num_heads, transformer_options=transformer_options,
        )

    return self.o(x + img_x)


def _ltx_forward(self, mask_fn, x, context=None, mask=None, pe=None, k_pe=None, transformer_options={}):
    from comfy.ldm.lightricks.model import apply_rotary_emb

    is_self_attn = context is None
    context = x if is_self_attn else context

    q = self.q_norm(self.to_q(x))
    k = self.k_norm(self.to_k(context))
    v = self.to_v(context)

    if pe is not None:
        q = apply_rotary_emb(q, pe)
        k = apply_rotary_emb(k, pe if k_pe is None else k_pe)

    if not is_self_attn:
        temporal_mask = mask_fn(q, k, transformer_options)
        if temporal_mask is not None:
            mask = temporal_mask if mask is None else mask + temporal_mask

    if mask is None:
        out = comfy.ldm.modules.attention.optimized_attention(
            q, k, v, self.heads, attn_precision=self.attn_precision,
            transformer_options=transformer_options,
        )
    else:
        out = _masked_attention(q, k, v, self.heads, mask=mask,
                                attn_precision=self.attn_precision,
                                transformer_options=transformer_options)

    if self.to_gate_logits is not None:
        gate_logits = self.to_gate_logits(x)
        b, t, _ = out.shape
        out = out.view(b, t, self.heads, self.dim_head)
        out = out * (2.0 * torch.sigmoid(gate_logits)).unsqueeze(-1)
        out = out.view(b, t, self.heads * self.dim_head)

    return self.to_out(out)


class _CrossAttnPatch:
    def __init__(self, impl, mask_fn):
        self.impl = impl
        self.mask_fn = mask_fn

    def __get__(self, obj, objtype=None):
        impl, mask_fn = self.impl, self.mask_fn

        def wrapped(self_module, *args, **kwargs):
            return impl(self_module, mask_fn, *args, **kwargs)

        return types.MethodType(wrapped, obj)


def detect_model_type(model):
    diff_model = model.model.diffusion_model

    if hasattr(diff_model, "patch_size") and not hasattr(diff_model, "patchifier"):
        return "wan", tuple(diff_model.patch_size), 4

    if hasattr(diff_model, "patchifier"):
        return "ltx", (1, 1, 1), int(diff_model.vae_scale_factors[0])

    raise ValueError(f"Unsupported model type: {type(diff_model).__name__}.")


def _check_unpatched(model_clone, key):
    if key in getattr(model_clone, "object_patches", {}):
        raise RuntimeError(
            f"PromptRelay: cross-attention forward at '{key}' is already patched by "
            "another node. Stacking is not supported — remove the conflicting node."
        )


def apply_patches(model_clone, arch, mask_fn):
    diffusion_model = model_clone.get_model_object("diffusion_model")

    if arch == "wan":
        from comfy.ldm.wan.model import WanI2VCrossAttention
        for idx, block in enumerate(diffusion_model.blocks):
            key = f"diffusion_model.blocks.{idx}.cross_attn.forward"
            _check_unpatched(model_clone, key)
            cross_attn = block.cross_attn
            impl = _wan_i2v_forward if isinstance(cross_attn, WanI2VCrossAttention) else _wan_t2v_forward
            model_clone.add_object_patch(key, _CrossAttnPatch(impl, mask_fn).__get__(cross_attn, cross_attn.__class__))
        return

    if arch == "ltx":
        for idx, block in enumerate(diffusion_model.transformer_blocks):
            for attr in ("attn2", "audio_attn2"):
                module = getattr(block, attr, None)
                if module is None:
                    continue
                key = f"diffusion_model.transformer_blocks.{idx}.{attr}.forward"
                _check_unpatched(model_clone, key)
                model_clone.add_object_patch(key, _CrossAttnPatch(_ltx_forward, mask_fn).__get__(module, module.__class__))
        return

    raise ValueError(f"Unknown model arch: {arch}")


# ==============================================================================
# 3. ENCODE RELAY LOGIC (Penggabungan & Setup)
# ==============================================================================

def _convert_to_latent_lengths(pixel_lengths, temporal_stride, latent_frames):
    if not pixel_lengths:
        return []
    total_pixel = sum(pixel_lengths)
    if total_pixel <= 0:
        return [1] * len(pixel_lengths)

    naive_total = max(1, round(total_pixel / temporal_stride))
    target_total = min(latent_frames, naive_total)
    
    if target_total >= latent_frames - 1:
        target_total = latent_frames

    exact = [p * target_total / total_pixel for p in pixel_lengths]
    result = [int(e) for e in exact]
    diff = target_total - sum(result)
    if diff > 0:
        order = sorted(range(len(exact)), key=lambda i: -(exact[i] - int(exact[i])))
        for k in range(diff):
            result[order[k % len(order)]] += 1

    for i in range(len(result)):
        if result[i] < 1:
            max_idx = max(range(len(result)), key=lambda j: result[j])
            if result[max_idx] > 1:
                result[max_idx] -= 1
                result[i] = 1

    return result


def _encode_relay(model, clip, latent, global_prompt, local_prompts, segment_lengths, epsilon):
    for name, val in (("global_prompt", global_prompt),
                      ("local_prompts", local_prompts),
                      ("segment_lengths", segment_lengths)):
        if val is None:
            raise ValueError(f"PromptRelay: '{name}' arrived as None.")

    locals_list = [p.strip() for p in local_prompts.split("|") if p.strip()]
    if not locals_list:
        raise ValueError("At least one local prompt is required (separate with |)")

    arch, patch_size, temporal_stride = detect_model_type(model)

    samples = latent["samples"]
    latent_frames = samples.shape[2]
    tokens_per_frame = (samples.shape[3] // patch_size[1]) * (samples.shape[4] // patch_size[2])

    parsed_lengths = None
    if segment_lengths.strip():
        pixel_lengths = [int(x.strip()) for x in segment_lengths.split(",") if x.strip()]
        parsed_lengths = _convert_to_latent_lengths(pixel_lengths, temporal_stride, latent_frames)

    raw_tokenizer = get_raw_tokenizer(clip)
    full_prompt, token_ranges = map_token_indices(raw_tokenizer, global_prompt, locals_list)

    log.info("[PromptRelay] Global: tokens [0:%d] (%d tokens)", token_ranges[0][0], token_ranges[0][0])
    for i, (s, e) in enumerate(token_ranges):
        log.info("[PromptRelay] Segment %d: tokens [%d:%d] (%d tokens)", i, s, e, e - s)

    conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(full_prompt))

    effective_lengths = distribute_segment_lengths(len(locals_list), latent_frames, parsed_lengths)

    log.info(
        "[PromptRelay] Latent: %d frames, %d tokens/frame, segments: %s",
        latent_frames, tokens_per_frame, effective_lengths,
    )

    q_token_idx = build_segments(token_ranges, effective_lengths, epsilon, None)
    mask_fn = create_mask_fn(q_token_idx, tokens_per_frame, latent_frames)

    patched = model.clone()
    apply_patches(patched, arch, mask_fn)

    patched.model_options["pr_segments"] = q_token_idx
    patched.model_options["pr_latent_frames"] = latent_frames
    patched.model_options["pr_tokens_per_frame"] = tokens_per_frame

    return patched, conditioning


# ==============================================================================
# 4. TEMPORAL LORA LOGIC 
# ==============================================================================

class _TemporalGatedLoRA(nn.Module):
    def __init__(self, original, lora_down, lora_up, scale, strength, gate_weights, tokens_per_frame):
        super().__init__()
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

        if x.dim() < 2:
            return base + delta

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
    midpoint = float(seg["midpoint"])
    window = float(seg["window"])
    sigma = float(seg["sigma"])

    weights = []
    for frame in range(latent_frames):
        d = abs(frame - midpoint)
        cost = (max(d - window, 0.0) ** 2) / (2.0 * sigma ** 2)
        weights.append(math.exp(-cost))

    return torch.tensor(weights, dtype=torch.float32)


def _get_module(diffusion_model, local_path):
    m = diffusion_model
    for part in local_path.split("."):
        m = getattr(m, part)
    return m


def _apply_gated_lora(model_clone, lora_file, segment, strength, tokens_per_frame, latent_frames):
    try:
        import comfy.loras
    except ImportError:
        log.error("[PromptRelay] comfy.loras not available in this ComfyUI build — cannot apply LoRA gate.")
        return

    key_map = {}
    try:
        comfy.loras.model_lora_keys_unet(model_clone.model, key_map)
    except Exception as e:
        log.error("[PromptRelay] Failed to build LoRA key map: %s", e)
        return

    try:
        patches = comfy.loras.load_lora(lora_file, key_map)
    except Exception as e:
        log.error("[PromptRelay] Failed to parse LoRA file: %s", e)
        return

    gate_weights = _build_gate_weights(segment, latent_frames)
    diffusion_model = model_clone.get_model_object("diffusion_model")

    applied = 0
    skipped = 0

    for param_key, patch_data in patches.items():
        if not param_key.endswith(".weight"):
            skipped += 1
            continue

        module_path = param_key[:-7]

        if not module_path.startswith("diffusion_model."):
            skipped += 1
            continue

        local_path = module_path[len("diffusion_model."):]

        try:
            alpha = patch_data[0]
            lora_up = patch_data[1]
            lora_down = patch_data[2]
        except (TypeError, IndexError):
            skipped += 1
            continue

        if lora_up is None or lora_down is None:
            skipped += 1
            continue

        if lora_down.dim() != 2 or lora_up.dim() != 2:
            skipped += 1
            continue

        rank = lora_down.shape[0]
        scale = float(alpha) / rank if alpha is not None else 1.0

        if module_path in model_clone.object_patches:
            target = model_clone.object_patches[module_path]
        else:
            try:
                target = _get_module(diffusion_model, local_path)
            except AttributeError:
                skipped += 1
                continue

        wrapped = _TemporalGatedLoRA(
            original=target,
            lora_down=lora_down,
            lora_up=lora_up,
            scale=scale,
            strength=strength,
            gate_weights=gate_weights,
            tokens_per_frame=tokens_per_frame,
        )

        model_clone.add_object_patch(module_path, wrapped)
        applied += 1

    log.info(
        "[PromptRelay] Segment midpoint=%.1f window=%.1f — applied %d LoRA patches, skipped %d",
        float(segment["midpoint"]), float(segment["window"]), applied, skipped,
    )


# ==============================================================================
# 5. COMFYUI NODE CLASSES
# ==============================================================================

class PromptRelayEncodeTimeline(io.ComfyNode):
    """WYSIWYG timeline variant — segments and lengths come from a visual editor in the node UI."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PromptRelayEncodeTimeline",
            display_name="Prompt Relay Encode (Timeline)",
            category="Rikannodes",
            description=(
                "Same as Prompt Relay Encode, but local prompts and segment lengths are edited "
                "visually as draggable blocks on a timeline. The max_frames input only sets the "
                "timeline scale (pixel space) — actual frame count is still read from the latent."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Latent.Input("latent", tooltip="Empty latent video — dimensions are read from its shape."),
                io.String.Input(
                    "global_prompt", multiline=True, default="",
                    tooltip="Conditions the entire video. Anchors persistent characters, objects, and scene context.",
                ),
                io.Int.Input(
                    "max_frames", default=129, min=1, max=10000, step=1,
                    tooltip="Total timeline length in pixel-space frames. Used by the editor for visual scale only.",
                ),
                io.String.Input(
                    "timeline_data", default="",
                    tooltip="JSON state of the timeline editor (auto-managed; do not edit by hand).",
                ),
                io.String.Input(
                    "local_prompts", multiline=True, default="",
                    tooltip="Auto-populated from the timeline editor.",
                ),
                io.String.Input(
                    "segment_lengths", default="",
                    tooltip="Auto-populated from the timeline editor (pixel-space frame counts).",
                ),
                io.Float.Input(
                    "epsilon", default=1e-3, min=1e-6, max=0.99, step=1e-4,
                    tooltip="Penalty decay parameter. Values below ~0.1 all produce sharp boundaries (paper default 0.001). For softer transitions, try 0.5 or higher.",
                ),
                io.Float.Input(
                    "fps", default=24.0, min=0.1, max=240.0, step=0.1, optional=True,
                    tooltip="Frames per second — only affects how time is displayed in the timeline editor when time_units is set to 'seconds'.",
                ),
                io.Combo.Input(
                    "time_units", options=["frames", "seconds"], default="frames", optional=True,
                    tooltip="Display the ruler, segment ranges, length input, and total in frames or seconds. Internal storage is always pixel-space frames.",
                ),
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
        patched, conditioning = _encode_relay(
            model, clip, latent, global_prompt, local_prompts, segment_lengths, epsilon,
        )
        return io.NodeOutput(patched, conditioning, float(fps), int(max_frames))


class PromptRelayLoraGate(io.ComfyNode):
    """Applies a LoRA to one temporal segment defined by a Prompt Relay node."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PromptRelayLoraGate",
            display_name="Prompt Relay LoRA Gate",
            category="Rikannodes",
            description=(
                "Applies a LoRA with Gaussian temporal gating to one Prompt Relay segment. "
                "Chain multiple gates — one per segment. Bypass unused gates. "
                "segment_index is zero-based and must match the segment order in the Prompt Relay node."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip", tooltip="CLIP model. (Note: Patched CLIP only affects text encoded downstream, like negative prompts).", optional=True),
                io.Combo.Input(
                    "lora_name",
                    folder_paths.get_filename_list("loras"),
                    tooltip="LoRA to apply to this segment.",
                ),
                io.Int.Input(
                    "segment_index", default=0, min=0, max=99, step=1,
                    tooltip="Zero-based index of the Prompt Relay segment this LoRA targets. Segment 0 is the first local prompt.",
                ),
                io.Float.Input(
                    "strength", default=0.8, min=-10.0, max=10.0, step=0.01,
                    tooltip="LoRA strength for the UNet at the segment centre. Fades to zero outside the segment window.",
                ),
                io.Float.Input(
                    "clip_strength", default=1.0, min=-10.0, max=10.0, step=0.01,
                    tooltip="LoRA strength for the CLIP model (applied globally to this CLIP stream).",
                ),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Clip.Output(display_name="clip"),
            ],
        )

    @classmethod
    def execute(cls, model, lora_name, segment_index, strength, clip_strength, clip=None) -> io.NodeOutput:
        if strength == 0.0 and clip_strength == 0.0:
            return io.NodeOutput(model, clip)

        lora_path = folder_paths.get_full_path("loras", lora_name)
        if lora_path is None:
            raise ValueError(f"[PromptRelay LoraGate] LoRA file not found: {lora_name}")

        import comfy.utils
        import comfy.sd
        
        # Load the file once for both Model and CLIP
        lora_file = comfy.utils.load_torch_file(lora_path, safe_load=True)

        # --- 1. Patch CLIP globally (Standard ComfyUI method) ---
        patched_clip = clip
        if clip is not None and clip_strength != 0.0:
            _, patched_clip = comfy.sd.load_lora_for_models(None, clip, lora_file, 0.0, clip_strength)

        # --- 2. Patch UNet temporally ---
        patched_model = model
        if strength != 0.0:
            segments = model.model_options.get("pr_segments")
            latent_frames = model.model_options.get("pr_latent_frames")
            tokens_per_frame = model.model_options.get("pr_tokens_per_frame")

            if segments is None:
                log.warning(
                    "[PromptRelay LoraGate] No Prompt Relay segment metadata on this model. "
                    "Ensure the model comes from a Prompt Relay Encode node. Passing through model unchanged."
                )
            elif segment_index >= len(segments):
                log.warning(
                    "[PromptRelay LoraGate] segment_index %d is out of range — only %d segment(s) defined. "
                    "Passing through model unchanged.",
                    segment_index, len(segments),
                )
            else:
                patched_model = model.clone()
                _apply_gated_lora(
                    patched_model,
                    lora_file,
                    segments[segment_index],
                    strength,
                    tokens_per_frame,
                    latent_frames,
                )
                
                # Forward metadata to the next node in the chain
                patched_model.model_options["pr_segments"] = segments
                patched_model.model_options["pr_latent_frames"] = latent_frames
                patched_model.model_options["pr_tokens_per_frame"] = tokens_per_frame

        return io.NodeOutput(patched_model, patched_clip)

class PromptRelayPowerLoraGate:
    """Applies multiple LoRAs to various temporal segments natively with toggles."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "clip": ("CLIP", {"tooltip": "Optional CLIP model for global text patching."}),
            },
            "hidden": {},
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "execute"
    CATEGORY = "Rikannodes"

    def execute(self, model, clip=None, **kwargs):
        segments = model.model_options.get("pr_segments")
        latent_frames = model.model_options.get("pr_latent_frames")
        tokens_per_frame = model.model_options.get("pr_tokens_per_frame")

        if segments is None:
            log.warning("[PromptRelay PowerLoraGate] No Prompt Relay segment metadata on this model. Passing through.")
            return (model, clip)

        patched_model = model.clone()
        patched_clip = clip

        import comfy.utils
        import comfy.sd

        # 1. Parsing widget dinamis yang dikirim dari antarmuka JS
        loras_to_load = {}
        for k, v in kwargs.items():
            parts = k.split("_")
            if len(parts) < 2:
                continue
            
            prefix = parts[0]
            idx = parts[1]

            if idx not in loras_to_load:
                # Default value ditambah "enable"
                loras_to_load[idx] = {"name": None, "segment": 0, "model_str": 1.0, "clip_str": 1.0, "enable": True}

            if prefix == "lora" and isinstance(v, str):
                loras_to_load[idx]["name"] = v
            elif prefix == "segment":
                loras_to_load[idx]["segment"] = int(v)
            elif prefix == "modelStr":
                loras_to_load[idx]["model_str"] = float(v)
            elif prefix == "clipStr":
                loras_to_load[idx]["clip_str"] = float(v)
            elif prefix == "enable":
                loras_to_load[idx]["enable"] = bool(v)

        # 2. Loop dan terapkan semua LoRA ke segmen masing-masing
        for idx in sorted(loras_to_load.keys(), key=lambda x: int(x)):
            data = loras_to_load[idx]
            
            # Jika toggle dimatikan di UI, lewati LoRA ini
            if not data.get("enable", True):
                continue
                
            lora_name = data.get("name")
            seg_idx = data.get("segment", 0)
            strength_model = data.get("model_str", 1.0)
            strength_clip = data.get("clip_str", 1.0)

            if not lora_name or lora_name == "None":
                continue

            if strength_model == 0.0 and strength_clip == 0.0:
                continue

            if seg_idx >= len(segments):
                log.warning(f"[PromptRelay PowerLoraGate] segment_index {seg_idx} is out of range. Skipping LoRA {lora_name}.")
                continue

            lora_path = folder_paths.get_full_path("loras", lora_name)
            if not lora_path:
                log.warning(f"[PromptRelay PowerLoraGate] LoRA file not found: {lora_name}")
                continue

            lora_file = comfy.utils.load_torch_file(lora_path, safe_load=True)
            segment = segments[seg_idx]

            # Patch CLIP secara global
            if patched_clip is not None and strength_clip != 0.0:
                _, patched_clip = comfy.sd.load_lora_for_models(None, patched_clip, lora_file, 0.0, strength_clip)

            # Patch UNet secara temporal
            if strength_model != 0.0:
                _apply_gated_lora(
                    patched_model,
                    lora_file,
                    segment,
                    strength_model,
                    tokens_per_frame,
                    latent_frames,
                )

        patched_model.model_options["pr_segments"] = segments
        patched_model.model_options["pr_latent_frames"] = latent_frames
        patched_model.model_options["pr_tokens_per_frame"] = tokens_per_frame

        return (patched_model, patched_clip)


# ==============================================================================
# 6. MAPPINGS
# ==============================================================================

NODE_CLASS_MAPPINGS = {
    "PromptRelayEncodeTimeline": PromptRelayEncodeTimeline,
    "PromptRelayLoraGate": PromptRelayLoraGate,
    "PromptRelayPowerLoraGate": PromptRelayPowerLoraGate, 
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptRelayEncodeTimeline": "Prompt Relay Encode (Timeline)",
    "PromptRelayLoraGate": "Prompt Relay LoRA Gate",
    "PromptRelayPowerLoraGate": "Prompt Relay Power Lora Gate",
}
