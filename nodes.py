import logging
import math
import types
import torch
import torch.nn as nn
import numpy as np
import base64
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
    except ImportError: return

    lora_file = comfy.utils.load_torch_file(lora_path, safe_load=True)
    key_map = {}
    _lora_mod.model_lora_keys_unet(model_clone.model, key_map)
    patches = _lora_mod.load_lora(lora_file, key_map)
    
    gate_weights = _build_gate_weights(segment, latent_frames)
    diffusion_model = model_clone.get_model_object("diffusion_model")
    
    for param_key, patch_data in patches.items():
        if not param_key.endswith(".weight"): continue
        module_path = param_key[:-7]
        if not module_path.startswith("diffusion_model."): continue
        
        local_path = module_path[len("diffusion_model."):]
        try:
            lora_up, lora_down, alpha = patch_data.weights[0], patch_data.weights[1], patch_data.multiplier
        except: continue
        
        if lora_up is None or lora_down is None or lora_down.dim() != 2: continue
        
        scale = float(alpha) / lora_down.shape[0] if alpha is not None else 1.0
        target = model_clone.object_patches.get(module_path)
        if target is None:
            try: target = _get_module(diffusion_model, local_path)
            except AttributeError: continue

        wrapped = _TemporalGatedLoRA(target.forward, lora_down, lora_up, scale, strength, gate_weights, tokens_per_frame)
        model_clone.add_object_patch(module_path + ".forward", wrapped.forward)

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
        conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(full_prompt))
        effective_lengths = distribute_segment_lengths(len(locals_list), latent_frames, parsed_lengths)

        q_token_idx = build_segments(token_ranges, effective_lengths, epsilon)
        mask_fn = create_mask_fn(q_token_idx, tokens_per_frame, latent_frames)

        patched = model.clone()
        apply_patches(patched, arch, mask_fn)

        # Simpan metadata untuk LoRA Gate
        for key in ["rikan_pr_segments", "pr_segments"]: patched.model_options[key] = q_token_idx
        for key in ["rikan_pr_latent_frames", "pr_latent_frames"]: patched.model_options[key] = latent_frames
        for key in ["rikan_pr_tokens_per_frame", "pr_tokens_per_frame"]: patched.model_options[key] = tokens_per_frame

        print(f"\n[Timeline] SUCCESS: Injected {len(q_token_idx)} segments.\n")
        return io.NodeOutput(patched, conditioning, float(fps), int(max_frames))

# ── NEW: MULTI LORA GATE NODE ───────────────────────────────────

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
            log.warning("[Rikan MultiLoraGate] Metadata Prompt Relay tidak ditemukan.")
            return io.NodeOutput(model)

        try:
            loras_to_apply = json.loads(lora_data)
        except Exception as e:
            log.error(f"[Rikan MultiLoraGate] Gagal memproses data LoRA: {e}")
            return io.NodeOutput(model)

        patched = model.clone()

        for lora in loras_to_apply:
            if not lora.get("enable", False) or lora.get("name") == "None":
                continue

            lora_name = lora.get("name")
            # Index di UI sekarang dimulai dari 0, sejajar dengan index array
            seg_idx = int(lora.get("segment", 0)) 
            strength = float(lora.get("modelStr", 1.0))

            if seg_idx < 0 or seg_idx >= len(segments):
                log.warning(f"[Rikan MultiLoraGate] Segmen {seg_idx} di luar jangkauan timeline.")
                continue

            lora_path = folder_paths.get_full_path("loras", lora_name)
            if lora_path is None:
                log.warning(f"[Rikan MultiLoraGate] File LoRA tidak ditemukan: {lora_name}")
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
        
        positive_original, negative_original = positive, negative
        
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
                base_latent, gray_latent = concat_latent_image[:, :, 0:1], concat_latent_image[:, :, 1:]
                diff = gray_latent - base_latent
                scaled_latent = base_latent + (diff - diff.mean(dim=(1, 3, 4), keepdim=True)) * motion_amplitude + diff.mean(dim=(1, 3, 4), keepdim=True)
                concat_latent_image = torch.clamp(torch.cat([base_latent, scaled_latent], dim=2), -6, 6)

            positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask})
            negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask})
            
            ref_latent = vae.encode_tiled(start_image[:,:,:,:3], tile_x=tile_size, tile_y=tile_size, overlap=overlap, tile_t=temporal_size, overlap_t=temporal_overlap)
            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [ref_latent]}, append=True)
            negative = node_helpers.conditioning_set_values(negative, {"reference_latents": [torch.zeros_like(ref_latent)]}, append=True)

        out_latent = {"samples": latent}
        return io.NodeOutput(positive, negative, positive_original, negative_original, out_latent)


class RikanHiddenBase64ImageLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RikanHiddenBase64ImageLoader",
            display_name="Rikan Hidden Base64 Image Loader",
            category="Rikannodes",
            inputs=[io.String.Input("base64_data", multiline=True, default="")],
            outputs=[io.Image.Output(display_name="image"), io.String.Output(display_name="raw_base64")]
        )

    @classmethod
    def execute(cls, base64_data) -> io.NodeOutput:
        if not base64_data: return io.NodeOutput(torch.zeros((1, 64, 64, 3)), "")
        try:
            if "," in base64_data: base64_data = base64_data.split(",")[1]
            img = Image.open(pyio.BytesIO(base64.b64decode(base64_data))).convert("RGB")
            img = torch.from_numpy(np.array(img).astype(np.float32) / 255.0)[None,]
            return io.NodeOutput(img, base64_data)
        except: return io.NodeOutput(torch.zeros((1, 64, 64, 3)), "")

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
            RikanHiddenBase64ImageLoader
        ]

async def comfy_entrypoint() -> RikannodesExtension:
    return RikannodesExtension()

NODE_CLASS_MAPPINGS = {
    "RikanPromptRelayEncodeTimeline": RikanPromptRelayEncodeTimeline,
    "RikanPromptRelayMultiLoraGate": RikanPromptRelayMultiLoraGate,
    "rikan-i2vpainter-tiled-vae": RikanI2VPainterTiledVAE,
    "RikanHiddenBase64ImageLoader": RikanHiddenBase64ImageLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RikanPromptRelayEncodeTimeline": "Rikan Prompt Relay Encode (Timeline)",
    "RikanPromptRelayMultiLoraGate": "Rikan Prompt Relay Multi LoRA Gate",
    "rikan-i2vpainter-tiled-vae": "Rikan I2V Painter (Tiled VAE)",
    "RikanHiddenBase64ImageLoader": "Rikan Hidden Base64 Image Loader",
}
