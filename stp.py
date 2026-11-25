"""Full-dimensional Joint STP-8 with final two-view probability averaging."""

import math

import cma
import numpy as np
import torch


ROUNDS = 8
TOKENS_PER_MODALITY = 3
FORWARDS_PER_BATCH = 2 * ROUNDS + 2


def initial_prompt(model, seed):
    """Draw the same VPT initialization as the historical STP-8 runner."""
    width = model.pos_embed_a.shape[-1]
    prompts = []
    for index, embedding in enumerate((model.patch_embed_a, model.patch_embed_v)):
        size = embedding.patch_size
        area = size ** 2 if isinstance(size, int) else math.prod(size)
        bound = math.sqrt(6.0 / (3 * area + width))
        generator = torch.Generator(device="cpu").manual_seed(seed + index)
        prompt = torch.empty(TOKENS_PER_MODALITY * width).uniform_(
            -bound, bound, generator=generator
        )
        prompts.append(prompt)
    return torch.cat(prompts).numpy().astype(np.float64)


def weak_view(audio, video):
    """Delay the audio fbank by two frames and flip the video horizontally."""
    audio = torch.cat((audio[:, :1].expand(-1, 2, -1), audio[:, :-2]), dim=1)
    return audio, video.flip(-1)


def predict(model, audio, video, state):
    """Return log probabilities for one decoded shared prompt."""
    prompt = torch.as_tensor(state, dtype=torch.float32, device=audio.device)
    prompt_audio, prompt_video = prompt.reshape(2, TOKENS_PER_MODALITY, -1)
    output = model.forward_foa(
        audio, video,
        prompt_audio=prompt_audio, prompt_video=prompt_video,
        shift_vector=None, fusion_order="video_audio",
        prompt_type="tokens", feature_mode="vpt",
    )
    return output["logits"].float().log_softmax(dim=-1)


class STP:
    """Keep the winning raw prompt and RNG state across batches of one corruption."""

    def __init__(self, model, prompt_seed=2026, search_seed=2020):
        self.model = model.eval().requires_grad_(False)
        self.transform = cma.BoundTransform([-3.0, 3.0])
        self.raw = self.transform.inverse(initial_prompt(model, prompt_seed))
        self.rng = np.random.RandomState(search_seed)

    @torch.no_grad()
    def __call__(self, audio, video):
        if len(audio) == 0 or len(audio) != len(video):
            raise ValueError("Expected matching, nonempty audio and video batches")

        calls = 0

        def evaluate(audio_view, video_view, state):
            nonlocal calls
            calls += 1
            return predict(self.model, audio_view, video_view, state)

        def score(raw):
            # repair can modify its input, so keep the raw search coordinates separate.
            state = np.asarray(self.transform.repair(raw.copy()), dtype=np.float64)
            if not np.isfinite(state).all():
                raise FloatingPointError("Nonfinite prompt")
            logp = evaluate(audio, video, state)
            probabilities = logp.exp()
            loss = -(probabilities * logp).sum(-1).mean().item()
            if not math.isfinite(loss):
                raise FloatingPointError("Nonfinite prediction entropy")
            return loss, state, probabilities

        best_loss, best_state, best_probabilities = score(self.raw)
        for _ in range(ROUNDS):
            center = self.raw
            direction = self.rng.randn(center.size)
            for candidate in (center + direction, center - direction):
                loss, state, probabilities = score(candidate)
                if loss < best_loss:
                    self.raw = candidate
                    best_loss, best_state, best_probabilities = loss, state, probabilities

        augmented_audio, augmented_video = weak_view(audio, video)
        augmented_probabilities = evaluate(augmented_audio, augmented_video, best_state).exp()
        fused = (best_probabilities + augmented_probabilities) / 2
        if calls != FORWARDS_PER_BATCH:
            raise RuntimeError(f"Expected {FORWARDS_PER_BATCH} forwards, got {calls}")
        if not torch.isfinite(fused).all():
            raise FloatingPointError("Nonfinite fused prediction")
        fused_logp = fused.clamp_min(torch.finfo(fused.dtype).tiny).log()
        return fused_logp, best_probabilities, calls
