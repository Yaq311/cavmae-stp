"""Previous-batch prompt initialization using cosine similarity."""

import cma
import numpy as np
import torch
from torch.nn.functional import normalize

from stp import FORWARDS_PER_BATCH, ROUNDS, forward_prompt, fuse_predictions, initial_prompt, weak_view


class SimilarityTransfer:
    """Search each sample's prompt and retain the final prompts for the next batch."""

    def __init__(self, model, prompt_seed=2026, search_seed=2020):
        self.model = model.eval().requires_grad_(False)
        self.prompt_seed = prompt_seed
        self.reference = initial_prompt(model, prompt_seed)
        self.transform = cma.BoundTransform([-3.0, 3.0])
        self.rng = np.random.RandomState(search_seed)
        self.previous_features = self.previous_prompts = None

    @torch.no_grad()
    def __call__(self, audio, video):
        if len(audio) == 0 or len(audio) != len(video):
            raise ValueError("Expected matching, nonempty audio and video batches")

        calls = 0

        def evaluate(audio_view, video_view, state):
            nonlocal calls
            calls += 1
            return forward_prompt(self.model, audio_view, video_view, state)

        features = normalize(evaluate(audio, video, self.reference)["raw_fusion_feature"].float(), dim=-1)
        if self.previous_features is None:
            initial = initial_prompt(self.model, self.prompt_seed, count=len(audio))
        else:
            similarities = normalize(features, dim=-1) @ normalize(self.previous_features, dim=-1).T
            weights = (similarities / 0.07).softmax(dim=-1)
            initial = weights.double().cpu().numpy() @ self.previous_prompts
        raw = np.stack([self.transform.inverse(row.copy()) for row in initial])

        def score(candidate):
            state = np.stack([self.transform.repair(row.copy()) for row in candidate])
            if not np.isfinite(state).all():
                raise FloatingPointError("Nonfinite prompt")
            logp = evaluate(audio, video, state)["logits"].float().log_softmax(dim=-1)
            probabilities = logp.exp()
            losses = -(probabilities * logp).sum(-1)
            if not torch.isfinite(losses).all():
                raise FloatingPointError("Nonfinite prediction entropy")
            return losses, state, probabilities

        best_losses, best_state, best_probabilities = score(raw)
        for _ in range(ROUNDS):
            direction = self.rng.randn(*raw.shape)
            for candidate in (raw + direction, raw - direction):
                losses, state, probabilities = score(candidate)
                take = losses < best_losses
                rows = take.cpu().numpy()
                raw[rows] = candidate[rows]
                best_state[rows] = state[rows]
                best_losses = torch.where(take, losses, best_losses)
                best_probabilities = torch.where(take[:, None], probabilities, best_probabilities)

        augmented_audio, augmented_video = weak_view(audio, video)
        output = evaluate(augmented_audio, augmented_video, best_state)
        augmented_probabilities = output["logits"].float().log_softmax(dim=-1).exp()
        fused = fuse_predictions(best_probabilities, augmented_probabilities)
        if calls != FORWARDS_PER_BATCH + 1:
            raise RuntimeError(f"Expected {FORWARDS_PER_BATCH + 1} forwards, got {calls}")
        self.previous_features, self.previous_prompts = features, best_state
        return fused, best_probabilities, calls
