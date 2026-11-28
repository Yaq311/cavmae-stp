"""3×3 Joint STP-8 with cached modality encodings."""

import math

import numpy as np
import torch

from stp import ROUNDS, STP, TOKENS_PER_MODALITY, fuse_predictions, weak_view


# The center wins ties, followed by the two original STP directions.
CANDIDATES = ("++", "--", "+-", "-+", "+0", "-0", "0+", "0-")
ENCODINGS_PER_MODALITY = 2 + 2 * ROUNDS
FUSIONS_PER_BATCH = 2 + ROUNDS * len(CANDIDATES)


class GridSTP(STP):
    """Search independent audio and video directions at each round."""

    def _draw_direction(self):
        return self.rng.randn(self.raw.size)

    @torch.no_grad()
    def __call__(self, audio, video):
        if len(audio) == 0 or len(audio) != len(video):
            raise ValueError("Expected matching, nonempty audio and video batches")

        encoding_pairs = fusions = 0

        def encode(patches, raw):
            nonlocal encoding_pairs
            state = self._decode(raw)
            if not np.isfinite(state).all():
                raise FloatingPointError("Nonfinite prompt")
            prompts = torch.as_tensor(state, dtype=torch.float32, device=audio.device)
            audio_prompt, video_prompt = prompts.reshape(2, TOKENS_PER_MODALITY, -1)
            encoding_pairs += 1
            return (
                self.model.encode_foa_modality("audio", patches[0], audio_prompt),
                self.model.encode_foa_modality("video", patches[1], video_prompt),
            )

        def score(encodings):
            nonlocal fusions
            logits, _, _ = self.model.fuse_foa_encodings(*encodings)
            fusions += 1
            logp = logits.float().log_softmax(dim=-1)
            probabilities = logp.exp()
            return probabilities, -(probabilities * logp).sum(-1).mean()

        patches = self.model.embed_foa_inputs(audio, video)
        width = self.raw.size // 2
        incumbent = encode(patches, self.raw)
        best_probabilities, initial_loss = score(incumbent)
        best_loss = initial_loss.item()
        for _ in range(ROUNDS):
            direction = self._draw_direction()
            raw_options = {"0": self.raw, "+": self.raw + direction, "-": self.raw - direction}
            encodings = {
                "0": incumbent,
                "+": encode(patches, raw_options["+"]),
                "-": encode(patches, raw_options["-"]),
            }
            outputs = [score((encodings[a][0], encodings[v][1])) for a, v in CANDIDATES]
            losses = [best_loss, *torch.stack([loss for _, loss in outputs]).tolist()]
            if not all(math.isfinite(loss) for loss in losses):
                raise FloatingPointError("Nonfinite prediction entropy")
            winner = min(range(len(losses)), key=losses.__getitem__)
            if winner:
                a, v = CANDIDATES[winner - 1]
                self.raw = np.concatenate((raw_options[a][:width], raw_options[v][width:]))
                incumbent = encodings[a][0], encodings[v][1]
                best_loss = losses[winner]
                best_probabilities = outputs[winner - 1][0]

        augmented = self.model.embed_foa_inputs(*weak_view(audio, video))
        augmented_probabilities, _ = score(encode(augmented, self.raw))
        if (encoding_pairs, fusions) != (ENCODINGS_PER_MODALITY, FUSIONS_PER_BATCH):
            raise RuntimeError(f"Unexpected grid budget: {encoding_pairs} encoding pairs, {fusions} fusions")
        fused = fuse_predictions(best_probabilities, augmented_probabilities)
        return fused, best_probabilities, fusions


class GridSharedIncrementSTP(GridSTP):
    """Share one raw-coordinate increment across each modality's three tokens."""

    def _draw_direction(self):
        direction = super()._draw_direction().reshape(2, TOKENS_PER_MODALITY, -1)
        mean_direction = direction.mean(axis=1, keepdims=True)
        shared = np.repeat(mean_direction, TOKENS_PER_MODALITY, axis=1).reshape(2, -1)
        norms = np.linalg.norm(shared, axis=1)
        if (norms <= 1e-12).any():
            raise FloatingPointError("Degenerate shared search direction")
        # Match each modality's raw step RMS to the corresponding full search draw.
        shared *= (np.linalg.norm(direction.reshape(2, -1), axis=1) / norms)[:, None]
        return shared.ravel()
