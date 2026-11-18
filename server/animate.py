"""Sequential frame generation for animation: each frame is conditioned on
the previous one via img2img (edit_by_instruction)."""

from server.anim_presets import build_frame_prompt
from server.instruct import STYLE_SUFFIX


# loop closure reduces strength so the last frame stays close to frame 0
LOOP_STRENGTH_REDUCTION = 0.10


class AnimationGenerator:
    """Stateless generator: takes a pipeline instance, returns frames."""

    def generate_sequential(self, pipe, req, on_progress=None):
        """Generate frames sequentially, each conditioned on the previous.

        Returns (list[PIL.Image], list[int]) — frames and seeds.
        """
        anim = req.animation
        seeds = pipe.variant_seeds(req.seed, anim.frame_count)

        frames = []
        # frame 0: use provided image or generate via txt2img
        if req.frames and req.frames[0].image is not None:
            frames.append(req.frames[0].image)
            seeds[0] = seeds[0]  # keep the seed slot even if unused
        else:
            raw = pipe.txt2img(req.prompt, req.target_size, variants=1,
                               seeds=[seeds[0]])
            frames.append(raw[0])

        if on_progress:
            on_progress(1.0 / anim.frame_count)

        # frames 1..N-1: img2img from previous frame
        for i in range(1, anim.frame_count):
            prompt = build_frame_prompt(
                req.prompt, anim.preset, i, anim.frame_count, anim.loop)
            prompt += STYLE_SUFFIX

            strength = anim.strength
            if anim.loop and i == anim.frame_count - 1:
                strength = max(0.10, strength - LOOP_STRENGTH_REDUCTION)

            raw = pipe.edit_by_instruction(
                prompt, frames[i - 1], variants=1,
                seeds=[seeds[i]])
            frames.append(raw[0])

            if on_progress:
                on_progress((i + 1) / anim.frame_count)

        return frames, seeds
