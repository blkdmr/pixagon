"""Base pipeline: the contract every image backend must satisfy.

main.py calls only these methods, so a new backend
(a different diffusion model, an API provider, a mock for tests) only
needs to subclass BasePipeline and register itself in the models registry.
"""
from abc import ABC, abstractmethod


class BasePipeline(ABC):
    """Interface shared by every image backend (local GPU, cloud API, etc.)."""

    @abstractmethod
    def load(self):
        """Prepare the backend (download weights, open a connection, etc.).
        Called once before the first generation; must be idempotent."""

    @staticmethod
    def variant_seeds(seed, variants):
        """One seed per variant so a single good result can be reproduced."""
        import random
        if seed is None:
            return [random.randrange(2**32) for _ in range(variants)]
        return [(seed + i) % 2**32 for i in range(variants)]

    @abstractmethod
    def txt2img(self, prompt, target_size, variants=4, on_progress=None,
                seeds=None):
        """Text to image. Returns a list of RGBA PIL Images."""

    @abstractmethod
    def edit_by_instruction(self, instruction, image, variants=4,
                            on_progress=None, seeds=None):
        """Edit an image by a text instruction. Returns a list of RGBA PIL Images."""

    @abstractmethod
    def inpaint(self, prompt, image, mask, variants=4, on_progress=None,
                seeds=None):
        """Redraw the masked area. Returns a list of RGBA PIL Images."""
