"""Tests for the BasePipeline ABC and inheritance."""
from server.pipeline import BasePipeline


def test_base_pipeline_cannot_be_instantiated():
    try:
        BasePipeline()
        assert False, "should not instantiate an ABC"
    except TypeError:
        pass


def test_variant_seeds_deterministic():
    seeds = BasePipeline.variant_seeds(42, 4)
    assert seeds == [42, 43, 44, 45]


def test_variant_seeds_random_without_seed():
    seeds = BasePipeline.variant_seeds(None, 3)
    assert len(seeds) == 3
    assert all(0 <= s < 2**32 for s in seeds)


def test_klein_pipeline_is_subclass():
    from server.instruct import KleinPipeline
    assert issubclass(KleinPipeline, BasePipeline)


def test_openrouter_pipeline_is_subclass():
    from server.openrouter import OpenRouterPipeline
    assert issubclass(OpenRouterPipeline, BasePipeline)


def test_klein_inherits_variant_seeds():
    from server.instruct import KleinPipeline
    # variant_seeds should come from BasePipeline, not be defined on Klein
    assert "variant_seeds" not in KleinPipeline.__dict__
    seeds = KleinPipeline.variant_seeds(10, 2)
    assert seeds == [10, 11]


def test_openrouter_inherits_variant_seeds():
    from server.openrouter import OpenRouterPipeline
    assert "variant_seeds" not in OpenRouterPipeline.__dict__
    seeds = OpenRouterPipeline.variant_seeds(10, 2)
    assert seeds == [10, 11]
