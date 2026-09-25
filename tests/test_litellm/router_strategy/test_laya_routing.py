from unittest.mock import AsyncMock

import pytest

from litellm import Router
from litellm.router_strategy.laya_routing import (
    DEFAULT_TIERS,
    LayaRoutingStrategy,
    apply_laya_routing_strategy,
    laya_tiers_from_args,
)


class FakeLayaRouter:
    """Stands in for laya.Router so tests never load real checkpoints."""

    def __init__(self, score: float = 0.0):
        self.score = score
        self.calls = []

    def predict(self, state, questions):
        self.calls.append((state, questions))
        return {"answers": {"complexity": {"score": self.score}}, "routing": {"model": "english"}}


def _model_list():
    return [
        {
            "model_name": "laya-router",
            "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "fake-key"},
            "model_info": {"id": "tiny", "complexity_tier": "tiny"},
        },
        {
            "model_name": "laya-router",
            "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "fake-key"},
            "model_info": {"id": "small", "complexity_tier": "small"},
        },
        {
            "model_name": "laya-router",
            "litellm_params": {"model": "openai/gpt-4o", "api_key": "fake-key"},
            "model_info": {"id": "medium", "complexity_tier": "medium"},
        },
        {
            "model_name": "laya-router",
            "litellm_params": {"model": "openai/gpt-4o", "api_key": "fake-key"},
            "model_info": {"id": "large", "complexity_tier": "large"},
        },
    ]


def _create_test_router():
    return Router(model_list=_model_list())


def _strategy(score: float = 0.0, router=None):
    router = router or _create_test_router()
    return LayaRoutingStrategy(router_instance=router, laya_router=FakeLayaRouter(score=score))


@pytest.mark.asyncio
async def test_low_score_routes_to_tiny():
    strategy = _strategy(score=0.0)
    result = await strategy.async_get_available_deployment(
        model="laya-router",
        messages=[{"role": "user", "content": "what is 2+2"}],
    )
    assert result["model_info"]["complexity_tier"] == "tiny"


@pytest.mark.asyncio
async def test_high_score_routes_to_large():
    strategy = _strategy(score=3.0)
    result = await strategy.async_get_available_deployment(
        model="laya-router",
        messages=[{"role": "user", "content": "design a distributed rate limiter"}],
    )
    assert result["model_info"]["complexity_tier"] == "large"


@pytest.mark.asyncio
async def test_score_rounds_to_nearest_tier():
    strategy = _strategy(score=1.6)
    result = await strategy.async_get_available_deployment(
        model="laya-router",
        messages=[{"role": "user", "content": "write a short poem"}],
    )
    assert result["model_info"]["complexity_tier"] == "medium"


@pytest.mark.asyncio
async def test_out_of_range_score_is_clamped():
    strategy = _strategy(score=99.0)
    result = await strategy.async_get_available_deployment(
        model="laya-router",
        messages=[{"role": "user", "content": "hello"}],
    )
    assert result["model_info"]["complexity_tier"] == "large"


@pytest.mark.asyncio
async def test_no_prompt_text_defaults_to_first_tier_without_calling_laya():
    router = _create_test_router()
    laya = FakeLayaRouter(score=3.0)
    strategy = LayaRoutingStrategy(router_instance=router, laya_router=laya)

    result = await strategy.async_get_available_deployment(model="laya-router", messages=[])

    assert result["model_info"]["complexity_tier"] == "tiny"
    assert laya.calls == []


@pytest.mark.asyncio
async def test_prefers_last_user_message():
    router = _create_test_router()
    laya = FakeLayaRouter(score=0.0)
    strategy = LayaRoutingStrategy(router_instance=router, laya_router=laya)

    await strategy.async_get_available_deployment(
        model="laya-router",
        messages=[
            {"role": "system", "content": "you are a helpful assistant"},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second question"},
        ],
    )

    state, _ = laya.calls[0]
    assert state == {"body": "second question"}


@pytest.mark.asyncio
async def test_falls_back_to_embeddings_input():
    router = _create_test_router()
    laya = FakeLayaRouter(score=0.0)
    strategy = LayaRoutingStrategy(router_instance=router, laya_router=laya)

    await strategy.async_get_available_deployment(model="laya-router", input="embed this")

    state, _ = laya.calls[0]
    assert state == {"body": "embed this"}


@pytest.mark.asyncio
async def test_missing_tier_falls_back_with_warning(caplog):
    router = Router(
        model_list=[
            {
                "model_name": "laya-router",
                "litellm_params": {"model": "openai/gpt-4o", "api_key": "fake-key"},
                "model_info": {"id": "only-medium", "complexity_tier": "medium"},
            }
        ]
    )
    strategy = LayaRoutingStrategy(router_instance=router, laya_router=FakeLayaRouter(score=3.0))

    with caplog.at_level("WARNING"):
        result = await strategy.async_get_available_deployment(
            model="laya-router",
            messages=[{"role": "user", "content": "design a system"}],
        )

    assert result["model_info"]["complexity_tier"] == "medium"
    assert "no deployment for tier 'large'" in caplog.text


@pytest.mark.asyncio
async def test_no_router_returns_none():
    strategy = LayaRoutingStrategy(laya_router=FakeLayaRouter())
    result = await strategy.async_get_available_deployment(model="laya-router")
    assert result is None


@pytest.mark.asyncio
async def test_empty_healthy_deployments_returns_none():
    router = _create_test_router()
    router.async_get_healthy_deployments = AsyncMock(return_value=[])
    strategy = LayaRoutingStrategy(router_instance=router, laya_router=FakeLayaRouter())

    result = await strategy.async_get_available_deployment(model="laya-router")
    assert result is None


@pytest.mark.asyncio
async def test_specific_deployment_dict_short_circuit():
    deployment = {
        "model_info": {"complexity_tier": "large"},
        "litellm_params": {"model": "openai/gpt-4o"},
    }
    router = _create_test_router()
    router.async_get_healthy_deployments = AsyncMock(return_value=deployment)
    strategy = LayaRoutingStrategy(router_instance=router, laya_router=FakeLayaRouter())

    result = await strategy.async_get_available_deployment(model="laya-router", specific_deployment=True)
    assert result == deployment


@pytest.mark.asyncio
async def test_non_dict_healthy_deployment_returns_none():
    router = _create_test_router()
    router.async_get_healthy_deployments = AsyncMock(return_value=[None])
    strategy = LayaRoutingStrategy(router_instance=router, laya_router=FakeLayaRouter())

    result = await strategy.async_get_available_deployment(model="laya-router")
    assert result is None


def test_get_available_deployment_raises_not_implemented():
    strategy = LayaRoutingStrategy(laya_router=FakeLayaRouter())
    with pytest.raises(NotImplementedError, match="async routing"):
        strategy.get_available_deployment(model="laya-router")


def test_select_deployment_empty_list():
    strategy = LayaRoutingStrategy(laya_router=FakeLayaRouter())
    selected, exact_match = strategy._select_deployment("tiny", [])
    assert selected is None
    assert exact_match is False


def test_select_deployment_skips_non_dict_entries():
    strategy = LayaRoutingStrategy(laya_router=FakeLayaRouter())
    deployment = {"model_info": {"complexity_tier": "small"}}
    selected, exact_match = strategy._select_deployment("small", ["skip-me", deployment])
    assert selected == deployment
    assert exact_match is True


def test_select_deployment_fallback_uses_first_dict():
    strategy = LayaRoutingStrategy(laya_router=FakeLayaRouter())
    deployment = {"model_info": {"complexity_tier": "small"}}
    selected, exact_match = strategy._select_deployment("large", ["skip-me", deployment])
    assert selected == deployment
    assert exact_match is False


def test_laya_tiers_from_args_uses_defaults():
    assert laya_tiers_from_args({}) == [dict(t) for t in DEFAULT_TIERS]


def test_laya_tiers_from_args_parses_custom_tiers():
    tiers = laya_tiers_from_args(
        {
            "tiers": [
                {"name": "cheap", "description": "trivial"},
                {"name": "premium", "description": "hard"},
            ]
        }
    )
    assert tiers == [
        {"name": "cheap", "description": "trivial"},
        {"name": "premium", "description": "hard"},
    ]


def test_laya_tiers_from_args_drops_malformed_entries():
    tiers = laya_tiers_from_args(
        {
            "tiers": [
                {"name": "ok", "description": "fine"},
                {"name": "", "description": "blank name dropped"},
                "not-a-dict",
                {"name": "also-ok", "description": "fine too"},
            ]
        }
    )
    assert [t["name"] for t in tiers] == ["ok", "also-ok"]


def test_laya_tiers_from_args_rejects_too_few_tiers():
    with pytest.raises(ValueError, match="at least 2 tiers"):
        laya_tiers_from_args({"tiers": [{"name": "solo", "description": "only one"}]})


def test_laya_tiers_from_args_rejects_duplicate_names():
    with pytest.raises(ValueError, match="unique names"):
        laya_tiers_from_args(
            {
                "tiers": [
                    {"name": "dup", "description": "a"},
                    {"name": "dup", "description": "b"},
                ]
            }
        )


def test_apply_laya_routing_strategy_wires_custom_selector():
    router = Router(model_list=_model_list(), routing_strategy="simple-shuffle")
    apply_laya_routing_strategy(router, {})
    assert router.routing_strategy == "laya"
    with pytest.raises(NotImplementedError, match="async routing"):
        router.get_available_deployment(model="laya-router")


@pytest.mark.asyncio
async def test_router_init_with_laya_routing_strategy():
    router = Router(
        model_list=_model_list(),
        routing_strategy="laya",
        routing_strategy_args={"tiers": [t for t in laya_tiers_from_args({})]},
    )
    assert router.routing_strategy == "laya"

    strategy = router.async_get_available_deployment.__self__
    strategy._laya_router = FakeLayaRouter(score=3.0)

    result = await router.async_get_available_deployment(
        model="laya-router",
        messages=[{"role": "user", "content": "design a distributed system"}],
    )
    assert result["model_info"]["complexity_tier"] == "large"


@pytest.mark.asyncio
async def test_update_settings_switches_to_laya_routing():
    router = Router(model_list=_model_list(), routing_strategy="simple-shuffle")
    router.update_settings(routing_strategy="laya", routing_strategy_args={})

    strategy = router.async_get_available_deployment.__self__
    strategy._laya_router = FakeLayaRouter(score=0.0)

    result = await router.async_get_available_deployment(
        model="laya-router",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert router.routing_strategy == "laya"
    assert result["model_info"]["complexity_tier"] == "tiny"


def test_update_settings_switching_from_laya_restores_default_selectors():
    router = Router(model_list=_model_list(), routing_strategy="laya")

    router.update_settings(routing_strategy="simple-shuffle")

    assert router.routing_strategy == "simple-shuffle"
    assert "get_available_deployment" not in router.__dict__
    assert "async_get_available_deployment" not in router.__dict__
    result = router.get_available_deployment(model="laya-router")
    assert result["model_name"] == "laya-router"


def test_apply_laya_invalid_tiers_leaves_router_unchanged():
    router = Router(model_list=_model_list(), routing_strategy="simple-shuffle")

    with pytest.raises(ValueError, match="at least 2 tiers"):
        apply_laya_routing_strategy(router, {"tiers": [{"name": "solo", "description": "x"}]})

    assert router.routing_strategy == "simple-shuffle"
    assert "async_get_available_deployment" not in router.__dict__
