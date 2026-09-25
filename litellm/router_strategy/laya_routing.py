"""
Laya Complexity Routing Strategy

Routes each request to the deployment tier matching how complex the request
actually is, as scored by the `laya` decision model (convaiinnovations/laya
on Hugging Face: https://huggingface.co/convaiinnovations/laya).

Deployments opt into a tier via `model_info.complexity_tier` in the model_list
config. (Not `model_info.tier`: that field already exists, typed to just
"free"/"paid" for budget routing, so laya uses its own field name.) Tiers and
their descriptions are configurable via routing_strategy_args.

The `laya` package is imported lazily, on first prediction, so it is only
a hard dependency when this strategy is actually selected.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final

from litellm._logging import verbose_router_logger
from litellm.router import CustomRoutingStrategyBase

if TYPE_CHECKING:
    from litellm.router import Router

DEFAULT_TIERS: Final[tuple[dict[str, str], ...]] = (
    {
        "name": "tiny",
        "description": "single fact lookup, simple definition, basic arithmetic, or a one-line rewrite/translation",
    },
    {
        "name": "small",
        "description": "light reasoning, a short code snippet, a simple explanation, casual conversation",
    },
    {
        "name": "medium",
        "description": "multi-step reasoning, a moderately complex coding task, or analysis/writing that must stay coherent across paragraphs",
    },
    {
        "name": "large",
        "description": "hard multi-step reasoning, tricky math/logic, nontrivial system design, or anything where mistakes are costly and top quality is required",
    },
)


def laya_tiers_from_args(
    routing_strategy_args: Mapping[str, object] | None = None,
) -> list[dict[str, str]]:
    """Parse and validate the `tiers` list from routing_strategy_args, tiny -> large.

    Falls back to DEFAULT_TIERS when `tiers` is absent. Raises ValueError for a
    malformed or under-specified list rather than silently routing on a broken
    schema.
    """
    args: Final = routing_strategy_args or {}
    raw_tiers = args.get("tiers")
    if not raw_tiers:
        return [dict(t) for t in DEFAULT_TIERS]

    tiers: list[dict[str, str]] = []
    for entry in raw_tiers:
        if not isinstance(entry, Mapping):
            continue
        name = entry.get("name")
        description = entry.get("description")
        if isinstance(name, str) and name.strip() and isinstance(description, str) and description.strip():
            tiers.append({"name": name.strip(), "description": description.strip()})

    if len(tiers) < 2:
        raise ValueError(
            "laya routing_strategy_args.tiers must define at least 2 tiers, each with "
            "a non-empty 'name' and 'description', ordered from least to most complex."
        )

    names = [t["name"] for t in tiers]
    if len(set(names)) != len(names):
        raise ValueError(f"laya routing_strategy_args.tiers must have unique names, got {names}")

    return tiers


def _build_complexity_question(tiers: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    return {
        "complexity": {
            "type": "score",
            "instructions": (
                "How complex is this request to answer well? Consider the reasoning "
                "depth, domain difficulty, and quality bar required."
            ),
            "criteria": [tier["description"] for tier in tiers],
        }
    }


def _extract_prompt_text(
    messages: list[dict[str, str]] | None,
    input: str | list | None,
) -> str:
    """Best-effort extraction of the text laya should score for complexity.

    Prefers the latest user message (what the caller is actually asking for
    right now); falls back to the latest message of any role, then to the
    embeddings-style `input` argument.
    """
    if messages:
        for message in reversed(messages):
            if not isinstance(message, Mapping) or message.get("role") != "user":
                continue
            text = _flatten_content(message.get("content"))
            if text:
                return text
        for message in reversed(messages):
            if isinstance(message, Mapping):
                text = _flatten_content(message.get("content"))
                if text:
                    return text

    if isinstance(input, str) and input.strip():
        return input
    if isinstance(input, list):
        joined = " ".join(str(part) for part in input if isinstance(part, str))
        if joined.strip():
            return joined

    return ""


def _flatten_content(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [part.get("text", "") for part in content if isinstance(part, Mapping) and part.get("type") == "text"]
        return " ".join(p for p in parts if p).strip()
    return ""


class LayaRoutingStrategy(CustomRoutingStrategyBase):
    def __init__(
        self,
        router_instance: Router | None = None,
        tiers: list[dict[str, str]] | None = None,
        device: str | None = None,
        preload: bool = True,
        laya_router: Any | None = None,
    ):
        self._router = router_instance
        self.tiers = tiers or [dict(t) for t in DEFAULT_TIERS]
        self._question = _build_complexity_question(self.tiers)
        self._device = device
        self._preload = preload
        # Injected in tests to avoid depending on the real `laya` package/torch.
        self._laya_router = laya_router
        self._laya_router_lock = asyncio.Lock()

    async def _get_laya_router(self):
        if self._laya_router is not None:
            return self._laya_router
        async with self._laya_router_lock:
            if self._laya_router is None:
                from laya import Router as LayaModelRouter

                self._laya_router = await asyncio.to_thread(LayaModelRouter, preload=self._preload, device=self._device)
        return self._laya_router

    async def async_get_available_deployment(
        self,
        model: str,
        messages: list[dict[str, str]] | None = None,
        input: str | list | None = None,
        specific_deployment: bool | None = False,
        request_kwargs: dict | None = None,
    ):
        if request_kwargs is None:
            request_kwargs = {}
        if self._router is None:
            return None

        healthy: Final = await self._router.async_get_healthy_deployments(
            model=model,
            request_kwargs=request_kwargs,
            messages=messages,
            input=input,
            specific_deployment=specific_deployment,
        )
        if isinstance(healthy, dict):
            return healthy
        if not healthy:
            return None

        target = await self._classify_request(messages, input)
        selected, exact_match = self._select_deployment(target, healthy)

        if selected is None:
            return None
        if exact_match:
            verbose_router_logger.info("[laya] routed to tier=%s", target)
        else:
            actual_tier: Final = selected.get("model_info", {}).get("complexity_tier", "unknown")
            verbose_router_logger.warning(
                "[laya] no deployment for tier '%s', fallback to deployment tier '%s'", target, actual_tier
            )
        return selected

    async def _classify_request(
        self,
        messages: list[dict[str, str]] | None,
        input: str | list | None,
    ) -> str:
        prompt_text = _extract_prompt_text(messages, input)
        if not prompt_text:
            return self.tiers[0]["name"]

        laya_router = await self._get_laya_router()
        decision = await asyncio.to_thread(laya_router.predict, {"body": prompt_text}, self._question)
        score = decision["answers"]["complexity"]["score"]
        return self._tier_for_score(score)

    def _tier_for_score(self, score: float) -> str:
        index = round(score)
        index = max(0, min(index, len(self.tiers) - 1))
        return self.tiers[index]["name"]

    def _select_deployment(
        self,
        target_tier: str,
        deployments: list[dict],
    ) -> tuple[dict | None, bool]:
        if not deployments:
            return None, False

        for deployment in deployments:
            if not isinstance(deployment, dict):
                continue
            if deployment.get("model_info", {}).get("complexity_tier", "") == target_tier:
                return deployment, True

        for deployment in deployments:
            if isinstance(deployment, dict):
                return deployment, False

        return None, False

    def get_available_deployment(self, *args, **kwargs):
        raise NotImplementedError(
            "laya routing only supports async routing. Enable async_only_mode on the router or use acompletion."
        )


def apply_laya_routing_strategy(
    router: Router,
    routing_strategy_args: Mapping[str, object] | None = None,
) -> None:
    args: Final = routing_strategy_args or {}
    strategy: Final = LayaRoutingStrategy(
        router_instance=router,
        tiers=laya_tiers_from_args(args),
        device=args.get("device") if isinstance(args.get("device"), str) else None,
        preload=bool(args.get("preload", True)),
    )
    router.routing_strategy = "laya"
    router.set_custom_routing_strategy(strategy)
