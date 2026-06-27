import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import redis.asyncio as redis

@dataclass
class RoutingCriteria:
    task_type: Optional[str] = None
    max_cost_per_call: Optional[float] = None
    require_vision: bool = False
    require_long_context: bool = False
    latency_budget_ms: Optional[int] = None
    prefer_fine_tuned: bool = False

class ModelRouter:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url, decode_responses=True)

    async def get_models(self) -> List[Dict[str, Any]]:
        models_json = await self.redis_client.get("router:models")
        if not models_json:
            return []
        return json.loads(models_json)

    async def route(self, criteria: RoutingCriteria) -> str:
        models = await self.get_models()
        if not models:
            raise ValueError("No models found in registry.")

        # 4. evaluation -> always use judge model
        if criteria.task_type == "evaluation":
            # Just look for a judge model or fallback to a known judge like gpt-4o
            for m in models:
                if m.get("is_judge", False) or m["model"] == "gpt-4o":
                    return m["model"]
            return "gpt-4o" # default judge if not found

        filtered_models = models

        # 1. vision -> vision model
        if criteria.require_vision:
            filtered_models = [m for m in filtered_models if m.get("vision", False)]

        # 2. long context -> >100k model
        if criteria.require_long_context:
            filtered_models = [m for m in filtered_models if m.get("context", 0) > 100000]

        # 3. prefer fine-tuned -> route to fine-tuned
        if criteria.prefer_fine_tuned:
            fine_tuned_models = [m for m in filtered_models if m.get("fine_tuned", False)]
            if fine_tuned_models:
                filtered_models = fine_tuned_models

        # 5. max cost -> filter expensive models
        if criteria.max_cost_per_call is not None:
            filtered_models = [m for m in filtered_models if m.get("cost", float('inf')) <= criteria.max_cost_per_call]

        if not filtered_models:
            raise ValueError("No models satisfy the given routing criteria.")

        # 6. default -> cheapest satisfying model
        # Sort by cost
        filtered_models.sort(key=lambda m: m.get("cost", float('inf')))
        return filtered_models[0]["model"]
