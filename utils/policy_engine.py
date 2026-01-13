import time
from typing import Dict, Any


class PolicyEngine:
    def __init__(self, policies: list[dict]):
        self.policies = policies
        self._cooldowns = {}

    def evaluate(self, analysis: Dict[str, dict]):
        """
        Returns:
            action dict or None
        """
        now = time.time()

        for policy in self.policies:
            name = policy.get("name", "<unnamed>")
            when = policy.get("when", {})
            action = policy.get("action", {})

            region_name = when.get("region")
            if region_name not in analysis:
                continue

            region_state = analysis[region_name]

            # ---- match condition ----
            if when.get("matched") is not None:
                if region_state["matched"] != when["matched"]:
                    continue

            # ---- confidence condition ----
            conf_gte = when.get("confidence_gte")
            if conf_gte is not None:
                if region_state.get("confidence", 0.0) < conf_gte:
                    continue

            # ---- cooldown ----
            cooldown = action.get("cooldown", 0.0)
            last_fire = self._cooldowns.get(name, 0.0)
            if now - last_fire < cooldown:
                continue

            # ---- fire ----
            self._cooldowns[name] = now
            return {
                "policy": name,
                "action": action,
                "region": region_name,
                "confidence": region_state.get("confidence", 0.0),
            }

        return None
