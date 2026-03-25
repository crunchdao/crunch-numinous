from __future__ import annotations

from numinous.tracker import TrackerBase
from guoly_v32 import agent_main


class GuolyV32(TrackerBase):

    def _predict(self, subject: str, resolve_horizon_seconds: int, step_seconds: int) -> dict:
        data = self._get_data(subject)
        if not isinstance(data, dict):
            return {"event_id": subject, "prediction": 0.5}
        return agent_main(data)
