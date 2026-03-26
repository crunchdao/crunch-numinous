from __future__ import annotations

from numinous.tracker import TrackerBase
from openai_predictor import agent_main


class OpenaiPredictor(TrackerBase):

    def _predict(self, subject: str) -> dict:
        data = self._get_data(subject)
        if not isinstance(data, dict):
            return {"event_id": subject, "prediction": 0.5}
        return agent_main(data)
