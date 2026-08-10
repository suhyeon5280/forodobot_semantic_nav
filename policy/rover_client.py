"""
HTTP client for the Earth Rovers SDK server (main.py).

The policy runs in its own process so that restarting it does not drop the
rover's Agora session — re-joining costs a re-auth and, in mission mode, the
ride itself. That means everything goes over localhost HTTP.
"""

import base64
import io
import logging
from typing import NamedTuple, Optional

import requests
from PIL import Image

logger = logging.getLogger(__name__)


class Frame(NamedTuple):
    image: Image.Image
    base64: str  # kept as-is so it can be forwarded to /dataset/log-frame and the UI
    timestamp: float


class RoverClient:
    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    @staticmethod
    def _raise_for_status(response: requests.Response) -> None:
        """raise_for_status, but with the server's own explanation attached.

        A bare "400 Client Error" says nothing actionable — the SDK server puts
        the actual reason in FastAPI's {"detail": ...} body, so surface it.
        """
        if response.ok:
            return
        detail = None
        try:
            detail = response.json().get("detail")
        except Exception:  # noqa: BLE001 - body may not be JSON
            detail = (response.text or "").strip()[:200] or None

        hint = ""
        if detail and "start-mission" in str(detail):
            hint = (
                " -> remove MISSION_SLUG from .env and restart the SDK server;"
                " mission mode blocks every endpoint until /start-mission"
            )
        elif detail and "retrieve tokens" in str(detail):
            hint = " -> check SDK_API_TOKEN and BOT_SLUG in .env"

        raise requests.HTTPError(
            f"{response.status_code} from {response.request.method} "
            f"{response.url}: {detail or '(no detail)'}{hint}",
            response=response,
        )

    def front_frame(self) -> Optional[Frame]:
        """Latest front camera frame, or None if the stream is not up yet."""
        response = self._session.get(f"{self.base_url}/v2/front", timeout=self.timeout)
        if response.status_code == 404:
            return None
        self._raise_for_status(response)
        payload = response.json()
        encoded = payload["front_frame"]
        image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
        return Frame(image=image, base64=encoded, timestamp=payload["timestamp"])

    def data(self) -> dict:
        """Telemetry as broadcast by the rover over RTM (battery, GPS, IMU...)."""
        response = self._session.get(f"{self.base_url}/data", timeout=self.timeout)
        self._raise_for_status(response)
        return response.json() or {}

    def control(self, linear: float, angular: float) -> None:
        """Send a drive command. linear/angular are normalized to [-1, 1]."""
        response = self._session.post(
            f"{self.base_url}/control",
            json={"command": {"linear": linear, "angular": angular}},
            timeout=self.timeout,
        )
        self._raise_for_status(response)

    def stop(self) -> None:
        """Best-effort halt. Never raises — it runs on the way out of the loop."""
        try:
            self.control(0.0, 0.0)
        except Exception as exc:  # noqa: BLE001 - last-ditch safety path
            logger.error("failed to send stop command: %s", exc)

    # -- dataset recording (reuses the existing /dataset/* endpoints) ------

    def dataset_start(self) -> dict:
        response = self._session.post(
            f"{self.base_url}/dataset/start", timeout=self.timeout
        )
        self._raise_for_status(response)
        return response.json()

    def dataset_log_frame(
        self, timestamp: float, linear: float, angular: float, image_base64: str
    ) -> None:
        self._session.post(
            f"{self.base_url}/dataset/log-frame",
            json={
                "timestamp": timestamp,
                "linear": linear,
                "angular": angular,
                "image_base64": image_base64,
            },
            timeout=self.timeout,
        )

    def dataset_stop(self) -> dict:
        response = self._session.post(
            f"{self.base_url}/dataset/stop", timeout=self.timeout
        )
        self._raise_for_status(response)
        return response.json()

    def wait_until_ready(self, attempts: int = 30, delay: float = 2.0) -> bool:
        """Poll until the server's headless browser has a frame for us.

        The first request is what triggers the browser launch and Agora join, so
        this can take a while on a cold server.
        """
        import time

        for attempt in range(attempts):
            try:
                if self.front_frame() is not None:
                    return True
                logger.info("waiting for video stream (%d/%d)", attempt + 1, attempts)
            except Exception as exc:  # noqa: BLE001 - server may still be booting
                logger.info(
                    "waiting for server (%d/%d): %s", attempt + 1, attempts, exc
                )
            time.sleep(delay)
        return False
