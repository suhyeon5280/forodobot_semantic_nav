"""Headless browser that owns the rover's Agora connection.

Driven by Playwright, matching upstream earth-rovers-sdk. The previous version
used pyppeteer, which is unmaintained: it pins urllib3<2 and websockets<11, has
no Python 3.13 wheels, and upstream dropped it. Nothing here needs a specific
Python version any more.

The public API (data/front/rear/send_message/speak/take_screenshot) is unchanged,
so main.py did not have to be touched.
"""

import asyncio
import logging
import os
import time

from dotenv import load_dotenv
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import async_playwright

load_dotenv()

logger = logging.getLogger("browser_service")

# Configuration from environment variables with defaults
FORMAT = os.getenv("IMAGE_FORMAT", "jpeg")
QUALITY = float(os.getenv("IMAGE_QUALITY", "0.8"))
HAS_REAR_CAMERA = os.getenv("HAS_REAR_CAMERA", "False").lower() == "true"

if FORMAT not in ["png", "jpeg", "webp"]:
    raise ValueError("Invalid image format. Supported formats: png, jpeg, webp")

if QUALITY < 0 or QUALITY > 1:
    raise ValueError("Invalid image quality. Quality should be between 0 and 1")

SDK_PAGE_URL = os.getenv("SDK_PAGE_URL", "http://127.0.0.1:8000/sdk")

LAUNCH_ARGS = [
    "--ignore-certificate-errors",
    "--no-sandbox",
    "--autoplay-policy=no-user-gesture-required",
    "--use-fake-ui-for-media-stream",
    "--disable-application-cache",
    "--disk-cache-size=0",
]


class BrowserService:
    def __init__(self):
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._ready = False
        self._lock = None
        self._lock_loop = None
        self.last_error = None
        self._viewport = {"width": 1920, "height": 1200}

    def _get_lock(self) -> asyncio.Lock:
        # An asyncio.Lock binds to the loop that is running when it is built.
        # This service is instantiated at import time — before the ASGI server
        # creates its serving loop — so the lock must be created (and recreated
        # after a --reload loop swap) inside the running loop.
        running_loop = asyncio.get_running_loop()
        if self._lock is None or self._lock_loop is not running_loop:
            self._lock = asyncio.Lock()
            self._lock_loop = running_loop
        return self._lock

    @property
    def is_ready(self) -> bool:
        return bool(
            self._ready
            and self._page
            and not self._page.is_closed()
            and self._browser
            and self._browser.is_connected()
        )

    async def ensure_page(self):
        # Lock-free fast path: concurrent /control and /v2 calls must not
        # serialize on the init lock once the page is up.
        if self.is_ready:
            return self._page
        async with self._get_lock():
            if self.is_ready:
                return self._page
            await self._teardown()
            await self._launch()
            return self._page

    # Kept so existing callers/read-throughs of the old name still work.
    async def initialize_browser(self):
        await self.ensure_page()

    async def _launch_browser(self):
        """Pick a browser: explicit path, then installed Chrome, then bundled."""
        executable_path = os.getenv("CHROME_EXECUTABLE_PATH") or None
        if executable_path:
            logger.info("Using browser from CHROME_EXECUTABLE_PATH")
            return await self._playwright.chromium.launch(
                executable_path=executable_path, headless=True, args=LAUNCH_ARGS
            )
        # Prefer installed Google Chrome: it ships the H.264/AAC codecs some
        # rover streams need; Playwright's open-source Chromium does not and
        # decodes those streams as 0x0.
        try:
            browser = await self._playwright.chromium.launch(
                channel="chrome", headless=True, args=LAUNCH_ARGS
            )
            logger.info("Using installed Google Chrome (all codecs)")
            return browser
        except PlaywrightError:
            if not os.path.exists(self._playwright.chromium.executable_path):
                raise RuntimeError(
                    "No usable browser found. Run:"
                    " python -m playwright install chromium"
                    " (or install Google Chrome, or set CHROME_EXECUTABLE_PATH)"
                )
            logger.warning(
                "Using Playwright's bundled Chromium (no Google Chrome found)."
                " If video frames stay empty, the stream may need H.264:"
                " install Chrome or set CHROME_EXECUTABLE_PATH"
            )
            return await self._playwright.chromium.launch(
                headless=True, args=LAUNCH_ARGS
            )

    async def _launch(self):
        self._ready = False
        try:
            if self._playwright is None:
                self._playwright = await async_playwright().start()

            self._browser = await self._launch_browser()
            self._context = await self._browser.new_context(
                viewport=self._viewport,
                extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
            )
            self._page = await self._context.new_page()
            # Surface page-side RTM logs (connection state, send failures) in
            # the server's stdout.
            self._page.on(
                "console",
                lambda msg: print(f"[browser console:{msg.type}] {msg.text}"),
            )
            await self._page.goto(SDK_PAGE_URL, wait_until="domcontentloaded")
            await self._page.click("#join")
            # Wait on RTM readiness, not on a <video> element: control and
            # telemetry must still come up when a camera is offline.
            await self._page.wait_for_function(
                "() => typeof window.sendMessage === 'function'", timeout=30000
            )
            await self._page.wait_for_selector("#map", timeout=30000)
            # Let the Agora video tracks attach before the first frame grab.
            await self._page.wait_for_timeout(2000)

            call = f"""() => {{
                window.initializeImageParams({{
                    imageFormat: "{FORMAT}",
                    imageQuality: {QUALITY}
                }});
            }}"""
            await self._page.evaluate(call)
            self._ready = True
            self.last_error = None
            logger.info("Headless browser connected to %s", SDK_PAGE_URL)
        except Exception as e:
            self.last_error = str(e).split("\n", 1)[0]
            logger.error("Error initializing browser: %s", e)
            await self._teardown()
            # A failed Playwright transport cannot recover by reusing the same
            # driver instance. Recreate it on the next attempt.
            if self._playwright:
                try:
                    await self._playwright.stop()
                except Exception:
                    pass
                self._playwright = None
            raise

    async def _teardown(self):
        self._ready = False
        for target in (self._page, self._context, self._browser):
            if target:
                try:
                    await target.close()
                except Exception:
                    pass
        self._page = None
        self._context = None
        self._browser = None

    async def _invalidate(self, failed_page):
        """Tear down only if the failed page is still the active generation."""
        async with self._get_lock():
            if self._page is failed_page:
                await self._teardown()

    async def _run(self, fn, *, retry_on_disconnect: bool = True):
        page = await self.ensure_page()
        try:
            return await fn(page)
        except PlaywrightError as e:
            disconnected = page.is_closed() or not (
                self._browser and self._browser.is_connected()
            )
            if not disconnected:
                # A JavaScript error is not a browser crash — a failed
                # sendMessage must surface to /control, not silently relaunch.
                raise
            logger.warning("Browser disconnected (%s); relaunching", e)
            await self._invalidate(page)
            if not retry_on_disconnect:
                raise
            page = await self.ensure_page()
            return await fn(page)

    async def warmup(self, max_attempts: int = 5):
        """Bring the browser up before the first request, with backoff."""
        delay = 2
        for attempt in range(1, max_attempts + 1):
            try:
                await self.ensure_page()
                return True
            except Exception as e:
                logger.warning(
                    "Browser warm-up attempt %s/%s failed: %s", attempt, max_attempts, e
                )
                if attempt < max_attempts:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 30)
        logger.warning(
            "Browser warm-up gave up after %s attempts;"
            " it will initialize lazily on the next request",
            max_attempts,
        )
        return False

    async def take_screenshot(self, video_output_folder: str, elements: list):
        element_map = {"front": "#player-1000", "rear": "#player-1001", "map": "#map"}

        async def capture(page):
            screenshots = {}
            for name in elements:
                if name not in element_map:
                    logger.warning("Invalid element name: %s", name)
                    continue
                locator = page.locator(element_map[name])
                if await locator.count() == 0:
                    logger.warning("Element %s not found", element_map[name])
                    continue
                output_path = os.path.join(video_output_folder, f"{name}.png")
                start_time = time.time()
                image = await locator.screenshot(type="png", timeout=5000)
                elapsed_ms = (time.time() - start_time) * 1000
                logger.info("Screenshot for %s took %.2f ms", name, elapsed_ms)
                await asyncio.to_thread(self._write_file, output_path, image)
                screenshots[name] = output_path
            return screenshots

        return await self._run(capture)

    @staticmethod
    def _write_file(path: str, content: bytes):
        with open(path, "wb") as output:
            output.write(content)

    async def data(self) -> dict:
        return await self._run(lambda page: page.evaluate("() => window.rtm_data"))

    async def front(self) -> str:
        return await self._run(
            lambda page: page.evaluate("() => getLastBase64Frame(1000) || null")
        )

    async def rear(self) -> str:
        return await self._run(
            lambda page: page.evaluate("() => getLastBase64Frame(1001) || null")
        )

    async def send_message(self, message: dict):
        # window.sendMessage returns a promise that rejects on RTM failure;
        # Playwright awaits it, so a failed send raises here and surfaces as a
        # 500 from /control instead of a silent 200.
        return await self._run(
            lambda page: page.evaluate(
                "(message) => window.sendMessage(message)", message
            ),
            retry_on_disconnect=False,
        )

    async def speak(self, audio_url: str):
        return await self._run(
            lambda page: page.evaluate(
                "async (audioUrl) => await window.playAudioToRover(audioUrl)", audio_url
            ),
            retry_on_disconnect=False,
        )

    async def reset(self):
        """Force a relaunch on next use — rebuilds the page, the Agora
        connections, and the RTM session."""
        async with self._get_lock():
            await self._teardown()

    async def close(self):
        async with self._get_lock():
            await self._teardown()
            if self._playwright:
                try:
                    await self._playwright.stop()
                except Exception:
                    pass
                self._playwright = None

    # Backward-compatible alias
    async def close_browser(self):
        await self.close()
