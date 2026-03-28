"""PixelPal — MCP server for Android phone control via droidrun.

Exposes droidrun's device I/O (accessibility tree, tap, swipe, type,
screenshot) as MCP tools so any LLM agent can control a phone without
droidrun's built-in LLM orchestration.

Usage:
    # stdio (for Claude Code, Cursor, etc.)
    python server.py

    # Or via the installed entry point
    pixelpal
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

from async_adbutils import adb
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.utilities.types import Image

from droidrun.portal import ensure_portal_ready
from droidrun.tools.driver.android import AndroidDriver
from droidrun.tools.filters import ConciseFilter
from droidrun.tools.formatters import IndexedFormatter
from droidrun.tools.ui.provider import AndroidStateProvider
from droidrun.tools.ui.state import UIState

logger = logging.getLogger(__name__)

# Default settle time (seconds) after actions before refreshing UI
_SETTLE_DELAY = 1.0


# ---------------------------------------------------------------------------
# Lazy-connecting device state (survives MCP server startup without a phone)
# ---------------------------------------------------------------------------


@dataclass
class DeviceState:
    _driver: Optional[AndroidDriver] = field(default=None, repr=False)
    _provider: Optional[AndroidStateProvider] = field(default=None, repr=False)
    ui: Optional[UIState] = field(default=None, repr=False)
    _connected: bool = False

    async def connect(self) -> None:
        if self._connected:
            return

        serial = os.environ.get("PIXELPAL_SERIAL", os.environ.get("DROIDRUN_SERIAL"))
        use_tcp_env = os.environ.get("PIXELPAL_TCP", os.environ.get("DROIDRUN_TCP", ""))
        use_tcp = use_tcp_env.lower() in ("1", "true") if use_tcp_env else False

        if serial is None:
            devices = await adb.list()
            if not devices:
                raise RuntimeError(
                    "No connected Android devices found. "
                    "Plug in your phone and enable USB debugging."
                )
            serial = devices[0].serial

        device_obj = await adb.device(serial=serial)
        await ensure_portal_ready(device_obj, debug=False)

        self._driver = AndroidDriver(serial=serial, use_tcp=use_tcp)
        await self._driver.connect()

        self._provider = AndroidStateProvider(
            self._driver,
            tree_filter=ConciseFilter(),
            tree_formatter=IndexedFormatter(),
        )
        self._connected = True

    @property
    def driver(self) -> AndroidDriver:
        assert self._driver is not None, "Not connected — call connect() first"
        return self._driver

    @property
    def state_provider(self) -> AndroidStateProvider:
        assert self._provider is not None, "Not connected — call connect() first"
        return self._provider

    async def refresh_ui(self) -> str:
        """Fetch the current UI tree and cache it. Returns formatted text."""
        ui = await self.state_provider.get_state()
        self.ui = ui
        result = ui.formatted_text
        if ui.phone_state:
            result += f"\n\nPhone state: {json.dumps(ui.phone_state)}"
        return result

    async def shutdown(self) -> None:
        if self._driver and self._driver.device:
            try:
                await self._driver.device.shell(
                    "ime disable com.droidrun.portal/.input.DroidrunKeyboardIME"
                )
            except Exception:
                pass


@asynccontextmanager
async def device_lifespan(server: FastMCP) -> AsyncIterator[DeviceState]:
    """Yield a DeviceState that lazy-connects on first tool call."""
    state = DeviceState()
    try:
        yield state
    finally:
        await state.shutdown()


# ---------------------------------------------------------------------------
# FastMCP server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="pixelpal",
    instructions=(
        "Control an Android phone via PixelPal. "
        "Call get_ui to see what's on screen (returns an indexed accessibility tree), "
        "then use tap_element with the element index to interact. "
        "Most action tools (tap_element, scroll, press_button, etc.) return the "
        "updated UI automatically, so you rarely need to call get_ui separately. "
        "Use screenshot only when you need to visually inspect the screen."
    ),
    lifespan=device_lifespan,
)


def _get_state(ctx: Context) -> DeviceState:
    """Extract DeviceState from the MCP request context."""
    return ctx.request_context.lifespan_context


async def _ensure_connected(ctx: Context) -> DeviceState:
    """Get state and ensure device is connected (lazy init)."""
    state = _get_state(ctx)
    await state.connect()
    return state


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_ui(ctx: Context) -> str:
    """Get the current phone screen as an indexed accessibility tree.

    Returns a structured text listing all visible UI elements with indices,
    class names, resource IDs, text content, and pixel bounds.
    Use the element indices with tap_element to interact.
    Also returns phone state (current app, keyboard visibility, etc.).
    """
    state = await _ensure_connected(ctx)
    return await state.refresh_ui()


@mcp.tool()
async def tap_element(index: int, ctx: Context) -> str:
    """Tap a UI element by its index from the get_ui output.

    Call get_ui first to see available elements and their indices.
    The tap will land at the centre of the element's bounds.
    Automatically returns the updated UI tree after tapping.
    """
    state = await _ensure_connected(ctx)
    if state.ui is None:
        return "Error: call get_ui first to load the UI tree."
    try:
        x, y = state.ui.get_element_coords(index)
        info = state.ui.get_element_info(index)
        text = info.get("text", "")
        await state.driver.tap(x, y)
        await asyncio.sleep(_SETTLE_DELAY)
        ui_text = await state.refresh_ui()
        return f"Tapped '{text}' at ({x}, {y})\n\n{ui_text}"
    except (ValueError, IndexError) as e:
        return f"Error: {e}"


@mcp.tool()
async def tap_xy(x: int, y: int, ctx: Context) -> str:
    """Tap at exact screen coordinates (pixels).
    Automatically returns the updated UI tree after tapping.
    """
    state = await _ensure_connected(ctx)
    await state.driver.tap(x, y)
    await asyncio.sleep(_SETTLE_DELAY)
    ui_text = await state.refresh_ui()
    return f"Tapped ({x}, {y})\n\n{ui_text}"


@mcp.tool()
async def scroll(
    direction: str, amount: int = 3, ctx: Context = None
) -> str:
    """Scroll the screen in a direction. Much simpler than raw swipe.

    Args:
        direction: "up", "down", "left", or "right"
        amount: 1-5 (small to large scroll distance). Default 3.

    Automatically returns the updated UI tree after scrolling.
    """
    state = await _ensure_connected(ctx)

    # Use screen dimensions from last UI fetch, or defaults for common phones
    w = state.ui.screen_width if state.ui else 1080
    h = state.ui.screen_height if state.ui else 2400

    cx, cy = w // 2, h // 2
    # Scale distance: amount 1 = 20% of screen, amount 5 = 60%
    frac = 0.12 + (amount * 0.10)
    dist_x = int(w * frac)
    dist_y = int(h * frac)

    swipes = {
        "down": (cx, cy + dist_y // 2, cx, cy - dist_y // 2),
        "up": (cx, cy - dist_y // 2, cx, cy + dist_y // 2),
        "left": (cx + dist_x // 2, cy, cx - dist_x // 2, cy),
        "right": (cx - dist_x // 2, cy, cx + dist_x // 2, cy),
    }
    direction = direction.lower()
    if direction not in swipes:
        return f"Error: direction must be up/down/left/right, got '{direction}'"

    x1, y1, x2, y2 = swipes[direction]
    await state.driver.swipe(x1, y1, x2, y2, duration_ms=800)
    await asyncio.sleep(_SETTLE_DELAY)
    ui_text = await state.refresh_ui()
    return f"Scrolled {direction} (amount={amount})\n\n{ui_text}"


@mcp.tool()
async def swipe(
    x1: int, y1: int, x2: int, y2: int, duration_ms: int = 1000, ctx: Context = None
) -> str:
    """Swipe from (x1, y1) to (x2, y2) for precise control.

    For simple scrolling, prefer the scroll() tool instead.
    """
    state = await _ensure_connected(ctx)
    await state.driver.swipe(x1, y1, x2, y2, duration_ms=duration_ms)
    await asyncio.sleep(_SETTLE_DELAY)
    ui_text = await state.refresh_ui()
    return f"Swiped ({x1},{y1}) -> ({x2},{y2})\n\n{ui_text}"


@mcp.tool()
async def input_text(text: str, clear: bool = False, ctx: Context = None) -> str:
    """Type text into the currently focused input field.

    Use tap_element to focus an input field first, then call this.
    Set clear=True to clear existing text before typing.
    Automatically returns the updated UI tree after typing.
    """
    state = await _ensure_connected(ctx)
    success = await state.driver.input_text(text, clear)
    if not success:
        return "Error: failed to type text. Is an input field focused?"
    await asyncio.sleep(_SETTLE_DELAY)
    ui_text = await state.refresh_ui()
    label = f"Typed: '{text}'" + (" (cleared first)" if clear else "")
    return f"{label}\n\n{ui_text}"


@mcp.tool()
async def press_button(button: str, ctx: Context = None) -> str:
    """Press a system button.

    Supported buttons: back, home, enter
    Automatically returns the updated UI tree after pressing.
    """
    button = button.lower()
    if button not in ("back", "home", "enter"):
        return f"Error: unknown button '{button}'. Use: back, home, enter"
    state = await _ensure_connected(ctx)
    await state.driver.press_button(button)
    await asyncio.sleep(_SETTLE_DELAY)
    ui_text = await state.refresh_ui()
    return f"Pressed {button}\n\n{ui_text}"


@mcp.tool()
async def screenshot(ctx: Context) -> Image:
    """Take a screenshot of the phone screen.

    Returns a PNG image. Prefer get_ui for structured information;
    use screenshot only when you need to visually inspect the screen.
    """
    state = await _ensure_connected(ctx)
    png_bytes = await state.driver.screenshot()
    return Image(data=png_bytes, format="png")


@mcp.tool()
async def get_apps(include_system: bool = False, ctx: Context = None) -> str:
    """List installed apps on the device.

    Returns JSON array of {package, label} for each app.
    Use the package name with start_app to launch an app.
    """
    state = await _ensure_connected(ctx)
    apps = await state.driver.get_apps(include_system=include_system)
    return json.dumps(apps, indent=2)


@mcp.tool()
async def start_app(package: str, ctx: Context = None) -> str:
    """Launch an app by its package name.

    Call get_apps first to find the package name of the app you want.
    Example: start_app("com.google.android.apps.maps")
    Automatically returns the updated UI tree after launching.
    """
    state = await _ensure_connected(ctx)
    result = await state.driver.start_app(package)
    await asyncio.sleep(2.0)  # apps take longer to launch
    ui_text = await state.refresh_ui()
    return f"{result or f'Started {package}'}\n\n{ui_text}"


@mcp.tool()
async def get_element_info(index: int, ctx: Context = None) -> str:
    """Get detailed info about a UI element by index (from cached get_ui).

    Returns JSON with text, className, type, and child texts.
    Does NOT re-fetch the UI tree - uses the last get_ui snapshot.
    """
    state = await _ensure_connected(ctx)
    if state.ui is None:
        return "Error: call get_ui first to load the UI tree."
    info = state.ui.get_element_info(index)
    if not info:
        return f"Error: no element found with index {index}"
    return json.dumps(info, indent=2)


@mcp.tool()
async def drag(
    x1: int, y1: int, x2: int, y2: int, duration_s: float = 3.0, ctx: Context = None
) -> str:
    """Drag from (x1, y1) to (x2, y2). Useful for sliders, drag-and-drop, etc.

    Slower than swipe by default (3s) to ensure the drag registers.
    Automatically returns the updated UI tree after dragging.
    """
    state = await _ensure_connected(ctx)
    await state.driver.drag(x1, y1, x2, y2, duration=duration_s)
    await asyncio.sleep(_SETTLE_DELAY)
    ui_text = await state.refresh_ui()
    return f"Dragged ({x1},{y1}) -> ({x2},{y2}) over {duration_s}s\n\n{ui_text}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    """Run the MCP server over stdio transport."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
