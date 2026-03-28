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
    ui = await state.state_provider.get_state()
    state.ui = ui

    result = ui.formatted_text
    if ui.phone_state:
        result += f"\n\nPhone state: {json.dumps(ui.phone_state)}"
    return result


@mcp.tool()
async def tap_element(index: int, ctx: Context) -> str:
    """Tap a UI element by its index from the get_ui output.

    Call get_ui first to see available elements and their indices.
    The tap will land at the centre of the element's bounds.
    """
    state = await _ensure_connected(ctx)
    if state.ui is None:
        return "Error: call get_ui first to load the UI tree."
    try:
        x, y = state.ui.get_element_coords(index)
        await state.driver.tap(x, y)
        info = state.ui.get_element_info(index)
        text = info.get("text", "")
        cls = info.get("className", "")
        return f"Tapped element {index} ({cls}: '{text}') at ({x}, {y})"
    except (ValueError, IndexError) as e:
        return f"Error: {e}"


@mcp.tool()
async def tap_xy(x: int, y: int, ctx: Context) -> str:
    """Tap at exact screen coordinates (pixels)."""
    state = await _ensure_connected(ctx)
    await state.driver.tap(x, y)
    return f"Tapped ({x}, {y})"


@mcp.tool()
async def swipe(
    x1: int, y1: int, x2: int, y2: int, duration_ms: int = 1000, ctx: Context = None
) -> str:
    """Swipe from (x1, y1) to (x2, y2).

    Common patterns:
    - Scroll down: swipe(540, 1500, 540, 500)
    - Scroll up: swipe(540, 500, 540, 1500)
    - Swipe left: swipe(900, 1200, 100, 1200)
    """
    state = await _ensure_connected(ctx)
    await state.driver.swipe(x1, y1, x2, y2, duration_ms=duration_ms)
    return f"Swiped ({x1},{y1}) -> ({x2},{y2}) over {duration_ms}ms"


@mcp.tool()
async def input_text(text: str, clear: bool = False, ctx: Context = None) -> str:
    """Type text into the currently focused input field.

    Use tap_element to focus an input field first, then call this.
    Set clear=True to clear existing text before typing.
    """
    state = await _ensure_connected(ctx)
    success = await state.driver.input_text(text, clear)
    if success:
        return f"Typed: '{text}'" + (" (cleared first)" if clear else "")
    return "Error: failed to type text. Is an input field focused?"


@mcp.tool()
async def press_button(button: str, ctx: Context = None) -> str:
    """Press a system button.

    Supported buttons: back, home, enter
    """
    button = button.lower()
    if button not in ("back", "home", "enter"):
        return f"Error: unknown button '{button}'. Use: back, home, enter"
    state = await _ensure_connected(ctx)
    await state.driver.press_button(button)
    return f"Pressed {button}"


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
    """
    state = await _ensure_connected(ctx)
    result = await state.driver.start_app(package)
    return result or f"Started {package}"


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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    """Run the MCP server over stdio transport."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
