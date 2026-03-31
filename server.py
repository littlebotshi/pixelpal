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
import io
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator, List, Optional, Tuple, Dict, Any

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

# Max screenshot dimension for the compact screenshot tool
_SCREENSHOT_MAX_DIM = 720

# If compact output has fewer text elements than this, auto-attach a screenshot
_MIN_CONTENT_ELEMENTS = 5

# Smart settle: max wait, poll interval, and min wait
_SETTLE_MAX_WAIT = 3.0
_SETTLE_POLL_INTERVAL = 0.4
_SETTLE_MIN_WAIT = 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _search_elements(
    elements: List[Dict[str, Any]], pattern: re.Pattern
) -> List[Dict[str, Any]]:
    """Recursively search UIState elements for text matching pattern.

    Searches text, contentDescription, and child texts.
    Returns list of matching element dicts.
    """
    matches = []
    for elem in elements:
        text = elem.get("text", "")
        desc = elem.get("contentDescription", "")
        searchable = f"{text} {desc}"
        if pattern.search(searchable):
            matches.append(elem)
        # Recurse into children
        children = elem.get("children", [])
        if children:
            matches.extend(_search_elements(children, pattern))
    return matches


def _elem_center(elem: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """Get center coordinates from an element's bounds string.

    Handles both formats: "left,top,right,bottom" and "[left,top][right,bottom]"
    """
    bounds = elem.get("bounds", "")
    if not bounds:
        return None
    nums = re.findall(r"\d+", bounds)
    if len(nums) < 4:
        return None
    left, top, right, bottom = int(nums[0]), int(nums[1]), int(nums[2]), int(nums[3])
    return (left + right) // 2, (top + bottom) // 2


def _is_noise_text(text: str) -> bool:
    """Return True if text is just a class name, resource ID, or other noise."""
    if not text:
        return True
    # Android/Java class names
    if text.startswith(("android.", "androidx.", "com.")) and ("." in text[4:]):
        return True
    # Resource IDs
    if ":id/" in text:
        return True
    return False


def _format_compact(ui: UIState) -> str:
    """Format UIState as a compact list, deduped by bounds.

    Filters out class names, resource IDs, and wrapper elements.
    When multiple elements share the same bounds, keeps only the one
    with the longest meaningful text (the leaf with real content).
    """
    all_elems = UIState._collect_all(ui.elements)

    # Group by bounds, keeping only elements with real text
    by_bounds: dict[str, list[tuple[int, str]]] = {}
    for elem in all_elems:
        idx = elem.get("index")
        if idx is None:
            continue
        text = elem.get("text", "")
        if _is_noise_text(text):
            continue
        bounds = elem.get("bounds", "")
        key = bounds or str(idx)
        by_bounds.setdefault(key, []).append((idx, text))

    # For each bounds group, pick the entry with the longest text
    # Truncate very long text to save tokens (150 chars keeps enough context)
    lines = []
    for bounds_key, entries in by_bounds.items():
        idx, text = max(entries, key=lambda e: len(e[1]))
        if len(text) > 150:
            text = text[:147] + "..."
        lines.append((idx, f"[{idx}] '{text}' ({bounds_key})"))

    # Sort by index for stable ordering
    lines.sort(key=lambda x: x[0])

    header = ""
    if ui.phone_state:
        app = ui.phone_state.get("currentApp", "?")
        kb = "KB:visible" if ui.phone_state.get("keyboardVisible") else "KB:hidden"
        header = f"App: {app} | {kb}\n"

    result = [l[1] for l in lines]
    return header + "\n".join(result) if result else header + "(no text elements)"


_GENERIC_LABELS = {
    "back", "home", "menu", "search", "close", "cancel", "ok", "done",
    "send a like", "open camera.", "open photo gallery.", "open audio recorder.",
    "open sticker, emoji and gif keyboard.", "message", "tap to send",
    "chat profile", "view buyer profile", "marketplace listing",
    "tap to refresh", "send", "profile picture",
}


def _summarize_ui(ui: UIState, max_items: int = 6, max_len: int = 250) -> str:
    """Generate a 1-2 line breadcrumb summary of the current screen.

    Extracts the most informative text elements: messages with content,
    prices, names, statuses — anything an agent might need to recall later.
    Skips generic UI labels (Back, Menu, Send) and deduplicates by content.
    """
    all_elems = UIState._collect_all(ui.elements)

    # Collect meaningful text, deduped by bounds AND content
    seen_bounds: set[str] = set()
    seen_text: set[str] = set()
    items: list[str] = []
    for elem in all_elems:
        text = elem.get("text", "")
        if _is_noise_text(text):
            continue
        bounds = elem.get("bounds", "")
        if bounds in seen_bounds:
            continue
        seen_bounds.add(bounds)
        # Skip very short or generic labels
        if len(text) < 4 or text.lower().strip() in _GENERIC_LABELS:
            continue
        # Skip if a longer item already contains this text (substring dedup)
        text_lower = text.lower()
        if any(text_lower in s for s in seen_text):
            continue
        # Remove items that are substrings of this new text
        seen_text = {s for s in seen_text if s not in text_lower}
        seen_text.add(text_lower)
        # Truncate individual items
        if len(text) > 80:
            text = text[:77] + "..."
        items.append(text)

    if not items:
        return "(empty screen)"

    # Build the app context
    app = ""
    if ui.phone_state:
        app = ui.phone_state.get("currentApp", "")

    # Take the top N items, join with semicolons
    summary_items = items[:max_items]
    summary = "; ".join(summary_items)
    if len(items) > max_items:
        summary += f" (+{len(items) - max_items} more)"

    # Cap total length
    if len(summary) > max_len:
        summary = summary[:max_len - 3] + "..."

    prefix = f"[{app}] " if app else ""
    return prefix + summary


def _count_app_elements(ui: UIState) -> int:
    """Count text elements that belong to the app (not Chrome toolbar, etc.)."""
    chrome_ids = {"com.android.chrome:id/toolbar", "com.android.chrome:id/toolbar_container",
                  "com.android.chrome:id/location_bar", "com.android.chrome:id/url_bar",
                  "com.android.chrome:id/tab_switcher_button", "com.android.chrome:id/menu_button",
                  "com.android.chrome:id/home_button", "com.android.chrome:id/location_bar_status",
                  "com.android.chrome:id/location_bar_status_icon", "com.android.chrome:id/optional_toolbar_button",
                  "com.android.chrome:id/bar_items_view"}
    count = 0
    all_elems = UIState._collect_all(ui.elements)
    for elem in all_elems:
        text = elem.get("text", "")
        if not text:
            continue
        rid = elem.get("resourceId", "")
        # Skip Chrome toolbar elements
        if rid in chrome_ids:
            continue
        # Skip elements that look like Chrome toolbar (URL bar, tab count, etc.)
        if "chrome" in rid.lower():
            continue
        count += 1
    return count


# ---------------------------------------------------------------------------
# Danger gate — blocks irreversible payment/order taps without explicit opt-in
# ---------------------------------------------------------------------------

# Patterns that indicate a payment or irreversible order confirmation.
# Matched case-insensitively against element text and contentDescription.
_DANGER_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"^place order$",
        r"^place my order$",
        r"^confirm order$",
        r"^confirm and pay",
        r"^confirm purchase$",
        r"^complete order$",
        r"^complete purchase$",
        r"^submit order$",
        r"^pay now$",
        r"^pay \$",          # "Pay $12.50"
        r"^buy now$",
        r"^purchase$",
        r"^checkout$",
        r"^check out$",
        r"^order now$",
        r"^yes,? place",     # "Yes, place my order"
        r"^subscribe and pay",
        r"^start (free )?trial",
    ]
]


def _is_dangerous(text: str) -> bool:
    """Return True if the element text matches a payment/order danger pattern."""
    t = (text or "").strip()
    return any(p.search(t) for p in _DANGER_PATTERNS)


def _danger_block_message(element_text: str) -> str:
    return (
        f"🚫 BLOCKED — '{element_text}' is a payment/order confirmation button.\n"
        f"This action is irreversible. You MUST ask the user for explicit approval "
        f"before proceeding.\n"
        f"Once the user confirms, re-call this tool with confirmed=True to execute."
    )


def _resize_png_safe(png_bytes: bytes, max_dim: int) -> Tuple[bytes, str]:
    """Resize a PNG to JPEG. Returns (bytes, format).

    Falls back to raw PNG if Pillow fails.
    """
    try:
        from PIL import Image as PILImage

        img = PILImage.open(io.BytesIO(png_bytes))
        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), PILImage.LANCZOS)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=60)
        return buf.getvalue(), "jpeg"
    except Exception as e:
        logger.warning(f"Pillow resize failed ({e}), returning raw PNG")
        return png_bytes, "png"


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
        """Fetch the current UI tree and cache it. Returns full formatted text."""
        ui = await self.state_provider.get_state()
        self.ui = ui
        result = ui.formatted_text
        if ui.phone_state:
            result += f"\n\nPhone state: {json.dumps(ui.phone_state)}"
        return result

    async def refresh_ui_compact(self):
        """Fetch the current UI tree and cache it. Returns compact text.

        Only shows elements that have text, in format: [index] 'text' (bounds)
        Much smaller than full format — saves ~70% context tokens.

        Auto-fallback: if the app content is thin (e.g., WebView dropped elements),
        automatically attaches a compressed screenshot so the caller isn't blind.
        Returns str normally, or [str, Image] when fallback is triggered.
        """
        ui = await self.state_provider.get_state()
        self.ui = ui
        compact = _format_compact(ui)

        # Check if content is thin (likely WebView elements dropped off)
        app_elements = _count_app_elements(ui)
        if app_elements < _MIN_CONTENT_ELEMENTS:
            # Auto-attach a screenshot for visual fallback
            try:
                png_bytes = await self.driver.screenshot()
                img_bytes, fmt = _resize_png_safe(png_bytes, _SCREENSHOT_MAX_DIM)
                compact += "\n\n(!) Few UI elements detected — auto-attaching screenshot"
                return [compact, Image(data=img_bytes, format=fmt)]
            except Exception:
                compact += "\n\n(!) Few UI elements detected — screenshot failed"
        return compact

    async def ensure_ui(self) -> None:
        """Ensure UI cache is populated. Auto-refresh if stale/None."""
        if self.ui is None:
            await self.refresh_ui()

    async def smart_settle(self) -> None:
        """Wait for UI to stabilize after an action.

        Polls the UI tree and waits until it stops changing,
        with a minimum wait and maximum timeout.
        """
        await asyncio.sleep(_SETTLE_MIN_WAIT)
        prev_text = None
        elapsed = _SETTLE_MIN_WAIT
        while elapsed < _SETTLE_MAX_WAIT:
            ui = await self.state_provider.get_state()
            curr_text = ui.formatted_text
            if prev_text is not None and curr_text == prev_text:
                # UI stabilized
                self.ui = ui
                return
            prev_text = curr_text
            self.ui = ui
            await asyncio.sleep(_SETTLE_POLL_INTERVAL)
            elapsed += _SETTLE_POLL_INTERVAL

    async def long_press(self, x: int, y: int, duration_ms: int = 1000) -> None:
        """Long press via ADB swipe-in-place (no move = long press)."""
        await self.driver.device.shell(
            f"input swipe {x} {y} {x} {y} {duration_ms}"
        )

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
        "Call get_ui to see what's on screen (returns full indexed accessibility tree). "
        "Action tools (tap_element, scroll, press_button, etc.) return a COMPACT "
        "UI summary with just [index] 'text' (bounds) — call get_ui if you need "
        "full element details like classNames and resourceIds. "
        "Use find_and_tap to tap elements by text when you don't have the index. "
        "Use scroll_to_text to scroll until specific text appears on screen. "
        "Prefer screenshot_small over screenshot to save context window space. "
        "IMPORTANT: Do NOT call nonexistent tools. The correct tool names are: "
        "tap_xy (NOT 'tap'), press_button (NOT 'press_key'), start_app (NOT 'launch_app'). "
        "All text parameters must be strings, not numbers (use '98122' not 98122). "
        "For start_app/stop_app/launch_app, use 'package' not 'package_name' — "
        "though both are accepted. "
        "To switch apps stuck in foreground: press_button('home') first, then stop_app, then start_app. "
        "In messaging apps (Messenger, WhatsApp, iMessage), Enter does NOT send — "
        "use input_text() then send_current_input() to send messages. "
        "ALWAYS verify writes (sent messages, submitted forms) via get_ui before declaring success. "
        "PAYMENT GATE: tap_element, find_and_tap, and tap_xy will BLOCK any tap on a payment or "
        "order confirmation button (Place Order, Pay Now, Confirm Purchase, etc.) and return an error. "
        "You MUST ask the user for approval first, then re-call with confirmed=True."
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


def _action_result(prefix: str, ui_result, state: "DeviceState" = None) -> str | list:
    """Combine an action prefix, breadcrumb summary, and compact UI result.

    Output format:
        {action_line}
        BREADCRUMB: {short semantic summary of current screen}

        {full compact UI tree}

    The BREADCRUMB line is designed for the context-trim plugin: when the
    tool result is persisted to the transcript, the plugin keeps everything
    up to and including BREADCRUMB, and drops the full UI below.
    This gives future turns a useful summary without the full tree.
    """
    breadcrumb = ""
    if state and state.ui:
        breadcrumb = f"\nBREADCRUMB: {_summarize_ui(state.ui)}"

    if isinstance(ui_result, list):
        # ui_result is [compact_text, Image(...)]
        text_part = ui_result[0] if ui_result else ""
        rest = ui_result[1:] if len(ui_result) > 1 else []
        return [f"{prefix}{breadcrumb}\n\n{text_part}"] + rest
    return f"{prefix}{breadcrumb}\n\n{ui_result}"


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
async def tap_element(
    index: int,
    ctx: Context,
    use_clear_point: bool = False,
    confirmed: bool = False,
) -> str:
    """Tap a UI element by its index from the get_ui output.

    Call get_ui first to see available elements and their indices.
    The tap will land at the centre of the element's bounds.

    Args:
        index: Element index from get_ui output.
        use_clear_point: If True, find a tap point that avoids overlapping
            elements (useful for crowded UIs like lists with favorite buttons).
        confirmed: Must be True to tap payment/order confirmation buttons
            (e.g. "Place Order", "Pay Now", "Confirm Purchase"). The server
            blocks these automatically — only pass confirmed=True after the
            user has explicitly approved the action.
    """
    state = await _ensure_connected(ctx)
    await state.ensure_ui()
    try:
        if use_clear_point:
            x, y = state.ui.get_clear_point(index)
        else:
            x, y = state.ui.get_element_coords(index)
        info = state.ui.get_element_info(index)
        text = info.get("text", "")

        # Danger gate: block payment/order confirmations without explicit opt-in
        if _is_dangerous(text) and not confirmed:
            return _danger_block_message(text)

        await state.driver.tap(x, y)
        await state.smart_settle()
        ui_text = await state.refresh_ui_compact()
        return _action_result(f"Tapped '{text}' at ({x}, {y})", ui_text, state)
    except (ValueError, IndexError) as e:
        return f"Error: {e}"


@mcp.tool()
async def tap_xy(
    x: int, y: int, ctx: Context, confirmed: bool = False
) -> str:
    """Tap at exact screen coordinates (pixels).
    Automatically returns the updated UI tree after tapping.

    Args:
        confirmed: Must be True if the coordinates correspond to a payment or
            order confirmation button. The server will warn you if it detects
            a danger element near these coordinates.
    """
    state = await _ensure_connected(ctx)

    # Soft danger check: look for danger elements near these coordinates
    if state.ui and not confirmed:
        all_elems = UIState._collect_all(state.ui.elements)
        for elem in all_elems:
            center = _elem_center(elem)
            if center is None:
                continue
            ex, ey = center
            # Within 100px of tap point
            if abs(ex - x) < 100 and abs(ey - y) < 100:
                text = elem.get("text", "")
                if _is_dangerous(text):
                    return _danger_block_message(text)

    await state.driver.tap(x, y)
    await state.smart_settle()
    ui_text = await state.refresh_ui_compact()
    return _action_result(f"Tapped ({x}, {y})", ui_text, state)


# Alias: models often hallucinate "tap" instead of "tap_xy"
@mcp.tool()
async def tap(x: int, y: int, ctx: Context) -> str:
    """Alias for tap_xy. Tap at exact screen coordinates (pixels).

    NOTE: Prefer tap_element(index) when you have an element index,
    or find_and_tap(text) when you know the text. Use this only
    when you need to tap specific pixel coordinates.
    """
    return await tap_xy(x, y, ctx)


@mcp.tool()
async def long_press(
    x: int, y: int, duration_ms: int = 1000, ctx: Context = None
) -> str:
    """Long press at screen coordinates. Useful for context menus, drag initiation, etc.

    Args:
        x: X coordinate in pixels.
        y: Y coordinate in pixels.
        duration_ms: How long to hold (default 1000ms).
    """
    state = await _ensure_connected(ctx)
    await state.long_press(x, y, duration_ms)
    await state.smart_settle()
    ui_text = await state.refresh_ui_compact()
    return _action_result(f"Long pressed ({x}, {y}) for {duration_ms}ms", ui_text, state)


@mcp.tool()
async def long_press_element(
    index: int, duration_ms: int = 1000, ctx: Context = None
) -> str:
    """Long press a UI element by its index from get_ui output.

    Args:
        index: Element index from get_ui output.
        duration_ms: How long to hold (default 1000ms).
    """
    state = await _ensure_connected(ctx)
    await state.ensure_ui()
    try:
        x, y = state.ui.get_element_coords(index)
        info = state.ui.get_element_info(index)
        text = info.get("text", "")
        await state.long_press(x, y, duration_ms)
        await state.smart_settle()
        ui_text = await state.refresh_ui_compact()
        return _action_result(f"Long pressed '{text}' at ({x}, {y}) for {duration_ms}ms", ui_text, state)
    except (ValueError, IndexError) as e:
        return f"Error: {e}"


@mcp.tool()
async def find_and_tap(
    text: str, ctx: Context, occurrence: int = 1, confirmed: bool = False
) -> str:
    """Find a UI element by its text content and tap it.

    Searches the current UI tree for elements matching the given text
    (case-insensitive substring match). Much more reliable than guessing
    coordinates or scrolling through indices.

    Args:
        text: Text to search for (case-insensitive substring match).
        occurrence: Which match to tap if multiple found (1 = first, 2 = second, etc.).
        confirmed: Must be True to tap payment/order confirmation buttons
            (e.g. "Place Order", "Pay Now"). Only pass after user approval.
    """
    # Type coercion: models often pass int instead of str
    text = str(text)

    state = await _ensure_connected(ctx)
    await state.refresh_ui()
    if state.ui is None:
        return "Error: could not fetch UI tree."

    pattern = re.compile(re.escape(text), re.IGNORECASE)
    matches = _search_elements(state.ui.elements, pattern)

    if not matches:
        return f"No element found with text matching '{text}'. Current UI:\n\n{_format_compact(state.ui)}"

    if occurrence > len(matches):
        names = [m.get("text", "?") for m in matches]
        return f"Only {len(matches)} match(es) for '{text}': {names}. Requested occurrence {occurrence}."

    target = matches[occurrence - 1]
    center = _elem_center(target)
    if center is None:
        return f"Found '{text}' but element has no bounds."

    matched_text = target.get("text", text)

    # Danger gate: block payment/order confirmations without explicit opt-in
    if _is_dangerous(matched_text) and not confirmed:
        return _danger_block_message(matched_text)

    x, y = center
    await state.driver.tap(x, y)
    await state.smart_settle()
    ui_text = await state.refresh_ui_compact()
    return _action_result(f"Found and tapped '{matched_text}' at ({x}, {y})", ui_text, state)


@mcp.tool()
async def scroll_to_text(
    text: str,
    direction: str = "down",
    max_scrolls: int = 10,
    ctx: Context = None,
) -> str:
    """Scroll until an element with the given text appears on screen.

    Repeatedly scrolls and checks the UI tree for matching text.
    Stops when found or after max_scrolls attempts.

    Args:
        text: Text to search for (case-insensitive substring match).
        direction: Scroll direction — "up" or "down" (default "down").
        max_scrolls: Maximum number of scroll attempts (default 10).

    Returns the UI tree when found, or an error if not found.
    """
    text = str(text)
    state = await _ensure_connected(ctx)
    pattern = re.compile(re.escape(text), re.IGNORECASE)

    w = state.ui.screen_width if state.ui else 1080
    h = state.ui.screen_height if state.ui else 2400
    cx = w // 2
    dist = int(h * 0.35)

    ui_text = ""
    for attempt in range(max_scrolls + 1):
        await state.refresh_ui()
        if state.ui is None:
            return "Error: could not fetch UI tree."

        # Search the cached elements for the text
        matches = _search_elements(state.ui.elements, pattern)
        ui_text = state.ui.formatted_text
        if state.ui.phone_state:
            ui_text += f"\n\nPhone state: {json.dumps(state.ui.phone_state)}"

        if matches:
            return f"Found '{text}' after {attempt} scroll(s)\n\n{ui_text}"

        if attempt < max_scrolls:
            if direction == "down":
                await state.driver.swipe(cx, h // 2 + dist // 2, cx, h // 2 - dist // 2, duration_ms=800)
            else:
                await state.driver.swipe(cx, h // 2 - dist // 2, cx, h // 2 + dist // 2, duration_ms=800)
            await asyncio.sleep(_SETTLE_MIN_WAIT)

    return f"'{text}' not found after {max_scrolls} scrolls ({direction}). Last UI:\n\n{ui_text}"


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

    w = state.ui.screen_width if state.ui else 1080
    h = state.ui.screen_height if state.ui else 2400

    cx, cy = w // 2, h // 2
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
    await state.smart_settle()
    ui_text = await state.refresh_ui_compact()
    return _action_result(f"Scrolled {direction} (amount={amount})", ui_text, state)


@mcp.tool()
async def scroll_element(
    index: int,
    direction: str = "down",
    amount: int = 3,
    ctx: Context = None,
) -> str:
    """Scroll within a specific element's bounds (e.g., a horizontal tab bar or list).

    This is critical for horizontal tab bars and scrollable containers where a
    full-screen scroll would miss the target area.

    Args:
        index: Element index from get_ui to scroll within.
        direction: "up", "down", "left", or "right".
        amount: 1-5 (small to large scroll distance). Default 3.
    """
    state = await _ensure_connected(ctx)
    await state.ensure_ui()

    try:
        elem = state.ui.get_element(index)
        if not elem:
            return f"Error: no element with index {index}"

        center = _elem_center(elem)
        if center is None:
            return f"Error: can't parse bounds for element {index}"

        bounds = elem.get("bounds", "")
        nums = re.findall(r"\d+", bounds)
        left, top, right, bottom = int(nums[0]), int(nums[1]), int(nums[2]), int(nums[3])
        cx, cy = center
        w = right - left
        h = bottom - top

        frac = 0.15 + (amount * 0.12)
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
        await state.driver.swipe(x1, y1, x2, y2, duration_ms=600)
        await state.smart_settle()
        ui_text = await state.refresh_ui_compact()
        return _action_result(f"Scrolled {direction} within element {index}", ui_text, state)
    except (ValueError, IndexError) as e:
        return f"Error: {e}"


@mcp.tool()
async def swipe(
    x1: int, y1: int, x2: int, y2: int, duration_ms: int = 1000, ctx: Context = None
) -> str:
    """Swipe from (x1, y1) to (x2, y2) for precise control.

    For simple scrolling, prefer the scroll() tool instead.
    """
    state = await _ensure_connected(ctx)
    await state.driver.swipe(x1, y1, x2, y2, duration_ms=duration_ms)
    await state.smart_settle()
    ui_text = await state.refresh_ui_compact()
    return _action_result(f"Swiped ({x1},{y1}) -> ({x2},{y2})", ui_text, state)


@mcp.tool()
async def input_text(text: str, clear: bool = False, ctx: Context = None) -> str:
    """Type text into the currently focused input field.

    Use tap_element to focus an input field first, then call this.
    Set clear=True to clear existing text before typing.
    Automatically returns the updated UI tree after typing.

    ⚠️  WARNING — messaging apps (Messenger, WhatsApp, iMessage, Instagram DMs):
    pressing Enter does NOT send the message — it only adds a newline.
    After typing, call send_current_input() to find and tap the Send button,
    then verify the message appears in the chat via get_ui before declaring success.
    """
    # Type coercion: models often pass int instead of str
    text = str(text)

    state = await _ensure_connected(ctx)

    # Auto-focus: find and tap an EditText before typing.
    # This ensures the input field is properly focused, which prevents
    # text from disappearing when the accessibility tree is refreshed.
    if state.ui:
        all_elems = UIState._collect_all(state.ui.elements)
        for elem in all_elems:
            cls = elem.get("className", "").lower()
            if "edittext" in cls:
                center = _elem_center(elem)
                if center:
                    await state.driver.tap(center[0], center[1])
                    await asyncio.sleep(0.5)
                    break

    success = await state.driver.input_text(text, clear)
    if not success:
        return "Error: failed to type text. Is an input field focused?"
    await state.smart_settle()
    ui_text = await state.refresh_ui_compact()
    label = f"Typed: '{text}'" + (" (cleared first)" if clear else "")
    return _action_result(label, ui_text, state)


@mcp.tool()
async def send_current_input(ctx: Context, verify_text: Optional[str] = None) -> str:
    """Find and tap the Send button after typing a message. Then verify it was sent.

    Use this after input_text() in any messaging app (Messenger, WhatsApp,
    iMessage, Instagram DMs, etc.) where pressing Enter adds a newline instead
    of sending. This tool:
      1. Searches the current UI for common Send button patterns
      2. Taps the first match
      3. Waits for the UI to settle
      4. Verifies the message was sent (optional: check for verify_text in chat)

    Args:
        verify_text: Optional snippet of the message you just typed. If provided,
            the tool checks the post-send UI for this text to confirm delivery.
            If absent, the tool still taps Send but skips text verification.

    Returns a success/failure message plus the updated compact UI.
    """
    state = await _ensure_connected(ctx)
    # Refresh UI to get the current state — after input_text with auto-focus,
    # the text should persist and "Send a like" should have changed to "Send"
    await state.refresh_ui()

    # SEND BUTTON DETECTION
    # In messaging apps, the send button is typically:
    # - "Send a like" when input is empty (tapping sends a like/thumbs up)
    # - "Send" or a blue arrow when text is typed (tapping sends the message)
    # Both are the SAME button at the SAME position — just different labels.
    # Since we use cached UI (pre-input), we look for EITHER label.
    SEND_LABELS = {"send", "send a like", "send message", "send a thumbs up"}

    all_elems = UIState._collect_all(state.ui.elements)
    send_elem = None

    # Strategy 1: Find element with send-like text
    for elem in all_elems:
        text = elem.get("text", "").lower().strip()
        desc = elem.get("contentDescription", "").lower().strip()
        rid = elem.get("resourceId", "").lower()
        if text in SEND_LABELS or desc in SEND_LABELS:
            send_elem = elem
            break
        if any(p in rid for p in ("send_button", "btn_send", "action_send")):
            send_elem = elem
            break

    # Strategy 2: Find EditText (input field), then rightmost element in same row
    if send_elem is None:
        input_elem = None
        for elem in all_elems:
            cls = elem.get("className", "").lower()
            if "edittext" in cls:
                input_elem = elem
                break
        if input_elem:
            input_bounds = input_elem.get("bounds", "")
            nums = re.findall(r"\d+", input_bounds)
            if len(nums) >= 4:
                input_cy = (int(nums[1]) + int(nums[3])) // 2
                right_of_input = [
                    e for e in all_elems
                    if _elem_center(e) is not None
                    and _elem_center(e)[0] > int(nums[2])
                    and abs(_elem_center(e)[1] - input_cy) < 100
                ]
                if right_of_input:
                    send_elem = max(right_of_input, key=lambda e: _elem_center(e)[0])

    # Strategy 3: Rightmost bottom element
    if send_elem is None:
        w = state.ui.screen_width if state.ui else 1080
        h = state.ui.screen_height if state.ui else 2400
        candidates = [
            e for e in all_elems
            if _elem_center(e) is not None
            and _elem_center(e)[0] > w * 0.8
            and _elem_center(e)[1] > h * 0.85
            and not _is_noise_text(e.get("text", ""))
        ]
        if candidates:
            send_elem = max(candidates, key=lambda e: _elem_center(e)[1])

    if send_elem is None:
        return (
            "Could not find a Send button in the current UI. "
            "Try tap_xy on the send button coordinates from a screenshot, "
            "or use find_and_tap('Send')."
        )

    center = _elem_center(send_elem)
    if center is None:
        return "Found a Send button candidate but could not get its coordinates."

    x, y = center
    rid = send_elem.get("resourceId", "")
    label = send_elem.get("contentDescription", send_elem.get("text", rid or "button"))

    await state.driver.tap(x, y)
    await state.smart_settle()
    ui_text = await state.refresh_ui_compact()

    # Optional verification: check if verify_text appears in the refreshed UI
    if verify_text:
        verify_text = str(verify_text)
        pattern = re.compile(re.escape(verify_text), re.IGNORECASE)
        if state.ui and _search_elements(state.ui.elements, pattern):
            result = f"✓ Sent and verified — '{verify_text}' appears in chat (tapped '{label}' at {x},{y})"
        else:
            result = (
                f"⚠ Tapped Send ('{label}' at {x},{y}) but '{verify_text}' not found in UI. "
                f"The message may not have sent — check with get_ui or screenshot_small."
            )
    else:
        result = f"Tapped Send button ('{label}' at {x},{y}) — verify with get_ui that message appears in chat"

    return _action_result(result, ui_text, state)


@mcp.tool()
async def press_button(button: str, ctx: Context = None) -> str:
    """Press a system button.

    Supported buttons: back, home, enter
    Automatically returns the updated UI tree after pressing.
    """
    button = str(button).lower()
    if button not in ("back", "home", "enter"):
        return f"Error: unknown button '{button}'. Use: back, home, enter"
    state = await _ensure_connected(ctx)
    await state.driver.press_button(button)
    await state.smart_settle()
    ui_text = await state.refresh_ui_compact()
    return _action_result(f"Pressed {button}", ui_text, state)


# Alias: models often hallucinate "press_key" instead of "press_button"
@mcp.tool()
async def press_key(button: str, ctx: Context = None) -> str:
    """Alias for press_button. Press a system button (back, home, enter)."""
    return await press_button(button, ctx)


@mcp.tool()
async def screenshot(ctx: Context) -> Image:
    """Take a screenshot of the phone screen.

    Returns a PNG image. Prefer get_ui for structured information;
    use screenshot only when you need to visually inspect the screen.
    WARNING: Full-resolution screenshots are large and consume context quickly.
    Prefer screenshot_small for a compressed version.
    """
    state = await _ensure_connected(ctx)
    png_bytes = await state.driver.screenshot()
    return Image(data=png_bytes, format="png")


@mcp.tool()
async def screenshot_small(ctx: Context) -> Image:
    """Take a compressed, resized screenshot of the phone screen.

    Returns a JPEG image resized to fit within 720px on its longest side.
    Much smaller than full screenshot (~50-80KB vs ~1-3MB), saving context space.
    Use this instead of screenshot unless you need full resolution.
    """
    state = await _ensure_connected(ctx)
    png_bytes = await state.driver.screenshot()
    img_bytes, fmt = _resize_png_safe(png_bytes, _SCREENSHOT_MAX_DIM)
    return Image(data=img_bytes, format=fmt)


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
async def start_app(
    package: str, ctx: Context = None, package_name: Optional[str] = None
) -> str:
    """Launch an app by its package name.

    Call get_apps first to find the package name of the app you want.
    Example: start_app("com.google.android.apps.maps")
    Automatically returns the updated UI tree after launching.

    Args:
        package: The app package name (e.g. "com.facebook.katana").
        package_name: Alias for package — accepted to handle common model mistakes.
    """
    # Accept package_name as fallback if package was not provided
    package = str(package or package_name or "").strip()
    if not package:
        return "Error: provide a package name, e.g. start_app('com.facebook.katana')"
    state = await _ensure_connected(ctx)
    result = await state.driver.start_app(package)
    await asyncio.sleep(2.0)  # apps take longer to launch
    ui_text = await state.refresh_ui_compact()
    return _action_result(result or f"Started {package}", ui_text, state)


# Alias: models often hallucinate "launch_app" instead of "start_app"
@mcp.tool()
async def launch_app(
    package: str, ctx: Context = None, package_name: Optional[str] = None
) -> str:
    """Alias for start_app. Launch an app by its package name."""
    return await start_app(package, ctx, package_name)


@mcp.tool()
async def stop_app(
    package: str, ctx: Context = None, package_name: Optional[str] = None
) -> str:
    """Force stop an app by its package name.

    Useful for clearing a stuck foreground app before launching another.
    Tip: call press_button('home') first to dismiss any full-screen overlay,
    then stop_app, then start_app for the new app.

    Args:
        package: The app package name (e.g. "com.starbucks.mobilecard").
        package_name: Alias for package — accepted to handle common model mistakes.
    """
    # Accept package_name as fallback if package was not provided
    package = str(package or package_name or "").strip()
    if not package:
        return "Error: provide a package name, e.g. stop_app('com.starbucks.mobilecard')"
    state = await _ensure_connected(ctx)
    await state.driver.device.shell(f"am force-stop {package}")
    await asyncio.sleep(0.5)
    ui_text = await state.refresh_ui_compact()
    return _action_result(f"Stopped {package}", ui_text, state)


@mcp.tool()
async def get_element_info(index: int, ctx: Context = None) -> str:
    """Get detailed info about a UI element by index (from cached get_ui).

    Returns JSON with text, className, type, and child texts.
    Does NOT re-fetch the UI tree - uses the last get_ui snapshot.
    """
    state = await _ensure_connected(ctx)
    await state.ensure_ui()
    info = state.ui.get_element_info(index)
    if not info:
        return f"Error: no element found with index {index}"
    return json.dumps(info, indent=2)


@mcp.tool()
async def find_text(text: str, ctx: Context = None) -> str:
    """Search for text in the current UI tree without tapping.

    Returns all matching elements with their indices, bounds, and text.
    Useful for checking if something is on screen before acting.

    Args:
        text: Text to search for (case-insensitive substring match).
    """
    text = str(text)
    state = await _ensure_connected(ctx)
    await state.refresh_ui()
    if state.ui is None:
        return "Error: could not fetch UI tree."

    # Search the cached UIState elements (not the raw tree!)
    pattern = re.compile(re.escape(text), re.IGNORECASE)
    matches = _search_elements(state.ui.elements, pattern)

    if not matches:
        return f"No elements found matching '{text}'."

    results = []
    for m in matches:
        t = m.get("text", m.get("contentDescription", ""))
        b = m.get("bounds", "")
        idx = m.get("index", "?")
        results.append(f"  [{idx}] '{t}' bounds={b}")

    return f"Found {len(matches)} match(es) for '{text}':\n" + "\n".join(results)


@mcp.tool()
async def drag(
    x1: int, y1: int, x2: int, y2: int, duration_s: float = 3.0, ctx: Context = None
) -> str:
    """Drag from (x1, y1) to (x2, y2). Useful for sliders, drag-and-drop, etc.

    Uses ADB swipe with slow duration to simulate drag.
    Automatically returns the updated UI tree after dragging.
    """
    state = await _ensure_connected(ctx)
    duration_ms = int(duration_s * 1000)
    await state.driver.device.shell(
        f"input swipe {x1} {y1} {x2} {y2} {duration_ms}"
    )
    await state.smart_settle()
    ui_text = await state.refresh_ui_compact()
    return _action_result(f"Dragged ({x1},{y1}) -> ({x2},{y2}) over {duration_s}s", ui_text, state)


@mcp.tool()
async def open_url(url: str, ctx: Context = None) -> str:
    """Open a URL or deep link on the device.

    Works with web URLs (https://...) and app deep links (uber-eats://...).
    The device will open the URL in the default handler app.

    Args:
        url: URL or deep link to open.
    """
    state = await _ensure_connected(ctx)
    await state.driver.device.shell(
        f'am start -a android.intent.action.VIEW -d "{url}"'
    )
    await asyncio.sleep(2.0)  # page loads take time
    ui_text = await state.refresh_ui_compact()
    return _action_result(f"Opened: {url}", ui_text, state)


@mcp.tool()
async def wait_for(
    text: str,
    timeout_s: float = 10.0,
    poll_interval_s: float = 1.0,
    ctx: Context = None,
) -> str:
    """Wait until specific text appears on screen (no scrolling).

    Repeatedly polls the UI tree until the text is found or timeout is reached.
    Useful for waiting for pages to load, dialogs to appear, etc.

    Args:
        text: Text to wait for (case-insensitive substring match).
        timeout_s: Maximum time to wait in seconds (default 10).
        poll_interval_s: How often to check in seconds (default 1).
    """
    text = str(text)
    state = await _ensure_connected(ctx)
    pattern = re.compile(re.escape(text), re.IGNORECASE)
    elapsed = 0.0

    while elapsed < timeout_s:
        await state.refresh_ui()
        if state.ui is None:
            return "Error: could not fetch UI tree."

        matches = _search_elements(state.ui.elements, pattern)
        ui_text = state.ui.formatted_text
        if state.ui.phone_state:
            ui_text += f"\n\nPhone state: {json.dumps(state.ui.phone_state)}"

        if matches:
            return f"'{text}' appeared after {elapsed:.1f}s\n\n{ui_text}"

        await asyncio.sleep(poll_interval_s)
        elapsed += poll_interval_s

    ui_text = state.ui.formatted_text if state.ui else "(no UI)"
    if state.ui and state.ui.phone_state:
        ui_text += f"\n\nPhone state: {json.dumps(state.ui.phone_state)}"
    return f"Timeout: '{text}' not found after {timeout_s}s. Current UI:\n\n{ui_text}"


@mcp.tool()
async def wake_phone(ctx: Context) -> str:
    """Wake the phone and restart the accessibility service if needed.

    Call this FIRST if other tools return errors like "No active window",
    "Portal returned error", or hang without returning.

    This tool:
    1. Wakes the screen (KEYCODE_WAKEUP)
    2. Dismisses the lock screen (KEYCODE_MENU swipe)
    3. Force-restarts the Portal accessibility service
    4. Verifies the UI tree is accessible

    Returns the current UI tree on success, or an error if recovery failed.
    """
    state = await _ensure_connected(ctx)
    driver = state.driver

    # Wake screen
    await driver.device.shell("input keyevent KEYCODE_WAKEUP")
    await asyncio.sleep(0.5)
    await driver.device.shell("input keyevent KEYCODE_MENU")
    await asyncio.sleep(1.0)

    # Force restart Portal accessibility service
    await driver.device.shell("am force-stop com.droidrun.portal")
    await asyncio.sleep(1.0)
    await driver.device.shell(
        'settings put secure enabled_accessibility_services '
        '"com.droidrun.portal/com.droidrun.portal.service.DroidrunAccessibilityService"'
    )
    await asyncio.sleep(3.0)

    # Try to get UI tree
    try:
        ui_text = await state.refresh_ui()
        return f"Phone woken and Portal restarted successfully.\n\n{ui_text}"
    except Exception as e:
        return f"Phone woken but UI tree fetch failed: {e}. Try again in a few seconds."


# ---------------------------------------------------------------------------
# Conversation reading
# ---------------------------------------------------------------------------


def _extract_messages_from_ui(ui: UIState) -> List[Dict[str, Any]]:
    """Extract message-like elements from a chat UI.

    Returns list of dicts with 'text', 'top' (y-position), 'bounds', and 'sender'.
    Sender is inferred from horizontal position:
      - Right-aligned (center x > 60% of screen) → "You"
      - Left-aligned (center x < 40% of screen) → "Them"
      - Center → "System" (timestamps, status messages)
    Filters out UI chrome (toolbar, input fields, generic labels).
    """
    all_elems = UIState._collect_all(ui.elements)
    messages = []
    seen_text: set = set()
    screen_w = ui.screen_width if ui else 1080

    for elem in all_elems:
        text = elem.get("text", "")
        if not text or _is_noise_text(text):
            continue
        # Skip generic UI labels
        if text.lower().strip() in _GENERIC_LABELS:
            continue
        # Skip very short text that's likely a button
        if len(text) <= 2 and not text[0].isdigit():
            continue

        bounds = elem.get("bounds", "")
        nums = re.findall(r"\d+", bounds)
        if len(nums) < 4:
            continue

        left, top, right = int(nums[0]), int(nums[1]), int(nums[2])
        center_x = (left + right) // 2

        # Detect timestamps — universal across chat apps
        # Patterns: "5:30 PM", "SUN AT 2:49", "Yesterday 10:10 PM", "Mar 28, 2026"
        is_timestamp = bool(re.match(
            r'^('
            r'(MON|TUE|WED|THU|FRI|SAT|SUN|Mon|Tue|Wed|Thu|Fri|Sat|Sun)'  # day names
            r'|(\d{1,2}:\d{2}\s*(AM|PM|am|pm)?)'  # time only
            r'|(Today|Yesterday|Just now|now)'  # relative time
            r'|(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'  # month names
            r'|\d{1,2}/\d{1,2}/\d{2,4}'  # date formats
            r')',
            text.strip()
        ))

        # Infer sender from horizontal position
        # Right side = phone owner, Left side = the other person
        if is_timestamp:
            sender = "Time"
        elif center_x > screen_w * 0.60:
            sender = "This phone"
        elif center_x < screen_w * 0.40:
            sender = "Other"
        else:
            sender = "System"

        # Dedup by exact text + similar position (within 20px)
        dedup_key = f"{text}:{top // 20}"
        if dedup_key in seen_text:
            continue
        seen_text.add(dedup_key)

        messages.append({
            "text": text,
            "top": top,
            "bounds": bounds,
            "sender": sender,
        })

    # Sort by vertical position (top of screen first)
    messages.sort(key=lambda m: m["top"])
    return messages


@mcp.tool()
async def read_conversation(
    scrolls_up: int = 3,
    scrolls_down: int = 3,
    ctx: Context = None,
) -> str:
    """Read chat messages by scrolling up and down from the current position.

    Must be called when already inside a chat/conversation screen.
    Scrolls up a few screens to catch recent history, then scrolls back down
    to the latest messages. Captures all text at each position and deduplicates.

    Much faster and cheaper than reading the entire conversation — use this
    instead of multiple get_ui + scroll combos.

    Args:
        scrolls_up: How many screens to scroll up from current position (default 3).
        scrolls_down: How many screens to scroll back down (default 3).
            Set higher than scrolls_up to also read below current position.
    """
    state = await _ensure_connected(ctx)

    w = state.ui.screen_width if state.ui else 1080
    h = state.ui.screen_height if state.ui else 2400
    cx = w // 2
    scroll_dist = int(h * 0.30)

    all_messages: List[Dict[str, Any]] = []
    seen_texts: set = set()
    order_counter = 0

    def _add_new_messages(msgs: List[Dict[str, Any]]) -> int:
        nonlocal order_counter
        new_count = 0
        for m in msgs:
            key = m["text"].strip()
            if key not in seen_texts and len(key) > 2:
                seen_texts.add(key)
                m["order"] = order_counter
                order_counter += 1
                all_messages.append(m)
                new_count += 1
        return new_count

    # Capture current position first
    await state.refresh_ui()
    if state.ui:
        _add_new_messages(_extract_messages_from_ui(state.ui))

    # Phase 1: Scroll UP to catch recent history
    for i in range(scrolls_up):
        await state.driver.swipe(cx, h // 4, cx, h // 4 + scroll_dist, duration_ms=500)
        await asyncio.sleep(0.7)
        await state.refresh_ui()
        if state.ui:
            _add_new_messages(_extract_messages_from_ui(state.ui))

    # Phase 2: Scroll back DOWN past original position
    total_down = scrolls_up + scrolls_down
    for i in range(total_down):
        await state.driver.swipe(cx, h // 2 + scroll_dist // 2, cx, h // 2 - scroll_dist // 2, duration_ms=500)
        await asyncio.sleep(0.7)
        await state.refresh_ui()
        if state.ui:
            curr_msgs = _extract_messages_from_ui(state.ui)
            new_count = _add_new_messages(curr_msgs)
            # If scrolling down past the bottom, stop early
            if new_count == 0 and i >= scrolls_up:
                break

    if not all_messages:
        return "No messages found. Make sure you're inside a chat conversation before calling this tool."

    # Extract page header from the latest UI snapshot
    # The header is typically the topmost meaningful text on screen
    header = ""
    if state.ui:
        all_elems = UIState._collect_all(state.ui.elements)
        for elem in sorted(all_elems, key=lambda e: int(re.findall(r"\d+", e.get("bounds", "0,9999,0,0"))[1]) if re.findall(r"\d+", e.get("bounds", "")) else 9999):
            t = elem.get("text", "")
            if t and not _is_noise_text(t) and t.lower().strip() not in _GENERIC_LABELS and len(t) > 3:
                header = t
                break

    # Sort by order of discovery (top-to-bottom as scrolled)
    all_messages.sort(key=lambda m: m["order"])

    # Format output with sender labels
    lines = []
    for i, m in enumerate(all_messages, 1):
        text = m["text"]
        if len(text) > 300:
            text = text[:297] + "..."
        sender = m.get("sender", "?")
        prefix = {"This phone": "[This phone]", "Other": "[Other]", "Time": "[Time]", "System": "[System]"}.get(sender, "[?]")
        lines.append(f"{i}. {prefix} {text}")

    header_line = f"Page: {header}\n" if header else ""
    return f"{header_line}Found {len(all_messages)} messages in conversation:\n\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    """Run the MCP server over stdio transport."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
