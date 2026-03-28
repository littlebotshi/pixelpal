# PixelPal Agent Guide

How to control an Android phone through PixelPal's MCP tools.

## Setup

PixelPal connects to an Android device via ADB. The phone must have USB debugging enabled and be connected via USB or TCP.

```json
// Claude Code MCP config (~/.claude/settings.json)
{
  "mcpServers": {
    "pixelpal": {
      "command": "/path/to/python",
      "args": ["/path/to/pixelpal/server.py"]
    }
  }
}
```

Environment variables (optional):
- `PIXELPAL_SERIAL` — target a specific device by ADB serial
- `PIXELPAL_TCP=true` — use TCP connection instead of USB

## Core Workflow

### 1. See what's on screen

```
get_ui()  →  full accessibility tree with indices, classNames, resourceIds, bounds
```

This is your primary way to understand the screen. Every element gets an index you can use with other tools.

### 2. Interact with elements

**By index** (when you already called `get_ui`):
```
tap_element(index=14)                          # tap element 14
tap_element(index=14, use_clear_point=True)    # avoid overlapping elements
long_press_element(index=7)                    # context menu, etc.
```

**By text** (when you know what text to look for):
```
find_and_tap("Alcohol")           # finds and taps first match
find_and_tap("Add", occurrence=2) # taps the second "Add" button
```

**By coordinates** (last resort):
```
tap_xy(x=540, y=1200)
long_press(x=540, y=1200, duration_ms=1500)
```

### 3. Navigate

```
scroll(direction="down", amount=3)          # scroll screen (1-5 intensity)
scroll_element(index=22, direction="left")  # scroll within a specific element
scroll_to_text("Settings")                  # auto-scroll until text appears
press_button("back")                        # system buttons: back, home, enter
```

### 4. Type text

```
tap_element(index=25)                    # focus the input field first
input_text("hello world")               # type into focused field
input_text("new text", clear=True)      # clear field first, then type
```

### 5. Visual inspection

```
screenshot_small()   # compressed JPEG (~50-80KB) — use this by default
screenshot()         # full PNG (~1-3MB) — only when you need high resolution
```

## Response Format

**`get_ui`** returns the full tree:
```
1. ViewGroup: "com.app:id/container" - (0,0,1080,2400)
2. TextView: "com.app:id/title", "Hello World" - (42,100,500,160)
...
```

**Action tools** (tap, scroll, type, etc.) return compact output:
```
Tapped 'Hello World' at (271, 130)

App: MyApp | KB:hidden
[2] 'Hello World' (42,100,500,160)
[5] 'Submit' (42,200,300,260)
[8] 'Cancel' (400,200,600,260)
```

Only elements with text are shown. Call `get_ui` if you need full details.

### Auto-screenshot fallback

When web content drops from the accessibility tree (common in WebViews after scrolling), action tools automatically attach a compressed screenshot. You'll see:

```
(!) Few UI elements detected — auto-attaching screenshot
```

The screenshot appears as an image alongside the text. Use it to visually read the page content.

## Tool Reference

### Observation
| Tool | Purpose |
|------|---------|
| `get_ui()` | Full accessibility tree (indices, classes, IDs, bounds) |
| `get_element_info(index)` | Detailed info for one element (cached, no refresh) |
| `find_text("query")` | Search UI for text without tapping |
| `screenshot_small()` | Compressed screenshot (~50-80KB) |
| `screenshot()` | Full resolution screenshot (~1-3MB) |
| `get_apps()` | List installed apps with package names |

### Interaction
| Tool | Purpose |
|------|---------|
| `tap_element(index)` | Tap by element index |
| `tap_xy(x, y)` | Tap by coordinates |
| `tap(x, y)` | Alias for `tap_xy` |
| `find_and_tap("text")` | Find element by text and tap it |
| `long_press(x, y)` | Long press at coordinates |
| `long_press_element(index)` | Long press by element index |
| `input_text("text")` | Type into focused input |
| `press_button("back")` | System button (back/home/enter) |
| `press_key("back")` | Alias for `press_button` |

### Navigation
| Tool | Purpose |
|------|---------|
| `scroll(direction, amount)` | Scroll screen (up/down/left/right, 1-5) |
| `scroll_element(index, direction)` | Scroll within element bounds |
| `scroll_to_text("text")` | Auto-scroll until text found |
| `swipe(x1, y1, x2, y2)` | Raw swipe for precise control |
| `drag(x1, y1, x2, y2)` | Slow drag for sliders |
| `start_app("com.package.name")` | Launch an app |
| `launch_app("com.package.name")` | Alias for `start_app` |
| `stop_app("com.package.name")` | Force stop an app |
| `open_url("https://...")` | Open URL or deep link |

### Waiting
| Tool | Purpose |
|------|---------|
| `wait_for("text", timeout_s=10)` | Poll until text appears on screen |

## Patterns and Tips

### Typical app interaction flow

```
1. get_ui()                           # see what's on screen
2. tap_element(index)                 # tap something
3. (read compact response)            # action tools return compact UI
4. get_ui()                           # if you need full details again
```

### Searching and tapping by text

Prefer `find_and_tap` over `get_ui` + `tap_element` when you know the text:
```
find_and_tap("Search")       # instead of get_ui → find index → tap_element
```

### Scrolling to find content

```
scroll_to_text("Checkout")   # scrolls down until "Checkout" appears (max 10 scrolls)
```

This is better than blind scrolling — it stops as soon as the text is found and returns the full UI tree.

### Horizontal scrolling within containers

For tab bars, carousels, or horizontal lists, use `scroll_element`:
```
get_ui()                                    # find the tab bar element index
scroll_element(index=22, direction="left")  # scroll within that element only
```

### Handling WebView content dropout

Complex web apps (Google Flights, WebView-heavy apps) sometimes drop accessibility tree elements after scrolling. Signs:
- Compact output shows only Chrome toolbar elements
- `(!) Few UI elements detected` message appears

When this happens:
1. The auto-screenshot fallback provides visual context
2. Use `screenshot_small()` for additional visual checks
3. Use `scroll("up")` to scroll back to where elements are visible
4. Consider using `open_url()` with app deep links instead of web navigation

### Avoiding misclicks in crowded UIs

Lists with overlapping buttons (like favorite/heart icons over cards):
```
tap_element(index=14, use_clear_point=True)  # finds a tap point avoiding overlaps
```

### Entering text in search fields

```
tap_element(index=25)                     # tap the search field
input_text("pizza", clear=True)           # clear any existing text, type new text
press_button("enter")                     # submit the search
# OR
find_and_tap("Search result item")        # tap a search suggestion
```

### Dealing with popups and dialogs

System popups (autofill, permissions) can intercept your taps:
```
press_button("back")   # dismiss the popup
# then retry your original action
```

### Context-saving tips

- Use `screenshot_small` instead of `screenshot` (20x smaller)
- Action tools return compact output by default — only call `get_ui` when you need full element details
- Use `find_and_tap` to combine two steps (get_ui + tap) into one
- Use `scroll_to_text` instead of multiple `scroll` + `get_ui` calls

## Common Mistakes (IMPORTANT)

### Wrong tool names

These tool names DO NOT EXIST. Use the correct names:

| Wrong (will fail) | Correct |
|---|---|
| `tap(x, y)` | `tap_xy(x, y)` |
| `press_key("back")` | `press_button("back")` |
| `launch_app("com.foo")` | `start_app("com.foo")` |
| `stop_app("com.foo")` | `stop_app("com.foo")` (this one does exist) |

Note: `tap`, `press_key`, and `launch_app` are registered as aliases and will work, but prefer the canonical names above.

### Wrong parameter types

All `text` parameters MUST be strings, not numbers:

```
# WRONG — will cause validation error
find_and_tap(text=98122)
input_text(text=98122)

# CORRECT
find_and_tap(text="98122")
input_text(text="98122")
```

### Wrong parameter names

```
# WRONG — drag uses x1/y1/x2/y2, not start_x/start_y/end_x/end_y
drag(start_x=100, start_y=200, end_x=300, end_y=400)

# CORRECT
drag(x1=100, y1=200, x2=300, y2=400)
```

### Calling tap_element without get_ui

`tap_element` now auto-refreshes the UI if the cache is empty, so this is no longer an error. But it's still faster to call `get_ui` first so you know what indices are available.

### Confusing find_and_tap with tap_xy

```
# WRONG — find_and_tap takes TEXT, not coordinates
find_and_tap(x=540, y=700)

# CORRECT — use tap_xy for coordinates
tap_xy(x=540, y=700)

# CORRECT — use find_and_tap for text
find_and_tap(text="Order here")
```

### Looping on the same failing strategy

If an action fails 3+ times with the same approach, try a different strategy:
1. Use `screenshot_small()` to visually check what's on screen
2. Try `press_button("back")` and approach from a different screen
3. Use `stop_app` + `start_app` to restart the app
4. Try `open_url` with a deep link instead of navigating through menus
