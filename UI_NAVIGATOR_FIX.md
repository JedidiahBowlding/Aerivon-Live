# UI Navigator - New Tab Navigation Fix

## Testing the UI Navigator

**Access the UI:**
```
http://localhost:5174/ui_navigator.html
```

**Test Case for New Tab Navigation:**
```
URL: https://example.com
Task: Click the More information link
Expected: Screenshot shows IANA page (not example.com)
```

**Other Test Cases:**
```
URL: https://news.ycombinator.com
Task: Click the first story link

URL: https://agentikvault.com
Task: Click the Get Started button
```

## The Key Fix - Popup Handling

**What was broken:** Clicks on `target="_blank"` links didn't follow to the new tab.

**The fix in backend/server.py (lines 476-504):**
```python
if t == "click":
    x = int(action.get("x"))
    y = int(action.get("y"))
    
    # Check if click opens a new tab/window
    import sys
    print(f"[UI NAV DEBUG] Click executed at ({x}, {y}), checking for popup...", file=sys.stderr, flush=True)
    
    # Try to detect popup
    try:
        async with page.expect_popup(timeout=1000) as popup_info:
            await page.mouse.click(x, y)
        popup = await popup_info.value
        page = popup  # Switch to new tab
        await page.wait_for_load_state("domcontentloaded")
        print(f"[UI NAV DEBUG] Switched to popup: {page.url}", file=sys.stderr, flush=True)
    except Exception:
        # No popup, just a regular click
        print(f"[UI NAV DEBUG] No popup, waiting for navigation...", file=sys.stderr, flush=True)
        await page.mouse.click(x, y)
        try:
            await page.wait_for_load_state("networkidle", timeout=3000)
            print("[UI NAV DEBUG] Navigation detected, page is now idle", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[UI NAV DEBUG] No navigation (expected): {type(e).__name__}, waiting 1s for dynamic updates", file=sys.stderr, flush=True)
            await page.wait_for_timeout(1000)
    
    return {"ok": True, "type": "click", "x": x, "y": y}
```

**Key insight:** Use `page.expect_popup()` as a context manager BEFORE clicking, then switch to the popup if one opens. The context manager waits for the popup event, and if the click doesn't create a popup, it times out and we proceed with regular navigation handling.

**Also required:** Added `page` to the nonlocal declaration in `execute_action()`:
```python
async def execute_action(action: dict[str, Any]) -> dict[str, Any]:
    nonlocal cancel_flag, page  # Added 'page' here
```

## WebSocket API Usage

**Client-side connection:**
```javascript
const ws = new WebSocket('ws://localhost:8081/ws/ui');

// Open URL
ws.send(JSON.stringify({
  type: "open",
  url: "https://example.com"
}));

// Run task
ws.send(JSON.stringify({
  type: "task", 
  text: "Click the login button"
}));

// Receive events
ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  if (msg.type === "screenshot") {
    console.log("Screenshot URL:", msg.url);
    // msg.data_b64 contains base64 PNG
  }
  if (msg.type === "actions") {
    console.log("Plan:", msg.plan);
  }
  if (msg.type === "action_result") {
    console.log("Action result:", msg.result);
  }
};
```

## Environment Variables Required

```bash
export GOOGLE_GENAI_USE_VERTEXAI=True
export GOOGLE_CLOUD_PROJECT="aerivon-live-agent"
export GOOGLE_CLOUD_LOCATION="us-central1"
```

## Starting the Server

```bash
# Easy way with launcher
export AERIVON_BACKEND_PORT=8081
export AERIVON_FRONTEND_PORT=5174
./aerivon

# Manual way - Backend
export GOOGLE_GENAI_USE_VERTEXAI=True
export GOOGLE_CLOUD_PROJECT="aerivon-live-agent"
export GOOGLE_CLOUD_LOCATION="us-central1"
.venv/bin/uvicorn server:app \
  --app-dir backend \
  --host 127.0.0.1 \
  --port 8081 \
  --log-level info

# Manual way - Frontend
.venv/bin/uvicorn server:app \
  --app-dir frontend \
  --host 127.0.0.1 \
  --port 5174
```

## Logs Location

```bash
# Backend logs
tail -f /tmp/aerivon_backend.log

# Frontend logs  
tail -f /tmp/aerivon_frontend.log

# Check for debug messages
grep "UI NAV DEBUG" /tmp/aerivon_backend.log

# See what happened during last test
tail -20 /tmp/aerivon_backend.log
```

## How It Works

1. **User opens URL** → Backend uses Playwright to navigate and take screenshot
2. **User gives task** → Screenshot sent to Gemini multimodal vision model
3. **Gemini analyzes** → Returns JSON action plan with click coordinates, text to type, etc.
4. **Backend executes actions** → Playwright performs the actions in real browser
5. **New screenshot** → Process repeats until task is complete

## Debugging Tips

**Check if servers are running:**
```bash
lsof -nP -iTCP:8081,5174 -sTCP:LISTEN
```

**Kill stuck processes:**
```bash
pkill -f 'uvicorn.*cookbook'
```

**Verify environment variables:**
```bash
ps eww -p $(pgrep -f 'uvicorn.*backend') | tr ' ' '\n' | grep GOOGLE
```

**Test WebSocket connection:**
```bash
# Should return connection upgrade
curl -i -N -H "Connection: Upgrade" \
  -H "Upgrade: websocket" \
  -H "Sec-WebSocket-Version: 13" \
  -H "Sec-WebSocket-Key: test" \
  http://localhost:8081/ws/ui
```
