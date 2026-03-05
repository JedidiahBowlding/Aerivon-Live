Aerivon Unified Fix Patch

Objective

Resolve the following Aerivon issues in a single deployment update:

1. Conversational responses instead of execution behavior
2. Incorrect intent routing for navigation requests
3. Story interrupt failures
4. Quota/timeout instability
5. Planner output inconsistencies

Aerivon must behave as a deterministic autonomous execution agent, not a conversational assistant.

---

Fix 1 — Force Execution Mode

Update the system instruction used in:

- "handle_ui_request"
- "handle_hybrid_request"
- "_ui_plan_actions"
- "handle_story_request"

Replace existing instructions with:

You are Aerivon, an autonomous execution engine.

You are not a chatbot.

You never:
- explain capabilities
- refuse requests
- ask permission
- say "I can't", "I cannot", or "would you like"

Your job is to execute the user’s request.

If the request involves websites:
1. Navigate to the website
2. Identify UI elements
3. Perform required interactions

If the request involves storytelling:
1. Generate narrative scenes
2. Generate illustrations

If both navigation and storytelling are requested:
1. Navigate first
2. Extract relevant information
3. Generate the story and images

Always return actions or generated outputs.

Never produce conversational filler text.

---

Fix 2 — Strengthen Intent Detection

Update "detect_intent()" so navigation requests cannot fall back to conversational mode.

Replace UI detection logic with:

navigation_verbs = [
    "open",
    "go to",
    "visit",
    "navigate",
    "click",
    "press",
    "scroll",
    "search",
    "type",
]

has_navigation = any(v in text for v in navigation_verbs)

has_domain = bool(re.search(r'\b\w+\.(com|org|net|io|gov|edu)\b', text))
has_url = bool(re.search(r'https?://', text))

if has_navigation or has_domain or has_url:
    return {
        "intent": "ui",
        "confidence": 0.95,
        "reason": "navigation request detected",
        "requires_browser": True,
        "requires_images": False
    }

---

Fix 3 — Hybrid Intent Override

If both story keywords and navigation keywords exist:

if has_story and (has_navigation or has_domain or has_url):
    return {
        "intent": "hybrid",
        "confidence": 0.95,
        "requires_browser": True,
        "requires_images": True
    }

Hybrid execution order must be:

1. Browser navigation
2. Content extraction
3. Story generation
4. Image generation

---

Fix 4 — Strict Planner JSON Output

Update "_ui_plan_actions()" prompt:

You are a browser automation planner.

Return ONLY JSON.

Allowed actions:
- navigate
- click
- type
- scroll
- wait
- extract

Example output:

{
  "actions": [
    {"action": "navigate", "url": "https://example.com"},
    {"action": "click", "selector": "a[href*='domains']"}
  ]
}

Reject any planner output that is not valid JSON and retry generation.

---

Fix 5 — Conversational Guard

Inside "handle_conversational()" add a routing guard:

navigation_words = ["open","click","go to","visit","navigate",".com",".org",".net"]

if any(word in text for word in navigation_words):
    return await handle_ui_request(text)

This prevents accidental conversational routing.

---

Fix 6 — Story Interrupt Reliability

Story endpoint must use the same concurrency model as "/ws/aerivon".

Architecture must include:

- "message_listener"
- "main_processor"
- "message_queue"
- "active_task" cancellation

Interrupt flow must be:

1. Receive interrupt
2. Send ACK immediately
3. Cancel active task
4. Reset state

Interrupt ACK must always send:

{"type":"interrupted","source":"client"}

---

Fix 7 — Limit Story Concurrency

Prevent quota exhaustion.

Add:

STORY_SEMAPHORE = asyncio.Semaphore(1)

Wrap story generation:

async with STORY_SEMAPHORE:
    parts = await run_story_generation()

---

Fix 8 — Reduce Token Usage

Replace:

max_output_tokens=4096

with:

max_output_tokens=2048

---

Fix 9 — Retry Quota Errors

Add retry logic for:

- "429 RESOURCE_EXHAUSTED"
- timeouts

Backoff pattern:

1 second
2 seconds
4 seconds

Maximum retries: 3

---

Fix 10 — Response Safety Filter

Reject outputs containing refusal phrases.

blocked_phrases = [
"I can't",
"I cannot",
"I'm unable",
"would you like",
"I do not have access"
]

If detected, retry generation.

---

Expected Behavior

User request:

Go to example.com and click learn more

Agent output:

🌐 Opening https://example.com
🖱 Clicking Learn More
📄 Navigating to Domains page

User request:

Go to example.com domains page and create an illustrated story

Agent output:

🌐 Opening example.com
📄 Extracting domain content
📖 Generating illustrated story
🖼 Scene image
🎙 Narration

No conversational responses should appear.

---

Deployment

1. Apply code updates
2. Deploy new Cloud Run revision
3. Run stability tests

Expected test result:

/ws/aerivon 10/10
/ws/ui 10/10
/ws/story 10/10