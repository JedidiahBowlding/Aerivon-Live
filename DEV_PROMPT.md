# Developer Prompts for Gemini Live Agent Challenge Submissions

---

## CATEGORY 1: LIVE AGENT - Developer Prompt

```
You are building a LIVE AGENT for the Gemini Live Agent Challenge.

REQUIREMENTS:
1. Your agent must accept real-time input (voice or text) from users
2. It must respond in real-time with interruptible output (user can cut off mid-response)
3. It must maintain conversation context across multiple turns
4. It must handle edge cases and gracefully handle interruptions

TECHNICAL CONSTRAINTS:
- Use Gemini 3 or Gemini Nano from Google AI SDK or Agent Development Kit
- Deploy on Google Cloud (Cloud Run recommended for REST API backend)
- Use at least one Google Cloud service from: Firestore (context storage), Cloud SQL (conversation logs), Cloud Storage (asset storage)
- Implement streaming mode to support interruptions

IMPLEMENTATION CHECKLIST:
- [ ] Set up Gemini streaming API with streaming_mode enabled
- [ ] Create a WebSocket or Server-Sent Events (SSE) endpoint for real-time communication
- [ ] Store conversation context in Firestore (user_id → conversation history)
- [ ] Implement interrupt handling: on new user input, cancel current response stream
- [ ] Add latency optimization: response time < 500ms for natural feel
- [ ] Deploy to Cloud Run with auto-scaling enabled
- [ ] Create Cloud Run service that accepts POST requests with {"user_id": "...", "message": "..."}
- [ ] Response should stream chunks via Server-Sent Events or WebSocket

RESPONSE FORMAT:
For each user turn:
1. Load conversation context from Firestore using user_id
2. Send context + new message to Gemini with streaming=True
3. Stream chunks back to client
4. On interruption (new request), cancel stream and start fresh
5. Save full conversation to Firestore after completion

REAL-WORLD IMPLEMENTATIONS:

### Option 1A: Real-Time Translator
```
User says (voice): "How do you say hello in Spanish?"
Agent (streaming): "The Spanish word for hello is 'hola'. It's pronounced..."
User interrupts mid-speech: "Actually, what about French?"
Agent: Cancels stream, responds: "In French, you'd say 'bonjour'..."

Implementation:
- Input: Google Cloud Speech-to-Text (real-time audio)
- Processing: Gemini with streaming=True
- Output: Google Cloud Text-to-Speech (immediate playback)
- Context: Firestore stores translation history per user
- Interruption: WebSocket allows client to send new message, server cancels previous stream
```

### Option 1B: Vision-Enabled Tutor (Homework Helper)
```
Student: Shows phone camera at math homework (image)
Agent (vision): "I see quadratic equation: x² + 5x + 6 = 0. Let me explain..."
Student interrupts: "Wait, explain the discriminant first"
Agent: Provides detailed explanation of discriminant, then returns to original problem

Implementation:
- Input: Camera feed (WebRTC or JPEG uploads)
- Vision: Gemini Vision API analyzes homework image
- Processing: Gemini with streaming=True for explanation
- Output: Text + possibly generate diagram/solution steps
- Context: Firestore stores student learning progress, previous topics covered
- Interruption: Student can ask follow-up questions mid-explanation
```

### Option 1C: Customer Support Agent
```
Customer: "Why was my order cancelled?"
Agent streams: "Let me check your account... [looks up order] Your order was cancelled because..."
Customer interrupts: "Can you just re-order it for me?"
Agent: "Absolutely, I can help with that" [processes re-order instead of continuing original explanation]

Implementation:
- Input: Text chat or voice call
- Processing: Gemini accesses customer database via Cloud SQL
- Streaming: Response is interrupted by new customer request
- Context: Firestore or Cloud SQL stores customer history, order status
- Action: Can execute refund, re-order, escalate (triggers other APIs)
```

BONUS POINTS:
- Add voice input/output (Google Cloud Speech-to-Text, Text-to-Speech)
- Implement multi-language support
- Add conversation memory (remember previous messages in same session)
- Publish blog post: "Building a Real-Time AI Agent with Gemini and Google Cloud"

FILE STRUCTURE:
/your-repo
  /backend
    main.py (Cloud Run entry point, Flask/FastAPI)
    agent.py (Gemini streaming logic)
    firestore_handler.py (context storage)
  /frontend
    index.html (WebSocket client for real-time chat)
  ARCHITECTURE.md (diagram + explanation)
  SETUP.md (how to deploy to Cloud Run)
  demo_video.mp4 (< 4 min, real-time execution)
  README.md (overview, how to run locally)

VALIDATION:
Run locally: `python main.py` → open http://localhost:8080 → test interruption
Test on Cloud Run: Deploy and show Cloud Run dashboard in demo video
```

---

## CATEGORY 2: CREATIVE STORYTELLER - Developer Prompt

```
You are building a CREATIVE STORYTELLER AGENT for the Gemini Live Agent Challenge.

REQUIREMENTS:
1. Your agent must generate output that SEAMLESSLY INTERLEAVES text, images, audio, and video
2. Output is NOT sequential (text, then image, then audio) but rather mixed in ONE stream
3. The agent acts as a creative director, orchestrating all modalities together
4. All outputs must be thematically and tonally consistent

TECHNICAL CONSTRAINTS:
- Use Gemini 3 or Gemini Nano with native interleaved/multimodal output capability
- Deploy on Google Cloud: Cloud Run for orchestration
- Use at least one of: Cloud Storage (asset storage), Vertex AI (image/video generation), Cloud SQL (story templates)
- Output must be a single coherent asset (e.g., MP4 with narration, MP3 with transcript, interactive web page)

IMPLEMENTATION CHECKLIST:
- [ ] Use Gemini multimodal capabilities with streaming output
- [ ] Request interleaved output format: ['text_chunk', 'image_prompt', 'text_chunk', 'video_segment', ...]
- [ ] For images: call Vertex AI Imagen API or similar (parallel with text generation)
- [ ] For audio: call Google Cloud Text-to-Speech (parallel with generation)
- [ ] For video: use Vertex AI Video Generation API or pre-render templates
- [ ] Assemble all outputs: use FFmpeg on Cloud Run to combine audio + video + text overlays
- [ ] Store final assets in Cloud Storage
- [ ] Implement state machine: capture each modality as it's generated, queue for assembly

RESPONSE FORMAT:
User request → Gemini with {"task": "create interactive storybook", "theme": "robots", "format": "video"}
Gemini response includes:
  - Initial narration (voice-ready text chunk)
  - [IMAGE: A futuristic robot factory scene]
  - Next narration chunk
  - [VIDEO: 5-second animated clip of robots working]
  - Final narration with call-to-action

REAL-WORLD IMPLEMENTATIONS:

### Option 2A: Interactive Storybook Generator
```
Input: User selects theme "Dragons in Medieval Times"
Output: Single MP4 or Web page with:
  - Paragraph 1 (narration): "In the misty highlands, a dragon awoke..."
  - [GENERATED IMAGE: Medieval landscape with dragon]
  - Paragraph 2 (narration): "The knight heard tales of the creature..."
  - [ANIMATED VIDEO CLIP: Dragon flying through clouds, 5 sec]
  - Paragraph 3 (narration): "But the dragon was not evil, merely lonely..."
  - [GENERATED IMAGE: Knight meeting dragon peacefully]

Implementation:
- Gemini generates story with interleaved prompts for images/video
- Vertex AI Imagen generates images inline
- Vertex AI Video API generates short clips
- Google Cloud Text-to-Speech creates narration
- FFmpeg assembles all into cohesive video
- Cloud Storage stores final MP4
- User can download or share story immediately
```

### Option 2B: Marketing Asset Generator
```
Input: Marketer says "Create LinkedIn post about AI adoption in healthcare"
Output: Single asset containing:
  - LinkedIn caption: "AI is transforming healthcare..."
  - [BRANDED IMAGE: Healthcare + AI icons]
  - 10-sec video testimonial (AI-generated avatar speaking)
  - Call-to-action: "Learn more →"
  - Hashtags: #AI #Healthcare #Innovation
  - All in one video file ready to post

Implementation:
- Gemini generates copy, image prompts, video script in one stream
- Vertex AI Imagen creates brand-consistent images
- HeyGen or similar generates avatar video
- Google Cloud Text-to-Speech provides voiceover
- FFmpeg assembles: background + text overlay + avatar video + graphics
- Cloud Storage stores final MP4 or GIF
- Result: One ready-to-post asset, not separate text/image/video files
```

### Option 2C: Educational Explainer (Lecture Slides)
```
Input: Biology teacher: "Create lesson on photosynthesis"
Output: Single presentation with:
  - Narration intro: "Photosynthesis is the process plants use..."
  - [DIAGRAM: Plant cell with chloroplasts highlighted]
  - Narration segment 2: "Light enters the leaf through the stomata..."
  - [ANIMATION: Light rays entering leaf]
  - Narration segment 3: "Energy is converted to glucose molecules..."
  - [VISUALIZATION: Molecule transformation animation]
  - Summary slide with key takeaways

Implementation:
- Gemini generates educational script with diagram prompts
- Generates diagrams (via Imagen or Miro API for scientific accuracy)
- Generates animations (segments passed to animation service)
- Google Cloud Text-to-Speech for narration (high-quality educational voice)
- Assembles into MP4 or interactive web presentation
- Cloud Storage hosts final output
- Teacher can show in class, students can replay at home
```

BONUS POINTS:
- Implement real-time streaming assembly (stream MP4 chunks as they're generated)
- Train custom image generation model for brand consistency
- Add interactive elements (clickable links in web-based output)
- Publish blog: "Blending AI with Creativity: Building Multimodal Storytelling Agents"

FILE STRUCTURE:
/your-repo
  /backend
    main.py (Cloud Run Flask/FastAPI, routes for story generation)
    agent.py (Gemini orchestration + streaming)
    media_assembler.py (FFmpeg orchestration, image/audio generation)
    gcs_handler.py (Cloud Storage upload)
  /templates
    story_templates.json (narrative structures)
  ARCHITECTURE.md (diagram: Gemini → Imagen + TTS + Video → FFmpeg → Cloud Storage)
  SETUP.md (deploy to Cloud Run, configure APIs)
  demo_video.mp4 (show generated asset + Cloud Run dashboard)
  README.md (example: "Input: marketing brief" → "Output: MP4")

VALIDATION:
Test locally: `python main.py` → POST /generate_story → check output MP4
Test on Cloud Run: show generated video + Cloud Storage bucket

CRITICAL:
- Must show interleaved output (not sequential steps)
- Demo video must show final assembled output (MP4, web page, etc.), not just prompts
```

---

## CATEGORY 3: UI NAVIGATOR - Developer Prompt

```
You are building a UI NAVIGATOR AGENT for the Gemini Live Agent Challenge.

REQUIREMENTS:
1. Your agent must observe a browser or device display (screenshot)
2. It must interpret visual UI elements (buttons, text fields, labels, images)
3. It must make decisions about which UI element to interact with next
4. It must execute actions (click, type, scroll) based on user intent
5. It must adapt to UI changes and handle errors gracefully

TECHNICAL CONSTRAINTS:
- Use Gemini Vision API (Gemini 3 can analyze screenshots)
- Deploy on Google Cloud: Cloud Run for orchestration
- Use browser automation: Playwright or Selenium (not Puppeteer if possible, use Playwright)
- Use at least one of: Firestore (action history), Cloud Storage (screenshot logs), Cloud SQL (task templates)
- Agent must run in a loop: screenshot → Gemini Vision → action → repeat until goal

IMPLEMENTATION CHECKLIST:
- [ ] Set up Playwright (or Selenium) for browser control on Cloud Run
- [ ] Capture screenshots after each action
- [ ] Pass screenshots to Gemini Vision API with OCR + understanding
- [ ] Gemini analyzes screenshot and determines next action in natural language
- [ ] Parse Gemini response to extract action: {"action": "click", "element": "Sign in button", "coordinates": [500, 300]}
- [ ] Execute action via Playwright/Selenium
- [ ] Store action history in Firestore (for debugging, analytics)
- [ ] Handle edge cases: popup windows, unexpected UI, timeouts
- [ ] Implement max_steps limit (e.g., 20 actions max per task to prevent infinite loops)
- [ ] Log all screenshots and actions to Cloud Storage for review

RESPONSE FORMAT:
User request: "Book me a hotel in New York for March 15-17"
Agent loop:
  Step 1: Screenshot of Google → Analyze → "I see Google homepage, need to search for hotel booking"
  Step 2: Input "hotels in new york" → click search → take screenshot
  Step 3: Screenshot of results → "I see booking.com link, clicking..."
  Step 4: Screenshot of booking.com → "Form detected, filling dates..." → type dates
  Step 5: Continue until checkout...
Final: "Booking complete, confirmation #12345"

REAL-WORLD IMPLEMENTATIONS:

### Option 3A: Universal Web Navigator (Flight Booking)
```
User: "Book me a one-way flight from NYC to Los Angeles on March 15, economy, cheapest option"
Agent steps:
  1. Screenshot: Google homepage
     Gemini: "I see Google. Will search for flight booking sites."
     Action: Type "cheap flights NYC to LA" in search, click search

  2. Screenshot: Google search results
     Gemini: "I see multiple options: Kayak, Google Flights, Expedia. Clicking Google Flights for best prices."
     Action: Click "Google Flights" link

  3. Screenshot: Google Flights homepage
     Gemini: "I see flight search form. Filling in departure city (NYC)."
     Action: Click origin field, type "New York (JFK)"

  4. Screenshot: Location dropdown
     Gemini: "Autocomplete showing JFK option. Selecting it."
     Action: Click "JFK - John F Kennedy"

  5. Screenshot: Updated form
     Gemini: "Now filling destination: Los Angeles (LAX)."
     Action: Click destination field, type "Los Angeles (LAX)"

  6. Screenshot: Date picker
     Gemini: "Setting March 15 as departure date."
     Action: Click calendar, select March 15

  7. Screenshot: Search results
     Gemini: "Showing available flights. Finding cheapest option: Southwest at $89."
     Action: Click Southwest flight $89

  8. Screenshot: Booking confirmation
     Gemini: "Flight selected. Clicking 'Continue to checkout'."
     Action: Click continue button

  9. Screenshot: Passenger info form
     Gemini: "Now filling passenger details with name, email, phone."
     Action: Fill form fields with user info

  10. Screenshot: Payment page
      Gemini: "Payment required. Task complete - agent has navigated to checkout."
      Final status: "Booking prepared at payment stage, ready for user to enter card"

Implementation:
- Playwright controls browser in headless mode
- Screenshot captured after each action
- Gemini Vision API analyzes screenshot and UI elements
- Max 30 steps to prevent infinite loops
- Firestore logs all actions for debugging
- Cloud Storage saves screenshots for replay/audit
```

### Option 3B: Cross-Application Workflow Automator
```
User: "Export my Google Sheets budget to PDF and email it to finance@company.com"
Agent steps:
  1. Navigate to Google Sheets → find "Budget 2024" spreadsheet → open it
  2. Click "File" → "Download" → "Download as PDF"
  3. Wait for PDF to download
  4. Switch to Gmail
  5. Compose new email
  6. Recipient: finance@company.com
  7. Subject: "Monthly Budget Report"
  8. Attach the downloaded PDF
  9. Write body: "Please find attached the budget report"
  10. Click "Send"

Implementation:
- Playwright manages multiple browser tabs (Sheets + Gmail)
- Each action captured with screenshot
- Gemini Vision identifies buttons, form fields, tabs
- File download handling integrated
- Email attachment via drag-drop or file input
- Firestore records workflow steps
- Notification sent when complete
- Can be scheduled (e.g., every month on 1st)
```

### Option 3C: Visual QA Testing Agent
```
User: "Test the checkout flow on our e-commerce site - verify all buttons exist and are clickable"
Agent steps:
  1. Screenshot: Homepage
     Gemini: "Homepage loaded successfully. I see navbar with [Product, Cart, Account] buttons. All visible."
     Check: Compare against baseline screenshot (no visual regression)

  2. Click on "Products"
     Gemini: "Products page loaded. Showing product grid with price, image, 'Add to Cart' buttons visible."
     Check: Verify all product cards have required elements

  3. Click "Add to Cart" on first product
     Gemini: "Product added. Confirmation message appeared. Cart count increased."
     Check: Verify cart count updated (1 item)

  4. Click "Cart"
     Gemini: "Cart page showing 1 item. I see [Update Cart], [Checkout], [Continue Shopping] buttons."
     Check: All action buttons present and clickable

  5. Click "Checkout"
     Gemini: "Checkout page loaded. Forms for [Shipping Address], [Billing Address], [Payment Method]."
     Check: All fields with proper labels, no layout shifts

  6. Fill checkout form with test data
     Gemini: "Shipping address filled. Billing address filled. Payment method selected."
     Check: Form validation working, no error messages

  7. Click "Complete Purchase"
     Gemini: "Order confirmation page appeared with order number #12345."
     Check: Confirmation shows order details correctly

Final Report:
  ✅ Checkout flow: PASSED
  - All pages load correctly
  - No visual regressions detected
  - All buttons clickable
  - Form validation working
  - Confirmation page displays properly
  Issues found: None

Implementation:
- Screenshot baseline stored in Cloud Storage
- Gemini Vision compares current vs baseline (visual regression detection)
- Action log stored in Firestore (for debugging failures)
- Can run on schedule (nightly QA tests)
- Team notified of failures
- Test report generated automatically
```

BONUS POINTS:
- Implement parallel task execution (multiple browser instances)
- Add visual regression testing (compare expected vs actual UI)
- Implement recovery logic for error states
- Publish blog: "AI-Powered RPA with Gemini Vision and Cloud Run"

FILE STRUCTURE:
/your-repo
  /backend
    main.py (Cloud Run Flask/FastAPI, task creation endpoint)
    navigator.py (Gemini Vision + action loop)
    browser_controller.py (Playwright wrapper)
    firestore_handler.py (action history storage)
    screenshot_analyzer.py (parse Gemini Vision response)
  /utils
    gcs_logger.py (save screenshots to Cloud Storage)
    error_handler.py (handle UI errors, retries)
  ARCHITECTURE.md (diagram: User Intent → Screenshot → Gemini Vision → Action → Loop)
  SETUP.md (deploy to Cloud Run, configure Playwright, Firestore, Cloud Storage)
  demo_video.mp4 (show real browser automation, Cloud console)
  README.md (example tasks: booking, form-filling, data entry)

VALIDATION:
Test locally: `python main.py` → POST /execute_task with {"task": "..."}
Monitor: Cloud Run logs show action history, screenshots in Cloud Storage bucket
Test on Cloud Run: Show live URL executing task in demo video

CRITICAL:
- Demo video MUST show real browser interaction (not mockups)
- Must demonstrate handling of unexpected UI (popup, layout change)
- Must show screenshot → Gemini analysis → action loop in real-time
- Proof: Live Cloud Run URL + Cloud Storage bucket with action logs
```

---

## UNIVERSAL VALIDATION FOR ALL CATEGORIES

Before submitting your project:

1. **GitHub Repo**
   - [ ] Public repo with clear README
   - [ ] ARCHITECTURE.md explaining Gemini + Google Cloud integration
   - [ ] SETUP.md with step-by-step instructions to run locally
   - [ ] Code is clean, commented, and follows best practices

2. **Demo Video (< 4 minutes)**
   - [ ] Shows REAL execution (no mockups, no slides)
   - [ ] Shows live agent/output generation in progress
   - [ ] Includes screenshot of Cloud Run dashboard or live URL
   - [ ] Audio is clear, video quality is good
   - [ ] Include: "This is running on Google Cloud" proof

3. **Google Cloud Proof**
   - [ ] Cloud Run service deployed with live URL
   - [ ] Screenshot showing active Cloud Run service + logs
   - [ ] Or: Vertex AI endpoint with logs showing requests
   - [ ] Include URL in video or demo documentation

4. **Architecture Diagram** (in ARCHITECTURE.md)
   - [ ] Shows: User Input → Gemini Model → Google Cloud Service → Output
   - [ ] Labels all APIs used (Gemini, Cloud Run, Firestore, etc.)
   - [ ] Simple boxes + arrows is fine (even ASCII art acceptable)

5. **Code Quality**
   - [ ] No API keys exposed (use environment variables)
   - [ ] Code is runnable without manual fixes
   - [ ] Error handling for API failures
   - [ ] Logging for debugging

BONUS SUBMISSION:
- [ ] Blog post on Medium/Dev.to with #GeminiLiveAgentChallenge
- [ ] Deployment script (GitHub Actions or Terraform) for one-click deploy
- [ ] Unit tests for core logic

---

## SUMMARY: What Judges Are Looking For

1. **Innovation** - Does the solution demonstrate creative use of Gemini's capabilities?
2. **Technical Execution** - Is it properly deployed on Google Cloud with proof?
3. **User Experience** - Is it smooth, responsive, and usable?
4. **Code Quality** - Is it maintainable, well-documented, and production-ready?
5. **Bonus Engagement** - Did you share your work and use the hashtag?

**Remember:** The demo video is the most important artifact. Make it count.

```
