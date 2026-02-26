# Aerivon Live - Demo Script

**Total Time:** ~8-10 minutes  
**Demo URL:** https://aerivon-live-frontend-621908229087.us-central1.run.app

---

## Introduction (30 seconds)

"Hi, I'm here to show you Aerivon Live - a comprehensive platform I built to showcase the power of Google's Gemini Live API for real-time multimodal AI interactions.

What makes this different from traditional chatbots? Three things:
1. **Real-time voice** - not request-response, but streaming audio conversations
2. **Barge-in support** - you can interrupt the AI naturally while it's speaking
3. **True multimodal** - voice, vision, and actions in the same session

Let me show you what this can do across 8 different applications."

---

## Demo 1: Interactive Storybook (2 minutes) ⭐ PRIMARY FEATURE

**Navigate to:** Click "Interactive Storybook" from home page

"First, let me show you the Interactive Storybook - this is my primary feature and it demonstrates native multimodal generation.

**[Type a prompt]:** 'A brave knight who discovers they can talk to dragons'

**[Click 'Tell My Story']**

Watch what happens - Gemini is generating both the story text AND the illustrations in real-time. This isn't text with separate image generation - it's native multimodal output. The model creates text and images together, naturally interleaved.

**[Point out as it generates]:**
- See the dark mystical theme with golden accents
- Text streams in character by character
- AI-generated illustrations appear inline with the story
- Everything is happening in one API call to Gemini 2.5 Flash Image Preview

**[When complete, click Save Story]:**
This saves the entire story - text AND images - to Google Cloud Storage as a portable JSON file. The images are base64-encoded, so each story is completely self-contained.

This took me from a text-parsing approach with separate Imagen calls to this elegant native multimodal solution - much simpler and better results."

---

## Demo 2: Live Agent - Voice Interaction (1.5 minutes)

**Navigate to:** Click "Live Agent" from navigation

"Now let's try real-time voice. This is the Gemini Live API in action.

**[Click microphone button]**

**[Speak clearly]:** 'Hello! Can you tell me about the weather patterns that influence ocean currents?'

**[As AI responds, demonstrate barge-in]:**
**[Interrupt mid-response]:** 'Actually, focus on the Pacific Ocean specifically.'

**[Point out]:**
Did you see how I interrupted? The AI detected my voice, stopped speaking, and pivoted to answer my refined question. This is barge-in - a key feature of live agents.

**[Show the state indicators]:**
- Green: Listening
- Blue: Speaking
- Red: Interrupted
- Gray: Idle

The conversation state is persisted to Google Cloud Storage, so if I reconnect, the AI remembers our discussion."

---

## Demo 3: UI Navigator - Vision + Automation (1.5 minutes)

**Navigate to:** Click "UI Navigator"

"This combines Gemini's vision capabilities with browser automation via Playwright.

**[Enter URL]:** 'https://news.ycombinator.com'

**[Speak or type command]:** 'Click on the top story'

**[Show what happens]:**
The AI is taking screenshots of the browser, using vision to understand the page layout, identifying elements, and executing clicks through Playwright.

**[Point out the screenshot panel]:**
You can see what the AI sees - it's analyzing this visual information to navigate.

**[Try another command]:** 'Scroll down to show more stories'

This demonstrates how vision models can understand UI context and take action - perfect for test automation, accessibility tools, or workflow demos."

---

## Demo 4: Real-Time Translator (1 minute)

**Navigate to:** Click "Real-Time Translator"

"Let me show you multilingual live translation.

**[Select languages]:** English → Spanish

**[Click microphone]**

**[Speak]:** 'Good morning! I'm demonstrating a real-time translation system powered by Google's Gemini Live API.'

**[Show results]:**
- Original transcript appears on the left
- Translated audio plays automatically
- Translated text appears on the right

The translation happens in real-time with the same low latency as the live agent - perfect for international meetings or customer support."

---

## Demo 5: Vision Tutor (45 seconds)

**Navigate to:** Click "Vision Tutor"

"The Vision Tutor combines voice and visual understanding for education.

**[If you have a whiteboard or diagram ready, show it to camera]**

**[Speak]:** 'Can you explain what you see in this diagram?'

The AI analyzes the visual input and provides explanations. This is useful for tutoring, visual learning, or accessibility - the AI can describe and explain visual content in real-time."

---

## Demo 6: Customer Support Agent (45 seconds)

**Navigate to:** Click "Customer Support"

"This demonstrates automated customer service with multimodal context.

**[Speak]:** 'I need help with my account settings'

The agent can see your screen, hear your voice, and maintain conversation history. It's context-aware and can escalate to humans when needed.

**[Point out]:** In a production system, this would integrate with your CRM, knowledge base, and ticketing system."

---

## Demo 7: Workflow Automator (45 seconds)

**Navigate to:** Click "Workflow Automator"

"This is for voice-driven task automation.

**[Speak]:** 'Fill out the registration form with my information'

The AI can execute multi-step workflows based on voice commands - form filling, data entry, repetitive tasks. It uses vision to understand the UI and Playwright to execute actions.

This is enterprise automation made accessible through natural language."

---

## Demo 8: Visual Q&A (30 seconds)

**Navigate to:** Click "Visual Q&A"

"Finally, Visual Q&A - upload any image and ask questions about it.

**[If you have an image ready]:** Upload it

**[Ask]:** 'What are the key elements in this image?'

The AI analyzes the image and provides detailed insights. Great for image analysis, documentation, or accessibility."

---

## Technical Highlights (1 minute)

**[Navigate to GitHub repo or show architecture diagram]**

"Let me quickly highlight the technical architecture:

**AI Models:**
- Gemini 2.5 Flash Image Preview for multimodal story generation
- Gemini Live API for real-time voice interactions
- Gemini 2.0 Flash for vision-guided automation

**Backend:**
- Python FastAPI
- WebSocket streaming for low latency
- Playwright for browser automation
- Google Cloud Storage for persistence

**Infrastructure:**
- Deployed on Google Cloud Run
- Automated CI/CD with Cloud Build
- Vertex AI for model access
- Everything containerized and scalable

**[Show deployment script if time allows]:**
The entire deployment is automated - one script pushes to Cloud Run with environment configuration, health checks, and rollback support."

---

## Key Learnings (30 seconds)

"Three key lessons from building this:

1. **Use native capabilities** - I started with text parsing and image markers, but switching to native multimodal generation simplified everything and improved results.

2. **Streaming matters** - Users perceive incremental delivery as much faster than waiting for complete responses.

3. **Developer experience pays off** - I built the `./aerivon` launcher script early, which automated environment setup and made iteration much faster.

The code is open source on GitHub, and there's a full write-up in the PROJECT_DESCRIPTION.md file."

---

## Closing (20 seconds)

"That's Aerivon Live - eight applications demonstrating real-time multimodal AI with Google's Gemini Live API.

**Key takeaways:**
- Real-time voice with barge-in support
- Native multimodal generation  
- Vision-guided automation
- Production-ready deployment on Google Cloud

Questions?"

---

## Backup Talking Points

**If asked about challenges:**
- "Model availability varies by region - had to switch to global location for image preview"
- "Base64 image storage trades file size for portability - worth it for this use case"
- "WebSocket state management required careful tracking across message types"

**If asked about scale:**
- "Cloud Run scales automatically from zero"
- "Cost controls include timeouts, token limits, and session memory caps"
- "GCS provides infinite storage for generated content"

**If asked about security:**
- "SSRF protection on tool calls"
- "API rate limiting and size caps"
- "Tool allowlist and validation"
- "Prompt injection hardening"

---

## Demo Checklist

**Before demo:**
- [ ] Test live demo URL is accessible
- [ ] Backend is healthy (check Cloud Run logs)
- [ ] Microphone permissions granted
- [ ] Good internet connection for voice demos
- [ ] Optional: Prepare sample image for Visual Q&A
- [ ] Optional: Have whiteboard/diagram ready for Vision Tutor
- [ ] Clear browser cache if needed for fresh demo

**During demo:**
- [ ] Speak clearly for voice interactions
- [ ] Show visual feedback (state indicators, streaming text)
- [ ] Demonstrate barge-in during Live Agent
- [ ] Highlight the save functionality in Storybook
- [ ] Point out dark theme and UI polish

**After demo:**
- [ ] Share links: Live demo, GitHub repo, PROJECT_DESCRIPTION.md
- [ ] Offer to show code for specific features
- [ ] Mention blog post if published
