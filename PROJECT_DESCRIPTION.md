# Aerivon Live Agent - Project Description

## Overview

Aerivon Live Agent is a comprehensive demonstration platform showcasing the capabilities of Google's Gemini Live API for real-time, multimodal AI interactions. The project features nine distinct demo applications that highlight different use cases for live agent technology, from interactive storytelling to customer support automation.

## Features and Functionality

### 🌟 Unified Agent (Flagship Feature)
**Intelligent Multi-Modal Interface with Intent Detection**
- Single interface that combines voice and text input with automatic intent recognition
- **Intent Detection**: Automatically identifies user goals (story creation, UI navigation, or conversation)
- **Voice + Text Input**: Toggle microphone for voice commands or type messages naturally
- **Three-in-One Capability**:
  - **Story Generation**: Creates illustrated stories with AI-generated images using Gemini 2.5 Flash Image Preview
  - **UI Navigation**: Controls browser automation for web interactions via Playwright
  - **Natural Conversation**: Engages in contextual dialogue using Gemini Live API
- **Real-time Action Visualization**: Displays planned actions and execution status in dedicated sidebar
- **Streaming Content**: Text, images, and narration appear incrementally for responsive UX
- **Unified WebSocket Backend**: Single `/ws/aerivon` endpoint handles all interaction types
- **Cloud Run Ready**: Automatically detects backend URL for seamless deployment
- Elegant dark theme with golden accents matching the Aerivon brand

### 1. Interactive Storybook
**Multimodal Story Generation with Native Image Creation**
- Real-time story generation with interleaved text and AI-generated illustrations
- Uses Gemini 2.5 Flash Image Preview model with native multimodal output
- Stories feature dynamic narration with text-to-speech synthesis
- Visual feedback with progress indicators and streaming content display
- **Save Functionality**: Stories can be saved to Google Cloud Storage with full content preservation including base64-encoded images
- **Story Management**: Backend endpoints for saving (`POST /story/save`) and listing (`GET /story/list`) saved stories
- Dark mystical theme with golden accents for an immersive reading experience

### 2. Live Agent Interface
- Real-time voice interaction with Gemini Live API
- Bidirectional audio streaming via WebSocket
- Visual state indicators (idle, listening, speaking, interrupted)
- Low-latency response for natural conversation flow
- Memory persistence to Google Cloud Storage for conversation history

### 3. UI Navigator
- Voice-controlled web navigation assistant
- Integrates with Playwright for browser automation
- Natural language commands for clicking links and navigating pages
- Screenshot-based visual understanding for precise element identification
- Real-time feedback on navigation actions

### 4. Real-Time Translator
- Multilingual voice translation
- Supports streaming input and output
- Language detection and conversion
- Audio playback of translated text
- Transcript display for both source and target languages

### 5. Vision Tutor
- Educational AI that can see and explain visual content
- Screen sharing integration for real-time visual analysis
- Interactive Q&A based on visual context
- Useful for tutoring, learning, and visual explanations

### 6. Customer Support Agent
- Automated customer service with voice and vision capabilities
- Context-aware responses based on visual and verbal input
- Memory of interaction history
- Escalation detection and handling

### 7. Workflow Automator
- Voice-driven task automation
- Multi-step workflow execution
- Visual confirmation of actions
- Integration with browser automation tools

### 8. Visual Q&A
- Image-based question answering
- Upload or capture images for analysis
- Detailed explanations and insights
- Context-aware visual understanding

## Technologies Used

### Frontend Technologies
- **HTML5/CSS3/JavaScript**: Core web technologies for all demo interfaces
- **WebSocket API**: Real-time bidirectional communication for streaming audio and data
- **Web Audio API**: Audio playback and processing for voice interactions and TTS
- **MediaRecorder API**: Audio capture for voice input (where used in demos)
- **Fetch API**: HTTP requests for REST endpoints

### Backend Technologies
- **Python 3.11+**: Primary backend language
- **FastAPI**: Modern web framework for building APIs
- **Uvicorn**: ASGI server for running FastAPI applications
- **Google GenAI Python SDK**: Official SDK for Gemini API interactions
- **Google Cloud Storage Client**: For persistent storage of stories and agent memory
- **Playwright**: Browser automation for UI Navigator feature
- **asyncio**: Asynchronous programming for concurrent operations

### AI/ML Services
- **Gemini 2.5 Flash Image Preview**: Multimodal model with native text and image generation
- **Gemini Live API**: Real-time conversational AI with streaming capabilities
- **Vertex AI**: Google Cloud's AI platform for model deployment
- **Cloud Text-to-Speech API**: Neural voice synthesis (optional enhancement)
- **Imagen 3**: Image generation (legacy, now replaced by native Gemini multimodal)

### Cloud Infrastructure
- **Google Cloud Platform (GCP)**: Primary cloud provider
- **Cloud Run**: Serverless deployment platform
- **Cloud Storage**: Object storage for stories and agent memory
- **Artifact Registry**: Container image storage
- **Cloud Build**: CI/CD pipeline for automated deployment

### Development Tools
- **Git/GitHub**: Version control and collaboration
- **Docker**: Containerization for consistent deployment
- **Shell Scripts**: Automation for development workflows (./aerivon launcher)
- **Environment Variables**: Configuration management

## Data Sources

### Google Cloud Storage
**Bucket**: `aerivon-live-agent-memory-1771792693`

**Purpose**: Persistent storage for generated content and agent state

**Storage Structure**:
- `/memory/`: Agent conversation history and context
  - JSON files with conversation transcripts, tool usage history, and session metadata
  - Keys are hashed session IDs for privacy
  
- `/stories/`: Saved storybook generations
  - Each story stored as JSON with unique ID: `{timestamp}_{prompt_hash}.json`
  - Contains: story ID, user prompt, scene array (interleaved text/images), creation timestamp, save timestamp
  - Images stored as base64-encoded data within JSON for complete portability

**Data Format Example**:
```json
{
  "id": "1740547215_a3b8c9d2",
  "prompt": "A dragon who fears fire",
  "scenes": [
    {"type": "text", "content": "Once upon a time..."},
    {"type": "image", "data": "iVBORw0KGgo...", "mimeType": "image/png"},
    {"type": "text", "content": "The dragon lived..."}
  ],
  "created": "2026-02-26T09:30:15.234Z",
  "saved_at": "2026-02-26T09:31:42.567Z"
}
```

### External Web Content (UI Navigator)
- Dynamic web pages accessed via Playwright
- Screenshots captured for visual analysis
- HTML DOM structure for element identification
- Temporary data, not persisted

## Development Findings and Learnings

### 1. Multimodal Output Evolution
**Initial Approach**: Text parsing with [IMAGE: description] markers + separate Imagen 3 API calls
- Complex regex parsing required
- Two API calls per story (Gemini for text, Imagen for each image)
- Images not contextually integrated by the model

**Final Approach**: Native multimodal with `response_modalities=["TEXT", "IMAGE"]`
- Single API call to Gemini 2.5 Flash Image Preview
- Model generates text and images together, naturally interleaved
- Simpler code, better context awareness, more cohesive results
- **Lesson**: Use native multimodal capabilities when available; avoid workarounds

### 2. Model and Location Requirements
**Challenge**: Model availability varies by Google Cloud region
- Initial use of `gemini-2.0-flash-exp` in `us-central1` didn't support response_modalities
- Error: "Model not found" or "Feature not supported"

**Solution**: Switch to `gemini-2.5-flash-image-preview` with `global` location
- Image preview models require global location
- Different endpoints have different model availability
- **Lesson**: Always check model availability documentation for each region

### 3. Environment Configuration Management
**Challenge**: Multiple environment variables needed for Vertex AI, GCS, service configuration
- Manual export commands error-prone
- Different configurations for development vs. production

**Solution**: Created `./aerivon` launcher script
- Automatically sets all required environment variables
- Handles dependency installation and health checks
- Manages both backend and frontend processes
- **Lesson**: Invest in developer experience tooling early; it pays dividends

### 4. WebSocket Streaming Architecture
**Finding**: Streaming is crucial for perceived performance
- Users perceive faster response with incremental content delivery
- Text streaming + image insertion creates engaging experience
- Audio narration can overlap with visual content for multisensory experience

**Implementation**:
```python
for part in parts:
    if part["kind"] == "text":
        await send({"type": "text", "text": part["text"]})
    elif part["kind"] == "image":
        await send({"type": "image", "data_b64": part["data"]})
```

**Lesson**: Design for streaming from the start; retrofitting is harder

### 5. Frontend State Management
**Challenge**: Tracking dynamic story data across WebSocket messages
- Multiple message types (text, image, audio, status, done)
- Need to aggregate text chunks while preserving order
- Enable save button only when story complete

**Solution**: Centralized state object (`currentStoryData`)
```javascript
currentStoryData = {
    prompt: "User's input",
    scenes: [{type, content/data, mimeType}],
    created: "ISO timestamp"
}
```

**Lesson**: Simple state management beats complex frameworks for small apps

### 6. Base64 Image Storage Trade-offs
**Pros**:
- Complete portability (single JSON file contains everything)
- No broken image links
- Easy to transfer/share stories
- Simpler backend (no separate image serving)

**Cons**:
- Larger file sizes (~33% overhead from base64 encoding)
- Cannot use CDN caching as effectively
- JSON parsing slightly slower

**Decision**: Worth it for this use case (small number of images per story)
**Lesson**: Evaluate storage trade-offs based on actual usage patterns

### 7. API Error Handling and Debug Logging
**Best Practice**: Structured logging with prefixes
```python
print(f"[STORY DEBUG] Calling Gemini...", file=sys.stderr)
print(f"[STORY ERROR] {type(e).__name__}: {e}", file=sys.stderr)
```

**Benefits**:
- Easy to grep logs for specific components
- Type information helps debug
- stderr separation from stdout

**Lesson**: Invest in observability from day one

### 8. CSS Theming Consistency
**Challenge**: Eight separate HTML files with different styling approaches
**Solution**: Unified dark mystical theme with CSS variables
- Consistent color palette across all pages
- Golden accents (#c9a84c) for calls-to-action
- Georgia serif for storytelling feel
- Reusable button and input styles

**Lesson**: Design system consistency improves user experience significantly

### 9. Auto-Reload in Development
**FastAPI with Uvicorn watch mode**: Excellent developer experience
- Files automatically reload on changes
- WebSocket connections gracefully handled
- No manual restart needed during development

**Gotcha**: Multiple rapid changes can cause multiple reloads
- Wait for reload to complete before testing
- Check logs to confirm settled state

**Lesson**: Auto-reload is essential but requires patience during rapid iteration

### 10. Cloud Storage for Generated Content
**Why GCS over database**:
- Simple blob storage, no schema management
- Scales infinitely without provisioning
- Cost-effective for infrequent access
- JSON format allows future schema evolution

**Access Pattern**:
- Write: Infrequent (only when user saves)
- Read: On-demand (when user loads story)
- List: Occasional (browse saved stories)

**Lesson**: Match storage solution to access patterns; don't over-engineer

### 11. Unified Agent Architecture
**Challenge**: Users had to choose the right demo interface before starting
- Eight separate HTML pages for different capabilities
- Each required understanding which to use
- No seamless switching between story generation, UI navigation, and conversation

**Solution**: Built Unified Agent with intent detection
- Single `/ws/aerivon` WebSocket endpoint
- Backend analyzes user input to detect intent (story, navigate, chat)
- Frontend displays different UI components based on detected mode
- Real-time action plan visualization for UI navigation tasks

**Implementation Details**:
```javascript
// Frontend: Auto-detect backend URL for Cloud Run
function getBackendHttpBase() {
  // Check for server-injected backend
  if (window.AERIVON_BACKEND_BASE) return window.AERIVON_BACKEND_BASE;
  // Check query param override
  if (urlParams.get('backend')) return urlParams.get('backend');
  // Auto-detect Cloud Run (replace 'frontend' with 'agent')
  if (/\.run\.app$/i.test(location.hostname)) {
    return location.hostname.replace(/frontend/i, 'agent');
  }
  // Local dev default
  return `${location.protocol}//${location.hostname}:8081`;
}
```

**Backend Pattern**:
```python
# Single unified endpoint handles all interaction types
@app.websocket("/ws/aerivon")
async def unified_agent_websocket(websocket: WebSocket):
    # Detect intent from user message
    if "story" in message.lower() or "tale" in message.lower():
        await handle_story_generation()
    elif "navigate" in message.lower() or "open" in message.lower():
        await handle_ui_navigation()
    else:
        await handle_conversation()
```

**Benefits**:
- Reduced cognitive load for users (one interface, not eight)
- Better developer experience (single codebase to maintain for core features)
- Cloud Run deployment simplified (automatic backend URL detection)
- Seamless user experience across different AI capabilities

**Lesson**: Unified interfaces with intelligent routing provide better UX than forcing users to pre-select capabilities. Backend URL auto-detection is essential for Cloud Run deployments where service URLs are dynamic.

## Future Enhancements

1. **Story Gallery UI**: Browse and load saved stories with thumbnails
2. **Story Sharing**: Generate shareable URLs for public stories
3. **Advanced Editing**: Allow users to regenerate specific scenes or images
4. **Story Templates**: Pre-built story structures and genres
5. **Multi-language Support**: Generate stories in different languages
6. **Collaborative Storytelling**: Multiple users contributing to one story
7. **Export Options**: PDF, EPUB, or printed book formats
8. **Analytics Dashboard**: Track popular themes, average story length, etc.
9. **User Authentication**: Personal story libraries with auth
10. **Cost Optimization**: Caching frequently generated content, image compression

## Conclusion

Aerivon Live Agent demonstrates the power of Google's Gemini Live API for building rich, multimodal AI applications. The project showcases best practices for real-time streaming, multimodal content generation, cloud storage integration, and user experience design. Through iterative development, I learned the importance of using native model capabilities, proper environment configuration, thoughtful state management, and comprehensive error handling.

The **Unified Agent**, our flagship feature, demonstrates how intelligent intent detection can create a seamless user experience that adapts to different use cases—whether generating illustrated stories, navigating websites, or holding natural conversations. This unified interface reduces cognitive load and showcases the versatility of the Gemini Live API.

The storybook feature highlights how modern multimodal AI can create engaging, visually rich content in real-time, with the ability to preserve and share those creations through cloud-based storage solutions.
