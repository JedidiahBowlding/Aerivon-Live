# Aerivon Storybook - Multimodal AI Storytelling

## Overview

Aerivon Storybook is an interactive multimodal storytelling agent that leverages **Gemini's native interleaved output capabilities** to create immersive illustrated stories with AI-narrated audio in real-time.

## Features

### 🎨 Multimodal Content Generation
- **Interleaved Text + Images**: Gemini generates story narration and illustrations in a single, cohesive stream
- **AI Narration**: Google Cloud Text-to-Speech automatically narrates the story as it unfolds
- **Real-time Streaming**: Content streams to the client as it's generated for a dynamic experience

### 📚 Story Structure
- **4-6 Scene Format**: Each story is structured as multiple scenes
- **Scene Components**:
  - 2-3 sentences of vivid narration
  - Beautiful storybook-style illustration
- **Engaging Themes**: Dragons, space exploration, time travel, fantasy, sci-fi, and more

### 🎭 Creative Director Approach
The agent thinks and creates like a creative director by:
1. Understanding the user's story prompt
2. Planning a cohesive narrative arc
3. Generating synchronized text and visuals
4. Producing audio narration that accompanies the visuals
5. Delivering everything in one fluid, mixed-media experience

## Technical Implementation

### Backend: `/ws/story` WebSocket Endpoint

**Location**: `backend/server.py`

**Key Technologies**:
- **Gemini 2.0 Flash Exp** with multimodal output (`response_modalities=["TEXT", "IMAGE"]`)
- **Google Cloud Text-to-Speech** for narration
- **WebSocket** for real-time bidirectional communication
- **Vertex AI** integration

**Message Protocol**:

Client → Server:
```json
{"type": "prompt", "text": "A dragon who fears fire"}
{"type": "interrupt"}
```

Server → Client:
```json
{"type": "status", "status": "connected"|"generating"|"done"}
{"type": "text", "text": "Once upon a time...", "index": 0}
{"type": "image", "data_b64": "...", "mime_type": "image/png", "index": 1}
{"type": "audio", "data_b64": "...", "mime_type": "audio/mpeg", "index": 2}
{"type": "done"}
{"type": "error", "error": "..."}
```

**Flow**:
1. Client sends story prompt
2. Server builds creative prompt for Gemini
3. Gemini generates interleaved text/image parts
4. Server processes parts:
   - Text chunks are sent immediately
   - When an image is encountered, accumulated text is synthesized to MP3 via TTS
   - Audio and image are sent to client
5. Client displays text, images, and plays audio sequentially

### Frontend: `story.html`

**Location**: `frontend/story.html`

**Features**:
- **Elegant Storybook UI**: Dark theme with golden accents
- **Responsive Design**: Works on desktop and mobile
- **Audio Queue**: Sequential playback of narration chunks
- **Smooth Animations**: Fade-in effects for scenes
- **Example Prompts**: Quick-start story ideas
- **Progress Indicators**: Real-time status updates

**UI Components**:
- Story prompt textarea
- "Tell Story" / "Stop" buttons
- Example prompt chips
- Status bar with spinner
- Story canvas with text and images
- Hidden audio element for autoplay

## Usage

### Local Development

1. **Start the servers**:
```bash
export AERIVON_BACKEND_PORT=8081
export AERIVON_FRONTEND_PORT=5174
./aerivon
```

2. **Access the storybook**:
```
http://localhost:5174/story.html
```

3. **Create a story**:
   - Enter a prompt like "A young astronaut discovers a hidden ocean on Europa"
   - Click "Tell Story" or press Cmd/Ctrl+Enter
   - Watch as the story unfolds with text, images, and narration

### Cloud Deployment

**Backend**:
```bash
./scripts/deploy_cloud_run.sh
```

**Frontend**:
```bash
SOURCE_DIR=frontend SERVICE_NAME=aerivon-live-frontend ./scripts/deploy_cloud_run.sh
```

**Access**:
```
https://aerivon-live-frontend-[PROJECT-ID].us-central1.run.app/story.html
```

## Example Prompts

- "A dragon who fears fire"
- "Lost city beneath the Sahara"
- "Robot who learns to dream"
- "A lighthouse at the edge of the universe"
- "Time traveler stuck in ancient Rome"
- "Mermaid who wants to fly"
- "Space detective solving cosmic mysteries"
- "Enchanted library where books come alive"

## Hackathon Category Match

### Focus: Multimodal Storytelling with Interleaved Output ✅

**Requirements Met**:
- ✅ Uses Gemini's native interleaved output capabilities
- ✅ Seamlessly weaves text, images, and audio
- ✅ Single fluid output stream
- ✅ Rich mixed-media responses
- ✅ Hosted on Google Cloud (Cloud Run)

**Use Cases Demonstrated**:
- ✅ Interactive storybooks (text + generated illustrations inline)
- ✅ Educational explainers (narration woven with visuals)
- ⚡ Extensible to marketing assets, social content, etc.

## Architecture

```
┌─────────────┐         WebSocket          ┌──────────────────┐
│   Browser   │ ◄─────────────────────────► │   Backend WS     │
│ story.html  │    /ws/story               │   /ws/story      │
└─────────────┘                             └──────────────────┘
                                                     │
                                                     ▼
                                            ┌──────────────────┐
                                            │  Gemini 2.0 Flash│
                                            │  Vertex AI       │
                                            │  TEXT + IMAGE    │
                                            └──────────────────┘
                                                     │
                                                     ▼
                                            ┌──────────────────┐
                                            │  Google Cloud    │
                                            │  Text-to-Speech  │
                                            │  (MP3 narration) │
                                            └──────────────────┘
```

## Configuration

### Environment Variables

**Required for Vertex AI**:
```bash
export GOOGLE_GENAI_USE_VERTEXAI=True
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_LOCATION="us-central1"
```

**Optional**:
```bash
export AERIVON_BACKEND_PORT=8081
export AERIVON_FRONTEND_PORT=5174
```

### Dependencies

Already included in `backend/requirements.txt`:
- `google-genai` - Gemini SDK
- `google-cloud-texttospeech` - TTS synthesis
- `fastapi` - WebSocket server
- `playwright` - Browser automation (for UI Navigator)

## Performance

- **Story Generation Time**: 10-30 seconds (depends on length and complexity)
- **Image Generation**: Inline with Gemini's multimodal output
- **TTS Synthesis**: ~1-2 seconds per text chunk
- **Streaming**: Real-time as content is generated
- **Audio Playback**: Sequential queue to maintain narrative flow

## Future Enhancements

- [ ] Video generation integration (when available in Gemini)
- [ ] Custom voice selection for narration
- [ ] Multiple illustration styles (watercolor, cartoon, photorealistic)
- [ ] Story branching / choose-your-own-adventure
- [ ] Export stories as PDF or video
- [ ] Background music generation
- [ ] Multi-language support
- [ ] Character consistency across illustrations

## Credits

**Created for**: Google Gemini API Hackathon 2026  
**Category**: Multimodal Storytelling with Interleaved Output  
**Technologies**: Gemini 2.0 Flash, Vertex AI, Google Cloud TTS, Google Cloud Run  
**Agent**: Aerivon Live - Multimodal AI Platform
