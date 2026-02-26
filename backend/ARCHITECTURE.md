# Aerivon Live — Architecture (Hackathon Submission Version)

This architecture demonstrates a secure, autonomous, multimodal Gemini-powered AI agent deployed on Google Cloud.

---

## System Architecture Diagram

```mermaid
flowchart TB

  subgraph Client Layer
    U[User / Judge]
    C[Client Interface\ncurl / frontend]
  end

  subgraph API Layer
    API[FastAPI Backend\nCloud Run]
    Gate[Security Gateway\n• Rate limiting\n• Prompt injection filter\n• Size limits\n• SSRF protection]
  end

  subgraph Agent Layer
    Agent[AerivonLiveAgent\nAutonomous Agent Runtime]
    FW[Tool Firewall\n• Allowlist enforcement\n• Argument validation\n• Relevance gating\n• Max 6 tool calls\n• Timeout protection]
  end

  subgraph AI Layer
    GC[GeminiLiveClient]
    Vertex[Vertex AI\nGoogle Cloud]

    Live[Gemini Live API\nRealtime multimodal]
    Flash[Gemini Flash\nFallback multimodal model]
  end

  subgraph Tool Layer
    Tools[Tool Execution Runtime]

    Browser[Playwright Chromium\nVisual browsing\nScreenshot capture]

    Leads[Lead Extraction Engine\nBusiness discovery]

    Outreach[Outreach Generator\nAutomated messaging]
  end

  subgraph Storage Layer
    Tmp[/Ephemeral Artifact Storage<br/>/tmp screenshots]
  end

  U --> C
  C -->|HTTP JSON| API

  API --> Gate
  Gate --> Agent

  Agent --> GC
  GC --> Vertex

  Vertex --> Live
  Vertex --> Flash

  Live --> Agent
  Flash --> Agent

  Agent --> FW
  FW --> Tools

  Tools --> Browser
  Tools --> Leads
  Tools --> Outreach

  Browser --> Tmp

  Tools --> Agent
  Agent --> API
  API --> C
```

---

## Execution Flow

1. User sends request to `/agent/message`
2. API gateway enforces security and cost controls
3. Aerivon Live agent processes request
4. Gemini client connects to Vertex AI
5. Agent uses Gemini Live if available, otherwise falls back to Gemini Flash
6. Gemini may request tool execution
7. Tool firewall validates and executes tools securely
8. Tool results are wrapped as untrusted data
9. Gemini produces final response
10. API returns structured JSON response

---

## Google Cloud Integration

Aerivon Live runs fully on Google Cloud:

* Cloud Run — serverless agent hosting
* Vertex AI — Gemini model access
* Artifact Registry — container storage
* Cloud Build — automated deployment pipeline

This satisfies the hackathon requirement for Google Cloud hosting.

---

## Gemini Model Usage

Aerivon Live uses Gemini models through Vertex AI:

Primary mode:

* Gemini Live API (realtime multimodal interaction)

Fallback mode:

* Gemini Flash (multimodal reasoning)

Both modes support:

* text input/output
* structured tool invocation
* multimodal extensibility

---

## Autonomous Agent Capabilities

Aerivon Live is a fully autonomous agent capable of:

* browsing websites
* extracting business information
* generating outreach messages
* chaining tool execution
* reasoning over tool results

This demonstrates true agent autonomy beyond simple text generation.

---

## Security Architecture

Aerivon Live implements defense-in-depth protections:

### Prompt Injection Protection

Blocks attempts such as:

* "ignore previous instructions"
* "reveal system prompt"
* "export secrets"

### Tool Firewall

Prevents unsafe execution via:

* tool allowlist enforcement
* argument validation
* relevance filtering
* execution timeouts

### SSRF Protection

Blocks access to sensitive endpoints:

* metadata.google.internal
* localhost
* private IP ranges

### Cost and Abuse Protection

Limits resource usage via:

* rate limiting
* message size limits
* session caps
* tool call caps

---

## Diagnostic and Verification Endpoints

These endpoints allow judges to verify system operation:

| Endpoint              | Purpose                        |
| --------------------- | ------------------------------ |
| /health               | Backend health status          |
| /agent/startup-check  | Gemini availability check      |
| /agent/security-check | Security policy verification   |
| /agent/self-test      | Autonomous system verification |

---

## Fail-Safe Design

Aerivon Live implements automatic fallback:

If Gemini Live unavailable → fallback to Gemini Flash

This ensures continuous operation across environments.

---

## Hackathon Requirement Compliance

| Requirement                       | Status |
| --------------------------------- | ------ |
| Uses Gemini model                 | ✓      |
| Uses Google Cloud                 | ✓      |
| Uses multimodal-capable model     | ✓      |
| Uses agent architecture           | ✓      |
| Uses secure execution             | ✓      |
| Demonstrates autonomous reasoning | ✓      |

---

## Summary

Aerivon Live is a secure, autonomous, multimodal Gemini-powered AI agent deployed on Google Cloud that demonstrates real-world agent capabilities including browsing, lead generation, and automated outreach.
