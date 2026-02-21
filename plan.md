# Development Task: Modify Gemini "Browser_as_a_tool" Example into Aerivon Live Autonomous Agent

## Objective

Use the official Gemini example:

https://github.com/google-gemini/cookbook/blob/main/examples/Browser_as_a_tool.ipynb

and convert it into a production-ready autonomous multimodal business agent called **Aerivon Live**.

This agent must use Gemini Live API and function calling to execute real business automation tasks such as browsing websites, scraping leads, generating outreach messages, and executing workflows.

---

# Step 1 — Convert Notebook into Production Backend

Convert the notebook into Python backend files:

backend/

* agent.py
* gemini_client.py
* tools.py
* server.py

Remove all notebook-specific code.

Use Python 3.11+.

---

# Step 2 — Replace Basic Browser Tool with Production Tool System

The example currently includes:

def browse_url(url: str)

Expand this into a full toolset.

Create these tools in tools.py:

def browse_url(url: str)

def scrape_leads(location: str, business_type: str)

def extract_page_content()

def take_screenshot()

def generate_outreach_message(business_name: str, website: str, service: str)

Each tool must return structured JSON.

Example:

{
"business_name": "Example Dental",
"website": "example.com",
"email": "[contact@example.com](mailto:contact@example.com)"
}

---

# Step 3 — Modify Gemini Agent Configuration

In gemini_client.py:

Use Gemini Live API model:

gemini-2.5-flash-native-audio-preview

Enable tool calling:

config = {
"tools": tools,
"response_modalities": ["TEXT"]
}

Add system instruction:

system_instruction = """
You are Aerivon Live, an autonomous business agent.
Your role is to browse websites, scrape leads, generate outreach messages,
and execute business automation tasks using available tools.
Always prefer tool usage when possible.
"""

---

# Step 4 — Implement Autonomous Agent Loop

Modify the example loop into a reusable agent loop:

while True:

```
user_input = receive_input()

send input to Gemini Live API

if tool call returned:
    execute tool
    send tool result back to Gemini

else:
    return Gemini response
```

This must support continuous interaction.

---

# Step 5 — Add Lead Scraping Capability

Modify browse_url tool to support scraping:

Use Selenium or Playwright.

Extract:

* business name
* website
* email
* phone number

Return structured JSON.

---

# Step 6 — Add WebSocket Server

Create server.py using FastAPI.

Endpoints:

POST /agent/message
POST /agent/tool-result

This allows frontend connection.

---

# Step 7 — Prepare for Google Cloud Deployment

Add requirements.txt:

google-genai
fastapi
uvicorn
selenium
playwright

Add Dockerfile:

FROM python:3.11

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]

---

# Step 8 — Verify Tool Calling Works

Test example input:

"Find dentists in Miami and generate outreach messages."

Gemini should:

1. Call scrape_leads tool
2. Receive results
3. Call generate_outreach_message tool
4. Return final output

---

# Step 9 — Keep Compatibility with Gemini Live API

Must use:

from google import genai

and

client.aio.live.connect()

Do NOT use deprecated SDKs.

---

# Final Deliverables Required

Backend must support:

* Gemini Live API
* tool execution
* autonomous loop
* structured tool responses
* Cloud Run deployment ready

---

# Important

Do NOT change Gemini Live API architecture.

Only expand tools and productionize the example.

Use the official example as the base.

---

# Goal Outcome

Aerivon Live must function as a real autonomous multimodal agent capable of:

* browsing websites
* scraping business leads
* generating outreach messages
* executing workflows

using Gemini Live API tool calling.
