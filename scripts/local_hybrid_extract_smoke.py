from google import genai
from google.genai import types
from google.genai.types import HttpOptions

API_KEY = "AQ.Ab8RN6LSKosGd1MDvkOj-_yEOuWHS_Wq7Arq8gx4-jaRkRQCGA"
PROJECT = "snark-shot"
LOCATION = "global"

PROMPT = """Analyze this user request and extract two components:
1. Navigation goal: What website to visit and what actions to take
2. Story theme: What kind of illustrated story to create based on findings

User request: open example.com, click learn more, then create a story.

Respond in JSON format:
{
  \"navigation_goal\": \"specific navigation instructions\",
  \"story_theme\": \"theme for the illustrated story\"
}
"""


def main() -> int:
    print("--- API key generate_content test ---")
    api_client = genai.Client(api_key=API_KEY)
    try:
        r = api_client.models.generate_content(
            model="gemini-1.5-flash",
            contents=PROMPT,
            config=types.GenerateContentConfig(temperature=0.3, response_mime_type="application/json"),
        )
        print("API SUCCESS:", (r.text or "")[:120])
    except Exception as exc:
        print("API FAIL:", str(exc).splitlines()[0])

    print("--- Vertex generate_content test ---")
    vertex_client = genai.Client(
        vertexai=True,
        project=PROJECT,
        location=LOCATION,
        http_options=HttpOptions(api_version="v1"),
    )
    r2 = vertex_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=PROMPT,
        config=types.GenerateContentConfig(temperature=0.3, response_mime_type="application/json"),
    )
    print("VERTEX SUCCESS:", (r2.text or "")[:120])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
