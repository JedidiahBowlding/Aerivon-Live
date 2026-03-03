from google import genai
from google.genai import types
from google.genai.types import HttpOptions

PROJECT = "snark-shot"
LOCATION = "global"
MODEL = "gemini-2.5-flash-image"

PROMPT = "Create one very short scene and include one generated illustration."


def main() -> int:
    client = genai.Client(
        vertexai=True,
        project=PROJECT,
        location=LOCATION,
        http_options=HttpOptions(api_version="v1"),
    )

    resp = client.models.generate_content(
        model=MODEL,
        contents=PROMPT,
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
            temperature=0.7,
            max_output_tokens=400,
        ),
    )

    text_parts = 0
    image_parts = 0
    if resp.candidates:
        for cand in resp.candidates:
            if cand.content and cand.content.parts:
                for part in cand.content.parts:
                    if getattr(part, "text", None):
                        text_parts += 1
                    if getattr(part, "inline_data", None) and getattr(part.inline_data, "data", None):
                        image_parts += 1

    print(f"OK text_parts={text_parts} image_parts={image_parts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
