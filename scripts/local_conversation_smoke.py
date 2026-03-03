from google import genai
from google.genai.types import HttpOptions

API_KEY = "AQ.Ab8RN6LSKosGd1MDvkOj-_yEOuWHS_Wq7Arq8gx4-jaRkRQCGA"
PROJECT = "snark-shot"
LOCATION = "us-central1"
PROMPT = "Reply with exactly: conversation-ok"


def main() -> int:
    config = {"temperature": 0.1, "max_output_tokens": 64}

    api_client = genai.Client(api_key=API_KEY)
    for model_name in ["gemini-2.0-flash", "gemini-1.5-flash"]:
        try:
            resp = api_client.models.generate_content(model=model_name, contents=PROMPT, config=config)
            print("API SUCCESS", model_name, (resp.text or "").strip())
            return 0
        except Exception as exc:
            print("API FAIL", model_name, str(exc).splitlines()[0])

    vertex_client = genai.Client(
        vertexai=True,
        project=PROJECT,
        location=LOCATION,
        http_options=HttpOptions(api_version="v1"),
    )
    for model_name in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]:
        try:
            resp = vertex_client.models.generate_content(model=model_name, contents=PROMPT, config=config)
            print("VERTEX SUCCESS", model_name, (resp.text or "").strip())
            return 0
        except Exception as exc:
            print("VERTEX FAIL", model_name, str(exc).splitlines()[0])

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
