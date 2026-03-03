import asyncio
from google import genai
from google.genai import types
from google.genai.types import HttpOptions

API_KEY = "AQ.Ab8RN6LSKosGd1MDvkOj-_yEOuWHS_Wq7Arq8gx4-jaRkRQCGA"
PROJECT = "snark-shot"
LOCATION = "us-central1"


def is_api_key_live_error(exc: Exception) -> bool:
    msg = str(exc)
    return (
        "API keys are not supported" in msg
        or "1008" in msg
        or "UNAUTHENTICATED" in msg
    )


async def run_live_text(client: genai.Client, model: str, prompt: str) -> str:
    chunks: list[str] = []
    async with client.aio.live.connect(
        model=model,
        config=types.LiveConnectConfig(
            response_modalities=[types.Modality.TEXT],
            system_instruction="Reply with exactly: fallback-ok",
            temperature=0.0,
            max_output_tokens=32,
        ),
    ) as session:
        await session.send_client_content(
            turns=types.Content(role="user", parts=[types.Part.from_text(text=prompt)]),
            turn_complete=True,
        )
        async for r in session.receive():
            if r.text:
                chunks.append(r.text)
            if r.server_content and r.server_content.turn_complete:
                break
    return "".join(chunks).strip()


async def main() -> int:
    primary = genai.Client(api_key=API_KEY)
    try:
        text = await run_live_text(primary, "gemini-2.0-flash-live-preview-04-09", "hello")
        print("PRIMARY(API KEY) SUCCESS:", text)
        return 0
    except Exception as exc:
        print("PRIMARY(API KEY) FAIL:", str(exc).splitlines()[0])
        if not is_api_key_live_error(exc):
            return 1

    fallback = genai.Client(
        vertexai=True,
        project=PROJECT,
        location=LOCATION,
        http_options=HttpOptions(api_version="v1"),
    )
    text = await run_live_text(fallback, "gemini-2.0-flash-live-preview-04-09", "hello")
    print("FALLBACK(VERTEX) SUCCESS:", text)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
