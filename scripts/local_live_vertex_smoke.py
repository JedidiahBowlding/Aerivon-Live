import asyncio
from google import genai
from google.genai import types
from google.genai.types import HttpOptions

PROJECT = "snark-shot"
LOCATION = "us-central1"


async def main() -> int:
    client = genai.Client(
        vertexai=True,
        project=PROJECT,
        location=LOCATION,
        http_options=HttpOptions(api_version="v1"),
    )

    chunks: list[str] = []
    async with client.aio.live.connect(
        model="gemini-2.0-flash-exp",
        config=types.LiveConnectConfig(
            response_modalities=[types.Modality.TEXT],
            system_instruction="Reply with exactly: vertex-live-ok",
            temperature=0.0,
            max_output_tokens=32,
        ),
    ) as session:
        await session.send_client_content(
            turns=types.Content(role="user", parts=[types.Part.from_text(text="hello")]),
            turn_complete=True,
        )
        async for r in session.receive():
            if r.text:
                chunks.append(r.text)
            if r.server_content and r.server_content.turn_complete:
                break

    print("VERTEX LIVE:", "".join(chunks).strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
