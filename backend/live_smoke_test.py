import asyncio

from google import genai
from google.genai import types
from google.genai.types import HttpOptions

MODEL = "gemini-2.0-flash-live-preview-04-09"


async def main() -> None:
    client = genai.Client(http_options=HttpOptions(api_version="v1beta1"))
    config = types.LiveConnectConfig(
        response_modalities=[types.Modality.TEXT],
        system_instruction="Reply with OK",
    )

    try:
        async with client.aio.live.connect(model=MODEL, config=config) as stream:
            await stream.send(input="hi", end_of_turn=True)
            async for msg in stream.receive():
                if msg.text:
                    print("LIVE_CONNECT_OK", msg.text[:80])
                    return
            print("LIVE_CONNECT_OK_NO_TEXT")
    except Exception as exc:
        print("LIVE_CONNECT_FAILED", type(exc).__name__, str(exc)[:400])


if __name__ == "__main__":
    asyncio.run(main())
