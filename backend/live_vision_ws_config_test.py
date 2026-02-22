import asyncio
from pathlib import Path

from google import genai
from google.genai import types
from google.genai.types import HttpOptions

from gemini_client import SYSTEM_INSTRUCTION


MODEL = "gemini-2.0-flash-live-preview-04-09"
SCREENSHOT = "/tmp/aerivon_artifacts/screenshot_1771738708.png"


async def main() -> None:
    img = Path(SCREENSHOT).read_bytes()

    client = genai.Client(http_options=HttpOptions(api_version="v1beta1"))
    config = types.LiveConnectConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        response_modalities=[types.Modality.TEXT],
    )

    async with client.aio.live.connect(model=MODEL, config=config) as stream:
        parts = [
            types.Part.from_text(
                text="What domain is shown in this screenshot? Answer with the domain only."
            ),
            types.Part.from_bytes(data=img, mime_type="image/png"),
        ]
        await stream.send_client_content(
            turns=types.Content(role="user", parts=parts),
            turn_complete=True,
        )

        out: list[str] = []
        async for msg in stream.receive():
            if getattr(msg, "text", None):
                out.append(msg.text)
            sc = getattr(msg, "server_content", None)
            if sc is not None and getattr(sc, "turn_complete", None) is True:
                break

        print("OUT:", "".join(out).strip())


if __name__ == "__main__":
    asyncio.run(main())
