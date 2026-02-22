import asyncio
from pathlib import Path

from google import genai
from google.genai import types
from google.genai.types import HttpOptions


MODEL = "gemini-2.0-flash-live-preview-04-09"
IMG_PATH = "/tmp/aerivon_artifacts/screenshot_1771738708.png"


async def main() -> None:
    img = Path(IMG_PATH).read_bytes()

    client = genai.Client(http_options=HttpOptions(api_version="v1beta1"))
    cfg = types.LiveConnectConfig(response_modalities=[types.Modality.TEXT])

    async with client.aio.live.connect(model=MODEL, config=cfg) as stream:
        await stream.send_realtime_input(
            text="What domain is shown in this screenshot? Answer with the domain only."
        )
        await stream.send_realtime_input(media=types.Blob(data=img, mime_type="image/png"))

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
