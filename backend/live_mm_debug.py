import asyncio
from pathlib import Path

from google import genai
from google.genai import types
from google.genai.types import HttpOptions


async def main() -> None:
    img_path = Path("/tmp/aerivon_test.png")
    if not img_path.exists():
        raise SystemExit("Missing /tmp/aerivon_test.png; generate it first")

    img = img_path.read_bytes()

    client = genai.Client(http_options=HttpOptions(api_version="v1beta1"))
    cfg = types.LiveConnectConfig(response_modalities=[types.Modality.TEXT])

    async with client.aio.live.connect(
        model="gemini-2.0-flash-live-preview-04-09",
        config=cfg,
    ) as stream:
        parts = [
            types.Part.from_text(text="Describe what you see in the image in one short sentence."),
            types.Part.from_bytes(data=img, mime_type="image/png"),
        ]
        turn = types.Content(role="user", parts=parts)
        await stream.send(
            input=types.LiveClientContent(turns=[turn], turnComplete=True),
            end_of_turn=True,
        )

        i = 0
        async for msg in stream.receive():
            i += 1
            print("--- msg", i, "---")
            print("text:", getattr(msg, "text", None))

            server_content = getattr(msg, "server_content", None)
            if server_content is not None:
                print("server_content:", server_content)

            tool_call = getattr(msg, "tool_call", None)
            if tool_call is not None:
                print("tool_call:", tool_call)

            if i >= 8:
                break


if __name__ == "__main__":
    asyncio.run(main())
