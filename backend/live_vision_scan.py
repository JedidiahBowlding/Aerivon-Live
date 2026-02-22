import asyncio
import os
import struct
import zlib

from google import genai
from google.genai import types
from google.genai.types import HttpOptions


def short_model_name(full_name: str) -> str:
    for marker in ("/models/", "/publishers/google/models/"):
        if marker in full_name:
            return full_name.split(marker, 1)[1]
    return full_name


def make_probe_png_bytes(size: int = 32) -> bytes:
    width = height = max(8, int(size))
    row = b"\x00" + (b"\xff\x00\x00" * width)  # red RGB
    raw = row * height
    compressed = zlib.compress(raw)

    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + chunk_type
            + data
            + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        )

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # RGB
    return signature + chunk(b"IHDR", ihdr) + chunk(b"IDAT", compressed) + chunk(b"IEND", b"")


async def probe_vision(client: genai.Client, model: str, config: types.LiveConnectConfig) -> tuple[bool, str]:
    png = make_probe_png_bytes()
    prompt = "What color is the square in the provided image? Reply with exactly one word: RED or GREEN."
    out: list[str] = []

    try:
        async with client.aio.live.connect(model=model, config=config) as stream:
            await stream.send_realtime_input(text=prompt)
            await stream.send_realtime_input(media=types.Blob(data=png, mime_type="image/png"))

            start = asyncio.get_running_loop().time()
            async for msg in stream.receive():
                if getattr(msg, "text", None):
                    out.append(msg.text)

                sc = getattr(msg, "server_content", None)
                if sc is not None and getattr(sc, "turn_complete", None) is True:
                    break

                if asyncio.get_running_loop().time() - start > 6.0:
                    break

        text = "".join(out).strip()
        ok = text.strip().lower().startswith("red")
        return ok, text
    except Exception as exc:
        return False, f"ERROR: {exc}"


async def main() -> None:
    if os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").strip().lower() not in {"1", "true", "yes"}:
        raise SystemExit("Set GOOGLE_GENAI_USE_VERTEXAI=True")

    client = genai.Client(http_options=HttpOptions(api_version="v1beta1"))
    config = types.LiveConnectConfig(response_modalities=[types.Modality.TEXT])

    raw = []
    for m in client.models.list():
        name = getattr(m, "name", None)
        if not name:
            continue
        s = short_model_name(str(name))
        if "live" in s.lower():
            raw.append(s)

    # Always try the currently configured model first.
    configured = os.getenv("GEMINI_LIVE_MODEL", "gemini-2.0-flash-live-preview-04-09")
    candidates = [configured] + [x for x in raw if x != configured]

    print(f"Found {len(raw)} live-ish models via models.list(); probing up to 12...")

    for model in candidates[:12]:
        ok, text = await probe_vision(client, model, config)
        print("\nMODEL:", model)
        print("VISION_OK:", ok)
        print("OUTPUT:", text[:200].replace("\n", " "))


if __name__ == "__main__":
    asyncio.run(main())
