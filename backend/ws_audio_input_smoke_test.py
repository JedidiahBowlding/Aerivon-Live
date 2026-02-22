import asyncio
import json
import os

import websockets


WS_URL = os.getenv(
    "AERIVON_WS_URL_AUDIO_IN",
    "ws://127.0.0.1:8080/ws/live?output=text",
)


def make_silence_pcm(*, seconds: float = 0.25, sample_rate: int = 16000) -> bytes:
    # int16 little-endian silence
    samples = int(seconds * sample_rate)
    return b"\x00\x00" * samples


async def main() -> None:
    print("WS_URL:", WS_URL)
    pcm = make_silence_pcm(seconds=0.25)

    async with websockets.connect(WS_URL, max_size=8 * 1024 * 1024) as ws:
        # status
        while True:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=12.0))
            if msg.get("type") == "status":
                print("STATUS:", msg)
                break

        # prompt + mic-style audio chunk(s)
        await ws.send(
            json.dumps(
                {
                    "type": "text",
                    "text": "Transcribe the audio. If it is silent, reply with exactly: SILENCE",
                }
            )
        )

        # Send as a couple chunks
        half = len(pcm) // 2
        for chunk in (pcm[:half], pcm[half:]):
            await ws.send(
                json.dumps(
                    {
                        "type": "audio",
                        "mime_type": "audio/pcm",
                        "data_b64": __import__("base64").b64encode(chunk).decode("ascii"),
                    }
                )
            )
        await ws.send(json.dumps({"type": "audio_end"}))

        out = ""
        while True:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=25.0))
            if msg.get("type") == "text":
                out += msg.get("text") or ""
            if msg.get("type") == "turn_complete":
                break

        print("OUT:", out.strip())


if __name__ == "__main__":
    asyncio.run(main())
