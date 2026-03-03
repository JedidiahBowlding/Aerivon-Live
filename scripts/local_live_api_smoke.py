import io
import wave
import asyncio
from google import genai
from google.genai import types

API_KEY = "AQ.Ab8RN6LSKosGd1MDvkOj-_yEOuWHS_Wq7Arq8gx4-jaRkRQCGA"


def make_wav_bytes() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * int(16000 * 0.2))
    return buf.getvalue()


async def smoke_text(client: genai.Client) -> None:
    text_parts: list[str] = []
    async with client.aio.live.connect(
        model="gemini-2.0-flash-exp",
        config=types.LiveConnectConfig(
            response_modalities=[types.Modality.TEXT],
            system_instruction="Reply with exactly: live-text-ok",
            max_output_tokens=32,
            temperature=0.0,
        ),
    ) as session:
        await session.send_client_content(
            turns=types.Content(role="user", parts=[types.Part.from_text(text="hi")]),
            turn_complete=True,
        )
        async for r in session.receive():
            if r.text:
                text_parts.append(r.text)
            if r.server_content and r.server_content.turn_complete:
                break
    print("LIVE TEXT:", "".join(text_parts).strip())


async def smoke_audio_transcribe(client: genai.Client) -> None:
    text_parts: list[str] = []
    wav_bytes = make_wav_bytes()
    async with client.aio.live.connect(
        model="gemini-2.0-flash-exp",
        config=types.LiveConnectConfig(
            response_modalities=[types.Modality.TEXT],
            system_instruction=(
                "Transcribe user audio into text. Return ONLY the exact words spoken, nothing more."
            ),
            max_output_tokens=64,
            temperature=0.0,
        ),
    ) as session:
        await session.send_client_content(
            turns=types.Content(
                role="user",
                parts=[types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav")],
            ),
            turn_complete=True,
        )
        async for r in session.receive():
            if r.text:
                text_parts.append(r.text)
            if r.server_content and r.server_content.turn_complete:
                break
    print("LIVE AUDIO TRANSCRIBE:", "".join(text_parts).strip())


async def main() -> int:
    client = genai.Client(api_key=API_KEY)
    await smoke_text(client)
    await smoke_audio_transcribe(client)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
