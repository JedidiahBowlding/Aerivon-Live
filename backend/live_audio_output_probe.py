import asyncio
import os
from typing import Any

from google import genai
from google.genai import types
from google.genai.types import HttpOptions


MODEL = os.getenv("GEMINI_LIVE_MODEL", "gemini-2.0-flash-live-preview-04-09")
VOICE = os.getenv("AERIVON_LIVE_VOICE", "")
LANG = os.getenv("AERIVON_LIVE_VOICE_LANG", "en-US")


def _speech_config() -> types.SpeechConfig | None:
    if not VOICE:
        return None
    return types.SpeechConfig(
        language_code=LANG,
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=VOICE)
        ),
    )


async def main() -> None:
    if os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").strip().lower() not in {"1", "true", "yes"}:
        raise SystemExit("Set GOOGLE_GENAI_USE_VERTEXAI=True")

    client = genai.Client(http_options=HttpOptions(api_version="v1beta1"))
    config = types.LiveConnectConfig(
        system_instruction="You are a helpful assistant.",
        response_modalities=[types.Modality.AUDIO],
        speech_config=_speech_config(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
    )

    async with client.aio.live.connect(model=MODEL, config=config) as stream:
        await stream.send_client_content(
            turns=types.Content(
                role="user",
                parts=[types.Part.from_text(text="Say the word HELLO out loud, then also print it as text.")],
            ),
            turn_complete=True,
        )

        i = 0
        async for msg in stream.receive():
            i += 1
            print("--- msg", i, "---")
            print("msg.text:", getattr(msg, "text", None))

            # raw binary fields (best-effort)
            for attr in ("data", "mime_type", "mimeType"):
                if hasattr(msg, attr):
                    val = getattr(msg, attr)
                    if val is None:
                        continue
                    if isinstance(val, (bytes, bytearray)):
                        print(attr, "bytes", len(val))
                    else:
                        print(attr, type(val), str(val)[:120])

            sc = getattr(msg, "server_content", None)
            if sc is not None:
                print("server_content.interrupted:", getattr(sc, "interrupted", None))
                print("server_content.turn_complete:", getattr(sc, "turn_complete", None))
                print("server_content.generation_complete:", getattr(sc, "generation_complete", None))

                otx = getattr(sc, "output_transcription", None)
                if otx is not None:
                    print("output_transcription:", otx)

                mt = getattr(sc, "model_turn", None)
                parts = getattr(mt, "parts", None) if mt is not None else None
                if parts:
                    for p in parts:
                        inline = getattr(p, "inline_data", None)
                        txt = getattr(p, "text", None)
                        if txt:
                            print("part.text:", txt[:80])
                        if inline is not None:
                            mime = getattr(inline, "mime_type", None)
                            data = getattr(inline, "data", None)
                            print("part.inline_data:", mime, (len(data) if isinstance(data,(bytes,bytearray)) else type(data)))

            if i >= 12:
                break


if __name__ == "__main__":
    asyncio.run(main())
