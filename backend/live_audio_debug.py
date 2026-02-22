import asyncio
from pathlib import Path

from google import genai
from google.genai import types
from google.genai.types import HttpOptions


async def main() -> None:
    wav_path = Path("/tmp/aerivon_test.wav")
    if not wav_path.exists():
        raise SystemExit("Missing /tmp/aerivon_test.wav")

    audio = wav_path.read_bytes()

    client = genai.Client(http_options=HttpOptions(api_version="v1beta1"))
    cfg = types.LiveConnectConfig(response_modalities=[types.Modality.TEXT])

    async with client.aio.live.connect(
        model="gemini-2.0-flash-live-preview-04-09",
        config=cfg,
    ) as stream:
        # Live SDK currently allows only one argument per send_realtime_input call.
        await stream.send_realtime_input(text="Transcribe the audio. If it is silent, reply with exactly: SILENCE")
        await stream.send_realtime_input(audio=types.Blob(data=audio, mime_type="audio/wav"))
        await stream.send_realtime_input(audio_stream_end=True)

        i = 0
        async for msg in stream.receive():
            i += 1
            print("--- msg", i, "---")
            print("text:", getattr(msg, "text", None))

            sc = getattr(msg, "server_content", None)
            if sc is not None:
                itx = getattr(sc, "input_transcription", None)
                otx = getattr(sc, "output_transcription", None)
                mt = getattr(sc, "model_turn", None)
                tc = getattr(sc, "turn_complete", None)
                gc = getattr(sc, "generation_complete", None)
                print("server_content.turn_complete:", tc)
                print("server_content.generation_complete:", gc)
                if itx is not None:
                    print("input_transcription:", itx)
                if otx is not None:
                    print("output_transcription:", otx)
                if mt is not None:
                    print("model_turn:", mt)

            if i >= 20:
                break


if __name__ == "__main__":
    asyncio.run(main())
