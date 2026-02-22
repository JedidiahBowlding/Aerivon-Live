import asyncio
from pathlib import Path

from google.genai import types

from gemini_client import GeminiLiveClient


async def main() -> None:
    wav_path = Path("/tmp/aerivon_test.wav")
    audio = wav_path.read_bytes()

    glc = GeminiLiveClient()
    if not glc.use_vertex:
        raise SystemExit("This debug requires GOOGLE_GENAI_USE_VERTEXAI=True")

    config = glc.build_config()

    async with glc.client.aio.live.connect(model=glc.model, config=config) as stream:
        prompt = "Transcribe the audio. If it is silent, reply with exactly: SILENCE"
        await stream.send_realtime_input(text=prompt)
        await stream.send_realtime_input(audio=types.Blob(data=audio, mime_type="audio/wav"))
        await stream.send_realtime_input(audio_stream_end=True)

        i = 0
        async for msg in stream.receive():
            i += 1
            print('--- msg', i, '---')
            print('msg.text:', getattr(msg, 'text', None))
            sc = getattr(msg, 'server_content', None)
            if sc is not None:
                print('server_content.generation_complete:', getattr(sc, 'generation_complete', None))
                print('server_content.turn_complete:', getattr(sc, 'turn_complete', None))
                mt = getattr(sc, 'model_turn', None)
                if mt is not None:
                    print('model_turn:', mt)
                otx = getattr(sc, 'output_transcription', None)
                if otx is not None:
                    print('output_transcription:', otx)
            if i >= 12:
                break


if __name__ == '__main__':
    asyncio.run(main())
