import io
import wave

from google import genai
from google.genai import types
from google.genai.types import HttpOptions

API_KEY = "AQ.Ab8RN6LSKosGd1MDvkOj-_yEOuWHS_Wq7Arq8gx4-jaRkRQCGA"
PROJECT = "snark-shot"
LOCATION = "us-central1"


def make_wav_bytes() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * int(16000 * 0.2))
    return buf.getvalue()


def call(client: genai.Client, model: str, wav_bytes: bytes):
    return client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_text(text="Transcribe this audio into text. Return ONLY the exact words spoken."),
            types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"),
        ],
        config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=50),
    )


def main() -> int:
    wav_bytes = make_wav_bytes()

    api_client = genai.Client(api_key=API_KEY)
    for model_name in ["gemini-2.0-flash", "gemini-1.5-flash"]:
        try:
            resp = call(api_client, model_name, wav_bytes)
            print("API SUCCESS", model_name, (resp.text or "").strip())
            return 0
        except Exception as exc:
            print("API FAIL", model_name, str(exc).splitlines()[0])

    vertex_client = genai.Client(
        vertexai=True,
        project=PROJECT,
        location=LOCATION,
        http_options=HttpOptions(api_version="v1"),
    )
    for model_name in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]:
        try:
            resp = call(vertex_client, model_name, wav_bytes)
            print("VERTEX SUCCESS", model_name, (resp.text or "").strip())
            return 0
        except Exception as exc:
            print("VERTEX FAIL", model_name, str(exc).splitlines()[0])

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
