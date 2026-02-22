import inspect

from google.genai import types


def main() -> None:
    print("LiveConnectConfig sig:", inspect.signature(types.LiveConnectConfig))
    print("LiveConnectConfig annotations:", getattr(types.LiveConnectConfig, "__annotations__", None))

    print("\nSpeech-ish config types:")
    for name in sorted(dir(types)):
        if "speech" in name.lower() and "config" in name.lower():
            obj = getattr(types, name)
            if isinstance(obj, type):
                try:
                    print("-", name, "sig", inspect.signature(obj))
                except Exception:
                    print("-", name)

    print("\nCommon candidate types:")
    for n in [
        "GenerationConfig",
        "LiveGenerationConfig",
        "BidiGenerateContentSetup",
        "SpeechConfig",
        "VoiceConfig",
        "PrebuiltVoiceConfig",
    ]:
        obj = getattr(types, n, None)
        if obj is None:
            continue
        try:
            print(n, "sig", inspect.signature(obj))
        except Exception as e:
            print(n, "sig err", e)

    print("\nModality:", list(getattr(types, "Modality").__members__.keys()))


if __name__ == "__main__":
    main()
