# Incoming PCM Playback (Current Implementation)

This repo currently plays incoming WS audio as **raw PCM (`audio/pcm`, s16le, mono)** using a **WebAudio queue**.

## TL;DR

- Playback engine: **WebAudio API** (`AudioContext`, `AudioBuffer`, `AudioBufferSourceNode`)
- Transport payload: WS JSON with base64 PCM chunks
- Queue model: schedule each chunk at `max(playCursor, audioContext.currentTime + small_lead)`
- No `MediaSource`, no `<audio src=...>` streaming, no AudioWorklet in the current path

## Message Shape Expected

Incoming server message (audio chunk):

```json
{
  "type": "audio",
  "mime_type": "audio/pcm",
  "data_b64": "...",
  "session_id": 1
}
```

Optional config message (sample rate):

```json
{
  "type": "audio_config",
  "sample_rate": 24000,
  "format": "pcm_s16le",
  "channels": 1,
  "mime_type": "audio/pcm"
}
```

## Runtime Path (Live Agent / Navigator)

Implementation is in [frontend/main.js](frontend/main.js):

1. **Decode base64 → bytes**
   - `b64ToU8()` converts `data_b64` to `Uint8Array`.
2. **Interpret bytes as s16le PCM**
   - `new Int16Array(buffer, byteOffset, byteLength / 2)`.
3. **Convert to float samples**
   - `int16ToFloat32()` maps to [-1, 1].
4. **Create audio buffer**
   - `AudioBuffer(1 channel, sample_count, playbackSampleRate)`.
5. **Queue chunk in timeline**
   - `startAt = max(playCursor, currentTime + 0.02)`.
   - `playCursor = startAt + chunkDuration`.
6. **Track active sources**
   - `activeSources[]` stores currently scheduled/playing source nodes.
   - On `onended`, source is removed.

Key functions:

- `ensurePlaybackCtx()`
- `applyAudioConfig(sampleRate)`
- `enqueuePcmForPlayback(pcmBytes)`
- `stopPlayback()`
- `isAudioPlayingOrQueued()`

## Runtime Path (Translator Live)

Implementation is in [frontend/translator_live.js](frontend/translator_live.js):

- Same WebAudio queue pattern as above.
- Same PCM decode and schedule strategy.
- Uses:
  - `ensurePlaybackCtx()`
  - `enqueuePcmForPlayback()`
  - `stopPlayback()`
  - `isAudioPlayingOrQueued()`

## Interruption Behavior (Barge-in)

When user speech is detected during playback:

- Client sends `{ "type": "interrupt" }` over WS.
- Client immediately calls `stopPlayback()` to cut local audio.
- Server then emits interruption/turn restart events.

This is currently the mechanism that makes interruption feel immediate in browser.

## What This Is **Not**

- Not using `MediaSource` append buffers
- Not using `AudioWorklet` for playback
- Not using browser `<audio>` streaming for realtime PCM chunks

## Drop-in Integration Notes

If you need to plug another realtime backend into this frontend contract, you only need:

1. Send `type: "audio"` chunks as base64 s16le PCM.
2. (Recommended) send one `audio_config` with `sample_rate` early.
3. Keep chunks reasonably small (low latency) and in-order.
4. Emit interruption/turn-complete control events compatible with existing UI flow.
