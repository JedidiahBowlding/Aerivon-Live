// Elements are optional depending on which page is loaded.
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const uploadImgBtn = document.getElementById('uploadImgBtn');
const imageFileEl = document.getElementById('imageFile');
const statusEl = document.getElementById('status');
const stateBadgeEl = document.getElementById('stateBadge');
const transcriptEl = document.getElementById('transcript');
const logEl = document.getElementById('log');
const wsUrlEl = document.getElementById('wsUrl');

// UI Navigator elements
const navUrlEl = document.getElementById('navUrl');
const navTaskEl = document.getElementById('navTask');
const navOpenBtn = document.getElementById('navOpenBtn');
const navRunBtn = document.getElementById('navRunBtn');
const navSpeakBtn = document.getElementById('navSpeakBtn');
const navInterruptBtn = document.getElementById('navInterruptBtn');
const navImgEl = document.getElementById('navImg');
const navWsUrlEl = document.getElementById('navWsUrl');
const uploadPreviewEl = document.getElementById('uploadPreview');

const HAS_LIVE_AGENT_UI = !!(startBtn && stopBtn && statusEl && stateBadgeEl && transcriptEl && logEl && wsUrlEl);
const HAS_UI_NAVIGATOR_UI = !!(navUrlEl && navTaskEl && navOpenBtn && navRunBtn && navSpeakBtn && navInterruptBtn && navImgEl && navWsUrlEl);
const INTERRUPT_DEBUG = (() => {
  try {
    const qp = new URLSearchParams(location.search).get('debug');
    if (qp === '1' || qp === 'true') return true;
    const ls = localStorage.getItem('aerivon_interrupt_debug');
    return ls === '1' || ls === 'true';
  } catch {
    return false;
  }
})();

let navWs = null;
let navSpeechActive = false;

// Gemini Live STT (voice -> transcript -> forward to /ws/ui)
let navSttWs = null;
let navSttStream = null;
let navSttCtx = null;
let navSttProcessor = null;
let navSttSource = null;
let navSttInUtterance = false;
let navSttLastVoiceMs = 0;

// Gemini Live audio output for UI Navigator narration
let navTtsWs = null;
let navTtsSessionId = null;
let navTtsReady = false;
let navSpeakQueue = [];
let navSpeakBusy = false;

const TARGET_INPUT_SAMPLE_RATE = 16000;
let playbackSampleRate = 24000;

const AgentState = {
  IDLE: 'IDLE',
  LISTENING: 'LISTENING',
  SPEAKING: 'SPEAKING',
  INTERRUPTED: 'INTERRUPTED',
};

// Very small client-side VAD so Live reliably produces multiple turns.
// Without explicit audio_end, some Live configs will only answer once (or wait indefinitely).
// Lowered thresholds for wider mic compatibility (low-volume / aggressive noise suppression setups).
const VAD_SPEECH_RMS = 0.0025;     // start-of-utterance gate (was 0.010)
const VAD_SILENCE_MS = 700;        // end-of-utterance after this much silence

// When the model is speaking, the mic may pick up speaker leakage even with echoCancellation.
// Use a higher threshold and require consecutive frames to avoid self-interrupting mid-sentence.
const BARGE_IN_RMS = 0.045;
const BARGE_IN_FRAMES = 1;

const VAD_IDLE_THRESHOLD = 0.010;
const VAD_SPEAKING_THRESHOLD = 0.018;
const VAD_TRIGGER_MS = 90;
const VAD_COOLDOWN_MS = 350;
const VAD_MIN_TRIGGER_RMS = 0.0010; // allow barge-in detection at lower levels
const VAD_MAX_TRIGGER_RMS = 0.020;
const VAD_NOISE_MULTIPLIER = 2.0;
const AGGRESSIVE_INTERRUPT_RMS = 0.010;
const AGGRESSIVE_INTERRUPT_MS = 55;

// STT needs a longer silence window so natural pauses don't end the utterance mid-sentence.
const STT_SPEECH_RMS = 0.010;
const STT_SILENCE_MS = 2200;
const STT_MIN_UTTERANCE_MS = 900;

let ws = null;
let micStream = null;
let audioCtx = null;
let processor = null;
let sourceNode = null;
let wsReconnectTimer = null;
let wsReconnectAttempts = 0;
let isStoppingMic = false;

let playbackCtx = null;
let playCursor = 0;
let activeSources = [];
let playbackGainNode = null;

let inUtterance = false;
let lastVoiceMs = 0;
let agentState = AgentState.IDLE;
let lastInterruptAt = 0;
let interruptLock = false;
let vadActiveMs = 0;
let vadLastTs = performance.now();
let serverSessionId = null;
let suppressOutputUntilConnected = false;
let pendingTurnComplete = false;
let lastServerOutputAt = 0;
let lastServerAudioAt = 0;
let pendingResumeAfterUpstreamReconnect = false;
let vadNoiseFloor = 0.0015;
let aggressiveVadMs = 0;
let debugPanelEl = null;
let debugRowsEl = null;
let lastDebugMicTs = 0;

const UPSTREAM_RESUME_PROMPT = 'Connection restored. Continue only if you were mid-response, otherwise wait for the user.';

function hasActiveMicSession() {
  return !!(micStream || processor || sourceNode || audioCtx);
}

function clearWsReconnectTimer() {
  if (!wsReconnectTimer) return;
  clearTimeout(wsReconnectTimer);
  wsReconnectTimer = null;
}

function scheduleWsReconnect(reason = '') {
  if (isStoppingMic) return;
  if (!hasActiveMicSession()) return;
  if (wsReconnectTimer) return;

  const delayMs = Math.min(4000, 300 * (2 ** Math.min(5, wsReconnectAttempts)));
  wsReconnectAttempts += 1;
  log(`WS reconnect scheduled in ${delayMs}ms${reason ? ` reason=${reason}` : ''}`);

  wsReconnectTimer = setTimeout(async () => {
    wsReconnectTimer = null;
    if (isStoppingMic || !hasActiveMicSession()) return;
    try {
      await ensureWs();
    } catch (e) {
      log(`WS reconnect failed: ${e?.message || e}`);
      scheduleWsReconnect('retry_failed');
    }
  }, delayMs);
}

function ensureInterruptDebugPanel() {
  if (!INTERRUPT_DEBUG || debugPanelEl) return;
  const panel = document.createElement('div');
  panel.id = 'interruptDebugPanel';
  panel.style.position = 'fixed';
  panel.style.right = '12px';
  panel.style.bottom = '12px';
  panel.style.width = '340px';
  panel.style.maxHeight = '46vh';
  panel.style.overflow = 'auto';
  panel.style.zIndex = '99999';
  panel.style.background = 'rgba(17,17,17,0.92)';
  panel.style.color = '#e6e6e6';
  panel.style.border = '1px solid rgba(255,255,255,0.2)';
  panel.style.borderRadius = '8px';
  panel.style.padding = '8px';
  panel.style.font = '12px/1.35 ui-monospace, SFMono-Regular, Menlo, monospace';

  const title = document.createElement('div');
  title.textContent = 'Interrupt Debug';
  title.style.fontWeight = '700';
  title.style.marginBottom = '6px';
  panel.appendChild(title);

  const rows = document.createElement('pre');
  rows.style.margin = '0';
  rows.style.whiteSpace = 'pre-wrap';
  rows.style.wordBreak = 'break-word';
  panel.appendChild(rows);

  debugPanelEl = panel;
  debugRowsEl = rows;
  document.body.appendChild(panel);
}

function interruptDebug(kv = {}, event = '') {
  if (!INTERRUPT_DEBUG) return;
  ensureInterruptDebugPanel();
  if (!debugRowsEl) return;

  const merged = {
    state: agentState,
    ws: ws ? ws.readyState : -1,
    queued: isAudioPlayingOrQueued(),
    lock: interruptLock,
    vadMs: Math.round(vadActiveMs),
    ...kv,
  };
  const entries = Object.entries(merged).map(([k, v]) => `${k}=${v}`).join(' | ');
  const line = `[${nowMs()}] ${event || 'tick'} ${entries}`;
  debugRowsEl.textContent = `${line}\n${debugRowsEl.textContent}`.slice(0, 6000);
}

function rms(float32) {
  let sum = 0;
  for (let i = 0; i < float32.length; i++) {
    const v = float32[i];
    sum += v * v;
  }
  return Math.sqrt(sum / Math.max(1, float32.length));
}

function setState(state) {
  if (state === 'idle') {
    updateUIState(AgentState.IDLE);
    return;
  }
  if (state === 'speaking') {
    updateUIState(AgentState.SPEAKING);
    return;
  }
  if (state === 'interrupted') {
    updateUIState(AgentState.INTERRUPTED);
    return;
  }
  // idle/listening/thinking collapse to LISTENING for judge-facing state machine.
  updateUIState(AgentState.LISTENING);
}

function updateUIState(state) {
  agentState = state;

  const stateEl = document.getElementById('agentState');
  if (stateEl) {
    stateEl.textContent = state;
    stateEl.classList.remove('state-idle', 'state-speaking', 'state-interrupted', 'pulse-once');
    if (state === AgentState.LISTENING) {
      stateEl.classList.add('state-idle');
    } else if (state === AgentState.SPEAKING) {
      stateEl.classList.add('state-speaking');
    } else if (state === AgentState.INTERRUPTED) {
      stateEl.classList.add('state-interrupted');
      stateEl.offsetWidth;
      stateEl.classList.add('pulse-once');
    }
  }

  if (stateBadgeEl) {
    stateBadgeEl.className = 'muted';
    stateBadgeEl.style.color = '';
    if (state === AgentState.IDLE) stateBadgeEl.textContent = '⚪ Idle';
    else if (state === AgentState.LISTENING) stateBadgeEl.textContent = '🟢 Listening';
    else if (state === AgentState.SPEAKING) stateBadgeEl.textContent = '🔵 Speaking';
    else if (state === AgentState.INTERRUPTED) stateBadgeEl.textContent = '🟠 Interrupted';
    else stateBadgeEl.textContent = 'idle';
  }

  setDucking(state === AgentState.SPEAKING);
}

function maybeSendBargeInInterrupt() {
  triggerInterrupt(ws, 'barge-in');
}

function triggerInterrupt(targetWs, source = 'barge-in') {
  if (!targetWs || targetWs.readyState !== WebSocket.OPEN) {
    interruptDebug({ source }, 'interrupt_skip:ws_not_open');
    return false;
  }
  if (interruptLock) {
    interruptDebug({ source }, 'interrupt_skip:lock');
    return false;
  }

  const now = performance.now();
  if (now - lastInterruptAt < VAD_COOLDOWN_MS) {
    interruptDebug({ source, sinceMs: Math.round(now - lastInterruptAt) }, 'interrupt_skip:cooldown');
    return false;
  }
  interruptLock = true;
  lastInterruptAt = now;

  updateUIState(AgentState.INTERRUPTED);
  suppressOutputUntilConnected = true;
  stopPlayback();

  try {
    targetWs.send(JSON.stringify({ type: 'interrupt' }));
  } catch {
    // ignore
  }

  log(`Sent interrupt (${source})`);
  interruptDebug({ source }, 'interrupt_sent');
  updateUIState(AgentState.LISTENING);
  setTimeout(() => {
    interruptLock = false;
  }, 400);
  setTimeout(() => {
    console.assert(!isAudioPlayingOrQueued(), 'Audio still queued after interrupt!');
  }, 0);
  return true;
}

function handleMicFrameForBargeIn(frame, targetWs, source = 'barge-in') {
  const now = performance.now();
  const dt = now - vadLastTs;
  vadLastTs = now;

  const level = rms(frame);
  const speakingWindowActive = (now - lastServerAudioAt) < 1200;
  const speakingGateActive = agentState === AgentState.SPEAKING || isAudioPlayingOrQueued() || speakingWindowActive;

  // Learn ambient noise only when the agent is not speaking and no recent output is active.
  if (!speakingGateActive) {
    vadNoiseFloor = Math.min(0.02, Math.max(0.0005, vadNoiseFloor * 0.985 + level * 0.015));
  } else {
    // During speaking windows, only decay slowly to avoid echo-driven threshold inflation.
    vadNoiseFloor = Math.max(0.0005, vadNoiseFloor * 0.995);
  }

  const dynamicThreshold = Math.min(VAD_MAX_TRIGGER_RMS, Math.max(VAD_MIN_TRIGGER_RMS, vadNoiseFloor * VAD_NOISE_MULTIPLIER));
  const dynamicIdleThreshold = Math.max(0.002, dynamicThreshold * 0.65);

  if (INTERRUPT_DEBUG && now - lastDebugMicTs > 120) {
    lastDebugMicTs = now;
    interruptDebug(
      {
        source,
        level: level.toFixed(4),
        thr: BARGE_IN_RMS.toFixed(4),
        idleThr: dynamicIdleThreshold.toFixed(4),
        noiseFloor: vadNoiseFloor.toFixed(4),
        speakingWindowActive,
        dt: Math.round(dt),
      },
      'mic'
    );
  }

  if (!speakingGateActive) {
    aggressiveVadMs = 0;
    vadActiveMs = 0;
    if (INTERRUPT_DEBUG && now - lastDebugMicTs > 120) {
      interruptDebug({ source }, 'interrupt_skip:not_speaking_or_not_queued');
    }
    return false;
  }

  aggressiveVadMs = 0;

  if (now - lastInterruptAt < VAD_COOLDOWN_MS) {
    if (INTERRUPT_DEBUG && now - lastDebugMicTs > 120) {
      interruptDebug({ source, sinceMs: Math.round(now - lastInterruptAt) }, 'interrupt_skip:cooldown');
    }
    return false;
  }

  if (level > BARGE_IN_RMS) vadActiveMs += dt;
  else if (level < VAD_IDLE_THRESHOLD) vadActiveMs = Math.max(0, vadActiveMs - dt * 2);

  if (vadActiveMs >= VAD_TRIGGER_MS) {
    vadActiveMs = 0;
    return triggerInterrupt(targetWs, source);
  }

  if (INTERRUPT_DEBUG && now - lastDebugMicTs > 120) {
    interruptDebug({ source, level: level.toFixed(4) }, 'interrupt_hold:insufficient_vad');
  }

  return false;
}

function nowMs() {
  return new Date().toISOString().slice(11, 23);
}

function log(line) {
  if (!logEl) return;
  logEl.textContent += `[${nowMs()}] ${line}\n`;
  logEl.scrollTop = logEl.scrollHeight;
}

function setStatus(text, muted = false) {
  if (!statusEl) return;
  statusEl.textContent = text;
  statusEl.className = muted ? 'muted' : '';
}

function appendTranscript(text) {
  if (!transcriptEl) return;
  transcriptEl.textContent += text;
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

function u8ToB64(u8) {
  let s = '';
  const chunkSize = 0x8000;
  for (let i = 0; i < u8.length; i += chunkSize) {
    s += String.fromCharCode(...u8.subarray(i, i + chunkSize));
  }
  return btoa(s);
}

async function fileToDataUrl(file) {
  const buf = await file.arrayBuffer();
  const u8 = new Uint8Array(buf);
  return `data:${file.type || 'application/octet-stream'};base64,${u8ToB64(u8)}`;
}

async function loadImageFromFile(file) {
  // Prefer createImageBitmap when available; fall back to <img> decoding.
  if ('createImageBitmap' in window) {
    try {
      return await createImageBitmap(file);
    } catch {
      // fall through
    }
  }
  const url = URL.createObjectURL(file);
  try {
    const img = new Image();
    img.decoding = 'async';
    img.src = url;
    await img.decode();
    return img;
  } finally {
    URL.revokeObjectURL(url);
  }
}

async function canvasToBlob(canvas, mimeType, quality) {
  return await new Promise((resolve) => {
    canvas.toBlob((b) => resolve(b), mimeType, quality);
  });
}

async function prepareImageForWs(file) {
  // Backend enforces MAX_WS_MESSAGE_BYTES=256KB; keep some headroom.
  const TARGET_BYTES = 240 * 1024;

  // If it's already small enough, send as-is.
  if (typeof file.size === 'number' && file.size > 0 && file.size <= TARGET_BYTES) {
    const buf = await file.arrayBuffer();
    return { mime_type: file.type || 'image/png', u8: new Uint8Array(buf), note: 'original' };
  }

  // Compress/resize via canvas.
  const img = await loadImageFromFile(file);
  const srcW = img.width || 0;
  const srcH = img.height || 0;
  if (!srcW || !srcH) {
    const buf = await file.arrayBuffer();
    return { mime_type: file.type || 'image/png', u8: new Uint8Array(buf), note: 'fallback_no_dims' };
  }

  // Downscale if needed.
  let maxDim = 1280;
  let scale = Math.min(1, maxDim / Math.max(srcW, srcH));

  // Iteratively adjust size/quality until under target.
  let quality = 0.82;
  for (let attempt = 0; attempt < 8; attempt++) {
    const w = Math.max(1, Math.round(srcW * scale));
    const h = Math.max(1, Math.round(srcH * scale));

    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    if (!ctx) break;
    ctx.drawImage(img, 0, 0, w, h);

    const mime = 'image/jpeg';
    const blob = await canvasToBlob(canvas, mime, quality);
    if (blob && blob.size <= TARGET_BYTES) {
      const buf = await blob.arrayBuffer();
      return { mime_type: mime, u8: new Uint8Array(buf), note: `compressed ${w}x${h} q=${quality}` };
    }

    // Too big: first reduce quality, then reduce dimensions.
    if (quality > 0.55) quality -= 0.12;
    else scale *= 0.82;
  }

  // Last resort: send original.
  const buf = await file.arrayBuffer();
  return { mime_type: file.type || 'image/png', u8: new Uint8Array(buf), note: 'original_last_resort' };
}

async function sendVisionImage(file, promptText = 'What am I looking at?') {
  if (!file) return;
  if (!file.type || !file.type.toLowerCase().startsWith('image/')) {
    log('Upload rejected: not an image');
    return;
  }

  // Show local preview immediately.
  try {
    if (uploadPreviewEl) {
      const url = URL.createObjectURL(file);
      uploadPreviewEl.onload = () => {
        try { URL.revokeObjectURL(url); } catch {}
      };
      uploadPreviewEl.src = url;
    }
  } catch {
    // ignore preview failures
  }

  await ensureWs();
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    log('WS not connected; cannot send image');
    return;
  }

  const prepared = await prepareImageForWs(file);
  const u8 = prepared.u8;

  // Backend rejects >256KB; avoid sending a payload that will fail.
  if (u8.byteLength > 250 * 1024) {
    const msg = `Image is too large to send (${u8.byteLength} bytes after compression). Try a smaller image.`;
    log(msg);
    appendTranscript(`\n[upload] ${msg}\n`);
    return;
  }
  const payload = {
    type: 'image',
    mime_type: prepared.mime_type,
    data_b64: u8ToB64(u8),
    text: promptText,
  };

  try {
    suppressOutputUntilConnected = false;
    ws.send(JSON.stringify(payload));
    log(`Sent image (${prepared.mime_type}, ${u8.byteLength} bytes, ${prepared.note})`);
    setState('thinking');
  } catch (e) {
    log(`Failed to send image: ${e?.message || e}`);
  }
}

function b64ToU8(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

function floatTo16BitPCM(float32) {
  const out = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i++) {
    let s = Math.max(-1, Math.min(1, float32[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

function downsampleBuffer(input, inputRate, targetRate) {
  if (targetRate === inputRate) return input;
  if (targetRate > inputRate) throw new Error('targetRate must be <= inputRate');

  const ratio = inputRate / targetRate;
  const newLen = Math.round(input.length / ratio);
  const out = new Float32Array(newLen);

  let offsetResult = 0;
  let offsetBuffer = 0;

  while (offsetResult < out.length) {
    const nextOffsetBuffer = Math.round((offsetResult + 1) * ratio);
    let accum = 0;
    let count = 0;
    for (let i = offsetBuffer; i < nextOffsetBuffer && i < input.length; i++) {
      accum += input[i];
      count++;
    }
    out[offsetResult] = count ? accum / count : 0;
    offsetResult++;
    offsetBuffer = nextOffsetBuffer;
  }

  return out;
}

function int16ToFloat32(int16) {
  const out = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) out[i] = int16[i] / 0x8000;
  return out;
}

function stopPlayback() {
  if (!playbackCtx) {
    activeSources.length = 0;
    pendingTurnComplete = false;
    return;
  }

  activeSources.forEach((src) => {
    try { src.stop(0); } catch {}
  });
  activeSources.length = 0;
  playCursor = playbackCtx.currentTime;

  if (playbackCtx.state === 'running') {
    playbackCtx.suspend().then(() => playbackCtx.resume()).catch(() => {});
  }
  pendingTurnComplete = false;
}

function isAudioPlayingOrQueued() {
  if (!playbackCtx) return false;
  if (activeSources.length > 0) return true;
  return playCursor > playbackCtx.currentTime + 0.05;
}

function maybeSetStateAfterPlayback() {
  if (!pendingTurnComplete) return;
  if (isAudioPlayingOrQueued()) return;
  pendingTurnComplete = false;
  setState(micStream ? 'listening' : 'idle');
}

function ensurePlaybackCtx() {
  if (playbackCtx) return;
  const Ctx = window.AudioContext || window.webkitAudioContext;
  // Best-effort: request the server-provided sample rate.
  try {
    playbackCtx = new Ctx({ sampleRate: playbackSampleRate });
  } catch {
    playbackCtx = new Ctx();
  }
  playbackGainNode = playbackCtx.createGain();
  playbackGainNode.gain.value = 1.0;
  playbackGainNode.connect(playbackCtx.destination);
  playCursor = playbackCtx.currentTime;
}

function setDucking(on) {
  if (!playbackGainNode) return;
  playbackGainNode.gain.value = on ? 0.65 : 1.0;
}

async function applyAudioConfig(sampleRate) {
  const sr = Number(sampleRate);
  if (!Number.isFinite(sr) || sr <= 0) return;
  playbackSampleRate = sr;
  // If an AudioContext was already created at a different rate, recreate it.
  if (playbackCtx && playbackCtx.sampleRate !== playbackSampleRate) {
    stopPlayback();
    try { await playbackCtx.close(); } catch {}
    playbackCtx = null;
  }
}

function enqueuePcmForPlayback(pcmBytes) {
  ensurePlaybackCtx();

  const int16 = new Int16Array(pcmBytes.buffer, pcmBytes.byteOffset, Math.floor(pcmBytes.byteLength / 2));
  const floats = int16ToFloat32(int16);

  const buffer = playbackCtx.createBuffer(1, floats.length, playbackSampleRate);
  buffer.copyToChannel(floats, 0);

  const src = playbackCtx.createBufferSource();
  src.buffer = buffer;
  src.connect(playbackGainNode || playbackCtx.destination);

  src.onended = () => {
    activeSources = activeSources.filter((s) => s !== src);
    maybeSetStateAfterPlayback();
  };

  const startAt = Math.max(playCursor, playbackCtx.currentTime + 0.02);
  src.start(startAt);
  activeSources.push(src);
  playCursor = startAt + buffer.duration;
}

function getBackendHttpBase() {
  // If the hosting server injected a backend base (Cloud Run hosted frontend), prefer it.
  try {
    const injected = (window && window.AERIVON_BACKEND_BASE) ? String(window.AERIVON_BACKEND_BASE).trim() : '';
    if (injected) return injected.replace(/\/$/, '');
  } catch {
    // ignore
  }

  // Allow overriding backend base via query param for Cloud Run deployments.
  // Example: http://localhost:5173/?backend=https://SERVICE-xxxxx-uc.a.run.app
  try {
    const u = new URL(location.href);
    const qp = (u.searchParams.get('backend') || '').trim();
    if (qp) return qp.replace(/\/$/, '');
  } catch {
    // ignore
  }

  // Auto-detect: if the frontend itself is hosted on Cloud Run, use same origin.
  if (/\.run\.app$/i.test(location.hostname)) {
    return `${location.protocol}//${location.host}`;
  }

  // Local/dev default.
  return `${location.protocol}//${location.hostname}:8081`;
}

function httpBaseToWsBase(httpBase) {
  const base = String(httpBase || '').trim();
  if (!base) return '';
  return base.replace(/^https:/i, 'wss:').replace(/^http:/i, 'ws:');
}

function createLiveMemoryScope() {
  const suffix = (crypto && crypto.randomUUID)
    ? crypto.randomUUID()
    : `${Date.now()}_${Math.random().toString(36).slice(2, 10)}`;
  return `live_agent_${suffix}`;
}

let currentLiveMemoryScope = createLiveMemoryScope();

function buildWsUrl() {
  const wsBase = httpBaseToWsBase(getBackendHttpBase());
  const url = new URL(`${wsBase}/ws/live`);
  url.searchParams.set('output', 'audio');
  url.searchParams.set('user_id', getOrCreateUserId());
  url.searchParams.set('memory_scope', currentLiveMemoryScope);
  return url.toString();
}

function buildNavWsUrl() {
  const wsBase = httpBaseToWsBase(getBackendHttpBase());
  return `${wsBase}/ws/ui`;
}

function buildLiveSttWsUrl() {
  const wsBase = httpBaseToWsBase(getBackendHttpBase());
  const url = new URL(`${wsBase}/ws/live`);
  url.searchParams.set('output', 'text');
  url.searchParams.set('mode', 'stt');
  url.searchParams.set('user_id', getOrCreateUserId());
  url.searchParams.set('memory_scope', 'live_stt');
  return url.toString();
}

function buildLiveTtsWsUrl() {
  const wsBase = httpBaseToWsBase(getBackendHttpBase());
  const url = new URL(`${wsBase}/ws/live`);
  url.searchParams.set('output', 'audio');
  url.searchParams.set('mode', 'agent');
  url.searchParams.set('user_id', getOrCreateUserId());
  url.searchParams.set('memory_scope', 'ui_navigator_tts');
  return url.toString();
}

function getOrCreateUserId() {
  const key = 'aerivon_user_id';
  try {
    const existing = localStorage.getItem(key);
    if (existing && typeof existing === 'string') return existing;
    const id = (crypto && crypto.randomUUID) ? crypto.randomUUID() : String(Math.random()).slice(2);
    localStorage.setItem(key, id);
    return id;
  } catch {
    return 'default';
  }
}

async function ensureNavTtsWs() {
  if (navTtsWs && (navTtsWs.readyState === WebSocket.OPEN || navTtsWs.readyState === WebSocket.CONNECTING)) return;
  navTtsReady = false;
  navTtsSessionId = null;

  const url = buildLiveTtsWsUrl();
  navTtsWs = new WebSocket(url);

  navTtsWs.onopen = () => {
    log('TTS WS connected');
  };
  navTtsWs.onclose = (evt) => {
    navTtsReady = false;
    log(`TTS WS closed code=${evt.code} reason=${evt.reason || '(none)'}`);
  };
  navTtsWs.onerror = () => {
    navTtsReady = false;
    log('TTS WS error');
  };
  navTtsWs.onmessage = (evt) => {
    try {
      const msg = JSON.parse(evt.data);
      if (!msg || typeof msg.type !== 'string') return;

      if (msg.type === 'status' && msg.status === 'connected') {
        if (typeof msg.session_id === 'number') navTtsSessionId = msg.session_id;
        navTtsReady = true;
        // apply audio config if present
        if (msg.audio_config && msg.audio_config.sample_rate) {
          applyAudioConfig(msg.audio_config.sample_rate);
        }
        return;
      }

      if (navTtsSessionId != null && typeof msg.session_id === 'number' && msg.session_id !== navTtsSessionId) {
        return;
      }

      if (msg.type === 'audio') {
        const u8 = b64ToU8(msg.data_b64 || '');
        updateUIState(AgentState.SPEAKING);
        enqueuePcmForPlayback(u8);
        return;
      }

      if (msg.type === 'interrupted') {
        updateUIState(AgentState.LISTENING);
        stopPlayback();
        return;
      }

      if (msg.type === 'turn_complete') {
        updateUIState(AgentState.LISTENING);
        return;
      }
    } catch {
      // ignore
    }
  };
}

function browserSpeak(text) {
  try {
    if (!('speechSynthesis' in window)) return false;
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 1.0;
    u.pitch = 1.0;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(u);
    return true;
  } catch {
    return false;
  }
}

async function navSpeakWithGemini(text) {
  const t = (text || '').trim();
  if (!t) return;
  navSpeakQueue.push(t);
  if (navSpeakBusy) return;
  navSpeakBusy = true;

  while (navSpeakQueue.length) {
    const next = navSpeakQueue.shift();
    try {
      await ensureNavTtsWs();
      await waitForOpen(navTtsWs, 2500);

      // If something is already playing, interrupt; otherwise don't (prevents self-cutoff).
      if (isAudioPlayingOrQueued()) {
        stopPlayback();
        try { navTtsWs.send(JSON.stringify({ type: 'interrupt' })); } catch {}
      }

      // Send the utterance.
      navTtsWs.send(JSON.stringify({ type: 'text', text: next }));
      log(`NAV speak: ${next}`);

      // If no audio arrives quickly, fall back to browser TTS.
      const start = performance.now();
      while (performance.now() - start < 1800) {
        if (isAudioPlayingOrQueued()) break;
        await new Promise((r) => setTimeout(r, 50));
      }
      if (!isAudioPlayingOrQueued()) {
        log('NAV speak: no Gemini audio received; using browser speechSynthesis fallback');
        browserSpeak(next);
      }

      // Wait for playback to drain before speaking the next queue item.
      while (isAudioPlayingOrQueued()) {
        await new Promise((r) => setTimeout(r, 75));
      }
    } catch (e) {
      log(`NAV speak failed: ${e?.message || e}`);
      browserSpeak(next);
    }
  }

  navSpeakBusy = false;
}

async function ensureNavSttWs() {
  if (navSttWs && (navSttWs.readyState === WebSocket.OPEN || navSttWs.readyState === WebSocket.CONNECTING)) return;
  const url = buildLiveSttWsUrl();
  navSttWs = new WebSocket(url);
  navSttWs.onopen = () => log('STT WS connected');
  navSttWs.onclose = (evt) => log(`STT WS closed code=${evt.code} reason=${evt.reason || '(none)'}`);
  navSttWs.onerror = () => log('STT WS error');
}

function waitForOpen(wsObj, timeoutMs = 8000) {
  return new Promise((resolve, reject) => {
    if (!wsObj) return reject(new Error('ws missing'));
    if (wsObj.readyState === WebSocket.OPEN) return resolve();
    const t = setTimeout(() => reject(new Error('ws open timeout')), timeoutMs);
    wsObj.addEventListener('open', () => {
      clearTimeout(t);
      resolve();
    }, { once: true });
    wsObj.addEventListener('error', () => {
      clearTimeout(t);
      reject(new Error('ws error'));
    }, { once: true });
  });
}

async function liveSttOnce() {
  // Capture one utterance, return transcript text.
  await ensureNavSttWs();
  await waitForOpen(navSttWs);

  // Consume initial status (and ignore).
  const transcriptParts = [];
  let gotTurnComplete = false;
  let sentAudioEnd = false;
  let utteranceStartMs = 0;

  const onMsg = (evt) => {
    try {
      const msg = JSON.parse(evt.data);
      if (msg.type === 'text' && typeof msg.text === 'string') transcriptParts.push(msg.text);
      if (msg.type === 'turn_complete') gotTurnComplete = true;
    } catch {
      // ignore
    }
  };
  navSttWs.addEventListener('message', onMsg);

  try {
    navSttStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    navSttCtx = new (window.AudioContext || window.webkitAudioContext)();
    navSttSource = navSttCtx.createMediaStreamSource(navSttStream);
    navSttProcessor = navSttCtx.createScriptProcessor(4096, 1, 1);

    navSttInUtterance = false;
    navSttLastVoiceMs = 0;

    navSttProcessor.onaudioprocess = (e) => {
      if (!navSttWs || navSttWs.readyState !== WebSocket.OPEN) return;

      if (sentAudioEnd) {
        // Don't accidentally start a second turn while waiting for turn_complete.
        return;
      }

      const input = e.inputBuffer.getChannelData(0);
      const down = downsampleBuffer(input, navSttCtx.sampleRate, TARGET_INPUT_SAMPLE_RATE);
      const level = rms(down);
      const t = performance.now();
      const speakingNow = level >= STT_SPEECH_RMS;

      // Auto barge-in for Navigator pages.
      const didInterrupt = handleMicFrameForBargeIn(down, navTtsWs, 'nav-barge-in');
      if (didInterrupt && navWs && navWs.readyState === WebSocket.OPEN) {
        try { navWs.send(JSON.stringify({ type: 'interrupt' })); } catch {}
      }

      if (speakingNow) {
        navSttLastVoiceMs = t;
        navSttInUtterance = true;
        if (!utteranceStartMs) utteranceStartMs = t;
      } else if (navSttInUtterance && t - navSttLastVoiceMs >= STT_SILENCE_MS) {
        if (utteranceStartMs && (t - utteranceStartMs) < STT_MIN_UTTERANCE_MS) {
          // Too short; ignore this silence edge.
          return;
        }
        try {
          navSttWs.send(JSON.stringify({ type: 'audio_end' }));
          log('STT: sent audio_end (silence)');
        } catch {}
        sentAudioEnd = true;
        navSttInUtterance = false;
        return;
      }

      if (!navSttInUtterance) return;

      const int16 = floatTo16BitPCM(down);
      const u8 = new Uint8Array(int16.buffer);
      try {
        navSttWs.send(JSON.stringify({ type: 'audio', mime_type: 'audio/pcm', data_b64: u8ToB64(u8) }));
      } catch {
        // ignore
      }
    };

    navSttSource.connect(navSttProcessor);
    navSttProcessor.connect(navSttCtx.destination);

    // Wait for turn_complete (or timeout)
    const start = performance.now();
    while (!gotTurnComplete && performance.now() - start < 20000) {
      await new Promise((r) => setTimeout(r, 50));
    }

    // Small grace period to collect trailing transcript text.
    await new Promise((r) => setTimeout(r, 250));

    return transcriptParts.join('').trim();
  } finally {
    navSttWs.removeEventListener('message', onMsg);
    try {
      if (navSttProcessor) {
        navSttProcessor.disconnect();
        navSttProcessor.onaudioprocess = null;
      }
      if (navSttSource) navSttSource.disconnect();
      if (navSttStream) for (const t of navSttStream.getTracks()) t.stop();
      if (navSttCtx) await navSttCtx.close();
    } catch {}
    navSttProcessor = null;
    navSttSource = null;
    navSttStream = null;
    navSttCtx = null;
  }
}

async function ensureNavWs() {
  if (navWs && (navWs.readyState === WebSocket.OPEN || navWs.readyState === WebSocket.CONNECTING)) return;

  const url = buildNavWsUrl();
  navWsUrlEl.textContent = url;
  navWs = new WebSocket(url);

  navWs.onopen = () => {
    log('NAV WS connected');
  };
  navWs.onclose = (evt) => {
    log(`NAV WS closed code=${evt.code} reason=${evt.reason || '(none)'}`);
  };
  navWs.onerror = () => {
    log('NAV WS error');
  };
  navWs.onmessage = (evt) => {
    try {
      const msg = JSON.parse(evt.data);
      if (!msg || typeof msg.type !== 'string') return;
      if (msg.type === 'screenshot' && msg.data_b64) {
        navImgEl.src = `data:${msg.mime_type || 'image/png'};base64,${msg.data_b64}`;
        log(`NAV screenshot url=${msg.url || ''}`);
        return;
      }
      if (msg.type === 'actions') {
        log(`NAV actions step=${msg.step} plan=${JSON.stringify(msg.plan)}`);
        return;
      }
      if (msg.type === 'action_result') {
        log(`NAV action_result ${JSON.stringify(msg.result)}`);
        return;
      }
      if (msg.type === 'status') {
        log(`NAV status: ${msg.status}${msg.note ? ` note=${msg.note}` : ''}`);
        if (msg.status === 'done' && msg.note) {
          navSpeakWithGemini(msg.note);
        }
        return;
      }
      if (msg.type === 'error') {
        log(`NAV error: ${msg.error}`);
        navSpeakWithGemini(String(msg.error || 'Error'));
      }
    } catch {
      // ignore
    }
  };
}

async function ensureWs() {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;

  const url = buildWsUrl();
  wsUrlEl.textContent = url;
  setStatus('connecting…', true);

  ws = new WebSocket(url);

  ws.onopen = () => {
    clearWsReconnectTimer();
    wsReconnectAttempts = 0;
    setStatus('connected');
    log('WS connected');
  };

  ws.onclose = (evt) => {
    const shouldReconnect = !isStoppingMic && hasActiveMicSession() && evt.code !== 1000;
    ws = null;
    setStatus('disconnected', true);
    updateUIState(AgentState.IDLE);
    log(`WS closed code=${evt.code} reason=${evt.reason || '(none)'}`);
    if (shouldReconnect) {
      scheduleWsReconnect(`close_${evt.code}`);
    }
  };

  ws.onerror = () => {
    log('WS error');
  };

  ws.onmessage = (evt) => {
    try {
      const msg = JSON.parse(evt.data);
      if (!msg || typeof msg.type !== 'string') return;

      // session_id lets us ignore late audio/transcripts from a previous Live session
      // after an interrupt-triggered restart.
      if (msg.type === 'status' && msg.status === 'connected') {
        if (typeof msg.session_id === 'number') serverSessionId = msg.session_id;
        suppressOutputUntilConnected = false;
        lastServerOutputAt = 0;
        lastServerAudioAt = 0;
        vadActiveMs = 0;
        aggressiveVadMs = 0;
        // Resume prompt removed: causes model to repeat completed responses.
        // Instead, let silence happen naturally; user will speak when ready.
        pendingResumeAfterUpstreamReconnect = false;
      } else if (serverSessionId != null && typeof msg.session_id === 'number' && msg.session_id !== serverSessionId) {
        return;
      } else if (serverSessionId != null && msg.type !== 'status' && msg.type !== 'error' && typeof msg.session_id !== 'number') {
        // Safety: once a session is established, ignore packets without session_id.
        return;
      }

      if (msg.type === 'status') {
        const extra = [];
        if (msg.reason) extra.push(`reason=${msg.reason}`);
        if (msg.detail) extra.push(`detail=${msg.detail}`);
        log(`status: ${msg.status} model=${msg.model || ''} vision=${msg.vision} output=${msg.output}${extra.length ? ' ' + extra.join(' ') : ''}`);
        interruptDebug({ status: msg.status, reason: msg.reason || '', detail: msg.detail || '' }, 'ws_status');
        if (msg.status === 'restarting') {
          suppressOutputUntilConnected = true;
          lastServerOutputAt = 0;
          lastServerAudioAt = 0;
          vadActiveMs = 0;
          aggressiveVadMs = 0;
          const restartingReason = String(msg.reason || '').trim();
          const hadRecentModelAudio = isAudioPlayingOrQueued()
            || agentState === AgentState.SPEAKING;
          pendingResumeAfterUpstreamReconnect = restartingReason === 'upstream_disconnected' && hadRecentModelAudio;
          if (restartingReason !== 'upstream_disconnected') {
            stopPlayback();
          }
          setState('thinking');
          return;
        }
        if (msg.audio_config && msg.audio_config.sample_rate) {
          applyAudioConfig(msg.audio_config.sample_rate);
          log(`audio_config: ${Number(msg.audio_config.sample_rate)}Hz ${msg.audio_config.format || ''}`);
        }
        setState('idle');
        return;
      }

      if (msg.type === 'audio_config') {
        applyAudioConfig(msg.sample_rate);
        log(`audio_config: ${Number(msg.sample_rate)}Hz ${msg.format || ''}`);
        return;
      }

      if (msg.type === 'text') {
        if (suppressOutputUntilConnected) return;
        updateUIState(AgentState.SPEAKING);
        lastServerOutputAt = performance.now();
        console.log(`[TEXT DEBUG] Received text message: "${msg.text}"`);
        appendTranscript(msg.text || '');
        return;
      }

      if (msg.type === 'transcript') {
        if (suppressOutputUntilConnected) return;
        updateUIState(AgentState.SPEAKING);
        lastServerOutputAt = performance.now();
        if (typeof msg.text === 'string') appendTranscript(msg.text);
        if (msg.finished) appendTranscript('\n');
        return;
      }

      if (msg.type === 'audio') {
        if (suppressOutputUntilConnected) return;
        updateUIState(AgentState.SPEAKING);
        lastServerOutputAt = performance.now();
        lastServerAudioAt = lastServerOutputAt;
        const u8 = b64ToU8(msg.data_b64 || '');
        console.log(`[AUDIO DEBUG] Received ${u8.length} bytes of PCM audio data`);
        enqueuePcmForPlayback(u8);
        return;
      }

      if (msg.type === 'interrupted') {
        log(`server reports interrupted${msg.source ? ` source=${msg.source}` : ''}`);
        suppressOutputUntilConnected = true;
        pendingResumeAfterUpstreamReconnect = false;
        lastServerOutputAt = 0;
        lastServerAudioAt = 0;
        vadActiveMs = 0;
        aggressiveVadMs = 0;
        updateUIState(AgentState.LISTENING);
        stopPlayback();
        return;
      }

      if (msg.type === 'turn_complete') {
        appendTranscript('\n');
        updateUIState(AgentState.LISTENING);
        // Don't flip to listening until queued audio finished playing.
        pendingTurnComplete = true;
        maybeSetStateAfterPlayback();
        return;
      }

      if (msg.type === 'error') {
        log(`server error: ${msg.error}`);
        appendTranscript(`\n[server error] ${msg.error}\n`);
        return;
      }
    } catch {
      // ignore
    }
  };
}

async function startMic() {
  isStoppingMic = false;
  currentLiveMemoryScope = createLiveMemoryScope();
  await ensureWs();

  micStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
      channelCount: 1,
    },
  });
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();

  sourceNode = audioCtx.createMediaStreamSource(micStream);
  processor = audioCtx.createScriptProcessor(4096, 1, 1);

  processor.onaudioprocess = (e) => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;

    const input = e.inputBuffer.getChannelData(0);
    const down = downsampleBuffer(input, audioCtx.sampleRate, TARGET_INPUT_SAMPLE_RATE);

    // Always evaluate barge-in on every frame while model audio may be active.
    handleMicFrameForBargeIn(down, ws, 'barge-in');

    // VAD: only stream frames when the system detects the user is speaking. When it detects
    // enough silence after speech, send audio_end to close the utterance.
    const level = rms(down);
    const t = performance.now();
    // Use a dynamic threshold that adapts to the learned noise floor so quieter mics still trigger.
    const dynamicSpeechThr = Math.max(VAD_SPEECH_RMS, vadNoiseFloor * VAD_NOISE_MULTIPLIER);
    const speakingNow = level >= dynamicSpeechThr;

    if (speakingNow) {
      lastVoiceMs = t;

      inUtterance = true;
      setState('listening');
    } else if (inUtterance && t - lastVoiceMs >= VAD_SILENCE_MS) {
      // End the utterance.
      try {
        ws.send(JSON.stringify({ type: 'audio_end' }));
        setState('thinking');
      } catch {
        // ignore
      }
      inUtterance = false;
      vadActiveMs = 0;
      return;
    }

    if (!inUtterance) {
      // Don't stream silence; wait for speech.
      return;
    }

    const int16 = floatTo16BitPCM(down);
    const u8 = new Uint8Array(int16.buffer);

    const payload = {
      type: 'audio',
      mime_type: 'audio/pcm',
      data_b64: u8ToB64(u8),
    };

    ws.send(JSON.stringify(payload));
  };

  sourceNode.connect(processor);
  processor.connect(audioCtx.destination);

  startBtn.disabled = true;
  stopBtn.disabled = false;
  log('Mic started');
  updateUIState(AgentState.LISTENING);
}

async function stopMic() {
  isStoppingMic = true;
  clearWsReconnectTimer();
  try {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'audio_end' }));
      log('Sent audio_end');
      setState('thinking');
    }
  } catch {
    // ignore
  }

  if (processor) {
    processor.disconnect();
    processor.onaudioprocess = null;
  }
  if (sourceNode) sourceNode.disconnect();

  if (micStream) {
    for (const t of micStream.getTracks()) t.stop();
  }

  if (audioCtx) {
    await audioCtx.close();
  }

  processor = null;
  sourceNode = null;
  micStream = null;
  audioCtx = null;
  pendingResumeAfterUpstreamReconnect = false;

  startBtn.disabled = false;
  stopBtn.disabled = true;
  log('Mic stopped');
  setState('idle');

  // For demo reliability: fully close the WS on Stop so the server doesn't
  // keep restarting sessions in the background while the mic is off.
  try {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
      ws.close(1000, 'client_stop');
    }
  } catch {
    // ignore
  }
  isStoppingMic = false;
}

if (HAS_LIVE_AGENT_UI) {
  ensureInterruptDebugPanel();
  startBtn.addEventListener('click', async () => {
    try {
      await startMic();
    } catch (e) {
      log(`Start failed: ${e?.message || e}`);
      startBtn.disabled = false;
      stopBtn.disabled = true;
    }
  });

  stopBtn.addEventListener('click', async () => {
    await stopMic();
  });

  if (uploadImgBtn && imageFileEl) {
    uploadImgBtn.addEventListener('click', () => {
      imageFileEl.value = '';
      imageFileEl.click();
    });

    imageFileEl.addEventListener('change', async () => {
      const file = imageFileEl.files && imageFileEl.files[0];
      const promptEl = document.getElementById('visionPrompt');
      const prompt = promptEl && promptEl.value ? String(promptEl.value).trim() : '';
      await sendVisionImage(file, prompt || 'What am I looking at?');
    });
  }

  wsUrlEl.textContent = buildWsUrl();
  setStatus('disconnected', true);
  setState('idle');
  log('Ready');
}

if (HAS_UI_NAVIGATOR_UI) {
  ensureInterruptDebugPanel();
  navWsUrlEl.textContent = buildNavWsUrl();
  // Reuse shared log/status elements if present; otherwise navigator still works.
  log('NAV: Ready');

  navOpenBtn.addEventListener('click', async () => {
    await ensureNavWs();
    let url = (navUrlEl.value || '').trim();
    if (!url) {
      log('NAV: missing URL');
      return;
    }
    if (!/^https?:\/\//i.test(url)) url = `https://${url}`;
    navWs.send(JSON.stringify({ type: 'open', url }));
  });

  navRunBtn.addEventListener('click', async () => {
    await ensureNavWs();
    const text = (navTaskEl.value || '').trim();
    if (!text) {
      log('NAV: missing task');
      return;
    }
    // Demo-friendly: if a URL is present, open it first so the planner doesn't work against about:blank.
    let url = (navUrlEl.value || '').trim();
    if (url) {
      if (!/^https?:\/\//i.test(url)) url = `https://${url}`;
      navWs.send(JSON.stringify({ type: 'open', url }));
    }
    navWs.send(JSON.stringify({ type: 'task', text }));
  });
}

function getSpeechRecognition() {
  return window.SpeechRecognition || window.webkitSpeechRecognition || null;
}

async function speakTaskOnce() {
  if (navSpeechActive) return;
  navSpeechActive = true;
  navSpeakBtn.disabled = true;
  try {
    log('NAV: speak now (Gemini Live STT)…');
    const text = await liveSttOnce();
    if (!text) throw new Error('no transcript');
    navTaskEl.value = text;
    log(`NAV: transcript "${text}"`);

    await ensureNavWs();
    let url = (navUrlEl.value || '').trim();
    if (url) {
      if (!/^https?:\/\//i.test(url)) url = `https://${url}`;
      navWs.send(JSON.stringify({ type: 'open', url }));
    }
    navWs.send(JSON.stringify({ type: 'task', text }));
  } catch (e) {
    log(`NAV: Live STT failed (${e?.message || e}). Falling back to browser speech.`);
    const SR = getSpeechRecognition();
    if (!SR) {
      log('NAV: SpeechRecognition not supported in this browser');
      return;
    }

    const recog = new SR();
    recog.lang = 'en-US';
    recog.interimResults = true;
    recog.maxAlternatives = 1;

    let finalText = '';
    recog.onresult = (evt) => {
      let interim = '';
      for (let i = evt.resultIndex; i < evt.results.length; i++) {
        const r = evt.results[i];
        const t = (r[0] && r[0].transcript) ? r[0].transcript : '';
        if (r.isFinal) finalText += t;
        else interim += t;
      }
      const combined = (finalText + ' ' + interim).trim();
      if (combined) navTaskEl.value = combined;
    };
    recog.onend = async () => {
      const text = (navTaskEl.value || '').trim();
      if (!text) {
        log('NAV: no speech captured');
        return;
      }
      log(`NAV: heard "${text}"`);
      await ensureNavWs();
      let url = (navUrlEl.value || '').trim();
      if (url) {
        if (!/^https?:\/\//i.test(url)) url = `https://${url}`;
        navWs.send(JSON.stringify({ type: 'open', url }));
      }
      navWs.send(JSON.stringify({ type: 'task', text }));
    };
    try { recog.start(); } catch {}
  } finally {
    navSpeechActive = false;
    navSpeakBtn.disabled = false;
  }
}

if (HAS_UI_NAVIGATOR_UI) {
  navSpeakBtn.addEventListener('click', async () => {
    await speakTaskOnce();
  });

  navInterruptBtn.addEventListener('click', async () => {
    await ensureNavWs();
    navWs.send(JSON.stringify({ type: 'interrupt' }));
  });
}
