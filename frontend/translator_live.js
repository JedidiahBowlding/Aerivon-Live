function nowMs() {
  return new Date().toISOString().slice(11, 23);
}

function getBackendBase() {
  try {
    const injected = (window && window.AERIVON_BACKEND_BASE) ? String(window.AERIVON_BACKEND_BASE).trim() : '';
    if (injected) return injected.replace(/\/$/, '');
  } catch {
    // ignore
  }

  try {
    const u = new URL(location.href);
    const qp = (u.searchParams.get('backend') || '').trim();
    if (qp) return qp.replace(/\/$/, '');
  } catch {
    // ignore
  }

  return `${location.protocol}//${location.host}`;
}

function u8ToB64(u8) {
  let s = '';
  const chunkSize = 0x8000;
  for (let i = 0; i < u8.length; i += chunkSize) {
    s += String.fromCharCode(...u8.subarray(i, i + chunkSize));
  }
  return btoa(s);
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

const els = {
  backend: document.getElementById('backendBase'),
  wsUrl: document.getElementById('wsUrl'),
  status: document.getElementById('status'),
  start: document.getElementById('startBtn'),
  stop: document.getElementById('stopBtn'),
  interrupt: document.getElementById('interruptBtn'),
  userId: document.getElementById('userId'),
  target: document.getElementById('targetLang'),
  transcript: document.getElementById('transcript'),
  log: document.getElementById('log'),
};

function setStatus(text, muted = false) {
  if (!els.status) return;
  els.status.textContent = text;
  els.status.className = muted ? 'muted' : '';
}

function updateUIState(state) {
  agentState = state;
  if (els.status) {
    const isConnected = !!(ws && ws.readyState === WebSocket.OPEN);
    if (isConnected) {
      const low = state.toLowerCase();
      setStatus(`connected · ${low}`, false);
    } else {
      setStatus('disconnected', true);
    }
  }
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
  setDucking(state === AgentState.LISTENING);
}

function log(line) {
  if (!els.log) return;
  els.log.textContent += `[${nowMs()}] ${line}\n`;
  els.log.scrollTop = els.log.scrollHeight;
}

function appendTranscript(text) {
  if (!els.transcript) return;
  els.transcript.textContent += text;
  els.transcript.scrollTop = els.transcript.scrollHeight;
}

const backendBase = getBackendBase();
if (els.backend) els.backend.textContent = backendBase;

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

let ws = null;
let wsOpenPromise = null;
let wsOpenPromiseUrl = '';
let micStream = null;
let audioCtx = null;
let processor = null;
let sourceNode = null;

let playbackCtx = null;
let playCursor = 0;
let activeSources = [];
let playbackSampleRate = 24000;
let playbackGainNode = null;

const AgentState = {
  IDLE: 'IDLE',
  LISTENING: 'LISTENING',
  SPEAKING: 'SPEAKING',
  INTERRUPTED: 'INTERRUPTED',
};

let agentState = AgentState.IDLE;
let lastInterruptAt = 0;
let interruptLock = false;
let vadActiveMs = 0;
let vadLastTs = performance.now();

const VAD_SPEAKING_THRESHOLD = 0.010;
const VAD_IDLE_THRESHOLD = 0.006;
const VAD_TRIGGER_MS = 60;
const VAD_COOLDOWN_MS = 280;
const VAD_SPEECH_RMS = 0.012;
const VAD_SILENCE_MS = 850;
const VAD_MIN_TRIGGER_RMS = 0.0030;
const VAD_MAX_TRIGGER_RMS = 0.020;
const VAD_NOISE_MULTIPLIER = 2.0;
const AGGRESSIVE_INTERRUPT_RMS = 0.010;
const AGGRESSIVE_INTERRUPT_MS = 55;

let inUtterance = false;
let lastVoiceMs = 0;
let lastServerOutputAt = 0;
let lastServerAudioAt = 0;
let vadNoiseFloor = 0.0015;
let aggressiveVadMs = 0;
let debugPanelEl = null;
let debugRowsEl = null;
let lastDebugMicTs = 0;

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

function ensurePlaybackCtx() {
  if (playbackCtx) return;
  const Ctx = window.AudioContext || window.webkitAudioContext;
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

function applyAudioConfig(sampleRate) {
  const sr = Number(sampleRate);
  if (!Number.isFinite(sr) || sr <= 0) return;
  playbackSampleRate = sr;
  if (playbackCtx && playbackCtx.sampleRate !== playbackSampleRate) {
    stopPlayback();
    try { playbackCtx.close(); } catch {}
    playbackCtx = null;
    playbackGainNode = null;
  }
}

function stopPlayback() {
  if (!playbackCtx) {
    activeSources.length = 0;
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
}

function isAudioPlayingOrQueued() {
  if (!playbackCtx) return false;
  if (activeSources.length > 0) return true;
  return playCursor > playbackCtx.currentTime + 0.05;
}

function triggerInterrupt(source = 'barge-in') {
  if (!ws || ws.readyState !== WebSocket.OPEN) {
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
  stopPlayback();
  try {
    ws.send(JSON.stringify({ type: 'interrupt' }));
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

function handleMicFrame(frame) {
  const now = performance.now();
  const dt = now - vadLastTs;
  vadLastTs = now;

  const level = rms(frame);
  const speakingWindowActive = (now - lastServerAudioAt) < 1200;
  const speakingGateActive = agentState === AgentState.SPEAKING || isAudioPlayingOrQueued() || speakingWindowActive;

  if (!speakingGateActive) {
    vadNoiseFloor = Math.min(0.02, Math.max(0.0005, vadNoiseFloor * 0.985 + level * 0.015));
  } else {
    vadNoiseFloor = Math.max(0.0005, vadNoiseFloor * 0.995);
  }

  const dynamicThreshold = Math.min(VAD_MAX_TRIGGER_RMS, Math.max(VAD_MIN_TRIGGER_RMS, vadNoiseFloor * VAD_NOISE_MULTIPLIER));
  const dynamicIdleThreshold = Math.max(0.002, dynamicThreshold * 0.65);

  if (INTERRUPT_DEBUG && now - lastDebugMicTs > 120) {
    lastDebugMicTs = now;
    interruptDebug(
      {
        level: level.toFixed(4),
        thr: dynamicThreshold.toFixed(4),
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
      interruptDebug({}, 'interrupt_skip:not_speaking_or_not_queued');
    }
    return;
  }

  aggressiveVadMs = 0;

  if (now - lastInterruptAt < VAD_COOLDOWN_MS) {
    if (INTERRUPT_DEBUG && now - lastDebugMicTs > 120) {
      interruptDebug({ sinceMs: Math.round(now - lastInterruptAt) }, 'interrupt_skip:cooldown');
    }
    return;
  }

  if (level > dynamicThreshold) vadActiveMs += dt;
  else if (level < dynamicIdleThreshold) vadActiveMs = Math.max(0, vadActiveMs - dt * 2);

  if (vadActiveMs >= VAD_TRIGGER_MS) {
    vadActiveMs = 0;
    triggerInterrupt('barge-in');
    return;
  }

  if (INTERRUPT_DEBUG && now - lastDebugMicTs > 120) {
    interruptDebug({ level: level.toFixed(4) }, 'interrupt_hold:insufficient_vad');
  }
}

function enqueuePcmForPlayback(pcmBytes) {
  ensurePlaybackCtx();

  const int16 = new Int16Array(pcmBytes.buffer, pcmBytes.byteOffset, Math.floor(pcmBytes.byteLength / 2));
  const floats = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) floats[i] = int16[i] / 0x8000;

  const buffer = playbackCtx.createBuffer(1, floats.length, playbackSampleRate);
  buffer.copyToChannel(floats, 0);

  const src = playbackCtx.createBufferSource();
  src.buffer = buffer;
  src.connect(playbackGainNode || playbackCtx.destination);

  src.onended = () => {
    activeSources = activeSources.filter((s) => s !== src);
  };

  const startAt = Math.max(playCursor, playbackCtx.currentTime + 0.02);
  src.start(startAt);
  activeSources.push(src);
  playCursor = startAt + buffer.duration;
}

function buildWsUrl() {
  const userId = (els.userId && els.userId.value) ? String(els.userId.value).trim() : 'translator_demo';
  const target = (els.target && els.target.value) ? String(els.target.value).trim() : 'es-ES';

  const base = new URL(backendBase);
  const wsProto = base.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = new URL(base.toString());
  wsUrl.protocol = wsProto;
  wsUrl.pathname = '/ws/live';
  wsUrl.searchParams.set('output', 'audio');
  wsUrl.searchParams.set('user_id', userId);
  wsUrl.searchParams.set('memory_scope', 'translator_live');
  wsUrl.searchParams.set('lang', target);
  return wsUrl.toString();
}

function translatorInstruction() {
  const target = (els.target && els.target.value) ? String(els.target.value).trim() : 'es-ES';
  // Keep it very explicit: translate, speak translation only.
  return (
    `You are a real-time translator.\n` +
    `1) Transcribe what I say.\n` +
    `2) Translate it into ${target}.\n` +
    `3) Respond in AUDIO with only the translated sentence (no extra commentary).\n` +
    `4) Also send the translated sentence as text.\n`
  );
}

async function ensureWs() {
  const url = buildWsUrl();
  if (els.wsUrl) els.wsUrl.textContent = url;

  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
    // If there's already a connect-in-progress for the same URL, await it.
    if (wsOpenPromise && wsOpenPromiseUrl === url) {
      await wsOpenPromise;
    }
    return;
  }

  ws = new WebSocket(url);
  setStatus('connecting…', true);
  wsOpenPromiseUrl = url;
  wsOpenPromise = new Promise((resolve, reject) => {
    const t = setTimeout(() => reject(new Error('WS connect timeout')), 8000);
    ws.addEventListener('open', () => {
      clearTimeout(t);
      resolve(true);
    }, { once: true });
    ws.addEventListener('error', () => {
      clearTimeout(t);
      reject(new Error('WS error'));
    }, { once: true });
    ws.addEventListener('close', () => {
      clearTimeout(t);
      reject(new Error('WS closed'));
    }, { once: true });
  });

  ws.onopen = () => {
    updateUIState(AgentState.LISTENING);
    lastServerOutputAt = 0;
    lastServerAudioAt = 0;
    vadActiveMs = 0;
    aggressiveVadMs = 0;
    interruptDebug({}, 'ws_open');
    log('WS connected');
    try {
      ws.send(JSON.stringify({ type: 'text', text: translatorInstruction() }));
    } catch {
      // ignore
    }
  };

  ws.onclose = () => {
    updateUIState(AgentState.IDLE);
    interruptDebug({}, 'ws_close');
    log('WS closed');
    wsOpenPromise = null;
  };

  ws.onerror = () => {
    setStatus('error');
    log('WS error');
  };

  ws.onmessage = (evt) => {
    let msg;
    try { msg = JSON.parse(evt.data); } catch { return; }
    if (!msg || typeof msg !== 'object') return;

    if (msg.type === 'audio_config' && msg.sample_rate) {
      interruptDebug({ sampleRate: Number(msg.sample_rate) }, 'audio_config');
      applyAudioConfig(msg.sample_rate);
      return;
    }

    if (msg.type === 'audio' && msg.data_b64) {
      updateUIState(AgentState.SPEAKING);
      lastServerOutputAt = performance.now();
      lastServerAudioAt = lastServerOutputAt;
      const u8 = b64ToU8(String(msg.data_b64));
      enqueuePcmForPlayback(u8);
      return;
    }

    if (msg.type === 'text' && typeof msg.text === 'string') {
      lastServerOutputAt = performance.now();
      appendTranscript(msg.text);
      return;
    }

    if (msg.type === 'transcript' && typeof msg.text === 'string') {
      lastServerOutputAt = performance.now();
      appendTranscript(msg.text);
      return;
    }

    if (msg.type === 'interrupted') {
      log(`Interrupted (${msg.source || ''})`);
      lastServerOutputAt = 0;
      lastServerAudioAt = 0;
      vadActiveMs = 0;
      aggressiveVadMs = 0;
      updateUIState(AgentState.INTERRUPTED);
      stopPlayback();
      updateUIState(AgentState.LISTENING);
      return;
    }

    if (msg.type === 'error') {
      log(`Server error: ${msg.error || ''}`);
      return;
    }

    if (msg.type === 'turn_complete') {
      log('Turn complete');
      if (agentState !== AgentState.IDLE) updateUIState(AgentState.LISTENING);
    }
  };
}

async function startMic() {
  try {
    await ensureWs();
  } catch (e) {
    log(`WS connect failed: ${e?.message || e}`);
    return;
  }

  if (!ws || ws.readyState !== WebSocket.OPEN) {
    log('WS not open');
    return;
  }

  if (micStream) return;

  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  });
  micStream = stream;

  const Ctx = window.AudioContext || window.webkitAudioContext;
  audioCtx = new Ctx();
  sourceNode = audioCtx.createMediaStreamSource(stream);
  processor = audioCtx.createScriptProcessor(4096, 1, 1);

  const inputRate = audioCtx.sampleRate;
  const targetRate = 16000;

  processor.onaudioprocess = (e) => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;

    const input = e.inputBuffer.getChannelData(0);
    const now = performance.now();
    handleMicFrame(input);

    const level = rms(input);

    // Turn detection (voice activity): send audio_end after silence.
    if (level > VAD_SPEECH_RMS) {
      inUtterance = true;
      lastVoiceMs = now;
    }

    if (inUtterance && now - lastVoiceMs > VAD_SILENCE_MS) {
      try {
        ws.send(JSON.stringify({ type: 'audio_end' }));
      } catch {
        // ignore
      }
      inUtterance = false;
      log('Auto audio_end (silence)');
    }

    const down = downsampleBuffer(input, inputRate, targetRate);
    const int16 = floatTo16BitPCM(down);
    const u8 = new Uint8Array(int16.buffer);

    try {
      ws.send(JSON.stringify({ type: 'audio', mime_type: 'audio/pcm', data_b64: u8ToB64(u8) }));
    } catch {
      // ignore
    }
  };

  sourceNode.connect(processor);
  processor.connect(audioCtx.destination);

  if (els.start) els.start.disabled = true;
  if (els.stop) els.stop.disabled = false;
  updateUIState(AgentState.LISTENING);
  log('Mic started');
}

async function stopMic() {
  if (!micStream) return;

  try {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'audio_end' }));
    }
  } catch {
    // ignore
  }

  for (const t of micStream.getTracks()) t.stop();
  micStream = null;
  inUtterance = false;
  vadActiveMs = 0;

  try { processor.disconnect(); } catch {}
  try { sourceNode.disconnect(); } catch {}
  processor = null;
  sourceNode = null;

  try { await audioCtx.close(); } catch {}
  audioCtx = null;

  if (els.start) els.start.disabled = false;
  if (els.stop) els.stop.disabled = true;
  updateUIState(AgentState.IDLE);
  log('Mic stopped (sent audio_end)');
}

async function interrupt() {
  try {
    await ensureWs();
  } catch {
    // ignore
  }
  try {
    if (ws && ws.readyState === WebSocket.OPEN) {
      triggerInterrupt('manual');
    }
  } catch {
    // ignore
  }
}

if (els.start) {
  els.start.addEventListener('click', async () => {
    try { await startMic(); } catch (e) { log(`Start failed: ${e?.message || e}`); }
  });
}
if (els.stop) {
  els.stop.addEventListener('click', async () => {
    try { await stopMic(); } catch (e) { log(`Stop failed: ${e?.message || e}`); }
  });
}
if (els.interrupt) {
  els.interrupt.addEventListener('click', async () => {
    await interrupt();
  });
}
if (els.target) {
  els.target.addEventListener('change', () => {
    // Force reconnect so voice language takes effect.
    try { if (ws) ws.close(1000, 'lang_change'); } catch {}
    ws = null;
    stopPlayback();
    if (els.wsUrl) els.wsUrl.textContent = buildWsUrl();
    log('Language changed; reconnect on next start');
  });
}

if (els.wsUrl) els.wsUrl.textContent = buildWsUrl();
setStatus('disconnected', true);
updateUIState(AgentState.IDLE);
ensureInterruptDebugPanel();
log('Ready');
