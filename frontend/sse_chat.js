function nowMs() {
  return new Date().toISOString().slice(11, 23);
}

function getBackendHttpBase() {
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

function setText(el, text) {
  if (!el) return;
  el.textContent = text;
}

function appendPre(el, line) {
  if (!el) return;
  el.textContent += line;
  el.scrollTop = el.scrollHeight;
}

function sseParseLines(chunkText, state) {
  // Minimal SSE parser.
  // state: { buf: string, event: string|null }
  state.buf += chunkText;

  const events = [];
  while (true) {
    const idx = state.buf.indexOf('\n');
    if (idx < 0) break;

    const rawLine = state.buf.slice(0, idx);
    state.buf = state.buf.slice(idx + 1);

    const line = rawLine.replace(/\r$/, '');
    if (!line) {
      // dispatch event boundary
      state.event = null;
      continue;
    }

    if (line.startsWith('event:')) {
      state.event = line.slice('event:'.length).trim();
      continue;
    }

    if (line.startsWith('data:')) {
      const dataStr = line.slice('data:'.length).trim();
      events.push({ event: state.event || 'message', data: dataStr });
    }
  }

  return events;
}

function safeJsonParse(text) {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

const els = {
  backend: document.getElementById('backendBase'),
  userId: document.getElementById('userId'),
  message: document.getElementById('message'),
  send: document.getElementById('sendBtn'),
  stop: document.getElementById('stopBtn'),
  speak: document.getElementById('speakBtn'),
  micStatus: document.getElementById('micStatus'),
  voiceEnabled: document.getElementById('voiceEnabled'),
  voiceOut: document.getElementById('voiceOut'),
  output: document.getElementById('output'),
  log: document.getElementById('log'),
};

const backendBase = getBackendHttpBase();
setText(els.backend, backendBase);

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

const AgentState = {
  IDLE: 'IDLE',
  LISTENING: 'LISTENING',
  SPEAKING: 'SPEAKING',
  INTERRUPTED: 'INTERRUPTED',
};

let agentState = AgentState.IDLE;
let streamEpoch = 0;
let debugPanelEl = null;
let debugRowsEl = null;

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
    stream: controller ? 'active' : 'idle',
    ...kv,
  };
  const entries = Object.entries(merged).map(([k, v]) => `${k}=${v}`).join(' | ');
  const line = `[${nowMs()}] ${event || 'tick'} ${entries}`;
  debugRowsEl.textContent = `${line}\n${debugRowsEl.textContent}`.slice(0, 6000);
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
  interruptDebug({}, 'state');
}

let controller = null;
let currentOutputText = '';

function stopPlayback() {
  if (!els.voiceOut) return;
  try {
    els.voiceOut.pause();
    els.voiceOut.currentTime = 0;
    els.voiceOut.removeAttribute('src');
    els.voiceOut.load();
  } catch {
    // ignore
  }
  interruptDebug({}, 'stop_playback');
}

function handleInterrupt() {
  streamEpoch += 1;
  agentState = AgentState.INTERRUPTED;
  updateUIState(agentState);

  stopPlayback();

  if (controller) {
    try { controller.abort('interrupt'); } catch {}
    controller = null;
  }

  agentState = AgentState.LISTENING;
  updateUIState(agentState);
  interruptDebug({}, 'interrupt_sent');
}

async function playTtsForText(text) {
  if (!els.voiceOut) return;
  const enabled = els.voiceEnabled ? !!els.voiceEnabled.checked : false;
  if (!enabled) return;
  const clean = (text || '').trim();
  if (!clean) return;

  appendPre(els.log, `[${nowMs()}] TTS: synthesizing…\n`);

  const res = await fetch(`${backendBase}/agent/speak`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: clean, lang: 'en-US' }),
  });

  if (!res.ok) {
    const txt = await res.text().catch(() => '');
    appendPre(els.log, `[${nowMs()}] TTS error: HTTP ${res.status} ${txt}\n`);
    return;
  }

  const buf = await res.arrayBuffer();
  const blob = new Blob([buf], { type: 'audio/mpeg' });
  const url = URL.createObjectURL(blob);
  try {
    // Stop any current playback.
    try { els.voiceOut.pause(); } catch {}
    els.voiceOut.src = url;
    await els.voiceOut.play();
    appendPre(els.log, `[${nowMs()}] TTS: playing\n`);
  } catch (e) {
    appendPre(els.log, `[${nowMs()}] TTS play failed: ${e?.message || e}\n`);
  }
}

// Optional voice input for pages that include a Speak button.
let micStream = null;
let audioCtx = null;
let processor = null;
let sourceNode = null;
let sttWs = null;
let recording = false;

function u8ToB64(u8) {
  let s = '';
  const chunkSize = 0x8000;
  for (let i = 0; i < u8.length; i += chunkSize) {
    s += String.fromCharCode(...u8.subarray(i, i + chunkSize));
  }
  return btoa(s);
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

function buildSttWsUrl(userId) {
  const base = new URL(backendBase);
  const wsProto = base.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = new URL(base.toString());
  wsUrl.protocol = wsProto;
  wsUrl.pathname = '/ws/live';
  wsUrl.searchParams.set('mode', 'stt');
  wsUrl.searchParams.set('output', 'text');
  wsUrl.searchParams.set('user_id', userId || 'demo');
  return wsUrl.toString();
}

function setMicStatus(text) {
  if (!els.micStatus) return;
  els.micStatus.textContent = text;
}

async function stopRecording(sendAudioEnd = true) {
  recording = false;

  try {
    if (sendAudioEnd && sttWs && sttWs.readyState === WebSocket.OPEN) {
      sttWs.send(JSON.stringify({ type: 'audio_end' }));
    }
  } catch {
    // ignore
  }

  if (processor) {
    try { processor.disconnect(); } catch {}
  }
  if (sourceNode) {
    try { sourceNode.disconnect(); } catch {}
  }

  if (micStream) {
    try {
      for (const t of micStream.getTracks()) t.stop();
    } catch {}
  }

  if (audioCtx) {
    try { await audioCtx.close(); } catch {}
  }

  processor = null;
  sourceNode = null;
  micStream = null;
  audioCtx = null;

  try {
    if (sttWs && (sttWs.readyState === WebSocket.OPEN || sttWs.readyState === WebSocket.CONNECTING)) {
      sttWs.close(1000, 'client_stop');
    }
  } catch {
    // ignore
  }
  sttWs = null;

  if (els.speak) els.speak.textContent = '🎤 Speak';
  setMicStatus('idle');
}

async function recordOnceToTranscript() {
  const userId = (els.userId && els.userId.value) ? String(els.userId.value).trim() : 'demo';
  const wsUrl = buildSttWsUrl(userId);
  appendPre(els.log, `[${nowMs()}] STT WS: ${wsUrl}\n`);

  return await new Promise(async (resolve, reject) => {
    let transcript = '';
    let finished = false;

    try {
      sttWs = new WebSocket(wsUrl);
    } catch (e) {
      reject(new Error(e?.message || String(e)));
      return;
    }

    sttWs.onmessage = (evt) => {
      const msg = safeJsonParse(evt.data);
      if (!msg || typeof msg !== 'object') return;

      if (msg.type === 'text' && typeof msg.text === 'string') {
        transcript += msg.text;
        return;
      }

      if (msg.type === 'turn_complete') {
        finished = true;
        resolve(transcript.trim());
      }

      if (msg.type === 'error') {
        reject(new Error(String(msg.error || 'stt error')));
      }
    };

    sttWs.onerror = () => {
      reject(new Error('STT websocket error'));
    };

    sttWs.onclose = () => {
      if (!finished && !recording) {
        // If closed early, resolve whatever is available.
        resolve(transcript.trim());
      }
    };

    sttWs.onopen = async () => {
      try {
        micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const Ctx = window.AudioContext || window.webkitAudioContext;
        audioCtx = new Ctx();
        sourceNode = audioCtx.createMediaStreamSource(micStream);
        processor = audioCtx.createScriptProcessor(4096, 1, 1);

        const inputRate = audioCtx.sampleRate;
        const targetRate = 16000;

        processor.onaudioprocess = (e) => {
          if (!recording) return;
          if (!sttWs || sttWs.readyState !== WebSocket.OPEN) return;

          const input = e.inputBuffer.getChannelData(0);
          const down = downsampleBuffer(input, inputRate, targetRate);
          const int16 = floatTo16BitPCM(down);
          const u8 = new Uint8Array(int16.buffer);
          const payload = {
            type: 'audio',
            mime_type: 'audio/pcm',
            data_b64: u8ToB64(u8),
          };
          try {
            sttWs.send(JSON.stringify(payload));
          } catch {
            // ignore
          }
        };

        sourceNode.connect(processor);
        processor.connect(audioCtx.destination);

        recording = true;
        if (els.speak) els.speak.textContent = '⏹ Stop Recording';
        setMicStatus('recording…');
      } catch (e) {
        reject(new Error(e?.message || String(e)));
      }
    };
  });
}

function stopActive(reason = 'client_stop') {
  if (reason === 'new_request' || reason === 'voice_input' || reason === 'interrupt') {
    handleInterrupt();
    return;
  }

  streamEpoch += 1;
  if (controller) {
    try { controller.abort(reason); } catch {}
    controller = null;
  }
  stopPlayback();
  updateUIState(AgentState.IDLE);
  interruptDebug({ reason }, 'stop_active');
}

async function sendStreamingMessage() {
  const message = (els.message && els.message.value) ? String(els.message.value).trim() : '';
  if (!message) return;

  const userId = (els.userId && els.userId.value) ? String(els.userId.value).trim() : 'demo';

  // Interrupt any in-flight stream.
  stopActive('new_request');

  // Reset output for this turn.
  if (els.output) els.output.textContent = '';
  currentOutputText = '';
  updateUIState(AgentState.LISTENING);
  interruptDebug({}, 'stream_start');

  controller = new AbortController();
  const signal = controller.signal;
  const localEpoch = ++streamEpoch;

  appendPre(els.log, `[${nowMs()}] POST /agent/message-stream (user_id=${userId})\n`);

  const res = await fetch(`${backendBase}/agent/message-stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: userId, message }),
    signal,
  });

  if (!res.ok || !res.body) {
    const txt = await res.text().catch(() => '');
    appendPre(els.log, `[${nowMs()}] error: HTTP ${res.status} ${txt}\n`);
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder('utf-8');
  const state = { buf: '', event: null };

  try {
    while (true) {
      if (localEpoch !== streamEpoch) break;
      const { value, done } = await reader.read();
      if (done) break;
      if (localEpoch !== streamEpoch) break;
      const text = decoder.decode(value, { stream: true });
      const events = sseParseLines(text, state);

      for (const ev of events) {
        if (localEpoch !== streamEpoch) break;
        const payload = safeJsonParse(ev.data);
        if (!payload || typeof payload !== 'object') continue;

        if (payload.type === 'text' && typeof payload.text === 'string') {
          updateUIState(AgentState.SPEAKING);
          appendPre(els.output, payload.text);
          currentOutputText += payload.text;
        } else if (payload.type === 'status') {
          appendPre(els.log, `[${nowMs()}] status: ${payload.status || ''} model=${payload.model || ''}\n`);
          interruptDebug({ status: payload.status || '', model: payload.model || '' }, 'stream_status');
        } else if (payload.type === 'interrupted') {
          handleInterrupt();
          appendPre(els.log, `[${nowMs()}] interrupted (${payload.source || ''})\n`);
          interruptDebug({ source: payload.source || '' }, 'stream_interrupted');
        } else if (payload.type === 'error') {
          appendPre(els.log, `[${nowMs()}] error: ${payload.error || ''}\n`);
          interruptDebug({ error: payload.error || '' }, 'stream_error');
        } else if (payload.type === 'done') {
          if (localEpoch !== streamEpoch) break;
          appendPre(els.log, `[${nowMs()}] done\n`);
          await playTtsForText(currentOutputText);
          updateUIState(AgentState.LISTENING);
          interruptDebug({}, 'stream_done');
        }
      }
    }
  } catch (e) {
    if (signal.aborted) {
      appendPre(els.log, `[${nowMs()}] aborted\n`);
    } else {
      appendPre(els.log, `[${nowMs()}] stream error: ${e?.message || e}\n`);
    }
    updateUIState(AgentState.INTERRUPTED);
  } finally {
    try { reader.releaseLock(); } catch {}
    if (!signal.aborted && localEpoch === streamEpoch) updateUIState(AgentState.LISTENING);
    interruptDebug({ aborted: signal.aborted }, 'stream_end');
  }
}

if (els.send) {
  els.send.addEventListener('click', async () => {
    try {
      await sendStreamingMessage();
    } catch (e) {
      appendPre(els.log, `[${nowMs()}] send failed: ${e?.message || e}\n`);
    }
  });
}

if (els.stop) {
  els.stop.addEventListener('click', () => {
    stopActive('client_stop');
  });
}

if (els.speak) {
  els.speak.addEventListener('click', async () => {
    try {
      if (recording) {
        await stopRecording(true);
        return;
      }

      stopActive('voice_input');
      setMicStatus('connecting…');

      const transcript = await recordOnceToTranscript();
      await stopRecording(false);

      if (!transcript) {
        appendPre(els.log, `[${nowMs()}] no transcript\n`);
        return;
      }

      if (els.message) els.message.value = transcript;
      appendPre(els.log, `[${nowMs()}] transcript: ${transcript}\n`);

      await sendStreamingMessage();
    } catch (e) {
      await stopRecording(false);
      appendPre(els.log, `[${nowMs()}] voice failed: ${e?.message || e}\n`);
    }
  });
}

if (els.message) {
  els.message.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      try { els.send.click(); } catch {}
    }
  });
}

updateUIState(AgentState.IDLE);
ensureInterruptDebugPanel();
