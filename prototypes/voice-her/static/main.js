const micButton = document.getElementById("micButton");
const statusEl = document.getElementById("connectionStatus");
const latencyEl = document.getElementById("latencyStatus");
const hintEl = document.getElementById("hint");
const transcriptList = document.getElementById("transcriptList");
const textForm = document.getElementById("textForm");
const textInput = document.getElementById("textInput");

let socket;
let audioContext;
let mediaStream;
let captureNode;
let playbackTime = 0;
let active = false;
let firstAudioStartedAt = 0;

function setStatus(text, state = "") {
  statusEl.textContent = text;
  statusEl.className = `status-pill ${state}`.trim();
}

function appendTranscript(role, text) {
  const item = document.createElement("div");
  item.className = "line";
  const name = document.createElement("strong");
  name.textContent = role === "user" ? "Arthur" : "Her";
  const body = document.createElement("span");
  body.textContent = text;
  item.append(name, body);
  transcriptList.append(item);
  transcriptList.scrollTop = transcriptList.scrollHeight;
}

function downsample(buffer, fromRate, toRate) {
  if (fromRate === toRate) return buffer;
  const ratio = fromRate / toRate;
  const length = Math.round(buffer.length / ratio);
  const result = new Float32Array(length);
  let inputOffset = 0;
  for (let outputOffset = 0; outputOffset < length; outputOffset += 1) {
    const nextInputOffset = Math.round((outputOffset + 1) * ratio);
    let sum = 0;
    let count = 0;
    for (let i = inputOffset; i < nextInputOffset && i < buffer.length; i += 1) {
      sum += buffer[i];
      count += 1;
    }
    result[outputOffset] = count ? sum / count : 0;
    inputOffset = nextInputOffset;
  }
  return result;
}

function floatToPcm16(buffer) {
  const pcm = new Int16Array(buffer.length);
  for (let i = 0; i < buffer.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, buffer[i]));
    pcm[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
  }
  return pcm.buffer;
}

function playPcm24(arrayBuffer) {
  const pcm = new Int16Array(arrayBuffer);
  const floats = new Float32Array(pcm.length);
  for (let i = 0; i < pcm.length; i += 1) {
    floats[i] = pcm[i] / 32768;
  }
  const output = audioContext.createBuffer(1, floats.length, 24000);
  output.getChannelData(0).set(floats);
  const source = audioContext.createBufferSource();
  source.buffer = output;
  source.connect(audioContext.destination);
  const now = audioContext.currentTime;
  playbackTime = Math.max(playbackTime, now);
  source.start(playbackTime);
  playbackTime += output.duration;

  if (firstAudioStartedAt) {
    latencyEl.textContent = `Latence: ${Math.round(performance.now() - firstAudioStartedAt)} ms`;
    firstAudioStartedAt = 0;
  }
}

function wsUrl() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}/ws`;
}

async function connectSocket() {
  socket = new WebSocket(wsUrl());
  socket.binaryType = "arraybuffer";

  socket.onopen = () => {
    setStatus("Connecté", "connected");
    hintEl.textContent = "Micro actif.";
  };

  socket.onclose = () => {
    if (active) setStatus("Déconnecté", "error");
  };

  socket.onerror = () => {
    setStatus("Erreur connexion", "error");
  };

  socket.onmessage = (event) => {
    if (event.data instanceof ArrayBuffer) {
      playPcm24(event.data);
      return;
    }
    const message = JSON.parse(event.data);
    if (message.type === "transcript") appendTranscript(message.role, message.text);
    if (message.type === "interrupted") playbackTime = audioContext.currentTime;
    if (message.type === "error") {
      setStatus("Erreur session", "error");
      hintEl.textContent = message.message;
    }
  };
}

async function start() {
  audioContext = new AudioContext();
  await audioContext.audioWorklet.addModule("/static/pcm-processor.js");
  await connectSocket();

  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
      channelCount: 1,
    },
  });

  const source = audioContext.createMediaStreamSource(mediaStream);
  captureNode = new AudioWorkletNode(audioContext, "pcm-capture-processor");
  captureNode.port.onmessage = (event) => {
    if (!active || !socket || socket.readyState !== WebSocket.OPEN) return;
    const pcm = floatToPcm16(downsample(event.data, audioContext.sampleRate, 16000));
    socket.send(pcm);
    if (!firstAudioStartedAt) firstAudioStartedAt = performance.now();
  };

  const mute = audioContext.createGain();
  mute.gain.value = 0;
  source.connect(captureNode);
  captureNode.connect(mute);
  mute.connect(audioContext.destination);

  active = true;
  micButton.classList.add("active");
}

async function stop() {
  active = false;
  micButton.classList.remove("active");
  hintEl.textContent = "Maintenir une conversation vocale avec Her.";
  setStatus("Déconnecté");

  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ type: "stop_audio" }));
    socket.close();
  }
  if (captureNode) captureNode.disconnect();
  if (mediaStream) mediaStream.getTracks().forEach((track) => track.stop());
  if (audioContext) await audioContext.close();

  socket = null;
  captureNode = null;
  mediaStream = null;
  audioContext = null;
  playbackTime = 0;
  firstAudioStartedAt = 0;
}

micButton.addEventListener("click", async () => {
  try {
    if (active) {
      await stop();
    } else {
      await start();
    }
  } catch (error) {
    setStatus("Erreur micro", "error");
    hintEl.textContent = error.message;
    active = false;
    micButton.classList.remove("active");
  }
});

textForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = textInput.value.trim();
  if (!text || !socket || socket.readyState !== WebSocket.OPEN) return;
  socket.send(JSON.stringify({ type: "text", text }));
  textInput.value = "";
});
