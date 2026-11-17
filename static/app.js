const video = document.getElementById('video');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const startButton = document.getElementById('startButton');
const stopButton = document.getElementById('stopButton');
const cameraSelect = document.getElementById('cameraSelect');
const cameraStatus = document.getElementById('cameraStatus');
const cameraPlaceholder = document.getElementById('cameraPlaceholder');
const videoStage = document.getElementById('videoStage');
const message = document.getElementById('message');
const modelDevice = document.getElementById('modelDevice');

let stream = null;
let running = false;
let processing = false;
let selectedDeviceId = '';
let lastFaces = [];

const categoryElements = {
  balita: { count: 'countBalita', bar: 'barBalita' },
  anak_anak: { count: 'countAnak', bar: 'barAnak' },
  remaja: { count: 'countRemaja', bar: 'barRemaja' },
  dewasa: { count: 'countDewasa', bar: 'barDewasa' },
  lansia: { count: 'countLansia', bar: 'barLansia' },
};

function setMessage(text, isError = false) {
  message.textContent = text;
  message.style.color = isError ? '#ff9e9e' : '';
}

async function loadHealth() {
  try {
    const response = await fetch('/health');
    const data = await response.json();
    modelDevice.textContent = `Inference: ${data.device}`;
  } catch (_error) {
    modelDevice.textContent = 'Server belum tersambung';
  }
}

async function populateCameras() {
  if (!navigator.mediaDevices?.enumerateDevices) return;
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cameras = devices.filter(device => device.kind === 'videoinput');
    const previous = cameraSelect.value;
    cameraSelect.innerHTML = '<option value="">Kamera default</option>';
    cameras.forEach((camera, index) => {
      const option = document.createElement('option');
      option.value = camera.deviceId;
      option.textContent = camera.label || `Kamera ${index + 1}`;
      cameraSelect.appendChild(option);
    });
    if (previous) cameraSelect.value = previous;
  } catch (_error) {
    // The browser can hide camera labels until permission is granted.
  }
}

function resetStats() {
  document.getElementById('faceCount').textContent = '0';
  document.getElementById('categoryTotal').textContent = '0';
  document.getElementById('dominantAge').textContent = '—';
  Object.values(categoryElements).forEach(item => {
    document.getElementById(item.count).textContent = '0';
    document.getElementById(item.bar).style.width = '0%';
  });
}

function normalizedClass(name) {
  return String(name || '').toLowerCase().trim().replaceAll('-', '_').replaceAll(' ', '_');
}

function updateStats(faces) {
  const counts = { balita: 0, anak_anak: 0, remaja: 0, dewasa: 0, lansia: 0 };
  faces.forEach(face => {
    const key = normalizedClass(face.class_name);
    if (Object.prototype.hasOwnProperty.call(counts, key)) counts[key] += 1;
  });

  const total = faces.length;
  document.getElementById('faceCount').textContent = String(total);
  document.getElementById('categoryTotal').textContent = String(total);

  let dominant = '—';
  let dominantCount = 0;
  Object.entries(counts).forEach(([key, count]) => {
    const element = categoryElements[key];
    document.getElementById(element.count).textContent = String(count);
    document.getElementById(element.bar).style.width = total ? `${(count / total) * 100}%` : '0%';
    if (count > dominantCount) {
      dominantCount = count;
      dominant = key === 'anak_anak' ? 'ANAK-ANAK' : key.toUpperCase();
    }
  });
  document.getElementById('dominantAge').textContent = dominant;
}

function drawBoxes(faces, sourceWidth, sourceHeight) {
  if (!sourceWidth || !sourceHeight) return;
  if (canvas.width !== sourceWidth || canvas.height !== sourceHeight) {
    canvas.width = sourceWidth;
    canvas.height = sourceHeight;
  }
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  faces.forEach(face => {
    const x = face.x1;
    const y = face.y1;
    const width = face.x2 - face.x1;
    const height = face.y2 - face.y1;
    const color = '#62e5bd';
    const label = `${face.label} · ${Number(face.score).toFixed(1)}%`;

    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(2, canvas.width / 400);
    ctx.strokeRect(x, y, width, height);

    const fontSize = Math.max(13, canvas.width / 52);
    ctx.font = `700 ${fontSize}px Arial`;
    const textWidth = ctx.measureText(label).width;
    const boxHeight = fontSize + 13;
    const labelY = Math.max(boxHeight, y);
    ctx.fillStyle = color;
    ctx.fillRect(x, labelY - boxHeight, textWidth + 14, boxHeight);
    ctx.fillStyle = '#06251e';
    ctx.fillText(label, x + 7, labelY - 8);
  });
}

async function processFrame() {
  if (!running || processing || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
  processing = true;

  try {
    const capture = document.createElement('canvas');
    const maxWidth = 640;
    const ratio = Math.min(1, maxWidth / video.videoWidth);
    capture.width = Math.max(1, Math.round(video.videoWidth * ratio));
    capture.height = Math.max(1, Math.round(video.videoHeight * ratio));
    const captureContext = capture.getContext('2d');
    // Mirror the captured image to match the mirrored preview.
    captureContext.translate(capture.width, 0);
    captureContext.scale(-1, 1);
    captureContext.drawImage(video, 0, 0, capture.width, capture.height);

    const blob = await new Promise(resolve => capture.toBlob(resolve, 'image/jpeg', 0.76));
    if (!blob) throw new Error('Gagal mengambil frame kamera.');

    const formData = new FormData();
    formData.append('frame', blob, 'camera.jpg');
    const response = await fetch('/detect', { method: 'POST', body: formData });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Gagal memproses kamera.');

    lastFaces = data.faces || [];
    drawBoxes(lastFaces, data.width, data.height);
    updateStats(lastFaces);
    setMessage(lastFaces.length
      ? `${lastFaces.length} wajah sedang dianalisis. Frame diproses pada ${data.device}.`
      : 'Belum ada wajah terdeteksi. Hadapkan wajah ke kamera dan pastikan pencahayaan cukup.');
  } catch (error) {
    setMessage(error.message || 'Terjadi kesalahan saat memproses frame.', true);
  } finally {
    processing = false;
    if (running) window.setTimeout(processFrame, 250);
  }
}

async function startCamera() {
  if (running) return;
  try {
    const videoConstraints = selectedDeviceId
      ? { deviceId: { exact: selectedDeviceId }, width: { ideal: 640 }, height: { ideal: 480 } }
      : { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } };

    stream = await navigator.mediaDevices.getUserMedia({ video: videoConstraints, audio: false });
    video.srcObject = stream;
    await video.play();
    await new Promise(resolve => {
      if (video.videoWidth) resolve();
      else video.onloadedmetadata = resolve;
    });

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    // Match the preview panel to the camera aspect ratio so boxes stay aligned.
    videoStage.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
    running = true;
    cameraPlaceholder.hidden = true;
    videoStage.classList.add('active');
    cameraStatus.className = 'status-badge live';
    cameraStatus.innerHTML = '<span></span> Kamera aktif';
    startButton.disabled = true;
    stopButton.disabled = false;
    cameraSelect.disabled = true;
    setMessage('Kamera aktif. Mengirim frame ke server lokal untuk dianalisis...');
    await populateCameras();
    processFrame();
  } catch (error) {
    setMessage(`Tidak dapat mengakses kamera: ${error.message}`, true);
    stopCamera();
  }
}

function stopCamera() {
  running = false;
  processing = false;
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    stream = null;
  }
  video.srcObject = null;
  videoStage.style.aspectRatio = '4 / 3';
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  cameraPlaceholder.hidden = false;
  videoStage.classList.remove('active');
  cameraStatus.className = 'status-badge idle';
  cameraStatus.innerHTML = '<span></span> Kamera mati';
  startButton.disabled = false;
  stopButton.disabled = true;
  cameraSelect.disabled = false;
  lastFaces = [];
  resetStats();
  setMessage('Kamera dihentikan.');
}

startButton.addEventListener('click', startCamera);
stopButton.addEventListener('click', stopCamera);
cameraSelect.addEventListener('change', event => {
  selectedDeviceId = event.target.value;
});
window.addEventListener('beforeunload', () => {
  if (stream) stream.getTracks().forEach(track => track.stop());
});

loadHealth();
populateCameras();
