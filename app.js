import { ObjectDetector, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.2";

let objectDetector = undefined;
let runningMode = "VIDEO";
let webcamRunning = false;
let vision = undefined;

const video = document.getElementById("video");
const canvas = document.getElementById("outputCanvas");
const ctx = canvas.getContext("2d");
const status = document.getElementById("status");
const startButton = document.getElementById("startButton");
const stopButton = document.getElementById("stopButton");

let lastVideoTime = -1;
let currentSearchWord = "cup";
let lastSaveTime = 0; // 履歴の連続保存を防ぐためのタイマー

// MediaPipe公式のLiteモデルのパス
const modelPaths = {
  lite0: "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite",
  lite2: "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite2/float16/1/efficientdet_lite2.tflite"
};

// 初期AIロード
async function initializeAI() {
  status.innerText = "ステータス: AIモデルをロード中...";
  status.style.color = "#ffc107";

  try {
    if (!vision) {
      vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.2/wasm"
      );
    }
    const selectedModel = document.getElementById("modelSelect").value;
    objectDetector = await ObjectDetector.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: modelPaths[selectedModel],
        delegate: "GPU"
      },
      scoreThreshold: 0.3,
      runningMode: runningMode
    });

    startButton.disabled = false;
    startButton.innerText = "カメラ起動 & 探索スタート";
    status.innerText = "ステータス: 準備完了";
    status.style.color = "#00ff00";
  } catch (err) {
    console.error(err);
    status.innerText = "エラー: AIのロードに失敗しました";
    status.style.color = "#ff4444";
  }
}

initializeAI();

// 途中でモデルを切り替えたときの処理
window.changeModel = async function() {
  const selectedModel = document.getElementById("modelSelect").value;
  const wasRunning = webcamRunning;

  if (webcamRunning) {
    stopCamera();
  }

  status.innerText = `ステータス: ${selectedModel.toUpperCase()} に切り替え中...`;
  status.style.color = "#ffc107";

  try {
    objectDetector = await ObjectDetector.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: modelPaths[selectedModel],
        delegate: "GPU"
      },
      scoreThreshold: 0.3,
      runningMode: runningMode
    });

    status.innerText = `ステータス: 切り替え完了 (${selectedModel})`;
    status.style.color = "#00ff00";

    if (wasRunning) {
      startCamera();
    }
  } catch (err) {
    console.error(err);
    status.innerText = "エラー: モデルの切り替えに失敗しました";
    status.style.color = "#ff4444";
  }
};

window.updateTarget = function() {
  const inputVal = document.getElementById("targetObject").value.trim().toLowerCase();
  if (inputVal === "") {
    alert("探すものを入力してください！");
    return;
  }
  currentSearchWord = inputVal;
  status.innerText = `🎯 「${currentSearchWord}」を探します`;
  status.style.color = "#00ff00";
};

window.startCamera = async function() {
  if (!objectDetector) return;
  currentSearchWord = document.getElementById("targetObject").value.trim().toLowerCase();

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment' },
      audio: false
    });

    video.srcObject = stream;
    webcamRunning = true;
    startButton.style.display = "none";
    stopButton.style.display = "block";
    status.innerText = `ステータス: 探索中... (${currentSearchWord})`;

    video.onloadeddata = () => { predictWebcam(); };
  } catch (err) {
    console.error(err);
    alert("カメラを起動できませんでした。");
  }
};

window.stopCamera = function() {
  webcamRunning = false;
  const stream = video.srcObject;
  if (stream) stream.getTracks().forEach(track => track.stop());
  video.srcObject = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  stopButton.style.display = "none";
  startButton.style.display = "block";
  status.innerText = "ステータス: 停止しました（準備完了）";
  status.style.color = "#aaa";
};

// 発見したときの画像を履歴に保存する関数
function saveFoundImage(bbox) {
  const gallery = document.getElementById("imageGallery");
  if (!gallery) return;

  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = bbox.width;
  tempCanvas.height = bbox.height;
  const tempCtx = tempCanvas.getContext("2d");

  tempCtx.drawImage(
    video,
    bbox.originX, bbox.originY, bbox.width, bbox.height,
    0, 0, bbox.width, bbox.height
  );

  const img = document.createElement("img");
  img.src = tempCanvas.toDataURL("image/jpeg");
  img.style.width = "80px";
  img.style.height = "80px";
  img.style.objectFit = "cover";
  img.style.borderRadius = "8px";
  img.style.border = "2px solid #00ff00";

  gallery.prepend(img);
}

// ★追加：履歴をすべてクリアする関数
window.clearGallery = function() {
  const gallery = document.getElementById("imageGallery");
  if (gallery) {
    gallery.innerHTML = "";
  }
};

// リアルタイム推論
async function predictWebcam() {
  if (!webcamRunning) return;

  if (video.videoWidth > 0 && video.videoHeight > 0) {
    const displayWidth = video.clientWidth;
    const displayHeight = video.clientHeight;

    if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
      canvas.width = displayWidth;
      canvas.height = displayHeight;
    }
  }

  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;

    try {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      let foundTarget = false;

      const startTimeMs = performance.now();
      const detections = objectDetector.detectForVideo(video, startTimeMs);

      let detectedNames = [];
      if (detections && detections.detections) {
        const scaleX = canvas.width / video.videoWidth;
        const scaleY = canvas.height / video.videoHeight;

        for (let detection of detections.detections) {
          const categoryName = detection.categories[0].categoryName.toLowerCase();
          const score = Math.round(detection.categories[0].score * 100);
          detectedNames.push(`${categoryName}(${score}%)`);

          if (categoryName.includes(currentSearchWord)) {
            foundTarget = true;
            const bbox = detection.boundingBox;

            const x = bbox.originX * scaleX;
            const y = bbox.originY * scaleY;
            const width = bbox.width * scaleX;
            const height = bbox.height * scaleY;

            ctx.strokeStyle = "#00ff00";
            ctx.lineWidth = 4;
            ctx.strokeRect(x, y, width, height);

            ctx.fillStyle = "#00ff00";
            ctx.font = "16px sans-serif";
            ctx.fillText(`${categoryName} ${score}%`, x, y > 20 ? y - 5 : 20);

            const now = Date.now();
            if (now - lastSaveTime > 3000) {
              saveFoundImage(bbox);
              lastSaveTime = now;
            }
          }
        }
      }

      if (foundTarget) {
        status.innerText = `🎉 発見！ (${currentSearchWord})`;
        status.style.color = "#00ff00";
      } else {
        if (detectedNames.length > 0) {
          status.innerText = `👀 検出中: ${detectedNames.join(', ')} (探す物: ${currentSearchWord})`;
        } else {
          status.innerText = `🔍 探索中... (${currentSearchWord})`;
        }
        status.style.color = "#aaa";
      }

    } catch (e) {
      console.error("推論エラー:", e);
    }
  }

  if (webcamRunning) {
    window.requestAnimationFrame(predictWebcam);
  }
}

