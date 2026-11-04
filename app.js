// FINDRONE AI探偵カメラ
// ボタンの処理はすべてこのファイルの中で登録する(HTMLにonclickは書かない)

/* ================= 設定 ================= */
const MP_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.2";
const WASM_URL = MP_URL + "/wasm";
const DETECTOR_URLS = {
  lite0: "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite",
  lite2: "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite2/float16/1/efficientdet_lite2.tflite"
};
const EMBEDDER_URL = "https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_small/float32/1/mobilenet_v3_small.tflite";

const CUSTOM_THRESHOLD = 0.9;   // 登録した物体と判定する確率(0〜1)
const CUSTOM_HITS_NEEDED = 2;   // 何回続けてしきい値を超えたら「発見」にするか
const SAVE_INTERVAL_MS = 3000;  // 履歴の連続保存を防ぐ間隔
const STORAGE_KEY = "findrone_custom_v1";
const PC_ID_KEY = "findrone_pc_id";

// 標準AI(EfficientDet / COCO)が知っている80種類
const COCO_LABELS = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
  "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
  "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
  "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
  "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
  "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
  "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"];

// 日本語で入力したときの読み替え(カタカナ・ひらがなは区別しない)
const JP_RAW = {
  "傘": ["umbrella"], "かさ": ["umbrella"],
  "ペットボトル": ["bottle"], "ボトル": ["bottle"], "水筒": ["bottle"],
  "コップ": ["cup"], "カップ": ["cup"], "マグ": ["cup"], "マグカップ": ["cup"], "湯のみ": ["cup"],
  "スマホ": ["cell phone"], "スマートフォン": ["cell phone"], "携帯": ["cell phone"], "携帯電話": ["cell phone"], "ケータイ": ["cell phone"],
  "本": ["book"], "ほん": ["book"], "教科書": ["book"],
  "リモコン": ["remote"], "はさみ": ["scissors"], "ハサミ": ["scissors"],
  "マウス": ["mouse"], "キーボード": ["keyboard"],
  "パソコン": ["laptop"], "ノートパソコン": ["laptop"], "ノートPC": ["laptop"],
  "時計": ["clock"], "ぬいぐるみ": ["teddy bear"], "テディベア": ["teddy bear"],
  "歯ブラシ": ["toothbrush"], "ボール": ["sports ball"], "ラケット": ["tennis racket"],
  "グラス": ["wine glass"], "ワイングラス": ["wine glass"],
  "ボウル": ["bowl"], "お椀": ["bowl"], "茶碗": ["bowl"],
  "スプーン": ["spoon"], "フォーク": ["fork"], "ナイフ": ["knife"],
  "リュック": ["backpack"], "バックパック": ["backpack"],
  "ハンドバッグ": ["handbag"], "バッグ": ["handbag", "backpack", "suitcase"],
  "かばん": ["handbag", "backpack", "suitcase"], "鞄": ["handbag", "backpack", "suitcase"],
  "スーツケース": ["suitcase"], "ネクタイ": ["tie"],
  "いす": ["chair"], "椅子": ["chair"], "人": ["person"], "ひと": ["person"]
};

/* ================= 小さな道具 ================= */
const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const COLOR = { ok: "#00ff00", warn: "#ffc107", err: "#ff6b6b", dim: "#aaaaaa" };

function toHiragana(s) {
  return s.replace(/[\u30a1-\u30f6]/g, (c) => String.fromCharCode(c.charCodeAt(0) - 0x60));
}
const normKey = (s) => toHiragana(String(s).normalize("NFKC").trim().toLowerCase());

const JP = {};
for (const [k, v] of Object.entries(JP_RAW)) JP[normKey(k)] = v;

function setText(el, text, color) {
  el.textContent = text;
  if (color) el.style.color = color;
}
const setStatus = (t, c) => setText($("status"), t, c);
const setRegStatus = (t, c) => setText($("regStatus"), t, c);
const setConn = (t, c) => setText($("connectionStatus"), t, c);

function cameraErrorMessage(e) {
  if (e && e.name === "NotAllowedError") return "カメラの使用が許可されていません。ブラウザの設定で許可してください";
  if (e && e.name === "NotFoundError") return "カメラが見つかりません";
  if (e && e.name === "NotReadableError") return "カメラが他のアプリで使われています";
  return String((e && e.message) || e);
}

/* ================= 状態 ================= */
let mp = null, vision = null;
let detector = null, embedder = null, embedderPromise = null;
let aiReady = false;
let customModels = {};            // { 名前: { w:[...], b, n } }
let target = { kind: "none" };    // いま探しているもの
let searchRunning = false;
let lastVideoTime = -1;
let lastSaveTime = 0;
let customHits = 0, customBusy = false, lastCustomTime = 0;

const video = $("video");
const canvas = $("outputCanvas");
const ctx = canvas.getContext("2d");
const regVideo = $("regVideo");

// 特徴抽出用の作業用キャンバス(224x224)
const tmp = document.createElement("canvas");
tmp.width = tmp.height = 224;
const tctx = tmp.getContext("2d", { willReadFrequently: true });

/* ================= AIの読み込み ================= */
async function getVision() {
  if (!mp) mp = await import(MP_URL);
  if (!vision) vision = await mp.FilesetResolver.forVisionTasks(WASM_URL);
  return vision;
}

async function createDetector(modelKey) {
  const v = await getVision();
  const make = (delegate) => mp.ObjectDetector.createFromOptions(v, {
    baseOptions: { modelAssetPath: DETECTOR_URLS[modelKey], delegate },
    scoreThreshold: 0.3,
    runningMode: "VIDEO"
  });
  try {
    return await make("GPU");
  } catch (e) {
    console.warn("GPUで失敗したのでCPUに切り替えます", e);
    return await make("CPU");
  }
}

async function initializeAI() {
  setStatus("ステータス: AIモデルをロード中...", COLOR.warn);
  try {
    detector = await createDetector($("modelSelect").value);
    aiReady = true;
    $("startButton").disabled = false;
    $("startButton").textContent = "カメラ起動 & 探索スタート";
    setStatus("ステータス: 準備完了", COLOR.ok);
  } catch (err) {
    console.error(err);
    setStatus("エラー: AIのロードに失敗しました\n" + String(err.message || err), COLOR.err);
  }
}

function ensureEmbedder() {
  if (!embedderPromise) {
    embedderPromise = (async () => {
      const v = await getVision();
      if (!mp.ImageEmbedder) throw new Error("この版のMediaPipeにはImageEmbedderがありません");
      embedder = await mp.ImageEmbedder.createFromOptions(v, {
        baseOptions: { modelAssetPath: EMBEDDER_URL },
        runningMode: "IMAGE",
        l2Normalize: true
      });
      return embedder;
    })().catch((e) => { embedderPromise = null; throw e; });
  }
  return embedderPromise;
}

function getEmbedding(canvasEl) {
  const r = embedder.embed(canvasEl);
  return Float32Array.from(r.embeddings[0].floatEmbedding);
}

async function changeModel() {
  const wasRunning = searchRunning;
  if (wasRunning) stopCamera();
  setStatus("ステータス: " + $("modelSelect").value.toUpperCase() + " に切り替え中...", COLOR.warn);
  $("startButton").disabled = true;
  try {
    const old = detector;
    detector = await createDetector($("modelSelect").value);
    if (old && old.close) { try { old.close(); } catch (e) {} }
    $("startButton").disabled = false;
    setStatus("ステータス: 切り替え完了 (" + $("modelSelect").value + ")", COLOR.ok);
    if (wasRunning) startCamera();
  } catch (err) {
    console.error(err);
    setStatus("エラー: モデルの切り替えに失敗しました\n" + String(err.message || err), COLOR.err);
  }
}

/* ================= 探すものの決定 ================= */
function resolveTarget(raw) {
  const word = normKey(raw);
  if (!word) return { kind: "none", message: "探すものを入力してください" };

  for (const name of Object.keys(customModels)) {
    if (normKey(name) === word) return { kind: "custom", name, label: name };
  }
  if (JP[word]) return { kind: "coco", labels: new Set(JP[word]), label: raw.trim() };

  const variants = [word, word.replace(/s$/, "")];
  const hit = COCO_LABELS.filter((l) => variants.some((v) => l === v || l.split(" ").includes(v)));
  if (hit.length) return { kind: "coco", labels: new Set(hit), label: raw.trim() };

  return {
    kind: "none",
    message: "「" + raw.trim() + "」は標準AI(80種類)にありません。\n英語名(例: cup)を入れるか、「物体登録モード」で登録してください"
  };
}

async function applyTarget() {
  const t = resolveTarget($("targetObject").value);
  if (t.kind === "none") { setStatus(t.message, COLOR.err); return false; }
  if (t.kind === "custom") {
    setStatus("ステータス: 特徴抽出AIを準備中...", COLOR.warn);
    try { await ensureEmbedder(); }
    catch (e) { setStatus("エラー: 特徴抽出AIをロードできません\n" + String(e.message || e), COLOR.err); return false; }
  }
  target = t;
  customHits = 0;
  setStatus("🎯 「" + t.label + "」を探します", COLOR.ok);
  return true;
}

function renderQuickSelect() {
  const box = $("quickSelectList");
  box.replaceChildren();
  const words = Object.keys(customModels).concat(["cup", "bottle", "umbrella", "backpack", "cell phone"]);
  words.forEach((w) => {
    const b = document.createElement("button");
    b.className = "chip";
    b.textContent = w;
    b.addEventListener("click", async () => {
      $("targetObject").value = w;
      await applyTarget();
    });
    box.appendChild(b);
  });
}

/* ================= 探索カメラ ================= */
async function openCamera(videoEl) {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: { ideal: "environment" } },
    audio: false
  });
  videoEl.srcObject = stream;
  await videoEl.play();
}

function stopStream(videoEl) {
  const s = videoEl.srcObject;
  if (s) s.getTracks().forEach((t) => t.stop());
  videoEl.srcObject = null;
}

async function startCamera() {
  if (searchRunning) return;
  if (!(await applyTarget())) return;
  if (target.kind === "coco" && !detector) { setStatus("AIの準備ができていません", COLOR.err); return; }
  try {
    await openCamera(video);
  } catch (e) {
    console.error(e);
    setStatus("エラー: カメラを起動できません\n" + cameraErrorMessage(e), COLOR.err);
    return;
  }
  searchRunning = true;
  lastVideoTime = -1;
  $("startButton").hidden = true;
  $("stopButton").hidden = false;
  setStatus("ステータス: 探索中... (" + target.label + ")", COLOR.dim);
  requestAnimationFrame(loop);
}

function stopCamera() {
  searchRunning = false;
  stopStream(video);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  $("startButton").hidden = false;
  $("stopButton").hidden = true;
  if (aiReady) setStatus("ステータス: 停止しました(準備完了)", COLOR.dim);
}

function loop() {
  if (!searchRunning) return;
  try {
    if (target.kind === "coco") cocoStep();
    else if (target.kind === "custom") customStep();
  } catch (e) {
    console.error("推論エラー:", e);
  }
  requestAnimationFrame(loop);
}

function syncCanvas() {
  if (video.videoWidth && (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight)) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }
}

function drawBox(bb, text, color) {
  const lw = Math.max(3, canvas.width / 200);
  const fs = Math.max(14, Math.round(canvas.width / 35));
  ctx.strokeStyle = color;
  ctx.lineWidth = lw;
  ctx.strokeRect(bb.originX, bb.originY, bb.width, bb.height);
  ctx.fillStyle = color;
  ctx.font = "bold " + fs + "px sans-serif";
  ctx.fillText(text, bb.originX + 4, bb.originY > fs + 6 ? bb.originY - 6 : bb.originY + fs + 4);
}

/* ----- 標準AI(80種類)で探す ----- */
function cocoStep() {
  if (!detector || video.readyState < 2 || video.currentTime === lastVideoTime) return;
  lastVideoTime = video.currentTime;
  syncCanvas();
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const result = detector.detectForVideo(video, performance.now());
  const names = [];
  let best = null;
  for (const d of (result && result.detections) || []) {
    const cat = d.categories[0];
    const name = String(cat.categoryName || "").toLowerCase();
    const pct = Math.round(cat.score * 100);
    names.push(name + "(" + pct + "%)");
    if (target.labels.has(name)) {
      drawBox(d.boundingBox, name + " " + pct + "%", COLOR.ok);
      if (!best || cat.score > best.score) best = { name, score: cat.score, box: d.boundingBox };
    }
  }
  if (best) {
    setStatus("🎉 発見! (" + target.label + ")", COLOR.ok);
    handleFound(best.name, best.score, best.box);
  } else if (names.length) {
    setStatus("👀 検出中: " + names.join(", ") + "\n(探す物: " + target.label + ")", COLOR.dim);
  } else {
    setStatus("🔍 探索中... (" + target.label + ")", COLOR.dim);
  }
}

/* ----- 登録した物体で探す ----- */
let cropCache = { key: "", crops: [] };
function makeCrops(vw, vh) {
  const key = vw + "x" + vh;
  if (cropCache.key === key) return cropCache.crops;
  const m = Math.min(vw, vh);
  const crops = [];
  for (const side of [m, m * 0.6]) {
    const nx = vw - side > 4 ? 3 : 1;
    const ny = vh - side > 4 ? 3 : 1;
    for (let iy = 0; iy < ny; iy++) {
      for (let ix = 0; ix < nx; ix++) {
        const x = nx === 1 ? (vw - side) / 2 : ((vw - side) * ix) / (nx - 1);
        const y = ny === 1 ? (vh - side) / 2 : ((vh - side) * iy) / (ny - 1);
        crops.push({ originX: x, originY: y, width: side, height: side });
      }
    }
  }
  cropCache = { key, crops };
  return crops;
}

function drawCrop(videoEl, r) {
  tctx.drawImage(videoEl, r.originX, r.originY, r.width, r.height, 0, 0, 224, 224);
}

function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }
function predictLogistic(model, v) {
  const w = model.w;
  if (w.length !== v.length) return 0;
  let z = model.b;
  for (let i = 0; i < w.length; i++) z += w[i] * v[i];
  return sigmoid(z);
}

function customStep() {
  const now = performance.now();
  if (customBusy || !embedder || video.readyState < 2 || now - lastCustomTime < 200) return;
  customBusy = true;
  try { customDetect(); }
  finally { customBusy = false; lastCustomTime = performance.now(); }
}

function customDetect() {
  const model = customModels[target.name];
  if (!model) return;
  syncCanvas();
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  let best = null;       // いちばん確率が高い枠
  let smallest = null;   // しきい値を超えた中で、いちばん小さい枠
  for (const r of makeCrops(video.videoWidth, video.videoHeight)) {
    drawCrop(video, r);
    const p = predictLogistic(model, getEmbedding(tmp));
    if (!best || p > best.p) best = { p, r };
    if (p >= CUSTOM_THRESHOLD && (!smallest || r.width < smallest.r.width)) smallest = { p, r };
  }

  customHits = smallest ? customHits + 1 : 0;
  if (smallest) drawBox(smallest.r, target.name + " " + Math.round(smallest.p * 100) + "%", COLOR.ok);

  if (smallest && customHits >= CUSTOM_HITS_NEEDED) {
    setStatus("🎉 発見! (" + target.label + ")", COLOR.ok);
    handleFound(target.name, smallest.p, smallest.r);
  } else {
    setStatus("🔍 探索中... (" + target.label + ")\n最大の一致度: " + Math.round(best.p * 100) + "%", COLOR.dim);
  }
}

/* ================= 発見したとき ================= */
function cropToDataURL(videoEl, r, maxSide) {
  const ratio = r.width / r.height;
  const w = ratio >= 1 ? maxSide : Math.max(1, Math.round(maxSide * ratio));
  const h = ratio >= 1 ? Math.max(1, Math.round(maxSide / ratio)) : maxSide;
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  c.getContext("2d").drawImage(videoEl, r.originX, r.originY, r.width, r.height, 0, 0, w, h);
  return c.toDataURL("image/jpeg", 0.7);
}

function handleFound(label, score, box) {
  const now = Date.now();
  if (now - lastSaveTime < SAVE_INTERVAL_MS) return;
  lastSaveTime = now;

  const rec = {
    type: "found",
    label,
    score: Math.round(score * 100),
    place: $("placeInput").value.trim(),
    time: new Date().toISOString(),
    image: cropToDataURL(video, box, 160)
  };
  addGalleryItem(rec);
  sendToPC(rec);
}

function addGalleryItem(rec) {
  const g = $("imageGallery");
  const item = document.createElement("div");
  item.className = "gallery-item";
  const img = document.createElement("img");
  img.src = rec.image; img.alt = rec.label;
  const cap = document.createElement("div");
  cap.className = "cap";
  cap.textContent = rec.label + " " + rec.score + "%\n" + (rec.place || "場所未設定") + "\n" +
    new Date(rec.time).toLocaleTimeString("ja-JP");
  item.appendChild(img); item.appendChild(cap);
  g.prepend(item);
  while (g.children.length > 30) g.lastChild.remove();
}

/* ================= PCへの接続(任意) ================= */
let peer = null, conn = null;

function connectToPC() {
  const id = $("pcIdInput").value.trim().toUpperCase();
  if (!id) { setConn("PC側のIDを入力してください", COLOR.err); return; }
  if (typeof Peer === "undefined") { setConn("通信ライブラリを読み込めていません(ネットワークを確認)", COLOR.err); return; }
  $("pcIdInput").value = id;
  try { localStorage.setItem(PC_ID_KEY, id); } catch (e) {}
  setConn("接続中...", COLOR.warn);

  if (conn) { try { conn.close(); } catch (e) {} conn = null; }
  const open = () => {
    const c = peer.connect(id, { reliable: true });
    conn = c;
    const timer = setTimeout(() => {
      if (!c.open) setConn("つながりません。IDと、PC側の画面(pc.html)を確認してください", COLOR.err);
    }, 10000);
    c.on("open", () => { clearTimeout(timer); setConn("✅ PCに接続しました", COLOR.ok); });
    c.on("close", () => { setConn("切断されました", COLOR.dim); });
    c.on("error", (e) => { clearTimeout(timer); setConn("エラー: " + (e.type || e), COLOR.err); });
  };

  if (!peer || peer.destroyed) {
    peer = new Peer();
    peer.on("error", (err) => {
      setConn(err.type === "peer-unavailable" ? "そのIDのPCが見つかりません" : "エラー: " + err.type, COLOR.err);
    });
  }
  if (peer.open) open();
  else peer.once("open", open);
}

function sendToPC(payload) {
  try { if (conn && conn.open) conn.send(payload); }
  catch (e) { console.error("PCへの送信に失敗", e); }
}

/* ================= 物体登録モード ================= */
let objSamples = [], bgSamples = [];
let objPress = 0, bgPress = 0;

function loadCustomModels() {
  try { customModels = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}") || {}; }
  catch (e) { customModels = {}; }
}
function saveCustomModels() {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(customModels)); return true; }
  catch (e) { return false; }
}

async function startRegCamera() {
  setRegStatus("特徴抽出AIをロード中...", COLOR.warn);
  try { await ensureEmbedder(); }
  catch (e) { setRegStatus("エラー: 特徴抽出AIをロードできません\n" + String(e.message || e), COLOR.err); return; }
  try { await openCamera(regVideo); }
  catch (e) { setRegStatus("エラー: カメラを起動できません\n" + cameraErrorMessage(e), COLOR.err); return; }
  setRegStatus("カメラ起動OK。物体を黄色い枠に入れて「物体撮影」、物体のない背景で「背景撮影」を押してください", COLOR.ok);
}

function stopRegCamera() {
  stopStream(regVideo);
}

function regReady() {
  if (!regVideo.srcObject || !regVideo.videoWidth || !embedder) {
    setRegStatus("先に「登録カメラ起動」を押してください", COLOR.err);
    return false;
  }
  return true;
}

function guideCrop(scale) {
  const vw = regVideo.videoWidth, vh = regVideo.videoHeight;
  const cw = regVideo.clientWidth || vw;
  // 画面上の160pxの枠を、動画の画素に換算する
  const side = Math.min(vw, vh, (160 / cw) * vw * scale);
  return { originX: (vw - side) / 2, originY: (vh - side) / 2, width: side, height: side };
}

function addThumb(containerId) {
  const img = document.createElement("img");
  img.src = tmp.toDataURL("image/jpeg", 0.6);
  $(containerId).appendChild(img);
}

function updateCounts() {
  $("objCount").textContent = objPress;
  $("bgCount").textContent = bgPress;
}

function captureObject() {
  if (!regReady()) return;
  [1.0, 1.3, 0.8].forEach((scale, i) => {
    drawCrop(regVideo, guideCrop(scale));
    objSamples.push(getEmbedding(tmp));
    if (i === 0) addThumb("objThumbnails");
  });
  objPress++;
  updateCounts();
  setRegStatus("物体を撮影しました(" + objPress + "回)。角度や距離を変えて、あと数回撮ってください", COLOR.ok);
}

function captureBackground() {
  if (!regReady()) return;
  const crops = makeCrops(regVideo.videoWidth, regVideo.videoHeight);
  const picks = [];
  while (picks.length < Math.min(4, crops.length)) {
    const c = crops[Math.floor(Math.random() * crops.length)];
    if (!picks.includes(c)) picks.push(c);
  }
  picks.forEach((c, i) => {
    drawCrop(regVideo, c);
    bgSamples.push(getEmbedding(tmp));
    if (i === 0) addThumb("bgThumbnails");
  });
  bgPress++;
  updateCounts();
  setRegStatus("背景を撮影しました(" + bgPress + "回)。物体を置いていない場所を、向きを変えて撮ってください", COLOR.ok);
}

// 学習: 「物体か、背景か」を分ける小さな分類器(ロジスティック回帰)
async function trainLogistic(samples) {
  const D = samples[0].v.length, N = samples.length;
  const nPos = samples.filter((s) => s.y === 1).length;
  const nNeg = N - nPos;
  const wPos = N / (2 * nPos), wNeg = N / (2 * nNeg);
  const w = new Float64Array(D);
  let b = 0;
  const LR = 5, L2 = 1e-4, EPOCHS = 400;

  for (let ep = 0; ep < EPOCHS; ep++) {
    const gw = new Float64Array(D);
    let gb = 0;
    for (const s of samples) {
      let z = b;
      for (let i = 0; i < D; i++) z += w[i] * s.v[i];
      const e = ((sigmoid(z) - s.y) * (s.y === 1 ? wPos : wNeg)) / N;
      for (let i = 0; i < D; i++) gw[i] += e * s.v[i];
      gb += e;
    }
    for (let i = 0; i < D; i++) w[i] -= LR * (gw[i] + L2 * w[i]);
    b -= LR * gb;
    if (ep % 50 === 49) await sleep(0);
  }

  let ok = 0;
  for (const s of samples) {
    let z = b;
    for (let i = 0; i < D; i++) z += w[i] * s.v[i];
    if ((sigmoid(z) >= 0.5) === (s.y === 1)) ok++;
  }
  return { w, b, acc: ok / N };
}

async function trainAndRegister() {
  const name = $("registerName").value.trim();
  if (!name) { setRegStatus("登録する名前を入力してください", COLOR.err); return; }
  if (objPress < 3 || bgPress < 3) {
    setRegStatus("物体と背景を、それぞれ3回以上撮影してください\n(いま: 物体" + objPress + "回・背景" + bgPress + "回)", COLOR.err);
    return;
  }
  setRegStatus("学習中...", COLOR.warn);
  await sleep(30);

  const samples = objSamples.map((v) => ({ v, y: 1 })).concat(bgSamples.map((v) => ({ v, y: 0 })));
  const m = await trainLogistic(samples);

  // 同じ名前(大文字小文字・かな違いを含む)があれば置き換える
  for (const k of Object.keys(customModels)) if (normKey(k) === normKey(name)) delete customModels[k];
  customModels[name] = {
    w: Array.from(m.w, (x) => Math.round(x * 1000) / 1000),
    b: Math.round(m.b * 1000) / 1000,
    n: samples.length
  };
  if (!saveCustomModels()) {
    setRegStatus("学習はできましたが、ブラウザに保存できませんでした(容量不足の可能性)", COLOR.err);
  } else {
    setRegStatus("✅ 「" + name + "」を登録しました(学習データへの一致率 " + Math.round(m.acc * 100) +
      "%)\n探索モードで、この名前を入力して使えます", COLOR.ok);
  }

  objSamples = []; bgSamples = []; objPress = 0; bgPress = 0;
  $("objThumbnails").replaceChildren(); $("bgThumbnails").replaceChildren();
  $("registerName").value = "";
  updateCounts();
  renderRegisteredList();
  renderQuickSelect();
}

function renderRegisteredList() {
  const box = $("registeredList");
  box.replaceChildren();
  const names = Object.keys(customModels);
  if (!names.length) {
    const p = document.createElement("div");
    p.className = "note";
    p.textContent = "まだ登録されていません";
    box.appendChild(p);
    return;
  }
  names.forEach((name) => {
    const row = document.createElement("div");
    row.className = "reg-item";
    const label = document.createElement("div");
    label.className = "name";
    label.textContent = name;
    const use = document.createElement("button");
    use.className = "btn blue small";
    use.textContent = "探す";
    use.addEventListener("click", async () => {
      $("targetObject").value = name;
      switchTab("search");
      await applyTarget();
    });
    const del = document.createElement("button");
    del.className = "btn red small";
    del.textContent = "削除";
    del.addEventListener("click", () => {
      delete customModels[name];
      saveCustomModels();
      renderRegisteredList();
      renderQuickSelect();
    });
    row.append(label, use, del);
    box.appendChild(row);
  });
}

function clearCustomModels() {
  if (!confirm("登録した物体をすべて消去します。よろしいですか?")) return;
  customModels = {};
  saveCustomModels();
  renderRegisteredList();
  renderQuickSelect();
  setRegStatus("登録した物体を消去しました", COLOR.dim);
}

/* ================= タブの切り替え ================= */
function switchTab(name) {
  document.querySelectorAll(".tab-btn").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  $("searchTab").classList.toggle("active", name === "search");
  $("registerTab").classList.toggle("active", name === "register");
  stopCamera();       // カメラは同時に2つ使えないので、切り替えるときに止める
  stopRegCamera();
}

/* ================= ボタンの登録と起動 ================= */
document.querySelectorAll(".tab-btn").forEach((b) => b.addEventListener("click", () => switchTab(b.dataset.tab)));
$("connectButton").addEventListener("click", connectToPC);
$("targetButton").addEventListener("click", applyTarget);
$("targetObject").addEventListener("keydown", (e) => { if (e.key === "Enter") applyTarget(); });
$("modelSelect").addEventListener("change", changeModel);
$("startButton").addEventListener("click", startCamera);
$("stopButton").addEventListener("click", stopCamera);
$("clearGalleryButton").addEventListener("click", () => $("imageGallery").replaceChildren());
$("startRegButton").addEventListener("click", startRegCamera);
$("stopRegButton").addEventListener("click", stopRegCamera);
$("captureObjectButton").addEventListener("click", captureObject);
$("captureBgButton").addEventListener("click", captureBackground);
$("trainButton").addEventListener("click", trainAndRegister);
$("clearCustomButton").addEventListener("click", clearCustomModels);

loadCustomModels();
renderRegisteredList();
renderQuickSelect();
try { const saved = localStorage.getItem(PC_ID_KEY); if (saved) $("pcIdInput").value = saved; } catch (e) {}

// QRコード(または共有リンク)から開かれたとき: ?pc=ID の値で自動的につなぐ
(function () {
  const raw = new URLSearchParams(location.search).get("pc");
  if (!raw) return;
  const id = raw.trim().toUpperCase();
  if (!/^[A-Z0-9-]{3,20}$/.test(id)) return;
  $("pcIdInput").value = id;
  history.replaceState(null, "", location.pathname);
  connectToPC();
})();

if ("serviceWorker" in navigator) navigator.serviceWorker.register("sw.js").catch(() => {});

initializeAI();
