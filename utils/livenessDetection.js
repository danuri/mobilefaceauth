import * as FaceDetector from "expo-face-detector";
import * as ImageManipulator from "expo-image-manipulator";
import { Buffer } from "buffer";
import decodeJpeg from "jpeg-js";

// ─── Konfigurasi ────────────────────────────────────────────────
const LIVENESS_FRAME_COUNT = 5;
const LIVENESS_FRAME_DELAY_MS = 300; // ms antar frame
const LIVENESS_THRESHOLD = 0.35; // skor minimum keseluruhan
const LIVENESS_MIN_CHECKS_PASSED = 3; // minimal 3 dari 5 check harus lolos

// ─── Helper: Standard Deviation ─────────────────────────────────
const stdDev = (values) => {
  if (!values.length) return 0;
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const variance =
    values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / values.length;
  return Math.sqrt(variance);
};

const variance = (values) => {
  if (!values.length) return 0;
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  return values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / values.length;
};

// ─── 1. Capture Multiple Frames ─────────────────────────────────
const captureFrames = async (cameraRef, count = LIVENESS_FRAME_COUNT, delayMs = LIVENESS_FRAME_DELAY_MS) => {
  const frames = [];
  for (let i = 0; i < count; i++) {
    const photo = await cameraRef.current.takePictureAsync({ quality: 0.5 });
    frames.push(photo);
    if (i < count - 1) {
      await new Promise((resolve) => setTimeout(resolve, delayMs));
    }
  }
  return frames;
};

// ─── 2. Detect Faces with Full Landmarks & Classifications ──────
const detectFaceFull = async (photoUri) => {
  return await FaceDetector.detectFacesAsync(photoUri, {
    mode: FaceDetector.FaceDetectorMode.accurate,
    detectLandmarks: FaceDetector.FaceDetectorLandmarks.all,
    runClassifications: FaceDetector.FaceDetectorClassifications.all,
  });
};

// ═══════════════════════════════════════════════════════════════
//  CHECK 1: Micro-Movement Analysis
//  Wajah asli pasti ada gerakan mikro (napas, micro-expression).
//  Foto diam → variasi landmark ≈ 0
// ═══════════════════════════════════════════════════════════════
const checkMicroMovement = (detections) => {
  const landmarkKeys = [
    "leftEyePosition",
    "rightEyePosition",
    "noseBasePosition",
    "leftMouthPosition",
    "rightMouthPosition",
  ];

  let totalStd = 0;
  let count = 0;

  for (const key of landmarkKeys) {
    const xValues = detections
      .map((d) => d.faces[0]?.[key]?.x)
      .filter((v) => v != null);
    const yValues = detections
      .map((d) => d.faces[0]?.[key]?.y)
      .filter((v) => v != null);

    if (xValues.length >= 2) {
      totalStd += stdDev(xValues) + stdDev(yValues);
      count += 2;
    }
  }

  const avgVariation = count > 0 ? totalStd / count : 0;

  return {
    name: "Micro-Movement",
    score: Math.min(avgVariation / 2.0, 1.0),
    detail: `variasi=${avgVariation.toFixed(2)}px`,
    passed: avgVariation > 0.3,
  };
};

// ═══════════════════════════════════════════════════════════════
//  CHECK 2: Eye Blink / Eye Openness Variance
//  Mata manusia berkedip secara alami → probabilitas mata terbuka
//  berfluktuasi antar frame. Foto → konstan.
// ═══════════════════════════════════════════════════════════════
const checkEyeBlink = (detections) => {
  const leftProbs = detections
    .map((d) => d.faces[0]?.leftEyeOpenProbability)
    .filter((v) => v != null);
  const rightProbs = detections
    .map((d) => d.faces[0]?.rightEyeOpenProbability)
    .filter((v) => v != null);

  const leftVar = variance(leftProbs);
  const rightVar = variance(rightProbs);
  const avgVar = (leftVar + rightVar) / 2;

  return {
    name: "Eye Blink",
    score: Math.min(avgVar * 10, 1.0),
    detail: `var=${avgVar.toFixed(4)}`,
    passed: avgVar > 0.005,
  };
};

// ═══════════════════════════════════════════════════════════════
//  CHECK 3: Head Angle Variance (Roll / Yaw)
//  Kepala manusia punya gerakan halus → roll dan yaw berubah
//  sedikit antar frame. Foto statis → 0.
// ═══════════════════════════════════════════════════════════════
const checkAngleVariance = (detections) => {
  const rolls = detections
    .map((d) => d.faces[0]?.rollAngle)
    .filter((v) => v != null);
  const yaws = detections
    .map((d) => d.faces[0]?.yawAngle)
    .filter((v) => v != null);

  const rollStd = stdDev(rolls);
  const yawStd = stdDev(yaws);
  const avgStd = (rollStd + yawStd) / 2;

  return {
    name: "Angle Variance",
    score: Math.min(avgStd / 2.0, 1.0),
    detail: `roll_std=${rollStd.toFixed(2)}°, yaw_std=${yawStd.toFixed(2)}°`,
    passed: avgStd > 0.2,
  };
};

// ═══════════════════════════════════════════════════════════════
//  CHECK 4: Texture / Moiré Pattern Analysis
//  Layar HP/monitor menghasilkan pola moiré saat difoto.
//  Laplacian edge & periodicity analysis bisa mendeteksi ini.
// ═══════════════════════════════════════════════════════════════
const checkTexture = async (photoUri, faceData) => {
  try {
    // Crop area wajah, resize kecil untuk efisiensi
    const cropped = await ImageManipulator.manipulateAsync(
      photoUri,
      [
        {
          crop: {
            originX: Math.max(0, faceData.bounds.origin.x),
            originY: Math.max(0, faceData.bounds.origin.y),
            width: faceData.bounds.size.width,
            height: faceData.bounds.size.height,
          },
        },
        { resize: { width: 64, height: 64 } },
      ],
      { format: "jpeg", base64: true }
    );

    const rawData = Buffer.from(cropped.base64, "base64");
    const { data, width, height } = decodeJpeg.decode(rawData, {
      useTArray: true,
    });

    // ── Laplacian Edge Strength ──
    let edgeSum = 0;
    let edgeCount = 0;
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const idx = (y * width + x) * 4;
        const center = data[idx]; // R channel
        const left = data[idx - 4];
        const right = data[idx + 4];
        const up = data[idx - width * 4];
        const down = data[idx + width * 4];
        edgeSum += Math.abs(4 * center - left - right - up - down);
        edgeCount++;
      }
    }
    const avgEdge = edgeSum / edgeCount;

    // ── Horizontal Periodicity (moiré indicator) ──
    let periodicScore = 0;
    for (let y = 0; y < height; y++) {
      let signChanges = 0;
      let prevDiff = 0;
      for (let x = 1; x < width; x++) {
        const idx = (y * width + x) * 4;
        const prevIdx = (y * width + (x - 1)) * 4;
        const diff = data[idx] - data[prevIdx];
        if (prevDiff !== 0 && Math.sign(diff) !== Math.sign(prevDiff)) {
          signChanges++;
        }
        prevDiff = diff;
      }
      periodicScore += signChanges / (width - 1);
    }
    periodicScore /= height;

    const moireDetected = periodicScore > 0.7;

    return {
      name: "Texture",
      score: moireDetected ? 0 : Math.min(avgEdge / 30, 1.0),
      detail: `edge=${avgEdge.toFixed(1)}, periodic=${periodicScore.toFixed(2)}, moiré=${moireDetected ? "YA" : "TIDAK"}`,
      passed: !moireDetected && avgEdge > 5 && avgEdge < 80,
    };
  } catch (e) {
    return {
      name: "Texture",
      score: 0.5,
      detail: `error: ${e.message}`,
      passed: true, // skip check on error
    };
  }
};

// ═══════════════════════════════════════════════════════════════
//  CHECK 5: Color Distribution Analysis
//  Wajah asli punya distribusi warna kulit alami.
//  Layar/foto cetak punya pola warna yang lebih uniform/skewed.
// ═══════════════════════════════════════════════════════════════
const checkColorDistribution = async (photoUri, faceData) => {
  try {
    const cropped = await ImageManipulator.manipulateAsync(
      photoUri,
      [
        {
          crop: {
            originX: Math.max(0, faceData.bounds.origin.x),
            originY: Math.max(0, faceData.bounds.origin.y),
            width: faceData.bounds.size.width,
            height: faceData.bounds.size.height,
          },
        },
        { resize: { width: 32, height: 32 } },
      ],
      { format: "jpeg", base64: true }
    );

    const rawData = Buffer.from(cropped.base64, "base64");
    const { data, width, height } = decodeJpeg.decode(rawData, {
      useTArray: true,
    });

    let rSum = 0,
      gSum = 0,
      bSum = 0;
    let rSqSum = 0,
      gSqSum = 0,
      bSqSum = 0;
    const pixelCount = width * height;

    for (let i = 0; i < pixelCount; i++) {
      const r = data[i * 4];
      const g = data[i * 4 + 1];
      const b = data[i * 4 + 2];
      rSum += r;
      gSum += g;
      bSum += b;
      rSqSum += r * r;
      gSqSum += g * g;
      bSqSum += b * b;
    }

    const rMean = rSum / pixelCount;
    const gMean = gSum / pixelCount;
    const bMean = bSum / pixelCount;

    const rVar = rSqSum / pixelCount - rMean * rMean;
    const gVar = gSqSum / pixelCount - gMean * gMean;
    const bVar = bSqSum / pixelCount - bMean * bMean;

    const avgVariance = (rVar + gVar + bVar) / 3;

    return {
      name: "Color Distribution",
      score: avgVariance > 100 ? 1.0 : avgVariance / 100,
      detail: `var=${avgVariance.toFixed(1)}, R=${rMean.toFixed(0)} G=${gMean.toFixed(0)} B=${bMean.toFixed(0)}`,
      passed: avgVariance > 50,
    };
  } catch (e) {
    return {
      name: "Color Distribution",
      score: 0.5,
      detail: `error: ${e.message}`,
      passed: true, // skip check on error
    };
  }
};

// ═══════════════════════════════════════════════════════════════
//  MAIN: performLivenessCheck
//  Orkestrasi semua check dan hitung skor akhir.
// ═══════════════════════════════════════════════════════════════
/**
 * Jalankan passive liveness detection.
 *
 * @param {React.RefObject} cameraRef - Ref ke CameraView
 * @param {(msg: string) => void} onProgress - Callback progress log
 * @returns {Promise<{
 *   isLive: boolean,
 *   confidence: number,
 *   checks: object[],
 *   checksPassedCount: number,
 *   message: string,
 *   bestFrame: object|null,
 *   bestDetection: object|null,
 * }>}
 */
export const performLivenessCheck = async (cameraRef, onProgress) => {
  const result = {
    isLive: false,
    confidence: 0,
    checks: [],
    checksPassedCount: 0,
    message: "",
    bestFrame: null,
    bestDetection: null,
  };

  try {
    // ── Step 1: Capture beberapa frame ──
    onProgress?.("📸 Mengambil beberapa frame untuk analisis...");
    const frames = await captureFrames(cameraRef);

    // ── Step 2: Deteksi wajah lengkap di setiap frame ──
    onProgress?.("🔍 Mendeteksi landmark wajah...");
    const detections = [];
    for (const frame of frames) {
      const detection = await detectFaceFull(frame.uri);
      if (!detection.faces.length) {
        result.message = "❌ Wajah tidak terdeteksi di semua frame.";
        return result;
      }
      detections.push(detection);
    }

    // ── Step 3: Jalankan 5 check ──
    onProgress?.("🧪 Menjalankan analisis liveness...");

    const checks = [];

    // Check 1: Micro-Movement
    checks.push(checkMicroMovement(detections));

    // Check 2: Eye Blink
    checks.push(checkEyeBlink(detections));

    // Check 3: Angle Variance
    checks.push(checkAngleVariance(detections));

    // Ambil frame tengah untuk analisis pixel
    const midIdx = Math.floor(frames.length / 2);
    const midFrame = frames[midIdx];
    const midDetection = detections[midIdx];
    const midFace = midDetection.faces[0];

    onProgress?.("🎨 Menganalisis tekstur & warna...");

    // Check 4: Texture
    checks.push(await checkTexture(midFrame.uri, midFace));

    // Check 5: Color Distribution
    checks.push(await checkColorDistribution(midFrame.uri, midFace));

    // ── Step 4: Hitung skor keseluruhan ──
    // Bobot: Micro-Movement 30%, Eye Blink 20%, Angle 15%, Texture 20%, Color 15%
    const weights = [0.30, 0.20, 0.15, 0.20, 0.15];
    const overallScore = checks.reduce(
      (sum, check, i) => sum + check.score * weights[i],
      0
    );

    const passedCount = checks.filter((c) => c.passed).length;

    result.checks = checks;
    result.confidence = overallScore;
    result.checksPassedCount = passedCount;
    result.isLive =
      passedCount >= LIVENESS_MIN_CHECKS_PASSED &&
      overallScore > LIVENESS_THRESHOLD;
    result.bestFrame = midFrame;
    result.bestDetection = midDetection;

    result.message = result.isLive
      ? `✅ Liveness OK (${passedCount}/5, skor: ${(overallScore * 100).toFixed(1)}%)`
      : `❌ Liveness GAGAL (${passedCount}/5, skor: ${(overallScore * 100).toFixed(1)}%)`;

    return result;
  } catch (e) {
    result.message = "❌ Error liveness: " + e.message;
    return result;
  }
};
