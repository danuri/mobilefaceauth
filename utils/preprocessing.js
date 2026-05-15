import { Buffer } from "buffer";
import decodeJpeg from "jpeg-js";
import * as ImageManipulator from "expo-image-manipulator";
import * as FaceDetector from "expo-face-detector";

/**
 * Hitung Cosine Similarity antara dua vector
 * @param {number[]} a - Vector pertama
 * @param {number[]} b - Vector kedua
 * @returns {number} Nilai similarity (0-1)
 */
export const cosineSimilarity = (a, b) => {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
};

/**
 * Deteksi wajah dari foto
 * @param {string} photoUri - URI file foto
 * @returns {Promise<any>} Hasil deteksi wajah
 */
export const detectFace = async (photoUri) => {
  const detection = await FaceDetector.detectFacesAsync(photoUri, {
    mode: FaceDetector.FaceDetectorMode.fast,
  });
  return detection;
};

/**
 * Crop dan resize wajah ke 112x112
 * @param {string} photoUri - URI file foto asli
 * @param {any} faceData - Data bounds wajah dari deteksi
 * @returns {Promise<any>} Gambar yang sudah di-crop dan di-resize
 */
export const cropAndResizeFace = async (photoUri, faceData) => {
  const processed = await ImageManipulator.manipulateAsync(
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
      { resize: { width: 112, height: 112 } },
    ],
    { format: "jpeg", base64: true }
  );
  return processed;
};

/**
 * Normalisasi pixel dari JPEG ke tensor sesuai spek MobileFaceNet
 * Range pixel akan diubah dari [0, 255] -> [-1, 1]
 * @param {string} base64Data - Base64 string dari gambar JPEG
 * @returns {Float32Array} Tensor input untuk model
 */
export const preprocessPixels = (base64Data) => {
  const rawImageData = Buffer.from(base64Data, "base64");
  const { width, height, data } = decodeJpeg.decode(rawImageData, { useTArray: true });

  const inputTensor = new Float32Array(width * height * 3);
  for (let i = 0; i < width * height; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    // Normalisasi ke range -1 s/d 1 sesuai spek MobileFaceNet
    inputTensor[i * 3 + 0] = (r - 127.5) / 128.0;
    inputTensor[i * 3 + 1] = (g - 127.5) / 128.0;
    inputTensor[i * 3 + 2] = (b - 127.5) / 128.0;
  }

  return inputTensor;
};

/**
 * Jalankan preprocessing lengkap (crop + resize + normalisasi pixel)
 * @param {string} photoUri - URI file foto asli
 * @param {any} faceData - Data bounds wajah
 * @returns {Promise<Float32Array>} Tensor yang siap untuk inference
 */
export const preprocessFaceImage = async (photoUri, faceData) => {
  // Step 1: Crop & Resize
  const cropped = await cropAndResizeFace(photoUri, faceData);

  // Step 2: Normalisasi Pixel
  const inputTensor = preprocessPixels(cropped.base64);

  return inputTensor;
};
