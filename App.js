import React, { useEffect, useRef, useState } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ScrollView,
  Alert,
  ActivityIndicator,
} from "react-native";
import { CameraView, useCameraPermissions } from "expo-camera";
import * as SecureStore from "expo-secure-store";
import * as FileSystem from "expo-file-system/legacy";
import { loadTensorflowModel } from "react-native-fast-tflite";
import { Buffer } from "buffer";
import {
  cosineSimilarity,
  detectFace,
  preprocessFaceImage,
} from "./utils/preprocessing";
import { performLivenessCheck } from "./utils/livenessDetection";

global.Buffer = Buffer;

const MODEL_FILE = require("./assets/mobilefacenet.tflite");
const EMBEDDING_PATH = `${FileSystem.documentDirectory}face_master_data.json`;

export default function App() {
  const [permission, requestPermission] = useCameraPermissions();
  const [mode, setMode] = useState(null);
  const [log, setLog] = useState([]);
  const [model, setModel] = useState(null);
  const [loading, setLoading] = useState(false);
  const [livenessStatus, setLivenessStatus] = useState(null); // null | "checking" | "passed" | "failed"
  const cameraRef = useRef(null);

  useEffect(() => {
    (async () => {
      console.log("🛠 Inisialisasi Aplikasi...");
      const { status } = await requestPermission();
      try {
        const tflite = await loadTensorflowModel(MODEL_FILE);
        setModel(tflite);
        addLog("🚀 Model MobileFaceNet siap.");
      } catch (e) {
        console.error("❌ Model Load Error:", e);
        addLog("❌ Gagal load model.");
      }
    })();
  }, []);

  const addLog = (text) => {
    setLog((prev) => [`[${new Date().toLocaleTimeString()}] ${text}`, ...prev]);
  };

  /**
   * captureAndProcess — sekarang dengan Passive Liveness Detection.
   *
   * Alur baru:
   * 1. Ambil 5 frame berturut-turut (jeda 300ms per frame)
   * 2. Jalankan 5 passive liveness check:
   *    - Micro-movement (landmark variance antar frame)
   *    - Eye blink (probabilitas mata berkedip)
   *    - Angle variance (roll/yaw kepala)
   *    - Texture analysis (deteksi moiré / pola layar)
   *    - Color distribution (distribusi warna kulit)
   * 3. Jika liveness lolos → lanjut extract face embedding
   * 4. Jika gagal → tolak (anti-spoofing)
   */
  const captureAndProcess = async () => {
    if (!cameraRef.current || !model) return null;

    setLoading(true);
    setLivenessStatus("checking");
    console.group("🚀 [Face Recognition + Liveness Process]");

    try {
      // ═══════════════════════════════════════════════
      //  STEP 1: PASSIVE LIVENESS DETECTION
      // ═══════════════════════════════════════════════
      addLog("🔒 Memulai passive liveness detection...");
      console.time("🔒 Liveness Detection");

      const liveness = await performLivenessCheck(cameraRef, (msg) =>
        addLog(msg)
      );

      console.timeEnd("🔒 Liveness Detection");

      // Log detail setiap check
      for (const check of liveness.checks) {
        const icon = check.passed ? "✅" : "❌";
        addLog(`  ${icon} ${check.name}: ${check.detail}`);
        console.log(`  ${icon} ${check.name}:`, check.detail, `score=${check.score.toFixed(3)}`);
      }

      addLog(liveness.message);

      if (!liveness.isLive) {
        setLivenessStatus("failed");
        Alert.alert(
          "⚠️ Liveness Gagal",
          "Sistem mendeteksi bahwa wajah bukan dari orang hidup.\n\n" +
            "Pastikan:\n" +
            "• Anda langsung di depan kamera\n" +
            "• Bukan foto atau layar HP\n" +
            "• Pencahayaan cukup",
          [{ text: "OK" }]
        );
        console.groupEnd();
        return null;
      }

      setLivenessStatus("passed");
      addLog("🟢 Liveness terverifikasi, melanjutkan face recognition...");

      // ═══════════════════════════════════════════════
      //  STEP 2: FACE EMBEDDING EXTRACTION
      //  Gunakan frame terbaik dari liveness detection
      //  (tidak perlu capture ulang → lebih efisien)
      // ═══════════════════════════════════════════════
      const photo = liveness.bestFrame;
      const face = liveness.bestDetection.faces[0];

      // 2a. Crop, Resize & Pre-processing Pixel
      console.time("✂️ Crop & Preprocessing");
      const inputTensor = await preprocessFaceImage(photo.uri, face);
      console.timeEnd("✂️ Crop & Preprocessing");

      // 2b. Jalankan AI Inference
      console.time("🧠 AI Inference");
      const output = await model.run([inputTensor]);
      const embedding = Array.isArray(output) ? output[0] : output;
      console.timeEnd("🧠 AI Inference");

      // Log Vector Sample untuk Validasi
      const vectorArray = Object.values(embedding);
      console.log(
        "📊 Vector Sample (5 pertama):",
        JSON.stringify(vectorArray.slice(0, 5))
      );

      console.groupEnd();
      return vectorArray;
    } catch (e) {
      console.error("🚨 Error Detail:", e);
      console.groupEnd();
      addLog("❌ Error: " + e.message);
      return null;
    } finally {
      setLoading(false);
      // Reset liveness status after a delay
      setTimeout(() => setLivenessStatus(null), 3000);
    }
  };

  const onRegister = async () => {
    console.log("🚀 [Register Process]");
    const embedding = await captureAndProcess();
    if (!embedding) return;

    try {
      await FileSystem.writeAsStringAsync(
        EMBEDDING_PATH,
        JSON.stringify(embedding)
      );
      await SecureStore.setItemAsync("face_registered_status", "true");
      addLog("💾 Master wajah berhasil disimpan.");
      Alert.alert("Berhasil", "Wajah Anda telah terdaftar.");
      setMode(null);
    } catch (e) {
      addLog("❌ Gagal simpan: " + e.message);
    }
  };

  const onVerify = async () => {
    console.log("🚀 [Verify Process]");
    const embedding = await captureAndProcess();
    if (!embedding) return;

    try {
      const isRegistered = await SecureStore.getItemAsync(
        "face_registered_status"
      );
      if (!isRegistered) {
        addLog("⚠️ Belum ada wajah terdaftar.");
        return;
      }

      const savedJson = await FileSystem.readAsStringAsync(EMBEDDING_PATH);
      const savedEmbedding = JSON.parse(savedJson);

      const score = cosineSimilarity(embedding, savedEmbedding);
      console.log("🔢 Hasil Similarity:", score);
      addLog(`🔢 Skor Kemiripan: ${score.toFixed(4)}`);

      if (score > 0.75) {
        addLog("✅ VERIFIKASI BERHASIL");
        Alert.alert("Berhasil", `Wajah Cocok! (${(score * 100).toFixed(1)}%)`);
      } else {
        addLog("❌ TIDAK COCOK");
        Alert.alert("Ditolak", "Wajah tidak dikenal.");
      }
    } catch (e) {
      addLog("❌ Gagal verifikasi: " + e.message);
    }
  };

  // ─── Liveness Status Badge ────────────────────────────────
  const renderLivenessBadge = () => {
    if (!livenessStatus) return null;

    const config = {
      checking: { text: "🔍 Checking Liveness...", bg: "#f39c12" },
      passed: { text: "🟢 LIVE", bg: "#27ae60" },
      failed: { text: "🔴 SPOOF DETECTED", bg: "#e74c3c" },
    };

    const { text, bg } = config[livenessStatus];

    return (
      <View style={[styles.livenessBadge, { backgroundColor: bg }]}>
        <Text style={styles.livenessBadgeText}>{text}</Text>
      </View>
    );
  };

  if (!permission?.granted) {
    return (
      <View style={styles.center}>
        <TouchableOpacity onPress={requestPermission} style={styles.btn}>
          <Text style={styles.btnText}>Izinkan Kamera</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {!mode ? (
        <View style={styles.menu}>
          <Text style={styles.title}>Face ID Attendance</Text>
          <Text style={styles.subtitle}>🔒 Passive Liveness Detection</Text>
          <TouchableOpacity
            style={styles.btn}
            onPress={() => setMode("register")}
          >
            <Text style={styles.btnText}>Daftar Wajah</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.btn, { backgroundColor: "#27ae60", marginTop: 20 }]}
            onPress={() => setMode("verify")}
          >
            <Text style={styles.btnText}>Verifikasi Wajah</Text>
          </TouchableOpacity>
        </View>
      ) : (
        <>
          <View style={styles.cameraContainer}>
            <CameraView style={styles.camera} facing="front" ref={cameraRef} />
            {renderLivenessBadge()}
          </View>
          <View style={styles.overlay}>
            <TouchableOpacity
              disabled={loading}
              style={[
                styles.captureBtn,
                loading && { backgroundColor: "#555" },
              ]}
              onPress={mode === "register" ? onRegister : onVerify}
            >
              {loading ? (
                <View style={styles.loadingContainer}>
                  <ActivityIndicator color="#fff" />
                  <Text style={styles.loadingText}>Analisis Liveness...</Text>
                </View>
              ) : (
                <Text style={styles.btnText}>
                  {mode === "register" ? "Ambil Master" : "Scan Verifikasi"}
                </Text>
              )}
            </TouchableOpacity>
            <TouchableOpacity
              onPress={() => {
                setMode(null);
                setLivenessStatus(null);
              }}
              style={{ marginTop: 20 }}
            >
              <Text style={{ color: "#aaa" }}>Batal</Text>
            </TouchableOpacity>
          </View>
          <View style={styles.logArea}>
            <ScrollView>
              {log.map((item, index) => (
                <Text key={index} style={styles.logText}>
                  {item}
                </Text>
              ))}
            </ScrollView>
          </View>
        </>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#000" },
  center: { flex: 1, justifyContent: "center", alignItems: "center" },
  menu: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    padding: 30,
  },
  title: {
    color: "#fff",
    fontSize: 24,
    fontWeight: "bold",
    marginBottom: 8,
  },
  subtitle: {
    color: "#aaa",
    fontSize: 14,
    marginBottom: 50,
  },
  cameraContainer: {
    flex: 2,
    position: "relative",
  },
  camera: { flex: 1 },
  overlay: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#0a0a0a",
  },
  btn: {
    backgroundColor: "#2980b9",
    padding: 18,
    borderRadius: 12,
    width: "100%",
    alignItems: "center",
  },
  btnText: { color: "#fff", fontWeight: "bold", fontSize: 16 },
  captureBtn: {
    backgroundColor: "#c0392b",
    padding: 22,
    borderRadius: 60,
    width: "85%",
    alignItems: "center",
  },
  logArea: {
    flex: 0.8,
    backgroundColor: "#000",
    padding: 12,
    borderTopWidth: 1,
    borderTopColor: "#222",
  },
  logText: {
    color: "#2ecc71",
    fontSize: 12,
    fontFamily: "monospace",
    marginBottom: 3,
  },

  // ── Liveness Badge ──
  livenessBadge: {
    position: "absolute",
    top: 50,
    alignSelf: "center",
    paddingHorizontal: 20,
    paddingVertical: 8,
    borderRadius: 20,
    zIndex: 10,
  },
  livenessBadgeText: {
    color: "#fff",
    fontWeight: "bold",
    fontSize: 14,
  },
  loadingContainer: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
  },
  loadingText: {
    color: "#fff",
    fontSize: 14,
    marginLeft: 8,
  },
});