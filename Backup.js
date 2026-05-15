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
import * as FaceDetector from "expo-face-detector";
import * as ImageManipulator from "expo-image-manipulator";
import * as SecureStore from "expo-secure-store";
import * as FileSystem from "expo-file-system/legacy";
import { loadTensorflowModel } from "react-native-fast-tflite";
import { Buffer } from "buffer";
import decodeJpeg from "jpeg-js";

global.Buffer = Buffer;

const MODEL_FILE = require("./assets/mobilefacenet.tflite");
const EMBEDDING_PATH = `${FileSystem.documentDirectory}face_master_data.json`;

export default function App() {
    const [permission, requestPermission] = useCameraPermissions();
    const [mode, setMode] = useState(null);
    const [log, setLog] = useState([]);
    const [model, setModel] = useState(null);
    const [loading, setLoading] = useState(false);
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

    const cosineSimilarity = (a, b) => {
        let dot = 0, normA = 0, normB = 0;
        for (let i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        return dot / (Math.sqrt(normA) * Math.sqrt(normB));
    };

    const captureAndProcess = async () => {
        if (!cameraRef.current || !model) return null;

        setLoading(true);
        console.group("🚀 [Face Recognition Process]");
        try {
            // 1. Ambil Gambar
            console.time("📸 1. Capture");
            const photo = await cameraRef.current.takePictureAsync({ quality: 0.7 });
            console.timeEnd("📸 1. Capture");

            // 2. Deteksi Wajah
            console.time("🔍 2. Detection");
            const detection = await FaceDetector.detectFacesAsync(photo.uri, {
                mode: FaceDetector.FaceDetectorMode.fast,
            });
            console.timeEnd("🔍 2. Detection");

            if (!detection.faces.length) {
                Alert.alert("Gagal", "Wajah tidak terdeteksi.");
                console.groupEnd();
                return null;
            }
            const face = detection.faces[0];

            // 3. Crop & Resize 112x112
            console.time("✂️ 3. Crop & Resize");
            const processed = await ImageManipulator.manipulateAsync(
                photo.uri,
                [
                    {
                        crop: {
                            originX: Math.max(0, face.bounds.origin.x),
                            originY: Math.max(0, face.bounds.origin.y),
                            width: face.bounds.size.width,
                            height: face.bounds.size.height,
                        },
                    },
                    { resize: { width: 112, height: 112 } },
                ],
                { format: "jpeg", base64: true }
            );
            console.timeEnd("✂️ 3. Crop & Resize");

            // 4. Pre-processing Pixel (PENTING: Agar hasil tidak identik terus)
            console.time("🧪 4. Pre-processing");
            const rawImageData = Buffer.from(processed.base64, 'base64');
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
            console.timeEnd("🧪 4. Pre-processing");

            // 5. Jalankan AI Inference (Wajib dalam array [inputTensor])
            console.time("🧠 5. AI Inference");
            const output = await model.run([inputTensor]);
            const embedding = Array.isArray(output) ? output[0] : output;
            console.timeEnd("🧠 5. AI Inference");

            // Log Vector Sample untuk Validasi
            const vectorArray = Object.values(embedding);
            console.log("📊 Vector Sample (5 pertama):", JSON.stringify(vectorArray.slice(0, 5)));

            console.groupEnd();
            return vectorArray;

        } catch (e) {
            console.error("🚨 Error Detail:", e);
            console.groupEnd();
            addLog("❌ Error: " + e.message);
            return null;
        } finally {
            setLoading(false);
        }
    };

    const onRegister = async () => {
        console.log("🚀 [Register Process]");
        const embedding = await captureAndProcess();
        if (!embedding) return;

        try {
            await FileSystem.writeAsStringAsync(EMBEDDING_PATH, JSON.stringify(embedding));
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
            const isRegistered = await SecureStore.getItemAsync("face_registered_status");
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
                    <TouchableOpacity style={styles.btn} onPress={() => setMode("register")}>
                        <Text style={styles.btnText}>Daftar Wajah</Text>
                    </TouchableOpacity>
                    <TouchableOpacity style={[styles.btn, { backgroundColor: "#27ae60", marginTop: 20 }]} onPress={() => setMode("verify")}>
                        <Text style={styles.btnText}>Verifikasi Wajah</Text>
                    </TouchableOpacity>
                </View>
            ) : (
                <>
                    <CameraView style={styles.camera} facing="front" ref={cameraRef} />
                    <View style={styles.overlay}>
                        <TouchableOpacity disabled={loading} style={styles.captureBtn} onPress={mode === "register" ? onRegister : onVerify}>
                            {loading ? <ActivityIndicator color="#fff" /> : <Text style={styles.btnText}>{mode === "register" ? "Ambil Master" : "Scan Verifikasi"}</Text>}
                        </TouchableOpacity>
                        <TouchableOpacity onPress={() => setMode(null)} style={{ marginTop: 20 }}><Text style={{ color: "#aaa" }}>Batal</Text></TouchableOpacity>
                    </View>
                    <View style={styles.logArea}>
                        <ScrollView>
                            {log.map((item, index) => <Text key={index} style={styles.logText}>{item}</Text>)}
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
    menu: { flex: 1, justifyContent: "center", alignItems: "center", padding: 30 },
    title: { color: "#fff", fontSize: 24, fontWeight: "bold", marginBottom: 50 },
    camera: { flex: 2 },
    overlay: { flex: 1, alignItems: "center", justifyContent: "center", backgroundColor: "#0a0a0a" },
    btn: { backgroundColor: "#2980b9", padding: 18, borderRadius: 12, width: "100%", alignItems: "center" },
    btnText: { color: "#fff", fontWeight: "bold", fontSize: 16 },
    captureBtn: { backgroundColor: "#c0392b", padding: 22, borderRadius: 60, width: "85%", alignItems: "center" },
    logArea: { flex: 0.8, backgroundColor: "#000", padding: 12, borderTopWidth: 1, borderTopColor: "#222" },
    logText: { color: "#2ecc71", fontSize: 12, fontFamily: "monospace", marginBottom: 3 },
});