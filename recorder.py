import streamlit as st
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import noisereduce as nr
import os
import requests
from dotenv import load_dotenv
from pyannote.audio import Pipeline
import tempfile
import soundfile as sf
import json
from ai_doctor_chat import ai_doctor_chat_popup  

# ---------------- CONFIG ---------------- #
st.set_page_config(page_title="Smart Medical Conversation Recorder",
                   layout="wide",
                   initial_sidebar_state="expanded")

st.title("🩺 Smart Medical Conversation Recorder")

# ---------------- SIDEBAR ---------------- #
st.sidebar.header("Settings")
duration = st.sidebar.slider("Recording duration (seconds)", 20, 180, 60)
start_recording = st.sidebar.button("Start Recording", key="record_button")
uploaded_file = st.sidebar.file_uploader("Upload Audio File (WAV/MP3)", type=["wav", "mp3"])

# ---------------- ENV KEYS ---------------- #
load_dotenv()
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

TEMP_FOLDER = r"D:\Speaker-Recognition-Main\temp"
os.makedirs(TEMP_FOLDER, exist_ok=True)

# ---------------- LOAD PYANNOTE ---------------- #
@st.cache_resource
def load_pipeline():
    return Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token=HF_TOKEN
    )

pipeline = load_pipeline()

# ---------------- PLACEHOLDERS ---------------- #
record_msg = st.empty()
diarize_msg = st.empty()
transcribe_msg = st.empty()

audio = None
audio_path_to_process = None

# ---------- UPLOAD AUDIO ----------
if uploaded_file is not None:
    uploaded_path = os.path.join(TEMP_FOLDER, uploaded_file.name)
    with open(uploaded_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.audio(uploaded_path)
    mono_audio, fs = sf.read(uploaded_path)
    audio = np.stack([mono_audio, mono_audio], axis=1) if len(mono_audio.shape) == 1 else mono_audio

# ---------- RECORD AUDIO ----------
if start_recording:
    fs = 16000
    record_msg.info("🎙 Recording... Speak normally.")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=2, dtype="float32")
    sd.wait()
    record_msg.empty()
    raw_path = os.path.join(TEMP_FOLDER, "stereo_raw.wav")
    write(raw_path, fs, audio)

# ---------- PROCESS ----------
if audio is not None:

    # Noise reduction
    st.info("🔇 Cleaning background noise...")
    clean = np.zeros_like(audio)
    for ch in range(audio.shape[1]):
        clean[:, ch] = nr.reduce_noise(y=audio[:, ch], sr=fs)

    mono = clean.mean(axis=1)
    mono = mono / np.max(np.abs(mono))

    clean_path = os.path.join(TEMP_FOLDER, "clean_mono.wav")
    sf.write(clean_path, mono, fs)
    st.audio(clean_path)

    # Diarization
    diarize_msg.info("🧠 Detecting speakers...")
    diarization = pipeline(clean_path)
    diarize_msg.empty()

    # Transcription
    def transcribe_segment(path):
        url = "https://api.deepgram.com/v1/listen?punctuate=true&model=nova-2"
        headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}
        with open(path, "rb") as f:
            response = requests.post(url, headers=headers, data=f)
        return response.json()

    transcript_output = ""
    transcribe_msg.info("📝 Transcribing conversation...")

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start = max(0, int((turn.start - 0.2) * fs))
        end = min(len(mono), int((turn.end + 0.2) * fs))
        segment = mono[start:end]

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            sf.write(tmp.name, segment, fs)
            tmp_path = tmp.name

        result = transcribe_segment(tmp_path)
        try:
            text = result["results"]["channels"][0]["alternatives"][0]["transcript"]
        except:
            text = ""

        role = "Doctor" if speaker == "SPEAKER_00" else "Patient"
        transcript_output += f"{role}: {text}\n\n"
        os.remove(tmp_path)

    transcribe_msg.empty()

    st.subheader("🗣 Speaker-wise Conversation")
    st.text_area("Transcript", transcript_output, height=300)

    # ---------------- SYMPTOM DETECTION ---------------- #
    with open("disease_db.json", "r") as f:
        data = json.load(f)

    detected_symptoms = []
    detected_diseases = []
    prescriptions = []

    transcript_lower = transcript_output.lower()

    # detect symptoms and related diseases
    for symptom, disease_list in data["symptom_disease_map"].items():
        if symptom in transcript_lower:
            detected_symptoms.append(symptom)
            for d in disease_list:
                detected_diseases.append(d["disease"])
                prescriptions.extend(d["prescription"])

    # Store results in session for sidebar and chat
    st.session_state.detected_diseases = detected_diseases
    st.session_state.detected_symptoms = detected_symptoms
    st.session_state.transcript_text = transcript_output

    # Sidebar results
    st.sidebar.markdown("---")
    st.sidebar.subheader("🩺 Detected Analysis")
    st.sidebar.write("**Symptoms:**", ", ".join(detected_symptoms))
    st.sidebar.write("**Possible Diseases:**", ", ".join(detected_diseases))
    st.sidebar.write("**Prescription Advice:**")
    for p in prescriptions:
        st.sidebar.write("•", p)

# ---------------- CHAT FEATURE ---------------- #
if "detected_diseases" in st.session_state and st.session_state.detected_diseases:
    st.sidebar.markdown("---")
    if st.sidebar.button("💬 Chat with AI Doctor"):
        # open AI doctor chat popup with current session data
        ai_doctor_chat_popup(
            st.session_state.transcript_text,
            st.session_state.detected_diseases,
            st.session_state.detected_symptoms
        )









