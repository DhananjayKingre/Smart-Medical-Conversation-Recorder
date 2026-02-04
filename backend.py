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
from groq import Groq

# ---------------- ENV ---------------- #
load_dotenv()
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

TEMP_FOLDER = r"D:\nextastra\temp"
os.makedirs(TEMP_FOLDER, exist_ok=True)

# ---------------- LOAD DIARIZATION ---------------- #
_pipeline = None

def load_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HF_TOKEN)
    return _pipeline

# ---------------- TRANSCRIBE FUNCTION ---------------- #
def transcribe_segment(path):
    url = "https://api.deepgram.com/v1/listen?punctuate=true&model=nova-2"
    headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}
    with open(path, "rb") as f:
        r = requests.post(url, headers=headers, data=f)
    return r.json()

# ---------------- AUDIO PROCESS ---------------- #
def process_audio(audio, fs):
    # Remove background noise
    clean = np.zeros_like(audio)
    for ch in range(audio.shape[1]):
        clean[:, ch] = nr.reduce_noise(y=audio[:, ch], sr=fs)

    mono = clean.mean(axis=1)
    mono = mono / np.max(np.abs(mono))
    clean_path = os.path.join(TEMP_FOLDER, "clean.wav")
    sf.write(clean_path, mono, fs)

    # -------- DIARIZATION + TRANSCRIPTION -------- #
    pipeline = load_pipeline()
    diarization = pipeline(clean_path)

    speaker_chunks = {}

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start = max(0, int((turn.start - 0.2) * fs))
        end = min(len(mono), int((turn.end + 0.2) * fs))
        segment = mono[start:end]

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            sf.write(tmp.name, segment, fs)
            result = transcribe_segment(tmp.name)

        try:
            text = result["results"]["channels"][0]["alternatives"][0]["transcript"]
        except:
            text = ""

        speaker_chunks.setdefault(speaker, []).append(text)
        os.remove(tmp.name)

    # ---------------- ROLE CLASSIFICATION ---------------- #
    role_prompt = f"""
Two speakers in a medical consultation.

Speaker A said:
{' '.join(speaker_chunks.get('SPEAKER_00', []))}

Speaker B said:
{' '.join(speaker_chunks.get('SPEAKER_01', []))}

Who is Doctor and who is Patient?

Return JSON:
{{
"SPEAKER_00": "Doctor or Patient",
"SPEAKER_01": "Doctor or Patient"
}}
"""

    role_response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": role_prompt}],
        temperature=0,
        response_format={"type": "json_object"}
    )

    roles = json.loads(role_response.choices[0].message.content)

    # ---------------- BUILD FINAL TRANSCRIPT ---------------- #
    transcript_output = ""

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start = max(0, int((turn.start - 0.2) * fs))
        end = min(len(mono), int((turn.end + 0.2) * fs))
        segment = mono[start:end]

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            sf.write(tmp.name, segment, fs)
            result = transcribe_segment(tmp.name)

        try:
            text = result["results"]["channels"][0]["alternatives"][0]["transcript"]
        except:
            text = ""

        role = roles.get(speaker, "Speaker")
        transcript_output += f"{role}: {text}\n\n"
        os.remove(tmp.name)

    return transcript_output, clean_path

# ---------------- RECORD AUDIO ---------------- #
def record_audio(duration, fs=16000):
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=2, dtype="float32")
    sd.wait()
    
    # Save the recorded audio
    recorded_path = os.path.join(TEMP_FOLDER, "recorded_audio.wav")
    sf.write(recorded_path, audio, fs)
    
    return audio, fs, recorded_path

# ---------------- LOAD AUDIO FROM FILE ---------------- #
def load_audio_file(file_path):
    mono_audio, fs = sf.read(file_path)
    audio = np.stack([mono_audio, mono_audio], axis=1) if len(mono_audio.shape) == 1 else mono_audio
    return audio, fs

# ================= LLM MEDICAL EXTRACTION ================= #
def extract_medical_insights(transcript):
    prompt = f"""
Extract structured clinical data from this doctor-patient conversation.

Return JSON:
chief_complaint, symptoms, associated_symptoms, symptom_duration,
medical_history, diagnosis, plan_of_care, prescriptions, follow_up, doctor_notes

Conversation:
{transcript}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"}
    )

    data = json.loads(response.choices[0].message.content)
    return data

# ---------------- RECORD SHORT VOICE FOR EDITING ---------------- #
def record_voice_input(duration=5, fs=16000):
    """Record short audio clip for voice input"""
    import time
    import uuid
    
    # Generate unique filename to avoid conflicts
    unique_id = str(uuid.uuid4())[:8]
    temp_path = os.path.join(TEMP_FOLDER, f"voice_input_{unique_id}.wav")
    
    # Record audio
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()
    
    # Save to file
    sf.write(temp_path, audio, fs)
    
    # Small delay to ensure file is written
    time.sleep(0.2)
    
    return temp_path

# ---------------- TRANSCRIBE VOICE INPUT ---------------- #
def transcribe_voice_input(audio_path):
    """Transcribe short voice input to text"""
    try:
        result = transcribe_segment(audio_path)
        text = result["results"]["channels"][0]["alternatives"][0]["transcript"]
        
        # Clean up the temporary file
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        return text.strip()
    except Exception as e:
        print(f"Transcription error: {e}")
        # Clean up the temporary file even on error
        if os.path.exists(audio_path):
            os.remove(audio_path)
        return ""




























