import streamlit as st
import os
import ast
import json
import time
from backend import (
    process_audio,
    record_audio,
    load_audio_file,
    extract_medical_insights,
    record_voice_input,
    transcribe_voice_input,
    TEMP_FOLDER
)

# ---------------- CONFIG ---------------- #
st.set_page_config(page_title="Smart Medical Conversation Recorder", layout="wide")
st.title("🩺 Smart Medical Conversation Recorder")

# ---------------- SESSION STATE ---------------- #
if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "audio_processed" not in st.session_state:
    st.session_state.audio_processed = False
if "audio_path" not in st.session_state:
    st.session_state.audio_path = None
if "recorded_audio_path" not in st.session_state:
    st.session_state.recorded_audio_path = None
if "extracted_data" not in st.session_state:
    st.session_state.extracted_data = {}
if "editing_mode" not in st.session_state:
    st.session_state.editing_mode = False
if "current_field_values" not in st.session_state:
    st.session_state.current_field_values = {}
if "voice_pending" not in st.session_state:
    st.session_state.voice_pending = None


# ---------------- DISPLAY RECORDED AUDIO (Right after title) ---------------- #
if st.session_state.recorded_audio_path and os.path.exists(st.session_state.recorded_audio_path):
    st.subheader("🎵 Recorded Audio")
    audio_file = open(st.session_state.recorded_audio_path, 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/wav')
    audio_file.close()

# ---------------- SIDEBAR ---------------- #
st.sidebar.header("Settings")
duration = st.sidebar.slider("Recording duration (up to 15 min) ", 20, 900, 120)
start_recording = st.sidebar.button("Start Recording")
uploaded_file = st.sidebar.file_uploader("Upload Audio File", type=["wav", "mp3"])

# ---------------- HANDLE INPUT ---------------- #
audio = None
fs = 16000

if uploaded_file and not st.session_state.audio_processed:
    path = os.path.join(TEMP_FOLDER, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    audio, fs = load_audio_file(path)

    st.info("🔇 Removing background noise...")
    st.info("🧠 Detecting speakers...")
    #st.info("🧠 Identifying Doctor vs Patient...")
    
    transcript, audio_path = process_audio(audio, fs)
    st.session_state.transcript = transcript
    st.session_state.audio_path = audio_path
    st.session_state.recorded_audio_path = path
    st.session_state.audio_processed = True
    st.rerun()

if start_recording and not st.session_state.audio_processed:
    st.info("🎙 Recording...")
    audio, fs, recorded_path = record_audio(duration)
    
    st.session_state.recorded_audio_path = recorded_path
    
    st.info("🔇 Removing background noise...")
    st.info("🧠 Detecting speakers...")
    #st.info("🧠 Identifying Doctor vs Patient...")
    
    transcript, audio_path = process_audio(audio, fs)
    st.session_state.transcript = transcript
    st.session_state.audio_path = audio_path
    st.session_state.audio_processed = True
    st.rerun()

# ---------------- DISPLAY TRANSCRIPT ---------------- #
if st.session_state.transcript:
    st.subheader("🗣 Transcript")
    st.text_area("Conversation", st.session_state.transcript, height=300)

# ================= LLM MEDICAL EXTRACTION ================= #
st.sidebar.header("SUMMARY")

if st.sidebar.button("Extract Medical Insights") and st.session_state.transcript:
    with st.spinner("Analyzing conversation..."):
        data = extract_medical_insights(st.session_state.transcript)
        st.session_state.extracted_data = data
        # Initialize current field values
        st.session_state.current_field_values = data.copy()
        st.session_state.editing_mode = False
        st.rerun()

# ================= DISPLAY AND EDIT EXTRACTED DATA ================= #
if st.session_state.extracted_data:
    st.sidebar.subheader("📋 Extracted Insights")

    # Edit Summary
    if st.sidebar.button("✏️ Enable Editing" if not st.session_state.editing_mode else "💾 Save & Exit Editing"):
        if not st.session_state.editing_mode:
            st.session_state.current_field_values = st.session_state.extracted_data.copy()
        else:
            st.session_state.extracted_data = st.session_state.current_field_values.copy()

        st.session_state.editing_mode = not st.session_state.editing_mode
        st.rerun()

    field_labels = {
        "chief_complaint": "Chief Complaint",
        "symptoms": "Symptoms",
        "associated_symptoms": "Associated Symptoms",
        "symptom_duration": "Symptom Duration",
        "medical_history": "Medical History",
        "diagnosis": "Diagnosis",
        "plan_of_care": "Plan Of Care",
        "prescriptions": "Prescriptions",
        "follow_up": "Follow Up",
        "doctor_notes": "Doctor Notes"
    }

    # -------- APPLY PENDING VOICE UPDATE SAFELY -------- #
    if st.session_state.voice_pending:
        pending_field = st.session_state.voice_pending["field"]
        pending_text = st.session_state.voice_pending["text"]

        current_val = st.session_state.current_field_values.get(pending_field, "")

        if isinstance(current_val, list):
            current_val.append(pending_text)
            updated_value = current_val
        elif current_val and current_val != "None":
            updated_value = f"{current_val}, {pending_text}"
        else:
            updated_value = pending_text

        st.session_state.current_field_values[pending_field] = updated_value
        st.session_state[f"input_{pending_field}"] = str(updated_value)

        st.session_state.voice_pending = None 

    # -------- DISPLAY FIELDS -------- #
    for key in st.session_state.extracted_data.keys():
        value = st.session_state.current_field_values.get(key, st.session_state.extracted_data[key])
        label = field_labels.get(key, key.replace('_', ' ').title())

        if st.session_state.editing_mode:
            st.sidebar.markdown(f"**{label}:**")
            col1, col2 = st.sidebar.columns([4, 1])

            if isinstance(value, list):
                current_value = str(value)
            else:
                current_value = str(value) if value else ""

            # Summary TEXT BOX
            with col1:
                edited_value = st.text_area(
                    f"edit_{key}",
                    value=current_value,
                    height=80,
                    label_visibility="collapsed",
                    key=f"input_{key}"
                )

                try:
                    if edited_value.startswith('[') and edited_value.endswith(']'):
                        st.session_state.current_field_values[key] = ast.literal_eval(edited_value)
                    else:
                        st.session_state.current_field_values[key] = edited_value
                except:
                    st.session_state.current_field_values[key] = edited_value

            # VOICE BUTTON for each field
            with col2:
                if st.button("🎤", key=f"voice_btn_{key}", help="Record for 10 seconds"):
                    status_placeholder = st.sidebar.empty()
                    status_placeholder.info("🎙 Recording...")

                    try:
                        voice_path = record_voice_input(duration=10)
                        status_placeholder.info("🔄 Transcribing...")
                        transcribed_text = transcribe_voice_input(voice_path)

                        if transcribed_text:
                            # STORE IN TEMP BUFFER (NOT DIRECTLY IN BOX)
                            st.session_state.voice_pending = {
                                "field": key,
                                "text": transcribed_text
                            }
                            st.rerun()
                        else:
                            status_placeholder.error("❌ Could not transcribe.")
                            time.sleep(2)

                    except Exception as e:
                        status_placeholder.error(f"❌ Error: {str(e)}")
                        time.sleep(2)

            st.sidebar.markdown("---")

        else:
            display_value = st.session_state.current_field_values.get(key, value)
            st.sidebar.markdown(f"**{label}**: {display_value}")

    # Download JSON
    if not st.session_state.editing_mode:
        st.sidebar.markdown("---")
        json_data = json.dumps(st.session_state.current_field_values, indent=2)
        st.sidebar.download_button(
            label="📥 Download JSON",
            data=json_data,
            file_name="medical_insights.json",
            mime="application/json"
        )

















