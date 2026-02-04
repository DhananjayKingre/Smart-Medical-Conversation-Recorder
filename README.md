# 🩺 Smart Medical Conversation Recorder

## Overview

The **Smart Medical Conversation Recorder** is an AI-powered system designed to record, process, and analyze medical conversations between doctors and patients. The system automatically performs:

- Noise reduction and audio enhancement  
- Speaker diarization (detecting who is speaking)  
- Transcription using **Deepgram API**  
- Symptom detection and disease prediction  
- AI-assisted chat with a medical knowledge base using **Groq API with LLaMA 3 model**

This project leverages **state-of-the-art audio processing and AI tools** to provide a seamless medical conversation experience.

---

## 1. Techniques and Tools Used

### 1.1 Core Technologies

- **Streamlit**: Web framework to create an interactive dashboard and user interface for recording, uploading audio, and chatting with AI.
- **sounddevice**: Library for recording audio in real-time from the microphone.
- **numpy**: For numerical processing and handling audio arrays.
- **scipy**: For reading/writing WAV files.
- **noisereduce**: Noise reduction to clean background noise from recorded audio.
- **pyannote.audio**: Speaker diarization library that identifies different speakers in a conversation.
- **Deepgram API**: Automatic speech-to-text transcription.
- **Groq AI with LLaMA 3 model**: Provides a medical RAG (Retrieval-Augmented Generation) system to answer questions using the knowledge base.
- **Faiss + Sentence Transformers**: For building vector-based search of medical PDFs to enable fast semantic retrieval of information.

### 1.2 Why These Techniques Are Used

- **Real-Time Audio Recording**: Using `sounddevice` allows capturing natural conversation between doctor and patient.
- **Noise Reduction**: `noisereduce` ensures better transcription accuracy by removing background noise and interruptions.
- **Speaker Diarization**: `pyannote.audio` helps distinguish between doctor and patient voices.
- **Deepgram API**: Provides highly accurate transcription with punctuation and formatting.
- **RAG with Groq & LLaMA 3**: Enables AI to answer patient questions with context from medical documents using advanced LLaMA 3 model for accurate medical responses.
- **Streamlit UI**: Provides a user-friendly interface to interact with the system and visualize results.

---

## 2. Noise Reduction and Interruptions Handling

Before transcription, the recorded audio is cleaned:

1. Each audio channel is processed independently using **noisereduce**.
2. Channels are combined to mono and normalized.
3. Noise removal ensures that background interruptions do not affect speaker diarization or transcription accuracy.

---

## 3. Project Workflow

### Step-by-Step Workflow:

1. **Audio Input**:  
   - Either record audio using the built-in microphone (`sounddevice`) or upload a WAV/MP3 file.
   
2. **Noise Reduction**:  
   - Audio is processed using `noisereduce` to remove background noise.
   
3. **Speaker Diarization**:  
   - `pyannote.audio` identifies the speakers in the conversation and separates their segments.

4. **Transcription**:  
   - Audio segments are sent to **Deepgram API** for speech-to-text conversion.

5. **Symptom Detection**:  
   - Extracted transcript is compared with `disease_db.json` to identify symptoms and possible diseases.

6. **Medical RAG System with LLaMA 3**:  
   - Using **Groq API with LLaMA 3 model**, the system answers user questions based on detected diseases, symptoms, and a medical PDF knowledge base.
   
7. **Interactive Chat**:  
   - Users can ask follow-up questions to the AI doctor through the Streamlit chat interface.
   
8. **Results Visualization**:  
   - The UI displays the cleaned audio, transcript, detected symptoms, possible diseases, and prescription advice.

---

## 4. How to Run the Project

### 4.1 Prerequisites

- Python 3.10+
- git clone https://github.com/DhananjayKingre/Smart-Medical-Conversation-Recorder.git

### 4.2 Create Virtual Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 4.3 Install Dependencies

```bash
pip install --upgrade pip
pip install streamlit sounddevice numpy scipy noisereduce pyannote.audio requests python-dotenv soundfile sentence-transformers faiss-cpu PyPDF2 groq
```

### 4.4 Set Environment Variables

Create a `.env` file in the project root:

```env
DEEPGRAM_API_KEY=your_deepgram_api_key
HF_TOKEN=your_huggingface_token
GROQ_API_KEY=your_groq_api_key
```

### 4.5 Run the Project

```bash
# Run the Streamlit app
python -m streamlit run ui_str.py
```

---

## 5. File Structure

```
Smart-Medical-Recorder/
│
├─ recorder.py                 # Main Streamlit app for recording and analysis
├─ ai_doctor_chat.py           # AI Doctor chat logic and RAG implementation using LLaMA 3
├─ disease_db.json             # Symptom-to-disease mapping
├─ .env                        # API keys
├─ temp/                       # Temporary folder for audio processing
└─ README.md                   # Project documentation
```

---

## 6. Notes

- Ensure your microphone is properly configured for recording.
- Deepgram API key, HuggingFace token, and Groq API key are required.
- Large PDFs may take a few minutes to process on first run (vector database creation).
- The AI doctor can answer symptom-related questions and provide references from the medical knowledge base using LLaMA 3.

---

## 7. References

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Pyannote Audio](https://github.com/pyannote/pyannote-audio)
- [Deepgram API](https://developers.deepgram.com/)
- [Noisereduce Library](https://github.com/timsainb/noisereduce)
- [Groq AI](https://www.groq.com/)
- [Faiss for Vector Search](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)

