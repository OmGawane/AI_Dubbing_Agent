# Bhaav: Meaning-Aware Video Localization Agent 🎬

**Bhaav** is an advanced multi-modal video localization and dubbing pipeline built for the Gnani.ai internship challenge. Unlike traditional flat-monologue dubbing tools, Bhaav preserves speaker identity, emotional intent, and temporal sync across multi-character dialogues using advanced speech-to-text, LLM reasoning, and text-to-speech architecture.

* **Live App URL:** [omgawane-ai-dubbing-agent-app-fb0xdi.streamlit.app](https://omgawane-ai-dubbing-agent-app-fb0xdi.streamlit.app/)
* **GitHub Repository:** [OmGawane/AI_Dubbing_Agent](https://github.com/OmGawane/AI_Dubbing_Agent)

---

## 📌 Solution Pitch (1-Page Summary for Evaluators)

### 1. Problem Statement
Standard video localization pipelines treat foreign-language video translation as a flat text conversion problem. This results in robotic monologues, loss of emotional nuance, speaker identity mismatch in multi-character scenes, and severe temporal drift over background tracks.

### 2. The Solution: Bhaav
Bhaav introduces an intelligent multi-modal localization framework that deconstructs video files into distinct acoustic and semantic layers. It performs **speaker diarization**, contextual intent analysis, and region-specific voice synthesis to deliver native-feeling dubs that respect character dynamics and scene context.

### 3. Core Technology Stack
* **Speech-to-Text (STT):** Prisma API for high-accuracy transcription and time-stamped word alignment.
* **LLM Reasoning & Diarization:** Evon open-source 30B model (`gnani/gnani-evon-v3.3-30B-A3B`) via OpenRouter/API backend for contextual translation and speaker turn-taking analysis.
* **Text-to-Speech (TTS):** Timbre API for natural, emotional, and distinct character voice generation.
* **Media Processing & Multiplexing:** MoviePy and FFmpeg for precise audio-video track splicing and zero-padded timestamp offsets.
* **Frontend & Deployment:** Streamlit Community Cloud.

### 4. Key Engineering Highlights
* **Context-Aware Translation:** Uses scene prompt injection (e.g., *"two old friends arguing at a tea stall"*) to guide the LLM's tonal translation.
* **Temporal Sync Safeguards:** Implements sliding-window diarization and timestamp alignment to prevent audio desynchronization over background audio.
* **Robust Cloud Deployment:** Fully containerized with FFmpeg system dependencies and secure Streamlit Secrets management (`GNANI_API_KEY`, `EVON_BASE_URL`).

### 5. Current Limitations & Active R&D (What We Are Solving)
While Bhaav successfully handles full multi-character localization, we are actively iterating on the following production bottlenecks:
* **Background Audio Bleed / Music Ducking:** When stripping and replacing dialogue tracks via FFmpeg, complex mixed audio containing heavy background music or sound effects can occasionally suffer from phase cancellation or require manual ducking adjustments.
* **Phoneme-Level Lip Sync:** Current timing relies on word-level timestamp boundaries from STT. Fine-grained viseme-to-phoneme mapping for true visual lip synchronization is currently limited to audio track replacement rather than mouth-shape video warping.
* **Inference Latency on Large Files:** Processing high-resolution video files >60 seconds under cloud memory caps requires chunked stream processing to prevent container timeout or memory limits.

---


## 🛠️ Local Installation & Development

To set up, run, and develop **Bhaav** locally on your machine, follow these steps:

### Prerequisites

- Python 3.10 or higher installed on your system.
- **FFmpeg** installed on your operating system and added to your system PATH (required for audio extraction and multi-track splicing).

### Step 1: Clone the Repository

Clone the project repository to your local machine and navigate into the project directory:

```bash
git clone https://github.com/OmGawane/AI_Dubbing_Agent.git
cd AI_Dubbing_Agent
```

### Step 2: Install Python Dependencies

Install all required Python libraries using pip:

```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables

Set up your local API keys and environment configurations. You can export them in your terminal or create a `.env` file in the root directory.

```bash
export GNANI_API_KEY="your_gnani_api_key_here"
export EVON_BASE_URL="https://openrouter.ai/api/v1"
export EVON_MODEL="qwen/qwen3.8-flash"
export EVON_API_KEY="your_openrouter_or_evon_key_here"
```

> **Note:** Replace the placeholder API keys with your actual credentials. Never commit your `.env` file or expose API keys in your public repository.

### Step 4: Run the Streamlit Application

Launch the local development server:

```bash
streamlit run app.py
```

### Verification

Once the application starts, Streamlit will display your local and network URLs.

Open the following URL in your browser to access the interactive Bhaav interface:

```text
http://localhost:8501
```

You're all set! 🎉


   
