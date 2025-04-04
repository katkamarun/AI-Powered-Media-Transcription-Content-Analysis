# AI-Powered Media Transcription & Content Analysis 

A web application that combines OpenAI's Whisper for speech-to-text transcription and Llama 3.2 for intelligent content analysis. This tool processes both audio and video files to generate accurate transcriptions, titles, and summaries.

## Features

- 🎤 Audio/Video file processing (WAV, MP3, M4A, MP4)
- 📝 Accurate speech-to-text transcription using Whisper
- 🧠 Intelligent title and summary generation using Llama 3.2
- 🎯 Clean, user-friendly interface
- ⚡ Real-time processing feedback
- 🎥 Video preview and audio extraction

## Technical Stack

- **Speech-to-Text**: OpenAI Whisper
- **Content Analysis**: Llama 3.2 via Ollama
- **Web Interface**: Streamlit
- **Audio Processing**: Pydub
- **Structured Output**: Instructor with Pydantic
- **Video Handling**: FFmpeg (via Pydub)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/katkamarun/audio-transcription-tool.git
cd audio-transcription-tool
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Ollama and pull the Llama 3.2 model:
```bash
# Install Ollama (follow instructions at https://ollama.ai)
ollama pull llama3.2
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run extract_transcription.py
```

2. Open your browser and navigate to the provided local URL (usually http://localhost:8501)

3. Upload an audio or video file and click "Process File"

## Project Structure

```
audio-transcription-tool/
├── extract_transcription.py  # Main application code
├── requirements.txt          # Project dependencies
└── README.md                # Project documentation
```

## How It Works

1. **File Processing**
   - Upload audio/video file through Streamlit interface
   - For videos: extract audio track automatically
   - Convert to mono channel and 16kHz sample rate
   - Normalize audio for optimal processing

2. **Transcription**
   - Use Whisper model to convert speech to text
   - Generate clean, readable transcript

3. **Content Analysis**
   - Send transcript to Llama 3.2 via Ollama
   - Generate title and summary
   - Display structured results

## Author

[Your Name]
- GitHub: [katkamarun](https://github.com/katkamarun)
- Upwork: [https://www.upwork.com/freelancers/~013c73287cb4edd7e0?mp_source=share] 