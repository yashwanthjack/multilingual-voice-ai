import whisper
import torch
import logging
import asyncio
import numpy as np
import sounddevice as sd
from pathlib import Path

# --- Logging setup ---e
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Whisper model global load ---
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("base", device=device)  # Or "large-v3"
    logger.info(f"🤖 Whisper model loaded on {device}")
except Exception as e:
    logger.error(f"❌ Error loading Whisper model: {e}")
    model = None

SAMPLE_RATE = 16000
BLOCK_DURATION = 5  # seconds of speech to collect before passing to Whisper

async def transcribe_microphone():
    """
    Transcribe live audio from microphone using Whisper.
    Returns: transcript_text, detected_language
    """
    try:
        if not model:
            raise Exception("Whisper model not loaded")
        
        logger.info("🎤 Listening to Microphone...")
        blocksize = int(SAMPLE_RATE * BLOCK_DURATION)
        loop = asyncio.get_event_loop()
        # Callback to collect BLOCK_DURATION seconds of audio
        audio_data = np.zeros((blocksize,), dtype=np.float32)

        def callback(indata, frames, time, status):
            if status:
                logger.warning(status)
            audio_data[:frames] = indata[:, 0]
        
        # Record
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback, blocksize=blocksize):
            await asyncio.sleep(BLOCK_DURATION)
        
        # Run inference
        audio_padded = whisper.pad_or_trim(audio_data)
        result = await loop.run_in_executor(
            None, lambda: model.transcribe(
                audio_padded,
                language=None,
                fp16=torch.cuda.is_available()
            )
        )
        logger.info(f"Raw Whisper result: {result}")
        transcript = result["text"].strip()
        detected_language = result["language"]

        # Map language code
        lang_mapping = {"hi": "hindi", "en": "english", "te": "telugu", "ta": "tamil", "kn": "kannada"}
        language = lang_mapping.get(detected_language, "english")
        logger.info(f"📝 Transcription: '{transcript}' (Language: {language})")
        return transcript, language
    except Exception as e:
        logger.exception(f"❌ Transcription error: {e}")
        return "मुझे समझ नहीं आया, कृपया दोबारा बोलिए", "hindi"

# --- Test block ---
async def test_transcribe():
    transcript, language = await transcribe_microphone()
    print(f"Transcript: {transcript}")
    print(f"Detected Language: {language}")

if __name__ == "__main__":
    asyncio.run(test_transcribe())
