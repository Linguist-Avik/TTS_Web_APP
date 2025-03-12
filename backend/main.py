from fastapi import FastAPI, HTTPException
from parler_tts import ParlerTTSForConditionalGeneration, ParlerTTSConfig
import torch
import numpy as np
import soundfile as sf
import io
import base64

app = FastAPI()

# Load Parler-TTS Model
try:
    config = ParlerTTSConfig()
    model = ParlerTTSForConditionalGeneration(config)
    model.eval()  # Set to evaluation mode (important for inference)
except Exception as e:
    print(f"Error loading TTS model: {e}")
    raise RuntimeError("Failed to load Parler-TTS model")

@app.get("/")
def home():
    return {"message": "Parler-TTS API is running successfully!"}

@app.post("/tts/")
async def generate_speech(text: str):
    """
    Convert text to speech using Parler-TTS and return the audio as a base64-encoded string.
    """
    try:
        if not text.strip():
            raise HTTPException(status_code=400, detail="Text input cannot be empty")

        # Prepare input for the model
        inputs = model.prepare_inputs_for_generation(text)

        with torch.no_grad():
            output = model(**inputs)

        # Convert output tensor to numpy array
        audio_data = output.cpu().numpy().astype(np.float32)

        # Save audio to a buffer
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, samplerate=22050, format="WAV")
        buffer.seek(0)

        # Encode the audio file as Base64
        audio_base64 = base64.b64encode(buffer.read()).decode("utf-8")

        return {"audio_base64": audio_base64}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")


