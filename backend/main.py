from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from parler_tts.modeling_parler_tts import ParlerTTSForConditionalGeneration  # Correct import
import soundfile as sf
import numpy as np

app = FastAPI()

# Load the Parler-TTS model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler_tts_indic").to(device)

class TTSRequest(BaseModel):
    text: str
    language: str  # Options: "en", "bn", "hi"

@app.post("/synthesize/")
async def synthesize_speech(request: TTSRequest):
    try:
        # Convert text to speech
        inputs = {"text": request.text, "language": request.language}
        with torch.no_grad():
            audio_tensor = model.generate(**inputs)

        # Convert tensor to numpy for saving
        audio_numpy = audio_tensor.cpu().numpy().squeeze()

        # Save as WAV file
        output_file = "output.wav"
        sf.write(output_file, audio_numpy, samplerate=24000)  # 24kHz is a standard TTS sample rate
        
        return {"message": "Speech generated successfully!", "file": output_file}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
