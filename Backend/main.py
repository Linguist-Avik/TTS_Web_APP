from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from parler_tts import ParlerTTS

app = FastAPI()

# Load the Parler-TTS model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ParlerTTS.from_pretrained("parler-tts/parler_tts_indic").to(device)

class TTSRequest(BaseModel):
    text: str
    language: str  # Options: "en", "bn", "hi"

@app.post("/synthesize/")
async def synthesize_speech(request: TTSRequest):
    try:
        # Generate speech
        audio_waveform = model.generate(request.text, language=request.language)
        output_file = "output.wav"
        torch.save(audio_waveform, output_file)
        
        return {"message": "Speech generated successfully!", "file": output_file}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
