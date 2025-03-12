from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torchaudio
from parler_tts.modeling_parler_tts import ParlerTTSForConditionalGeneration as ParlerTTS
from transformers import AutoTokenizer

app = FastAPI()

# Load the Parler-TTS model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "parler-tts/parler_tts_indic"
model = ParlerTTS.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

class TTSRequest(BaseModel):
    text: str
    language: str  # Supported: "en", "bn", "hi"

@app.post("/synthesize/")
async def synthesize_speech(request: TTSRequest):
    try:
        # Tokenize input text
        inputs = tokenizer(request.text, return_tensors="pt").to(device)

        # Generate speech
        with torch.no_grad():
            audio_waveform = model.generate(**inputs)

        # Convert to CPU tensor for saving
        audio_waveform = audio_waveform.cpu()

        # Save output as a WAV file
        output_file = "output.wav"
        sample_rate = 22050  # Standard TTS sample rate
        torchaudio.save(output_file, audio_waveform, sample_rate)

        return {"message": "Speech generated successfully!", "file_url": f"/download/{output_file}"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Parler-TTS Backend is Running!"}
