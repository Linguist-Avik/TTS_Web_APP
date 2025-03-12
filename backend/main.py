from fastapi import FastAPI
from parler_tts.modeling_parler_tts import ParlerTTSForConditionalGeneration as ParlerTTS
from transformers import AutoTokenizer
import torch

app = FastAPI()

# Load the Parler-TTS model and tokenizer
model_name = "parler-tts/parler-tts"
model = ParlerTTS.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

@app.get("/")
def read_root():
    return {"message": "Parler-TTS Backend is Running!"}

@app.post("/synthesize/")
async def synthesize_text(text: str):
    """Generate speech from text"""
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs)

    # Convert tensor output to a list of numbers (for audio processing)
    return {"audio_output": outputs.tolist()}
