from fastapi import FastAPI, UploadFile, File
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import io
import uvicorn

app = FastAPI()

# Shuruat mein inhen None rakhenge
processor = None
model = None

@app.get("/")
async def health():
    # Ye page turant khulega, Hugging Face ko lagega server healthy hai
    return {
        "status": "online", 
        "model_loaded": model is not None,
        "message": "Rudranex API is running. Model will load on first request."
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global processor, model
    
    # Check agar model loaded nahi hai toh pehli request par load karo
    if model is None:
        print("Lazy Loading starting... Model is heavy, please wait.")
        processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten', use_fast=True)
        model = VisionEncoderDecoderModel.from_pretrained(
            'microsoft/trocr-base-handwritten',
            low_cpu_mem_usage=True
        )
        model.eval()
        print("Model Loaded successfully on first request!")

    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        with torch.no_grad():
            pixel_values = processor(images=image, return_tensors="pt").pixel_values
            generated_ids = model.generate(pixel_values)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return {"text": generated_text}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)