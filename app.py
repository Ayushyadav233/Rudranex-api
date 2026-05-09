from fastapi import FastAPI, UploadFile, File
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import io
import uvicorn

app = FastAPI()

# Memory optimization: Load light-weight model and standard python tokenizer
print("Initializing Rudranex-API with TrOCR-Small...")
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-handwritten', use_fast=False)
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-handwritten', low_cpu_mem_usage=True)
model.eval()
print("Backend Ready!")

@app.get("/")
async def health():
    return {"status": "online", "model": "trocr-small-handwritten"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        
        with torch.no_grad():
            generated_ids = model.generate(pixel_values)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return {"text": generated_text}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)