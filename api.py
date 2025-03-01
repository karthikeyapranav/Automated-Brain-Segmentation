from fastapi import FastAPI, UploadFile, File
import uvicorn
import io
import torch
from model import get_model

app = FastAPI()

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(device)
model.load_state_dict(torch.load("brain_segmentation_model.pth"))
model.eval()

@app.post("/predict/")
async def predict_mri(file: UploadFile = File(...)):
    contents = await file.read()
    # Load MRI scan & run prediction...
    return {"prediction": "example"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
