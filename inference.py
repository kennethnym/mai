import modal
import torch
import numpy as np
from PIL import Image
from model import mai
from augmentation import preprocess_validation

MODEL_NAME = "mai_20240424_180855_4"

image = modal.Image.debian_slim().pip_install(
    "datasets==2.19.0",
    "albumentations==1.4.4",
    "numpy==1.26.4",
    "torch==2.2.2",
)
app = modal.App("multilayer-authenticity-identifier", image=image)
volume = modal.Volume.from_name("model-store")
model_store_path = "/vol/models"


@app.function(timeout=5000, gpu="T4", volumes={model_store_path: volume})
def load_model_and_run_inference(img):
    print(f"REMOTE: {img.shape}")
    mai.load_state_dict(torch.load(f"{model_store_path}/{MODEL_NAME}"))
    mai.eval()
    img_batch = np.expand_dims(img, axis=0)
    img_batch = torch.tensor(img_batch)
    prediction = mai(img_batch)
    prediction = torch.sigmoid(prediction)
    print(prediction)


@app.local_entrypoint()
def main():
    img = Image.open("test_images/dog.jpg")
    img = preprocess_validation(image=np.array(img))["image"]
    print(img.shape)
    load_model_and_run_inference.remote(img)
