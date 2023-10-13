import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from examples.advanced.sdxl.sdxl import SDXL
from pydantic import BaseModel
from io import BytesIO
from time import sleep
import os
# Importing the SDXL model
from examples.advanced.sdxl.sdxl import SDXL

class ImageRequest(BaseModel):
    prompt: str

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the SDXL model
model = SDXL()

@app.post("/generate-custom-image")
def generate_custom_image(request: ImageRequest):
    # Use the SDXL model to generate an image based on the provided prompt
    response = model.run(prompt=request.prompt)
    
    # Return the generated image
    return response
