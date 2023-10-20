import sys
import os
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from examples.advanced.sdxl.sdxl import SDXL
from pydantic import BaseModel
from io import BytesIO
from time import sleep
from typing import List
from examples.advanced.sdxl.sdxl import SDXL

class ImageRequest(BaseModel):
    prompt: str

class OpenAIRequest(BaseModel):
    prompt: str
    n: int
    size: str

class OpenAIResponse(BaseModel):
    created: int
    data: List[dict]


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

model = SDXL()

#Generate Image from GPUtopia
@app.post("/generate-custom-image")
def generate_custom_image(request: ImageRequest):
    # Use the SDXL model to generate an image based on the provided prompt
    response = model.run(prompt=request.prompt)
    # Return the generated image
    return response


#Generate image from GPUtopia API
#Request & Response in OpenAI API format
@app.post("/openai-compat", response_model=OpenAIResponse)
def openai_compat_endpoint(request: OpenAIRequest):

    # Note: May need to adjust this part based on how SDXL model handles multiple image generation (n parameter)
    images = [model.run(prompt=request.prompt) for _ in range(request.n)]
    
    # Construct the response
    response_data = [{"url": image_url} for image_url in images]

    response = {
        "created": int(time.time()),
        "data": response_data
    }
    
    return response

