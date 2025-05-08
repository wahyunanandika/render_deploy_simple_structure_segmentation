from fastapi import FastAPI, UploadFile,File,HTTPException, Depends, Request
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from slowapi import Limiter
from slowapi.util import get_remote_address
import uuid
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from huggingface_hub import hf_hub_download,login, HfApi
from dotenv import load_dotenv
from io import BytesIO
from datetime import datetime
import os
import logging
import shutil
from PIL import Image

# setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load ENV for loggin to hugging face

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
api_key_header = APIKeyHeader(name = "X-API-KEY")

# Rate limiting setup
limiter = Limiter(key_func=get_remote_address)

app = FastAPI()

# CORS configuration (Allow requests from front end domain)

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],# Allow all HTTP methods
    allow_headers=["*"],# Allow all headers
)

# handle preflight OPTIONS requests explicitly
@app.options("/predict/")
async def options_handler():
    return Response(status_code=200)

def authenticate(api_key: str = Depends(api_key_header)):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=401, detail="Unauthorized")

# Login to hugging face

login(token=hf_token)

# Download model from Hugging face hub

model_path = hf_hub_download(repo_id="wahyunan/simple_structure_model_segmentation",filename="best.pt")

# Load model YOLOv8 segmentation

model = YOLO(model_path)

# color for classes

CLASS_COLORS = {
    "window": (0,255,255,100),
    "door": (0,0,255,100)
}

# Custom font for text label (make sure you have the TTF file or use a default one)
def get_font(size:int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except IOError:
        logger.warning("Arial font not found, using default font.")
        return ImageFont.load_default()

@app.post("/predict/")
@limiter.limit("10/minute") # Limit to 10 requests per minute
async def predict(request:Request,file: UploadFile=File(...)):
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
        # # Save to memory
        # contents = await file.read()
        # try:
        #     image = Image.open(BytesIO(contents)).convert("RGB")  # Convert to RGB format
        # except IOError as e:
        #     logger.error(f"Error opening image: {str(e)}")
        #     raise HTTPException(status_code=400, detail="Invalid image file.")
         # Save to disk
        temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        results = model(temp_filename)[0]
        names = model.names

        # Overlay and process
        image = Image.open(temp_filename).convert("RGB")
        # # Save to BytesIO for inferences
        # temp_io = BytesIO(contents)
        # temp_io.name = f"{uuid.uuid4().hex}.jpg" # Yolo needs name
        # results = model(temp_io)[0]
        # names = model.names

        # result of segmentation process
        overlay = Image.new("RGBA", image.size)
        draw = ImageDraw.Draw(overlay)
        masks_polygons = []

        for seg, cls, conf in zip(results.masks.xy, results.boxes.cls, results.boxes.conf):
            class_id = int(cls)
            label = names[class_id]
            confidence = float(conf)
            polygon = [(float(x),float(y)) for x, y in seg]

            masks_polygons.append({
                "class": label,
                "confidence": round(confidence,3),
                "polygon": polygon
            })

            color = CLASS_COLORS.get(label, (255,255,0,100))
            draw.polygon(polygon, fill=color)
            # add text to label and confidence score
            font = get_font(18)
            if polygon:
                draw.text(polygon[0], f"{label} {confidence:.2f}", fill=(255,255,255,255),font=font)
        
        combined = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

        #Save result to memory and upload to HF
        result_io = BytesIO()
        combined.save(result_io, format="JPEG")
        result_io.seek(0)

        filename = f"result_{uuid.uuid4().hex}.jpg"
        hf_path = f"results/{datetime.now().strftime('%Y-%m-%d')}/{filename}"

        api = HfApi()

        uploaded_url = api.upload_file(
            path_or_fileobj=result_io,
            path_in_repo=hf_path,
            repo_id="wahyunan/simple_structure_model_segmentation",
            repo_type="model"
        )
        os.remove(temp_filename)
        return JSONResponse(content={
            "message": 'Segmentation Completed',
            "results" : masks_polygons,
            "huggingface_url": uploaded_url
        })
    except HTTPException as e:
        logger.error(f"HTTP Exception: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error: An error occurred during processing")


    