from fastapi import APIRouter, UploadFile, File, Form
import os
import shutil
import uuid
from typing import List

from app.services.embedding import get_embedding
from app.services.vector_store import add_vector, search_vector
from app.services.image_utils import generate_caption
from app.services.llm.factory import get_llm

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/add_item")
async def add_item(
    name: str = Form(...),
    description: str = Form(""),
    files: List[UploadFile] = File(..., description="Upload multiple files", media_type="image/*")
):
    file_paths = []
    for file in files:
        file_path = f"{UPLOAD_DIR}/{uuid.uuid4()}_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file_paths.append(file_path)

    llm = get_llm()
    enhanced_description = llm.enhance_description(image_paths=file_paths, item_name=name, description=description)

    final_text = f"""
        Item: {name}
        User description: {description}
        Enhanced: {enhanced_description}
        """

    embedding = get_embedding(final_text)

    metadata = {
        "name": name,
        "user_description": description,
        "enhanced_description": enhanced_description,
        "image_paths": file_paths
    }

    add_vector(embedding, metadata)

    return {
        "message": "Item added",
        "ai_description": enhanced_description
    }


@router.post("/search")
def search(text: str):
    embedding = get_embedding(text)
    results = search_vector(embedding)

    return {"results": results}