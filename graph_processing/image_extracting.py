import io
import os
import fitz
import torch
import tempfile
from pathlib import Path
from time import time
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from PIL import Image
from ultralytics import YOLO
from graph_processing.image_reasoning import extract_concentration_range, extract_table_markdown

# Загрузка переменных окружения из .env файла
load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
YOLO_PATH = os.getenv("YOLO_PATH")


# Функция для извлечения страниц с изображениями из PDF
def extract_image_pages(pdf_path):
    doc = fitz.open(pdf_path)
    image_pages = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        images = page.get_images(full=True)
        if images:
            image_pages.append(page_num)
    return image_pages


# Функция для извлечения содержимого страницы в виде изображения
def get_page_image(pdf_path, page_num):
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)
    pix = page.get_pixmap(dpi=300)
    img_data = io.BytesIO(pix.tobytes())
    image = Image.open(img_data)
    return image


# Функция для обрезки изображения по границам YOLO
def crop_images(image, boxes):
    cropped_images = []
    for box in boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        cropped_image = image.crop((xyxy[0], xyxy[1], xyxy[2], xyxy[3]))
        cropped_images.append(cropped_image)
    return cropped_images


# Функция для обработки изображений с помощью YOLO и обрезки по границам
def process_images_with_yolo(images, model_path):
    if not images:  # Check for empty list
        return [], []
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(model_path)
    model.to(device=device)

    processed_images = []
    table_images = []
    
    try:
        results = model(images)

        for i, res in enumerate(results):
            if not res.boxes:  # Skip if no detections
                continue
                
            for box, cls in zip(res.boxes, res.boxes.cls):
                cropped_image = crop_images(images[i], [box])[0]
                if int(cls) == 1:  # Table class
                    table_images.append(cropped_image)
                else:
                    processed_images.append(cropped_image)

    except Exception as e:
        print(f"Error in YOLO processing: {str(e)}")
        return [], []

    return processed_images, table_images


# Функция для анализа PDF и получения описаний
def pdf_analysis(pdf_path, yolo_model_path = YOLO_PATH):
    """
    Analyze PDF file and extract structured information about nanozymes from images
    and convert tables to markdown.
    Returns a dictionary with analysis results and table markdowns.
    """
    image_pages = extract_image_pages(pdf_path)
    images = []

    for page_num in image_pages:
        image = get_page_image(pdf_path, page_num)
        images.append(image)

    analyses = []
    tables = []
    
    if yolo_model_path and images:  # Check if we have any images to process
        try:
            graph_images, table_images = process_images_with_yolo(images, model_path=YOLO_PATH)
            
            # Process graphs
            for image in graph_images:
                analysis = extract_concentration_range(image)
                if analysis.image_type == "concentration_graph" and (
                    analysis.concentration_data or 
                    analysis.kinetic_parameters or 
                    (analysis.nanozyme_properties and any(v is not None for v in analysis.nanozyme_properties.dict().values()))
                ):
                    analyses.append(analysis.dict())
            
            # Process tables
            for table_image in table_images:
                table_markdown = extract_table_markdown(table_image)
                if table_markdown:
                    tables.append(table_markdown)
        except Exception as e:
            print(f"Error processing images: {str(e)}")
            # Return empty results if processing fails
            return {
                "analyses": [],
                "tables": [],
                "error": str(e)
            }
    
    return {
        "analyses": analyses,
        "tables": tables
    }
