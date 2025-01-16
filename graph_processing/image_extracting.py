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
from graph_processing.image_reasoning import extract_concentration_range

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


# Функция для анализа изображения и получения текстового описания с помощью OpenAI API
def analyze_image(image):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_image_path = f"{temp_dir}/temp_image.png"
        image.save(temp_image_path, format="PNG")
        image_documents = SimpleDirectoryReader(temp_dir).load_data()
        query = (
            "Extract relevant kinetic data from various types of visual materials (e.g., graphs, tables, diagrams) commonly found in the context of enzyme-like nanozyme activity studies. "
            "These may include steady-state kinetic plots, Michaelis–Menten plots, Lineweaver–Burk plots, reaction velocity graphs, or concentration versus rate graphs, as well as tables summarizing key kinetic parameters."
            "The model should first identify the type of visual representation. If a graph is detected, focus on graphs where concentration is plotted along the X-axis and reaction velocity is on the Y-axis. "
            "Make sure to identify the Cmin (lower left point) and Cmax (upper right point) on the X-axis for concentration ranges (typically in mM or μM). "
            "Pay attention to captions or labels that mention kinetic analysis, Michaelis-Menten, Lineweaver–Burk, or steady-state kinetics, to confirm the relevance of the graph."
            "Key terms and phrases to look for include: "
            "'Steady-state kinetic', 'Lineweaver–Burk plot', 'Michaelis-Menten model', 'reaction velocities vs. substrate concentration', "
            "'Kinetic analysis for PBNPs', 'Peroxidase-like activity', 'Michaelis–Menten behavior', 'substrate concentration change', 'nanoparticles'. "
            "If a table is detected, extract kinetic parameters such as Km, Vmax, and Kcat, paying close attention to units (e.g., mM, μM). Tables can provide summarized values that may not always be directly interpretable from graphs, so prioritize extracting exact values when available. "
            "When encountering diagrams or non-relevant visual data (e.g., non-kinetic graphs or diagrams), label them as such and pass them for further analysis but do not attempt to extract concentrations from them."
            "If neither graphs nor tables are detected, the model should flag the image as potentially irrelevant to the kinetic analysis task and move to the next one. "
            "In case the image appears to be part of a different section of the study (e.g., materials, methods, or background information), mark it accordingly."
            "Additionally, when working in conjunction with the main agent, ensure that any relevant kinetic context derived from the article’s text (e.g., nanoparticle type, specific formulas, or reaction conditions) is passed along. "
            "Use this context to guide the search for specific kinetic information within graphs or tables. If kinetic parameters or related data (e.g., specific substrate concentrations or reaction rates) are not found, flag this as a missing data point and request further investigation."
            "For inappropriate or unusable images (e.g., 1/[H2O2] or 1/[substrate]), mark them as non-relevant to the current task and continue the analysis. Avoid graphs that use inverse concentrations or other less directly interpretable data."
        )

        openai_mm_llm = OpenAIMultiModal(
            model="gpt-4o",
            api_key=OPENAI_API_KEY,
            temperature=0.0,
        )

        response = openai_mm_llm.complete(
            prompt=query,
            image_documents=image_documents,
        )
    return response


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(model_path)
    model.to(device=device)

    processed_images = []
    results = model(images)

    for i, res in enumerate(results):
        cropped_images = crop_images(images[i], res.boxes)
        processed_images.extend(cropped_images)

    return processed_images


# Функция для анализа PDF и получения описаний
def pdf_analysis(pdf_path, yolo_model_path = None):
    """
    Analyze PDF file and extract structured information about nanozymes from images.
    Returns a dictionary with analysis results for each page containing relevant information.
    """
    image_pages = extract_image_pages(pdf_path)
    analyses = {}
    images = []

    for page_num in image_pages:
        image = get_page_image(pdf_path, page_num)
        images.append(image)

    if yolo_model_path:
        images = process_images_with_yolo(images, yolo_model_path=YOLO_PATH)

    for i, image in enumerate(images):
        analysis = extract_concentration_range(image)
        
        # Only include pages with relevant nanozyme information
        if analysis.image_type != "error" and (
            analysis.concentration_data or 
            analysis.kinetic_parameters or 
            (analysis.nanozyme_properties and any(v is not None for v in analysis.nanozyme_properties.dict().values()))
        ):
            analyses[i] = analysis.dict()

    return analyses
