import base64
import io
import os
from typing import Dict, Tuple
import fitz
from PIL import Image
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def pdf_page_to_base64(pdf_path: str, page_number: int) -> str:
    """Convert PDF page to base64 string."""
    pdf_document = fitz.open(pdf_path)
    page = pdf_document.load_page(page_number - 1)  # input is one-indexed
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def extract_concentration_range(file_path: str, page_number: int) -> Dict[str, Tuple[float, float]]:
    """
    Extract C_min and C_max values from graphs in the PDF.
    Returns a dictionary with reaction types as keys and (C_min, C_max) tuples as values.
    """
    base64_image = pdf_page_to_base64(file_path, page_number)
    llm = ChatOpenAI(model="gpt-4o-2024-11-20", max_tokens=1024)

    system_prompt = """
    Analyze the velocity vs concentration graphs and extract the EXACT minimum (C_min) and maximum (C_max) concentration values from the ACTUAL DATA POINTS.
    
    Critical measurement rules:
    1. Look at the actual experimental points (black squares/dots with error bars) on the graph
    2. For EACH velocity vs concentration graph:
       - Find the LEFTMOST experimental point - this is C_min
       - Find the RIGHTMOST experimental point - this is C_max
       - Pay special attention to points
       - Include ALL visible data points
    
    Important details:
    - Look carefully at the axis scale and grid lines
    - Use proportional measurement for points between grid lines
    - Include the full range of experimental points 
    - Ignore Lineweaver-Burk plots (1/v vs 1/[S])
    - Concentration can be expressed in mM, mmol/L, or μM (1 μM = 0.001 mM)

    
    Measurement process:
    1. First identify the graph type by x-axis label:
       - TMB concentration (mM) → TMB+H2O2 reaction
       - H2O2 concentration (mM) → H2O2+TMB reaction
    2. For each identified graph:
       - Scan all points to find the rightmost one
       - Measure exact positions using grid lines
    3. Double-check all measurements
    
    Return all values in mM units.
    """

    query = f"""
    {system_prompt}
    
    For each velocity vs concentration graph (ignore Lineweaver-Burk plots), provide:
    1. Reaction type based on x-axis label
    2. C_min: EXACT x-coordinate of the LEFTMOST point
    3. C_max: EXACT x-coordinate of the RIGHTMOST point
    
    Format your response EXACTLY as follows (no additional text):

    For H2O2+TMB reaction:
    Reaction type: H2O2+TMB
    C_min: [value] mM
    C_max: [value] mM

    For TMB+H2O2 reaction:
    Reaction type: TMB+H2O2
    C_min: [value] mM
    C_max: [value] mM
    """

    message = HumanMessage(
        content=[
            {"type": "text", "text": query},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            },
        ]
    )
    
    print(f"Processing page {page_number} from file {file_path}")
    print("Image successfully converted to base64")
    print("Sending request to OpenAI...")
    response = llm.invoke([message])
    print("Got response from OpenAI")
    print("Raw response:", response.content)
    
    # Parse the response
    result = {}
    lines = response.content.split('\n')
    print('lines', lines)
    
    current_type = None
    current_c_min = None
    current_c_max = None
    
    for line in lines:
        line = line.strip()
        if not line:  # Skip empty lines
            continue
            
        if line.startswith('Reaction type:'):
            # Если у нас есть предыдущий набор данных, сохраним его
            if current_type and current_c_min is not None and current_c_max is not None:
                result[current_type] = (current_c_min, current_c_max)
            
            current_type = line.split('Reaction type:')[1].strip()
            current_c_min = None
            current_c_max = None
            
        elif line.startswith('C_min:'):
            try:
                value = line.split('C_min:')[1].split('mM')[0].strip()
                current_c_min = float(value)
            except (ValueError, IndexError):
                continue
                
        elif line.startswith('C_max:'):
            try:
                value = line.split('C_max:')[1].split('mM')[0].strip()
                current_c_max = float(value)
            except (ValueError, IndexError):
                continue
    
    # Добавляем последний набор данных
    if current_type and current_c_min is not None and current_c_max is not None:
        result[current_type] = (current_c_min, current_c_max)
    
    return response

