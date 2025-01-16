import base64
import io
import os
from typing import Dict, Tuple, Optional, List, Union
import fitz
from PIL import Image
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from openai import OpenAI
import json

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

class ConcentrationData(BaseModel):
    reaction_type: str = Field(description="Type of reaction (e.g. TMB+H2O2, H2O2+TMB)")
    c_min: float = Field(description="Minimum concentration value in mM")
    c_max: float = Field(description="Maximum concentration value in mM")
    co_substrate_concentration: Optional[float] = Field(None, description="Concentration of co-substrate in mM if specified")

class KineticParameters(BaseModel):
    km: Optional[float] = Field(None, description="Michaelis constant Km in mM")
    vmax: Optional[float] = Field(None, description="Maximum reaction rate Vmax in mM/s")
    kcat: Optional[float] = Field(None, description="Turnover number kcat in s^-1")

class NanozymeProperties(BaseModel):
    formula: Optional[str] = Field(None, description="Chemical formula of the nanozyme")
    activity: Optional[str] = Field(None, description="Type of activity (peroxidase, oxidase, etc.)")
    syngony: Optional[str] = Field(None, description="Crystal system")
    size: Optional[Dict[str, float]] = Field(None, description="Size parameters in nm (length, width, depth or diameter)")
    surface_chemistry: Optional[str] = Field(None, description="Surface modification")

class ImageAnalysis(BaseModel):
    image_type: str = Field(description="Type of image (concentration_graph, kinetic_table, tem_image, etc.)")
    nanozyme_properties: Optional[NanozymeProperties] = Field(None, description="Properties of nanozyme if mentioned")
    concentration_data: Optional[List[ConcentrationData]] = Field(None, description="Concentration data if present")
    kinetic_parameters: Optional[List[KineticParameters]] = Field(None, description="Kinetic parameters if present")
    description: str = Field(description="Brief description of what was found in the image")

def extract_concentration_range(image) -> ImageAnalysis:
    """
    Analyze image and extract structured information about nanozyme properties, concentrations and kinetic parameters.
    
    The function uses GPT-4V to analyze various types of images (graphs, tables, TEM images) and returns structured data.
    """
    
    system_prompt = """
    Analyze the image and extract information about nanozyme properties, concentrations and kinetic parameters.
    
    Pay attention to:
    1. Type of image (concentration_graph, kinetic_table, tem_image, etc.)
    2. For concentration graphs:
       - Look for actual experimental points (dots/squares with error bars)
       - Find leftmost (C_min) and rightmost (C_max) points on concentration axis
       - Identify reaction type and co-substrate concentration
       - Check if points at x=0 are present
       - Note typical ranges: TMB (0-1.0 mM), H2O2 (0-100 mM)
    3. For kinetic tables:
       - Extract Km, Vmax, kcat values with units
       - Match parameters to specific reaction types
    4. For TEM/SEM/other images:
       - Note nanozyme formula, activity type, crystal system
       - Extract size parameters (length, width, depth, diameter)
       - Note surface modifications if mentioned
    
    Ignore:
    - Lineweaver-Burk plots (1/v vs 1/[S])
    - Non-kinetic data
    - Images without nanozyme-related information
    """
    
    # Convert image to base64
    if isinstance(image, str):
        base64_image = pdf_page_to_base64(image)
    else:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    # Initialize OpenAI client
    client = OpenAI()
    
    # Prepare message
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user", 
            "content": [
                {"type": "text", "text": "Analyze this image and extract all relevant information about nanozymes."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }
    ]
    
    try:
        # Get response with structured output
        completion = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=messages,
            response_format=ImageAnalysis
        )
        return completion
    except Exception as e:
        # If parsing fails, return a basic analysis indicating an error
        return ImageAnalysis(
            image_type="error",
            description=f"Failed to parse image: {str(e)}"
        )

