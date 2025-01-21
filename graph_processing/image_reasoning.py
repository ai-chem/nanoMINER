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
    image_type: str = Field(description="Type of image (concentration_graph)")
    nanozyme_properties: Optional[NanozymeProperties] = Field(None, description="Properties of nanozyme if mentioned")
    concentration_data: Optional[List[ConcentrationData]] = Field(None, description="Concentration data if present")
    kinetic_parameters: Optional[List[KineticParameters]] = Field(None, description="Kinetic parameters if present")
    description: str = Field(description="Brief description of what was found in the image")

def extract_concentration_range(image) -> ImageAnalysis:
    """
    Analyze image and extract structured information about nanozyme properties, concentrations and kinetic parameters.
    
    The function uses GPT-4V to analyze various types of images (graphs) and returns structured data.
    """
    
    # Load and encode example image
    with open("graph_processing/conc_example.jpg", "rb") as image_file:
        example_base64 = base64.b64encode(image_file.read()).decode("utf-8")
    
    system_prompt = """
    Analyze the image and extract information about nanozyme properties, concentrations and kinetic parameters.
    
    Here is an example of a concentration graph that you should look for:
    <image>data:image/jpeg;base64,{example_base64}</image>
    
    Pay attention to:
    1. Type of image (concentration_graph)
    2. For concentration graphs:
       - y-axis is velocity (v)
       - x-axis is concentration (C), units are mM, µM, nM, etc.
       - Look for actual experimental points (dots/squares with error bars)
       - Find leftmost (C_min) and rightmost (C_max) points on concentration axis
       - Identify reaction type and co-substrate concentration
       - Check if points at x=0 are present
    
    IGNORE OTHER TYPES OF GRAPHS!
    Ignore also:
    - Lineweaver-Burk plots (1/v vs 1/[S])
    - Non-kinetic data
    - Images without nanozyme-related information
    # """
    
    # system_prompt = """
    # Analyze the image and extract information about nanozyme properties, concentrations and kinetic parameters.
    
    # IMPORTANT: Only analyze concentration vs velocity (also called "v") plots or kinetic measurements. All other types of graphs should be ignored completely.
    
    # Here is an example of a concentration graph that you should look for:
    # <image>data:image/jpeg;base64,{example_base64}</image>

    # Result of the analysis EXAMPLE:
    # TMB (a): Concentration range is from 0.1 to 1.3 mM (from 100 to 1300 µM).
    # H₂O₂ (b): Concentration range is from 10 to 130 mM.
    # ABTS (c): Concentration range is from 0.05 to 0.75 mM (from 50 to 750 µM).
    # H₂O₂ (d): Concentration range is from 10 to 130 mM.


        
    # Pay attention to:
    # 1. Type of image - MUST be a concentration vs velocity (also called "v") plot showing:
    #    - We need to look at the lines and find the concentration range for each of the substrates.
    #    - Concentration (mM) on X-axis 
    #    - Reaction VELOCITY (v) on Y-axis
    #    - Experimental data points (dots/squares with error bars)
    #    - Direct plot (not reciprocal/transformed data)
    
    # 2. For valid concentration graphs only:
    #    - Find leftmost (C_min) and rightmost (C_max) points on concentration axis
    #    - Identify reaction type and co-substrate concentration
    #    - Check if points at x=0 are present
    #    - Note typical units: μM (1000 μM = 1 mM), mM
    #    - Look for velocity (v) on Y-axis
       
    # 3. For kinetic tables:
    #    - Extract Km, Vmax, kcat values with units
    #    - Match parameters to specific reaction types
    
    # Strictly ignore and do not analyze:
    # - Lineweaver-Burk plots (1/v vs 1/[S])
    # - Non-kinetic data
    # - Images without nanozyme-related information
    # """
    
    # Convert image to base64
    if isinstance(image, str):
        base64_image = pdf_page_to_base64(image)
    else:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    # Initialize OpenAI client
    client = OpenAI()
    
    # Prepare message with formatted system prompt
    messages = [
        {"role": "system", "content": system_prompt.format(example_base64=example_base64)},
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
        # Extract ImageAnalysis object from ParsedChatCompletion
        return completion.choices[0].message.parsed
    except Exception as e:
        # If parsing fails, return a basic analysis indicating an error
        return ImageAnalysis(
            image_type="error",
            description=f"Failed to parse image: {str(e)}"
        )

def extract_table_markdown(image) -> Optional[str]:
    """
    Convert table image to markdown format using GPT-4V.
    Returns markdown string or None if conversion failed.
    """
    # Convert image to base64
    if isinstance(image, str):
        base64_image = pdf_page_to_base64(image)
    else:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    system_prompt = """
    You are a specialized assistant that converts tables from images into markdown format.
    - Create proper markdown tables with aligned columns
    - Preserve all headers and data exactly as shown
    - Include any table captions or notes
    - Maintain units and formatting
    - If the image is not a table, return "NOT_A_TABLE"
    """
    
    client = OpenAI()
    
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Convert this table to markdown format."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=messages,
            max_tokens=1000
        )
        result = response.choices[0].message.content
        return None if result == "NOT_A_TABLE" else result
    except Exception as e:
        print(f"Failed to convert table: {str(e)}")
        return None

