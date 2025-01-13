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
    Analyze the graph and extract the minimum (C_min) and maximum (C_max) concentration values from the ACTUAL DATA POINTS, not the axis limits.
    
    Important notes:
    - Look at the actual experimental points (black squares with error bars or graphic line) on the graph
    - Find the EXACT x-coordinate values for the leftmost and rightmost data points
    - Carefully check the scale and measure the exact position of experimental points
    - Concentration can be expressed in mM, mmol/L, or μM (1 μM = 0.001 mM)
    - For reaction type H2O2+TMB, look at x-axis with H2O2 concentration
    - For reaction type TMB+H2O2, look at x-axis with TMB concentration
    - Ignore graphs showing 1/H2O2 or 1/substrate
    - Only consider experiments with substrate, co-substrate and nanoparticles (no additional additives like Hg2+)
    
    For each graph:
    1. Identify the exact x-coordinate of the leftmost experimental point
    2. Identify the exact x-coordinate of the rightmost experimental point
    3. Double-check these values against the grid lines and scale
    
    Return the values in mM units.
    """

    query = f"""
    {system_prompt}
    
    Please identify:
    1. The reaction type (TMB+H2O2 or H2O2+TMB)
    2. C_min value in mM (EXACT x-coordinate of the leftmost experimental point)
    3. C_max value in mM (EXACT x-coordinate of the rightmost experimental point)
    
    Please double-check all values before submitting.
    carefully measured from leftmost point and from rightmost point
    Format your response EXACTLY as follows (no additional text or formatting):

    For H2O2+TMB reaction:
    Reaction type: H2O2+TMB
    C_min: [value] mM
    C_max: [value] mM

    For TMB+H2O2 reaction:
    Reaction type: TMB+H2O2
    C_min: [value] mM
    C_max: [value] mM
    

    For H2O2 reaction:
    Reaction type: H2O2
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
    
    return result

def main():
    file_path = "s00604-021-05112-5_si.pdf"
    results = extract_concentration_range(file_path, 7)
    print(results)
    for reaction_type, (c_min, c_max) in results.items():
        print(f"\nРеакция: {reaction_type}")
        print(f"C_min: {c_min} mM")
        print(f"C_max: {c_max} mM")
    
    file_path = "d1nj00819f.pdf"
    results = extract_concentration_range(file_path, 4)
    print(results)
    for reaction_type, (c_min, c_max) in results.items():
        print(f"\nРеакция: {reaction_type}")
        print(f"C_min: {c_min} mM")
        print(f"C_max: {c_max} mM")

    
    file_path = "D0TB00239A_si.pdf"
    results = extract_concentration_range(file_path, 4)
    print(results)
    for reaction_type, (c_min, c_max) in results.items():
        print(f"\nРеакция: {reaction_type}")
        print(f"C_min: {c_min} mM")
        print(f"C_max: {c_max} mM")

    file_path = "D0TB00239A.pdf"
    results = extract_concentration_range(file_path, 4)
    print(results)
    for reaction_type, (c_min, c_max) in results.items():
        print(f"\nРеакция: {reaction_type}")
        print(f"C_min: {c_min} mM")
        print(f"C_max: {c_max} mM")


    file_path = "jacs.5b12070_si.pdf"
    results = extract_concentration_range(file_path, 18)
    print(results)
    for reaction_type, (c_min, c_max) in results.items():
        print(f"\nРеакция: {reaction_type}")
        print(f"C_min: {c_min} mM")
        print(f"C_max: {c_max} mM")

    file_path = "am406033q_si.pdf"
    results = extract_concentration_range(file_path, 2)
    print(results)
    for reaction_type, (c_min, c_max) in results.items():
        print(f"\nРеакция: {reaction_type}")
        print(f"C_min: {c_min} mM")
        print(f"C_max: {c_max} mM")

    13 - acsanm.2c05400.pdf
    15 - am501830v_si.pdf
    16 - j.snb.2017.02.059.pdf
    19 - s00604-020-04399-0.pdf
    file_path = "acsanm.2c05400.pdf"
    results = extract_concentration_range(file_path, 4)
    print(results)
    for reaction_type, (c_min, c_max) in results.items():
        print(f"\nРеакция: {reaction_type}")
        print(f"C_min: {c_min} mM")
        print(f"C_max: {c_max} mM")


    file_path = "am501830v_si.pdf"
    results = extract_concentration_range(file_path, 3)
    print(results)
    for reaction_type, (c_min, c_max) in results.items():
        print(f"\nРеакция: {reaction_type}")
        print(f"C_min: {c_min} mM")
        print(f"C_max: {c_max} mM")
    
    results = extract_concentration_range(file_path, 4)
    print(results)
    for reaction_type, (c_min, c_max) in results.items():
        print(f"\nРеакция: {reaction_type}")
        print(f"C_min: {c_min} mM")
        print(f"C_max: {c_max} mM")
    

    file_path = "j.snb.2017.02.059.pdf"
    results = extract_concentration_range(file_path, 5)
    print(results)
    for reaction_type, (c_min, c_max) in results.items():
        print(f"\nРеакция: {reaction_type}")
        print(f"C_min: {c_min} mM")
        print(f"C_max: {c_max} mM")

    
    file_path = "s00604-020-04399-0.pdf"

    results = extract_concentration_range(file_path, 10)
    print(results)
    for reaction_type, (c_min, c_max) in results.items():
        print(f"\nРеакция: {reaction_type}")
        print(f"C_min: {c_min} mM")
        print(f"C_max: {c_max} mM")
if __name__ == "__main__":
    main()