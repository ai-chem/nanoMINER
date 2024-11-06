# nanoMINER: Multimodal Information Extraction for Nanomaterials

This repository contains a cutting-edge multi-agent system that integrates large language models with multimodal analysis to extract crucial information on nanomaterials from research articles. 

This system processes scientific documents end-to-end, leveraging tools such as the YOLO model for visual data extraction and GPT-4o for linking textual and visual information. The core of our architecture is the ReAct agent, which orchestrates various specialized agents, ensuring comprehensive and accurate data extraction. We demonstrate its efficacy through a case study in nanozyme research.

## Features

- Upload and process PDF files of scientific articles and supplementary information
- Extract text from PDF files
- Utilize an AI agent to answer questions about the uploaded articles
- Handle multiple file uploads, including separate article and supplement files

## Installation

1. Clone this repository:

```bash
git clone https://github.com/ai-chem/LLM-Pipeline-for-Automated-Extraction-of-Nanozyme-Data.git
cd LLM-Pipeline-for-Automated-Extraction-of-Nanozyme-Data
```

2. Install the required packages:

```bash
poetry install
```
This command will install all necessary dependencies from pyproject.toml file using poetry package manager.

3. Set up your OpenAI API key in a .env file:

```bash
OPENAI_API_KEY=your_api_key_here
```

## Usage

### Agent app

1. Run the Streamlit app:

```bash
poetry run streamlit run agent_app.py
```

2. Open the provided URL in your web browser.

3. Upload a PDF file of a scientific article (and optionally, a supplementary information file).

4. Once the files are processed, you can start asking questions about the article in the chat interface.

### Auto extraction

The auto_extraction.py script is designed for batch processing of PDF files to extract detailed information about nanozyme experiments. It uses a multi-agent system to analyze scientific articles and supplementary information files, extracting named entities and other relevant data.

- To run the script, use the following command:

    ```bash
    python auto_extraction.py <pdf_articles_dir> <pdf_supplements_dir> <ner_json_dir> <results_dir>
    ```
    Where:
    - **pdf_articles_dir**: Directory containing the PDF articles.
    - **pdf_supplements_dir**: Directory containing the supplementary PDF files.
    - **ner_json_dir**: Directory containing the JSON files with named entity recognition (NER) data.
    - **results_dir**: Directory where the results will be saved.

### Structured Output

The structured_output.ipynb Jupyter Notebook is designed for converting the markdown ReAct agent's answers into structured tabular data. This notebook also contains the calculation of the Jaccard Index on the full dataset.

## File Structure

- agent_app.py: Main Streamlit application file;
- auto_extraction.py: Script for automated extraction of information from multiple PDF files;
- structured_output.ipynb: Jupyter Notebook for structured output postprocessing and calculation of the Jaccard Index.
- pdf2txt.py: Module for extracting text from PDF files;
- utils.py: Utility functions;
- logger.py: Logging configuration;
- image_processing/: Directory containing image processing modules.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the [MIT License](LICENSE).
