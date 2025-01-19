import json
import os
from tempfile import NamedTemporaryFile

import streamlit as st
from dotenv import load_dotenv
from graph_processing.image_extracting import pdf_analysis
from langchain.agents import AgentType, initialize_agent, tool
from langchain.callbacks import StreamlitCallbackHandler
from langchain_openai import ChatOpenAI
from data_preproccessing.pdf2txt import extract_text_from_pdf

from logger import LOGGER
from utils import merge_dictionaries

load_dotenv(override=True)

LOGGER.info(f'OPENAI_API_KEY = "{os.getenv("OPENAI_API_KEY")}"')

st.set_page_config(page_title="Nanozyme AI Assistant")
st.header("Nanozyme AI Assistant")

st.session_state["main_container"] = st.container()

if "files" not in st.session_state:
    st.session_state["files"] = None

if "united_params" not in st.session_state:
    st.session_state["united_params"] = None

if "article_file" not in st.session_state and "supplement_file" not in st.session_state:
    st.session_state["article_file"] = None
    st.session_state["supplement_file"] = None

if prompt := st.chat_input():
    try:
        if st.session_state["article_text"]:
            st.session_state["main_container"] = st.container()
            with st.session_state["main_container"]:
                st.chat_message("user").write(prompt)
                try:
                    st.session_state["streamlit_callback_container"] = st.container()
                    st_callback = StreamlitCallbackHandler(
                        st.session_state["streamlit_callback_container"]
                    )
                    response = st.session_state["agent"].run(
                        prompt, callbacks=[st_callback]
                    )
                    st.chat_message("assistant").write(response)
                except Exception as e:
                    LOGGER.error(e)
        else:
            st.error("Agent is not initialized")
    except Exception as e:
        st.error("You should upload an article first")
        LOGGER.error(e)

if st.session_state["files"] is None:
    st.info("Upload an Article to init Agent")
files = st.file_uploader(
    "Upload an Article to init Agent", type=["pdf"], accept_multiple_files=True
)

if files != [] and st.session_state["files"] != files:
    LOGGER.info("File upload start")
    st.session_state["files"] = files
    LOGGER.info(files)

    article_file = None
    supplement_file = None

    if len(files) > 2:
        st.error(
            f"You should upload only 1 article and supplement information. Continue with only {files[0].name} file"
        )
        article_file = files[0]
    elif len(files) == 2:
        if files[0].name[-7:] == "_si.pdf":
            supplement_file = files[0]
            article_file = files[1]
        elif files[1].name[-7:] == "_si.pdf":
            supplement_file = files[1]
            article_file = files[0]
        else:
            st.error(
                f"Can't find suplement. Be sure that one of your files has _si.pdf suffix. Continue with only {files[0].name} file"
            )
            article_file = files[0]
    else:
        article_file = files[0]
        supplement_file = None

    st.session_state["article_file"] = article_file
    st.session_state["supplement_file"] = supplement_file

    files_dict = {
        "article": st.session_state["article_file"].name,
        "supplement": (
            st.session_state["supplement_file"].name
            if st.session_state["supplement_file"]
            else None
        ),
    }

    st.info("Uploaded information:\n```json\n" + str(files_dict) + "\n```")

    st.session_state["article_text"] = extract_text_from_pdf(article_file)
    st.session_state["supplement_text"] = (
        extract_text_from_pdf(supplement_file) if supplement_file else None
    )

    @tool("get_full_text")
    def get_full_text(query: str) -> str:
        "Returns full text of the article and supplement information if provided."

        full_text_dict = {}
        full_text_dict["article_text"] = st.session_state["article_text"]
        full_text_dict["supplement_text"] = st.session_state["supplement_text"]

        return "```json\n" + str(full_text_dict) + "\n```"

    @tool("analyze_images")
    def analyze_images(file_name: str) -> str:
        "Extracts the minimum (Cmin) and maximum (Cmax) substrate concentrations from kinetic data on pages including pictures and graphs. Also extracts tables in markdown format. It uses GPT-4 to analyze images. Returns Cmin and Cmax for each page of the article and supplementary if provided, along with any tables found."
        results_dict = {}

        with NamedTemporaryFile(dir=".", suffix=".pdf", delete=False) as f:
            f.write(st.session_state["article_file"].getbuffer())
            results_dict["article"] = pdf_analysis(f.name)
        os.remove(f.name)

        if st.session_state["supplement_file"] is not None:
            with NamedTemporaryFile(dir=".", suffix=".pdf", delete=False) as f:
                f.write(st.session_state["supplement_file"].getbuffer())
                results_dict["supplement"] = pdf_analysis(f.name)
            os.remove(f.name)
        
        return "```json\n" + str(results_dict) + "\n```"

    @tool("llm_extractor")
    def llm_extractor(file_name: str) -> str:
        "Extracts Named Entities from article with the help of mistral-7b and llama3-8b. Can be useful when you need to get parameters of the experiments. WARNING: this tool does not separate parameters by experiment and may miss some important parameters. You should use this tool first, and then be sure to check everything and fix it with the full text."

        article_name = st.session_state["article_file"].name[:-4]
        mistral_article_dict = {}
        mistral_si_dict = {}

        if os.path.exists("./mistral_json"):
            with open(
                f"./mistral_json/{article_name}_mistral.json", "r", encoding="utf-8"
            ) as file:
                mistral_article_dict = json.load(file)

            with open(
                f"./mistral_json/{article_name}_si_mistral.json", "r", encoding="utf-8"
            ) as file:
                mistral_si_dict = json.load(file)

        mistral_dict = merge_dictionaries([mistral_article_dict, mistral_si_dict])

        llama_article_dict = {}
        llama_si_dict = {}
        if os.path.exists("./llama_json"):
            with open(
                f"./llama_json/{article_name}_lama.json", "r", encoding="utf-8"
            ) as file:
                llama_article_dict = json.load(file)

            with open(
                f"./llama_json/{article_name}_si_lama.json", "r", encoding="utf-8"
            ) as file:
                llama_si_dict = json.load(file)

        llama_dict = merge_dictionaries([llama_article_dict, llama_si_dict])

        return (
            "```"
            + "\nmistral-7b answer:\n"
            + str(mistral_dict)
            + "\nllama3-8b answer:\n"
            + str(llama_dict)
            + "\n```"
        )

    tools = [get_full_text, analyze_images]
    if os.path.exists("./llama_json") or os.path.exists("./mistral_json"):
        tools.append(llm_extractor)

    agent_llm = ChatOpenAI(
        temperature=0,
        model="gpt-4o",
        streaming=True,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    prompt = """
    You are helpful assistant in chemistry, specializing in nanozymes. Your task is to analyze scientific articles and extract detailed information about various experiments with nanozymes. It is crucial for you to accurately and comprehensively describe each experiment separately, without referring to other experiments in the article.
    Usually, the articles contain several experiments with nanozymes with different parameters, such as formula, activity (usually peroxidase, oxidase, catalase or laccase), syngony (usually cubic, hexagonal, tetragonal, monoclinic, orthorhombic, trigonal, amorphous or triclinic), length, width and depth (or just size or diameter), surface chemistry (naked by default or poly(ethylene oxide), poly(N-Vinylpyrrolidone), Tetrakis(4-carboxyphenyl)porphine or other), polymer used in synthesis (none or poly(N-Vinylpyrrolidone), oleic acid, poly(ethylene oxide), BSA or other), surfactant (none or l-ascorbic acid, ethylene glycol, sodium citrate, cetrimonium bromide, citric acid, trisodium citrate, ascorbic acid or other), molar mass, Michaelis constant Km, molar maximum reaction rate Vmax, reaction type (substrat + co-substrat) (TMB + H2O2, H2O2 + TMB, TMB, ABTS + H2O2, H2O2, OPD + H2O2, H2O2 + GSH or other), minimum concentration of the substrate when measuring catalytic activity C_min (mM), maximum concentration of the substrate when measuring catalytic activity C_max (mM), concentration of the co-substrate when measuring the catalytic activity (mM), concentration of nanoparticles in the measurement of catalytic (mg/mL), pH at which the catalytic activity was measured and temperature at which the research was carried out (°C). You need to find all the experiments with different values mentioned in the article and write about each of them separately. It's imperative that each of these elements is addressed independently for every experiment, providing a complete and isolated description with accurate measurements in appropriate units. This approach will ensure a comprehensive and clear understanding of each experiment as an individual entity within the scientific literature on nanozymes.
    You should describe the experiments in the article separately in words, while keeping the numerical values in the right units of measurement. It is critically important to extract all the numerical values as in the example, especially important are formula, activity, syngony, length, width, depth (size or diameter), Km, Vmax, reaction type. Usually such parameters as Michaelis constant  Km (mM), Vmax, mM/s are obtained in two experiments for every type of nanoparticle. You must determine what type of reaction such parameters as Michaelis constant Km (mM), Vmax, mM/s belong to. Reaction type is H2O2+TMB when H2O2 is a substrate and TMB in co-cubstrate. Reaction type is TMB +H2O2 when  TMB  is a substrate and H2O2 in co-cubstrate.For example in pair H2O2 and TMB in first case (you call this case as Reaction type TMB+H2O2) H2O2  plays role as a co-substrate with  constant concentration(C(const), mM) and TMB   plays role as a substrate  with concentrations from  Cmin,mM to  Cmax, mM. In second case (you call this case as Reaction type H2O2+)TMB) TMB  plays role as a co-substrate with  constant concentration(C(const), mM) and H2O2    plays role as a substrate  with concentrations from  Cmin,mM to  Cmax, mM. Please divide all the data into 2 tracks: where H2O2 was a substrate and its concentration varied and where H2O2 was a co-substrate and had a constant concentration. Please show data only for those nanoparticles for which the kinetic assay was performed. All other parameters from the example are also important.
    To answer questions, you can use the following tools: get_full_text, analyze_images, llm_extractor. If you have not managed to obtain the information you needed to answer with the help of the tools, mention that in your final answer. Do not attempt to use knowledge you already have. Don't use tools if you don't need them. After you receive a response from the tool, be sure to write your thoughts. Answer in the language that the question is asked in. Before writing the final answer, don't forget to write 'Final answer:'. You need to write this before you write the text of the final answer.
    """
    st.session_state["agent"] = initialize_agent(
        tools,
        agent_llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        agent_kwargs={"prefix": prompt, "seed": 42},
    )

    st.toast("Successful Agent initialization")
    LOGGER.info("Successful Agent initialization")
