from pathlib import Path
THIS_DIR     = Path(__file__).resolve()

import oci
from oci.config import from_file
from oci.object_storage import ObjectStorageClient

import os
from pathlib import Path
from dotenv import load_dotenv

THIS_DIR     = Path(__file__).resolve()
PROJECT_ROOT = THIS_DIR.parent.parent.parent

load_dotenv(PROJECT_ROOT / "config/.env")  # expects vars in .env

# Set up the OCI GenAI Agents endpoint configuration
BUCKET_REGION = os.getenv("BUCKET_REGION")
print("BUCKET_REGION: " + BUCKET_REGION)
BUCKET_NAME = os.getenv("BUCKET_NAME")
print("BUCKET_NAME: " + BUCKET_NAME)
BUCKET_PDF_NAME = os.getenv("BUCKET_PDF_NAME")
print("BUCKET_PDF_NAME: " + BUCKET_PDF_NAME)
OCI_GENAI_LLM_COMPARTMENT_ID = os.getenv("OCI_GENAI_LLM_COMPARTMENT_ID")
print("OCI_GENAI_LLM_COMPARTMENT_ID: " + OCI_GENAI_LLM_COMPARTMENT_ID)
OCI_GENAI_EMBEDDINGS_MODEL_COMPARTMENT_ID = os.getenv("OCI_GENAI_EMBEDDINGS_MODEL_COMPARTMENT_ID")
print("OCI_GENAI_EMBEDDINGS_MODEL_COMPARTMENT_ID: " + OCI_GENAI_EMBEDDINGS_MODEL_COMPARTMENT_ID)
OCI_CONFIG_PROFILE_NAME = os.getenv("OCI_CONFIG_PROFILE_NAME")
print("OCI_CONFIG_PROFILE_NAME: " + OCI_CONFIG_PROFILE_NAME)
OBJECT_STORAGE_NAMESPACE_NAME = os.getenv("OBJECT_STORAGE_NAMESPACE_NAME")
print("OBJECT_STORAGE_NAMESPACE_NAME: " + OBJECT_STORAGE_NAMESPACE_NAME)
LOCAL_PDF_PATH = os.getenv("LOCAL_PDF_PATH")
print("LOCAL_PDF_PATH: " + LOCAL_PDF_PATH)
USE_LOCAL_PDF = os.getenv("USE_LOCAL_PDF")
print("USE_LOCAL_PDF: " + USE_LOCAL_PDF)

# ─── OCI Custom RAG Configuration --------

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader

## Initialize LLM

from langchain_community.chat_models import ChatOCIGenAI

# initialize interface
chat = ChatOCIGenAI(
    model_id="ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceyajqi26fkxly6qje5ysvezzrypapl7ujdnqfjq6hzo2loq",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id=OCI_GENAI_LLM_COMPARTMENT_ID,
    provider="meta",
    model_kwargs={
        "temperature": 1,
        "max_tokens": 600,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "top_p": 0.75
    },
    auth_type="API_KEY", # The authentication type to use, e.g., API_KEY (default), SECURITY_TOKEN, INSTANCE_PRINCIPAL, RESOURCE_PRINCIPAL.
    auth_profile=OCI_CONFIG_PROFILE_NAME
)

## Load PDF

if USE_LOCAL_PDF=="true":
    loader = PyPDFLoader(LOCAL_PDF_PATH)
    pages = loader.load()
else:
    import oci
    # from PyPDFLoader import PyPDFLoader
    from io import BytesIO

    # OCI Object Storage configuration
    config = oci.config.from_file()
    object_storage = oci.object_storage.ObjectStorageClient(config)

    # Bucket and object details
    # Download the PDF from OCI Object Storage
    response = object_storage.get_object(OBJECT_STORAGE_NAMESPACE_NAME, BUCKET_NAME, BUCKET_PDF_NAME)
    pdf_content = response.data.content

    # Load the PDF into PyPDFLoader
    pdf_bytes = BytesIO(pdf_content)

    ## Save pdf_bytes to a temporary file

    import tempfile
    from langchain_community.document_loaders import PyPDFLoader

    # pdf_bytes is a BytesIO object
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
        tmp_file.write(pdf_bytes.getvalue())  # Extract raw bytes
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    pages = loader.load()
    ## clean up temporary file
    import os
    os.remove(tmp_file_path)


## Combined raw text from all pages

raw_text = ''

for i, doc in enumerate(pages):
    text = doc.page_content
    if text:
        raw_text += text

## Text splitter

text_splitter = RecursiveCharacterTextSplitter(
    # chunk_size should enable as many characters as can fit in max number of tokens allowed for embedding model i.e. 512 characters. Source: https://docs.oracle.com/en-us/iaas/Content/generative-ai/use-playground-embed.htm
    chunk_size=2000,
    chunk_overlap=100
)

texts = text_splitter.split_text(raw_text)

## Text chunk into a document structure

from langchain_core.documents import Document

documents = []
for i in range(len(texts)):
    doc = Document(page_content=texts[i])
    # doc = texts[i]
    documents.append(doc)

from langchain_community.embeddings import OCIGenAIEmbeddings

embeddings = OCIGenAIEmbeddings(
    model_id="cohere.embed-english-v3.0",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    truncate="NONE",
    compartment_id=OCI_GENAI_EMBEDDINGS_MODEL_COMPARTMENT_ID,
    auth_type="API_KEY",
    auth_profile=OCI_CONFIG_PROFILE_NAME
)

## Initialize a vector database

embed_model = embeddings
vectordb = Chroma(
    collection_name='summaries',
    embedding_function=embed_model,
    persist_directory='./data'
)

MAX_NUMBER_DOCUMENTS_IN_VECTOR_DB = 5461

vectordb.add_documents(documents[:MAX_NUMBER_DOCUMENTS_IN_VECTOR_DB])

## Create a retriever
retriever = vectordb.as_retriever(search_kwargs={"k": 4})

## Define a prompt template

PRODUCT_BOT_PROMPT = """
    You are a smart assistant.
    Your response must only be in English.
    Ensure your answers are relevant to the query with reference to provided context and not outside the context.
    Your responses should be elaborate and up to the mark referring to the context only.
    Do not include the keyword context in the final answer

    CONTEXT:
    {context}

    QUESTION: {question}

    YOUR ANSWER:
"""

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(PRODUCT_BOT_PROMPT)

## Chain the inputs and outputs

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

llm = chat

# Define the full processing pipeline
# 1. Takes query input
# 2. Retrieves relevant context from vector DB
# 3. Fills in the prompt template
# 4. Sends it to the LLM
# 5. Parses the output string

chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

## Process the query

""" Example Questions to try
Who is Mr. Raza
What is Form 10 - K
Explain acquisitions
List different events from year 2020 to 2023
"""

query = "List different events from year 2020 to 2023"

result = chain.invoke(query)

# Print response
print("Response:", result)