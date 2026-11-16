# from pathlib import Path
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from typing import List
# HERE = Path().__file__()
import json
with open (r"config.json", 'r') as f:
    config = json.load(f)


pdf = r"./data/thesis.pdf"
reader = PdfReader(pdf)
docs: List[Document] = []


for page_number, page in enumerate(reader.pages):
    text = page.extract_text() or ''
    doc = Document(
        page_content=text, metadata={
            "source": "thesis.pdf",
            "page": page_number + 1
            }
        )
    docs.append(doc) #we create one doc per page

#Text splitters split text in a cool way, by respecting certain chars like \n\n, \n, ' ', etc
splitter = RecursiveCharacterTextSplitter(
    chunk_size=config["chunk_size"], 
    chunk_overlap=config["chunk_overlap"]
    )
chunks = splitter.split_documents(docs)
print(f"From {len(docs)} page we generated {len(chunks)} chunks")

embeddings = OllamaEmbeddings(
    model=config["embedding_model"]
    )
llm = ChatOllama(
    model=config["llm"], 
    temperature=0
    )
vector_store = Chroma(
    collection_name="thesis",
    embedding_function=embeddings,
    persist_directory=r"./data/chroma_db"
)
ids = [f"thesis_page_{chunk.metadata['page']}_chunk_{i + 1}" for i, chunk in enumerate(chunks)]
vector_store.add_documents(documents=chunks, ids=ids)


# ----------------- #
query = "What emdeding model did the user use for the SKU mapping?"
results = vector_store.similarity_search(query=query, k=5)
for doc in results:
    print(doc.page_content)
    print(doc.metadata)


