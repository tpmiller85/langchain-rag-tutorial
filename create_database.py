import argparse
# from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
# from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai 
from dotenv import load_dotenv
import os
import shutil

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()
#---- Set OpenAI API key 
# Change environment variable name from "OPENAI_API_KEY" to the name given in 
# your .env file.
openai.api_key = os.environ['OPENAI_API_KEY']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_subfolder", nargs='?', type=str, help="Subfolder for source data and db prefix.")
    args = parser.parse_args()
    if not args.data_subfolder:
        CHROMA_PATH = "chroma"
        DATA_PATH = "data/books"
    else:
        CHROMA_PATH = f"chroma_{args.data_subfolder}"
        DATA_PATH = f"data/{args.data_subfolder}"

    documents = load_documents(DATA_PATH)
    chunks = split_text(documents)
    save_to_chroma(CHROMA_PATH, chunks)


def load_documents(data_path: str):
    loader = DirectoryLoader(data_path, glob="*.md")
    documents = loader.load()
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=150,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks


def save_to_chroma(chroma_path: str, chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=chroma_path
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {chroma_path}.")


if __name__ == "__main__":
    main()
