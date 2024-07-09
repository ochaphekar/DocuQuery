from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from typing import List, Dict

import os
import shutil
import time
import random

CHROMA_PATH = "chroma"
DATA_PATH = "data/rav4"
BATCH_SIZE = 10  # You can adjust this based on your typical document size and API limits

def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf")
    documents = loader.load()
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def save_to_chroma(chunks: List[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    embeddings = OpenAIEmbeddings()  # Initialize embeddings once to use for all batches
    total_chunks = len(chunks)
    processed_chunks = 0

    while processed_chunks < total_chunks:
        batch = chunks[processed_chunks:min(processed_chunks + BATCH_SIZE, total_chunks)]
        processed_chunks += len(batch)
        retry_save_to_chroma(batch, embeddings)
    
    print(f"Saved all chunks to {CHROMA_PATH}.")

def retry_save_to_chroma(batch: List[Document], embeddings):
    max_retries = 5
    backoff_factor = 2

    for attempt in range(max_retries):
        try:
            db = Chroma.from_documents(batch, embeddings, persist_directory=CHROMA_PATH)
            db.persist()
            print(f"Processed and saved batch: {len(batch)} chunks.")
            return  # Successfully saved the batch
        except openai.RateLimitError as e:
            if attempt < max_retries - 1:
                sleep_time = (backoff_factor ** attempt) + random.uniform(0, 1)
                print(f"Rate limit exceeded. Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print("Failed to save documents after several retries.")
                raise e
        except Exception as e:
            print(f"Failed to process batch due to an error: {e}")
            raise e

if __name__ == "__main__":
    main()
