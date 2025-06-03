import os
from pydoc import doc
from typing import Any, List, Optional, Dict
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
import pypdf

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_DATA_DIR = "data/documents"
DEFAULT_URL_LIST_FILE = "app/data/urls.txt"


def load_document_pdf(
    file_path: str
) -> List[Document]:
    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == ".pdf":
            loader = pypdf.PyPDFLoader(file_path)
            documents = loader.load()
            return documents
    except Exception as e:
        raise ValueError(f"Failed to load PDF document from {file_path}: {str(e)}")
    

# Memproses dokumen dari URL untuk retrieval-augmented generation (RAG)
def process_document_for_rag(
    local_dir: Optional[str] = DEFAULT_DATA_DIR,
    url_list_file_path: Optional[str] = DEFAULT_URL_LIST_FILE,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    """
    Process a document from a web URL for retrieval-augmented generation (RAG).

    Args:
        url (str): The URL of the document to process.
        chunk_size (int): The size of each chunk to split the document into.

    Returns:
        Dict[str, Any]: A dictionary containing the processed document and metadata.
    """
    loaded_documents = []
    try:
        if local_dir:
            if not os.path.exists(local_dir):
                print(f"Tidak ada file di direktori: {local_dir}")
            else:
                for file_name in os.listdir(local_dir):
                    file_path = os.path.join(local_dir, file_name)
                    if os.path.isfile(file_path):
                        print(f"Memproses file: {file_path}")
                        documents = load_document_pdf(file_path)
                        if documents:
                            loaded_documents.extend(documents)

        if url_list_file_path:
            urls = read_urls_from_file(url_list_file_path)
            if urls:
                for i, url in enumerate(urls):
                    print(f"Processing URL {i + 1}/{len(urls)}: {url}")
                    documents = load_web_url_content(url)
                    loaded_documents.extend(documents)

        if not loaded_documents:
            return []

        chunked_documents = split_documents(loaded_documents, chunk_size, chunk_overlap)
        return chunked_documents
    except ValueError as e:
        raise ValueError(f"Error processing URLs: {str(e)}")

def read_urls_from_file(file_path: str) -> List[str]:
    urls = []

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                url = line.strip()
                if url:
                    urls.append(url)
    except FileNotFoundError:
        raise ValueError(f"File not found: {file_path}")

    return urls


# Memuat konten dari URL web
def load_web_url_content(
    url: str, custom_metadata: Optional[Dict[str, Any]] = None
) -> List[Document]:
    """
    Load content from a web URL.

    Args:
        url (str): The URL to load content from.
        custom_metadata (Optional[Dict[str, str]]): Optional custom metadata to include.

    Returns:
        str: The content loaded from the URL.
    """
    # Placeholder for actual implementation
    try:
        headers = {  # Beberapa website butuh User-Agent
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        loader = WebBaseLoader(url, header_template=headers)

        documents = loader.load()

        final_docs = []
        for doc in documents:
            doc_metadata = doc.metadata.copy()

            if "source" not in doc_metadata or not doc_metadata["source"]:
                doc_metadata["source"] = url
            if custom_metadata:
                doc_metadata.update(custom_metadata)
            final_docs.append(
                Document(page_content=doc.page_content, metadata=doc_metadata)
            )

        if not final_docs:
            raise ValueError(f"No content found at {url}")
        else:
            return final_docs

    except Exception as e:
        raise ValueError(f"Failed to load content from {url}: {str(e)}") from e


# Memecah dokumen menjadi potongan-potongan teks
def split_documents(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split text into chunks of a specified size.

    Args:
        text (str): The text to split.
        chunk_size (int): The size of each chunk.

    Returns:
        list: A list of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    split_docs = text_splitter.split_documents(documents)
    return split_docs


if __name__ == "__main__":
    # Example usage
    try:
        content = process_document_for_rag(
            url_list_file_path=DEFAULT_URL_LIST_FILE,
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP
        )
        print("Content loaded successfully:", content)
    except ValueError as e:
        print(e)
