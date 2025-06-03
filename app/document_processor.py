from typing import Any, List, Optional, Dict
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150


def process_document_for_rag(url: str, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[Document]:
    """
    Process a document from a web URL for retrieval-augmented generation (RAG).

    Args:
        url (str): The URL of the document to process.
        chunk_size (int): The size of each chunk to split the document into.

    Returns:
        Dict[str, Any]: A dictionary containing the processed document and metadata.
    """
    try:
        documents = load_web_url_content(url)
        if not documents:
            raise ValueError(f"No content found at {url}")

        split_docs = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        return split_docs

    except Exception as e:
        raise ValueError(f"Failed to process document from {url}: {str(e)}") from e

def load_web_url_content(url: str, custom_metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
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
        headers = { # Beberapa website butuh User-Agent
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        loader = WebBaseLoader(url, header_template=headers)

        documents = loader.load()

        final_docs = []
        for doc in documents:
            doc_metadata = doc.metadata.copy()

            if 'source' not in doc_metadata or not doc_metadata['source']:
                doc_metadata['source'] = url
            if custom_metadata:
                doc_metadata.update(custom_metadata)
            final_docs.append(Document(page_content=doc.page_content, metadata=doc_metadata))
        
        if not final_docs:
            raise ValueError(f"No content found at {url}")
        else:
            return final_docs
        
    except Exception as e:
        raise ValueError(f"Failed to load content from {url}: {str(e)}") from e

def split_documents(documents: List[Document], chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[Document]:
    """
    Split text into chunks of a specified size.

    Args:
        text (str): The text to split.
        chunk_size (int): The size of each chunk.

    Returns:
        list: A list of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len, add_start_index=True)
    split_docs = text_splitter.split_documents(documents)
    return split_docs

if __name__ == "__main__":
    # Example usage
    url = "https://informatika.umsida.ac.id/tentang-umsida/visi-dan-misi/"
    try:
        content = process_document_for_rag(url)
        print("Content loaded successfully:", content)
    except ValueError as e:
        print(e)