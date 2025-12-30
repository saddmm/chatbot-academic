from fileinput import filename
import json
import os
import re
from typing import Any, List, Optional, Dict
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_transformers import BeautifulSoupTransformer

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200  # UPGRADE: Overlap diperbesar agar konteks terjaga


def clean_text(text: str) -> str:
    """Membersihkan teks dari artefak PDF umum."""
    # Hapus header/footer halaman umum (contoh: "Halaman 1 dari 10")
    text = re.sub(r'Halaman \d+ dari \d+', '', text, flags=re.IGNORECASE)
    # Hapus spasi berlebih dan newline aneh
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_document_pdf(
    file_path: str
) -> List[Document]:
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # UPGRADE: Bersihkan teks dan tambah metadata filename
        filename = os.path.basename(file_path)
        for doc in documents:
            doc.page_content = clean_text(doc.page_content)
            doc.metadata["source"] = filename  # Pastikan source adalah nama file

        return documents
    except Exception as e:
        print(f"Error loading PDF {file_path}: {e}")
        return []


def load_custom_json(file_path: str) -> List[Document]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        for item in data:
            content = item.get("page_content")
            metadata = item.get("metadata", {})
            
            # --- PERBAIKAN DISINI ---
            # Gabungkan Title/Kategori ke dalam Content agar pencarian lebih kuat
            title = metadata.get("title", "")
            category = metadata.get("category", "")
            
            # Format baru: "[Kategori] Judul: Isi konten..."
            # Ini membantu vector store menemukan dokumen berdasarkan judulnya juga
            enriched_content = f"[{category}] {title}: {content}"
            
            if content:
                # Gunakan enriched_content sebagai page_content
                doc = Document(page_content=enriched_content, metadata=metadata)
                documents.append(doc)
                
        print(f"✅ Berhasil memuat {len(documents)} dokumen dari {os.path.basename(file_path)}")
        return documents
    except Exception as e:
        print(f"❌ Gagal memuat JSON {file_path}: {e}")
        return []


# Memproses dokumen dari URL untuk retrieval-augmented generation (RAG)
def process_document_for_rag(
    local_dir: Optional[str] = None,
    url_list_file_path: Optional[str] = None,
    json_file_path: Optional[str] = None,
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
                    # if file_name.endswith(".pdf"):
                    #     print(f"Memproses file: {file_path}")
                    #     documents = load_document_pdf(file_path)
                    #     if documents:
                    #         loaded_documents.extend(documents)
                    if file_name.endswith(".json"):
                        print(f"Memproses file JSON khusus: {file_path}")
                        documents = load_custom_json(file_path)
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


def process_urls_for_rag(urls: List[str]) -> List:
    """
    Mengambil konten dari daftar URL, membersihkannya, dan memecahnya menjadi
    dokumen yang siap untuk dimasukkan ke dalam RAG system.

    Args:
        urls (List[str]): Daftar URL yang akan di-scrape.

    Returns:
        List: Daftar objek Document yang sudah bersih dan terpecah (chunks).
    """
    if not urls:
        return []

    print(f"Memulai proses untuk {len(urls)} URL...")

    # 1. Loading: Memuat konten HTML dari URL.
    # Anda bisa memberikan beberapa URL sekaligus.
    loader = WebBaseLoader(urls)
    docs_raw = loader.load()
    print(f"Berhasil memuat konten mentah dari {len(docs_raw)} halaman.")

    # 2. Transforming: Membersihkan HTML dan mengekstrak teks yang relevan.
    # Kita hanya akan mengambil teks dari tag <main> untuk fokus pada konten utama.
    # Anda bisa menyesuaikan ini sesuai dengan struktur website target.
    # Contoh lain: ["article", "div.content", "p"]
    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(
        documents=docs_raw, tags_to_extract=["main", "article", "div.content", "p"]
    )
    print("Berhasil membersihkan HTML dan mengekstrak konten utama.")

    # 3. Splitting: Memecah teks bersih menjadi potongan-potongan (chunks).
    # Parameter ini bisa disesuaikan dengan yang Anda gunakan untuk dokumen lain.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    print(text_splitter)
    docs_split = text_splitter.split_documents(docs_transformed)
    print(f"Berhasil memecah konten menjadi {len(docs_split)} dokumen (chunks).")

    return docs_split


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
        separators=["\n\n", "\n", ". ", " ", ""]  # Prioritas pemisahan
    )
    split_docs = text_splitter.split_documents(documents)
    return split_docs


if __name__ == "__main__":
    # Example usage
    try:
        content = process_document_for_rag(
            local_dir="./documents",
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP
        )
        print("Content loaded successfully:", len(content), "documents found.")
        print("Example document content:", content[0].page_content[:100] + "...")  # Print first 100 characters of the first document
        print("Content loaded successfully:", content)
    except ValueError as e:
        print(e)
