# ingest.py
from app.document_processor import process_document_for_rag
from app.vectorstore import get_or_create_vector_store
from app.llm_config import get_embedding

if __name__ == "__main__":
    print("🔄 Memulai update dokumen...")
    chunks = process_document_for_rag(local_dir="./documents")
    embed_model = get_embedding()
    # Force rebuild = True untuk memastikan data bersih
    get_or_create_vector_store(embed_model, chunks, force_rebuild=True)
    print("✅ Update selesai!")