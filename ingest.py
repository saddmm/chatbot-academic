import os
from app.document_processor import process_document_for_rag
from app.vectorstore import get_or_create_vector_store
from app.llm_config import get_embedding

def main():
    print("🔄 Memulai proses Ingest Data ke Vector Store...")
    
    # 1. Proses Dokumen (PDF & JSON)
    # Pastikan folder documents ada dan berisi file yang ingin diproses
    local_docs_dir = "./documents"
    if not os.path.exists(local_docs_dir):
        print(f"❌ Direktori {local_docs_dir} tidak ditemukan.")
        return

    print(f"📂 Membaca dokumen dari {local_docs_dir}...")
    document_chunks = process_document_for_rag(local_dir=local_docs_dir)
    
    if not document_chunks:
        print("⚠️ Tidak ada dokumen yang ditemukan atau diproses.")
        return

    print(f"📄 Total chunks dokumen yang akan disimpan: {len(document_chunks)}")

    # 2. Inisialisasi Embedding Model
    embedding_model = get_embedding()
    if not embedding_model:
        print("❌ Gagal memuat model embedding.")
        return

    # 3. Simpan ke Vector Store (Force Rebuild untuk memastikan data bersih dan terupdate)
    # force_rebuild=True akan menghapus database lama dan membuat baru dengan data terbaru
    print("💾 Menyimpan ke ChromaDB...")
    vector_store = get_or_create_vector_store(
        embedding_model=embedding_model,
        documents=document_chunks,
        force_rebuild=True 
    )

    if vector_store:
        print("✅ Ingest Data Selesai! Database vector telah diperbarui.")
    else:
        print("❌ Gagal menyimpan ke vector store.")

if __name__ == "__main__":
    main()
