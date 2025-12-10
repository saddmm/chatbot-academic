import os
import shutil
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List, Optional
from langchain_ollama import OllamaEmbeddings
import chromadb
from chromadb.config import Settings

def get_or_create_vector_store(
    embedding_model: OllamaEmbeddings,
    documents: List[Document] = None,
    vector_store_dir: str = "vector_store",
    collection_name: str = "prodi_collection",
    force_rebuild: bool = False, # UPGRADE: Fitur reset database
) -> Optional[Chroma]:
    try:
        # 1. Reset Database jika diminta
        if force_rebuild and os.path.exists(vector_store_dir):
            print(f"⚠️ Force rebuild aktif. Menghapus database lama...")
            shutil.rmtree(vector_store_dir)

        persistent_client = chromadb.PersistentClient(
            path=vector_store_dir,
            settings=Settings(persist_directory=vector_store_dir)
        )

        collection = persistent_client.list_collections()
        collection_exists = any(c.name == collection_name for c in collection)

        # 2. Load jika sudah ada dan tidak rebuild
        if collection_exists and not force_rebuild:
            print(f"📂 Memuat koleksi '{collection_name}' yang sudah ada.")
            return Chroma(
                client=persistent_client,
                collection_name=collection_name,
                embedding_function=embedding_model,
            )
        
        # 3. Buat baru jika belum ada
        else:
            print(f"🆕 Membuat koleksi baru '{collection_name}'...")
            if not documents:
                print("⚠️ Tidak ada dokumen untuk inisialisasi vector store.")
                return None
            
            # Batch processing untuk efisiensi memori
            batch_size = 100
            print(f"⏳ Menyimpan {len(documents)} dokumen (Batch size: {batch_size})...")
            
            vector_store = Chroma(
                client=persistent_client,
                collection_name=collection_name,
                embedding_function=embedding_model,
                persist_directory=vector_store_dir
            )
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                vector_store.add_documents(batch)
                print(f"   ...tersimpan {min(i + batch_size, len(documents))}/{len(documents)}")
                
            print("✅ Vector store berhasil dibuat!")
            return vector_store

    except Exception as e:
        print(f"❌ Error Critical VectorStore: {e}")
        return None

def add_documents_to_vector_store(
    documents: List[Document],
    embedding_model: OllamaEmbeddings,
    vector_store_dir: str = "vector_store",
    collection_name: str = "prodi_collection",
) -> Optional[Chroma]:
    # Fungsi helper sederhana untuk menambah dokumen
    return get_or_create_vector_store(
        embedding_model, documents, vector_store_dir, collection_name, force_rebuild=False
    )

# PERBAIKAN: Uncomment dan rapikan indentasi
if __name__ == '__main__':
    print("--- TEST MODE ---")
    # Kode testing bisa diletakkan di sini