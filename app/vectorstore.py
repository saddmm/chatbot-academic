import os
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List, Optional
from langchain_ollama import OllamaEmbeddings
import chromadb
from chromadb.config import Settings

def get_or_create_vector_store(
    embedding_model: OllamaEmbeddings,
    documents: List[Document] = Optional[None],
    vector_store_dir: str = "vector_store",
    collection_name: str = "prodi_collection",
) -> Chroma:
    try:
        persistent_client = chromadb.PersistentClient(
            path=vector_store_dir,
            settings=Settings(
                persist_directory=vector_store_dir
            )
        )

        collection = persistent_client.list_collections()
        if any(c.name == collection_name for c in collection):
            print(f"Memuat koleksi '{collection_name}' yang sudah ada.")
            vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=embedding_model,
                client=persistent_client
            )
            return vector_store
        else:
            print(f"Membuat koleksi baru '{collection_name}' di direktori {vector_store_dir}.")
            if not documents:
                raise ValueError("Tidak ada dokumen yang diberikan untuk membuat vector store.")
            vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=embedding_model,
                client=persistent_client,
            )
            vector_store.add_documents(documents)
            return vector_store
    except Exception as e:
        print(f"Error saat membuat vector store: {e}")
        return None

def add_documents_to_vector_store(
    documents: List[Document],
    embedding_model: OllamaEmbeddings,
    vector_store_dir: str = "vector_store",
    collection_name: str = "prodi_collection",
) -> Chroma:
    try:
        print(documents)
        if os.path.exists(vector_store_dir) and os.path.exists(os.path.join(vector_store_dir, f"chroma.sqlite")):
            print(f"Memuat koleksi '{collection_name}' yang sudah ada.")
            vector_store = Chroma(
                persist_directory=vector_store_dir,
                collection_name=collection_name,
                embedding_function=embedding_model
            )
            return vector_store
        else:
            print(f"Membuat koleksi baru '{collection_name}' di direktori {vector_store_dir}.")
            if not documents:
                raise ValueError("Tidak ada dokumen yang diberikan untuk membuat vector store.")
            vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=embedding_model,
                persist_directory=vector_store_dir,
                documents=documents
            )
            return vector_store
    except Exception as e:
        print(f"Error saat membuat vector store: {e}")
        return None

# if __name__ == '__main__':
    print("="*50)
    print(" MENJALANKAN TES MODUL VECTORSTORE UTILS ".center(50, "="))
    print("="*50)

    # Impor dari modul lain yang sudah kita buat
    # Pastikan file-file ini ada di app/ dan bisa diimpor
    # Jika menjalankan ini langsung, Python mungkin butuh PYTHONPATH disesuaikan
    # Cara mudah: pastikan terminal ada di root folder proyek (chatbot_prodi_v2)
    # dan jalankan dengan: python -m app.vectorstore_utils
    try:
        from llm_config import get_embedding
        from document_processor import process_document_for_rag, DEFAULT_DATA_DIR
    except ImportError:
        print("Error: Gagal mengimpor modul 'llm_config' atau 'document_processor'.")
        print("Pastikan kamu menjalankan skrip ini dari direktori root proyek (misalnya, 'chatbot_prodi_v2')")
        print("dengan perintah: python -m app.vectorstore_utils")
        print("Atau pastikan PYTHONPATH sudah benar.")
        exit()

    print("\n--- Langkah 1: Mempersiapkan Dokumen dan Model Embedding ---")
    # Dapatkan model embedding
    try:
        test_embedding_model = get_embedding()
        if not test_embedding_model:
            print("Gagal mendapatkan model embedding. Tes dibatalkan.")
            exit()
    except Exception as e:
        print(f"Gagal menginisialisasi model embedding: {e}. Tes dibatalkan.")
        exit()

    # Proses dokumen untuk mendapatkan chunks
    # Gunakan file URL list yang sudah ada atau buat yang baru untuk tes ini
    test_url_list_file = "app/data/urls.txt"
    if not os.path.exists(test_url_list_file):
        with open(test_url_list_file, "w", encoding="utf-8") as f:
            f.write("https://arxiv.org/pdf/2307.09288\n") # PDF dari arXiv
            # Tambahkan 1-2 URL PDF lain yang valid untuk tes
        print(f"File daftar URL tes '{test_url_list_file}' dibuat.")

    # Kita hanya akan proses dari URL untuk tes ini agar lebih cepat
    # Kamu bisa juga mengaktifkan pemrosesan dari lokal jika mau
    test_documents_chunks = process_document_for_rag(
        local_dir=None, # Atau DEFAULT_DATA_DIR jika mau tes lokal juga
        url_list_file_path=test_url_list_file
    )

    if not test_documents_chunks:
        print("Tidak ada dokumen yang diproses dari document_processor. Tidak bisa membuat vector store. Tes dibatalkan.")
        exit()
    
    print(f"\n--- Langkah 2: Membuat atau Memuat Vector Store ---")
    # Tentukan path untuk menyimpan/memuat vector store tes
    test_vectorstore_dir = os.path.join("vector_store", "test_index") # Simpan di subfolder agar tidak bentrok
    test_index_name = "index_prodi"

    # Hapus index lama jika ada untuk memastikan pembuatan baru saat tes pertama
    # (Hati-hati jika kamu punya index penting dengan nama yang sama)
    # if os.path.exists(os.path.join(test_vectorstore_dir, f"{test_index_name}.faiss")):
    #     print(f"Menghapus index tes lama di {test_vectorstore_dir}/{test_index_name}...")
    #     os.remove(os.path.join(test_vectorstore_dir, f"{test_index_name}.faiss"))
    #     os.remove(os.path.join(test_vectorstore_dir, f"{test_index_name}.pkl"))

    vector_store_instance = get_or_create_vector_store(
        documents=test_documents_chunks,
        embedding_model=test_embedding_model,
        vector_store_dir=test_vectorstore_dir,
        index_name=test_index_name
    )

    if vector_store_instance:
        print("\n--- Langkah 3: Tes Vector Store (jika berhasil dibuat/dimuat) ---")
        print(f"Jumlah dokumen di vector store: {vector_store_instance.index.ntotal}")
        
        # Coba lakukan similarity search sederhana
        try:
            query = "visi informatika apa?"
            print(f"\nMencari dokumen yang mirip dengan query: '{query}'")
            # Jumlah dokumen yang ingin diambil (k)
            results = vector_store_instance.similarity_search(query, k=2)
            if results:
                print(f"Ditemukan {len(results)} hasil pencarian:")
                for i, doc_result in enumerate(results):
                    print(f"\n--- Hasil Pencarian {i+1} ---")
                    print(f"Sumber: {doc_result.metadata.get('source', 'N/A')}, Halaman: {doc_result.metadata.get('page', 'N/A')}")
                    print(f"Konten (awal): {doc_result.page_content[:150]}...")
            else:
                print("Tidak ada hasil similarity search.")
        except Exception as e:
            print(f"Error saat melakukan similarity search: {e}")
    else:
        print("Gagal membuat atau memuat vector store. Tes tidak dapat dilanjutkan sepenuhnya.")

    print("\n" + "="*50)
    print(" TES MODUL VECTORSTORE UTILS SELESAI ".center(50, "="))
    print("="*50)