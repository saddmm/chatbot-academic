import os
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing import List
from langchain_ollama import OllamaEmbeddings


def create_vector_store(
    documents: List[Document], embedding_model: OllamaEmbeddings, index_path_prefix: str
) -> FAISS:

    # Load documents
    if not documents:
        raise ValueError("No documents provided to create vector store.")

    try:
        vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=embedding_model,  # Replace with your embedding model
        )

        print(f"Vector store created with {len(documents)} documents.")
        index_directory = os.path.dirname(index_path_prefix)
        index_name = os.path.basename(index_path_prefix)
        if index_directory and not os.path.exists(index_directory):
            os.makedirs(index_directory)
            print(f"Created directory for index: {index_directory}")

        vectorstore.save_local(folder_path=index_directory, index_name=index_name)
        print(f"Vector store saved to {index_path_prefix}.faiss and {index_path_prefix}.pkl")
        return vectorstore
    except Exception as e:
        raise RuntimeError(f"Failed to create vector store: {str(e)}")


def load_vector_store(
    index_path_prefix: str, embedding_model: OllamaEmbeddings
) -> FAISS:
    """
    Load a vector store from a local index file.

    Args:
        index_path_prefix (str): The path prefix for the index file.
        embedding_model (OllamaEmbeddings): The embedding model to use.

    Returns:
        FAISS: The loaded vector store.
    """
    faiss_file = f"{index_path_prefix}.faiss"
    pkl_file = f"{index_path_prefix}.pkl"
    if os.path.exists(faiss_file) and os.path.exists(pkl_file):
        print(f"Loading vector store from {faiss_file} and {pkl_file}")
        try:
            vectorstore = FAISS.load_local(
                folder_path=os.path.dirname(index_path_prefix),
                embeddings=embedding_model,
                index_name=os.path.basename(index_path_prefix),
                allow_dangerous_deserialization=True,  # Set to True if you trust the source of the index
            )
            return vectorstore
        except Exception as e:
            print(f"Error saat memuat vector store FAISS dari {index_path_prefix}: {e}")
            print("Mungkin file index rusak atau model embedding tidak cocok.")
            return None
    else:
        print(f"File index FAISS tidak ditemukan di {index_path_prefix}.faiss atau {index_path_prefix}.pkl.")
        return None


def get_or_create_vector_store(
    documents: List[Document],
    embedding_model: OllamaEmbeddings,
    vector_store_dir: str = "vector_store",
    index_name: str = "index_prodi",
) -> FAISS:
    """
    Get or create a vector store from documents.

    Args:
        documents (List[Document]): The documents to create the vector store from.
        embedding_model (OllamaEmbeddings): The embedding model to use.
        index_path_prefix (str): The path prefix for the index file.

    Returns:
        FAISS: The vector store.
    """
    if not os.path.exists(vector_store_dir):
        os.makedirs(vector_store_dir)
        print(f"Created vector store directory: {vector_store_dir}")

    index_path_prefix = os.path.join(vector_store_dir, index_name)

    vector_store = load_vector_store(index_path_prefix, embedding_model)    

    if vector_store:
        return vector_store
    else:
        print("Vector store not found, creating a new one.")
        return create_vector_store(documents, embedding_model, index_path_prefix)


if __name__ == '__main__':
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