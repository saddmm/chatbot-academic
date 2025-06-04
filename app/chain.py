import os
from typing import Any
from langchain_core.language_models import BaseLanguageModel
from langchain_core.vectorstores import VectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from .document_processor import process_document_for_rag
from .llm_config import get_embedding, get_llm
from .prompt import RAG_PROMPT_TEMPLATE
from .vectorstore import get_or_create_vector_store


def format_retrieved_docs(docs: list[Document]) -> tuple[str, str]:
    if not docs:
        return "no context", "no sources"
    
    formatted_context = []
    unique_sources = set()

    for doc in docs:
        source = doc.metadata.get("source", "unknown source")
        page = doc.metadata.get("page", None)
        page_info = f" (page {page+1})" if page is not None else ""

        formatted_context.append(f"Kutipan dari {os.path.basename(source)}{page_info}: {doc.page_content}")
        unique_sources.add(os.path.basename(source))
    
    context_str = "\n\n---\n\n".join(formatted_context)
    sources_str = "\n".join(f"- {source}" for source in sorted(list(unique_sources)))

    return context_str, sources_str

def create_rag_chain(llm: BaseLanguageModel, retriever: VectorStore.as_retriever, prompt: ChatPromptTemplate):
    """
    Create a Retrieval Augmented Generation (RAG) chain using the provided LLM and vector store.

    Args:
        llm (BaseLanguageModel): The language model to use for generating responses.
        vectorstore (FAISS): The vector store containing the indexed documents.

    Returns:
        RetrievalQA: The RAG chain ready for use.
    """
    def retrieve_format_context(inputs: dict[str, Any]) -> dict[str, Any]:
        question = inputs["question"]
        retrieved_docs = retriever.invoke(question)
        context, sources = format_retrieved_docs(retrieved_docs)
        print(f"\nRetrieved {len(retrieved_docs)} documents for question: {question}")
        print(f"Context:{len(context)} : {context[:500]}...")
        return {
            "context": context,
            "sources": sources,
            "question": question
        }

    rag_chain_from_docs = (
        RunnableLambda(retrieve_format_context)
        | prompt
        | llm
        | StrOutputParser()
    )
        
    
    return rag_chain_from_docs


if __name__ == '__main__':
    print("="*50)
    print(" MENJALANKAN TES MODUL RAG CHAIN ".center(50, "="))
    print("="*50)

    # 1. Inisialisasi LLM dan Embedding Model
    print("\n--- Menginisialisasi Model ---")
    try:
        llm_instance = get_llm() # Gunakan model LLM default dari llm_config
        embedding_model_instance = get_embedding() # Gunakan model embedding default
        if not llm_instance or not embedding_model_instance:
            raise ValueError("Gagal menginisialisasi LLM atau Embedding model.")
        print("LLM dan Embedding model berhasil diinisialisasi.")
    except Exception as e:
        print(f"Error saat inisialisasi model: {e}")
        print("Pastikan Ollama server berjalan dan model sudah di-pull.")
        exit()

    # 2. Siapkan Vector Store dan Retriever
    #    Ini akan memuat dokumen dan membuat vector store jika belum ada.
    #    Gunakan path ke file URL yang berisi link PDF/Web yang valid untuk tes.
    print("\n--- Menyiapkan Vector Store & Retriever ---")
    # Path ke file yang berisi daftar URL (PDF atau Halaman Web)
    # Pastikan file ini ada dan berisi URL yang valid
    # test_url_list_file = "app/data/urls.txt" 

    # Proses dokumen (memuat dari URL dan memecahnya)
    # Untuk tes, kita hanya proses dari file URL, tidak dari direktori lokal
    # agar tidak terlalu lama jika data lokal banyak.
    try:
        document_chunks = process_document_for_rag(
            # local_directory_path=None, # Tidak pakai data lokal untuk tes cepat ini
            
        )
        if not document_chunks:
            print("Tidak ada dokumen yang diproses. Tidak bisa membuat vector store. Tes dibatalkan.")
            
        # Dapatkan atau buat vector store
        # Menggunakan nama direktori dan index default dari vectorstore_utils
        vector_store = get_or_create_vector_store(
            documents=document_chunks,
            embedding_model=embedding_model_instance,
            vector_store_dir=os.path.join("vector_store", "rag_test_index"), # Subdirektori tes
            index_name="rag_test_faiss"
        )
        if not vector_store:
            print("Gagal membuat atau memuat vector store. Tes dibatalkan.")
            exit()
        
        # Buat retriever dari vector store
        # 'k' adalah jumlah dokumen yang ingin diambil
        retriever_instance = vector_store.as_retriever(search_kwargs={"k": 3}) 
        print("Vector store dan retriever berhasil disiapkan.")

    except Exception as e:
        print(f"Error saat menyiapkan vector store atau retriever: {e}")
        import traceback
        traceback.print_exc()
        exit()

    # 3. Buat RAG Chain
    print("\n--- Membuat RAG Chain ---")
    try:
        rag_chain_instance = create_rag_chain(llm_instance, retriever_instance, RAG_PROMPT_TEMPLATE)
        print("RAG Chain berhasil dibuat.")
    except Exception as e:
        print(f"Error saat membuat RAG chain: {e}")
        exit()

    # 4. Tes RAG Chain dengan pertanyaan
    print("\n--- Menguji RAG Chain ---")
    # Pertanyaan ini harus relevan dengan isi dokumen di URL yang kamu berikan
    # Misalnya, jika salah satu URL adalah paper tentang LLM:
    test_question = "Apa saja misi dari informatika?" 
    # Atau jika URLnya tentang kurikulum ITS:
    # test_question = "Mata kuliah apa saja yang ada di semester 1 sarjana Informatika ITS?"

    print(f"\nPertanyaan: {test_question}")
    
    try:
        # Untuk melihat input yang masuk ke LLM, kita bisa lakukan ini (agak manual):
        # inputs_for_llm = retrieve_and_format_context_for_chain(
        #     {"pertanyaan_mahasiswa": test_question},
        #     retriever_instance
        # )
        # print("\n--- Input yang akan dikirim ke LLM (setelah retrieval & formatting) ---")
        # print(f"Konteks: {inputs_for_llm['konteks_dokumen'][:500]}...")
        # print(f"Sumber: {inputs_for_llm['sumber_dokumen_str']}")
        # print(f"Pertanyaan: {inputs_for_llm['pertanyaan_mahasiswa']}")
        # print("--------------------------------------------------------------------")

        print("\nMeminta jawaban dari RAG chain...")
        # Langsung invoke chain dengan pertanyaan
        # Chain LCEL menerima input dictionary jika langkah pertamanya adalah RunnableParallel atau RunnablePassthrough.assign
        # atau jika ada beberapa input ke prompt template.
        # Dalam kasus kita, retrieve_and_format_context mengharapkan dictionary dengan key "pertanyaan_mahasiswa"
        # Namun, karena retrieve_and_format_context adalah langkah pertama dalam chain setelah RunnableLambda,
        # kita bisa menyederhanakan pemanggilan invoke hanya dengan pertanyaan jika Lambda diatur untuk menerimanya
        # atau kita harus memastikan inputnya adalah dictionary.

        # Untuk chain yang kita buat, input awalnya adalah pertanyaan_mahasiswa,
        # lalu retrieve_and_format_context mengambilnya dari dictionary input.
        # Mari kita pastikan pemanggilannya benar.
        # Chain kita dimulai dengan RunnableLambda(retrieve_and_format_context)
        # retrieve_and_format_context mengharapkan input: {"pertanyaan_mahasiswa": question}
        
        response = rag_chain_instance.invoke({"question": test_question})
        if not response:
            print("Tidak ada jawaban yang diterima dari RAG chain.")
            exit()
        
        print("\nJawaban Bot:")
        print(response)
    except Exception as e:
        print(f"Error saat menjalankan RAG chain: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*50)
    print(" TES MODUL RAG CHAIN SELESAI ".center(50, "="))
    print("="*50)