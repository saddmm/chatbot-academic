import os
import sqlite3
from datetime import datetime
from flask import Flask, request, jsonify
from app.graph_builder import create_graph
from app.llm_config import get_embedding, get_groq_llm
from app.prompt import CLASSIFICATION_PROMPT_TEMPLATE, CONDENS_QUESTION_PROMPT_TEMPLATE, GENERAL_CHAT_PROMPT_TEMPLATE, RAG_PROMPT_TEMPLATE
from app.vectorstore import get_or_create_vector_store
from langchain_core.messages import HumanMessage
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langgraph.checkpoint.sqlite import SqliteSaver
from flask_cors import CORS

# Setup Cache agar hemat biaya API
set_llm_cache(SQLiteCache(database_path=".langchain_cache.sqlite"))

app = Flask(__name__)
CORS(app)

# Global variables
app_graph = None

def initialize_chatbot():
    """Inisialisasi komponen chatbot sekali saja saat startup."""
    print("🚀 Memulai inisialisasi Chatbot...")
    
    # 1. Setup LLM & Embedding
    llm = get_groq_llm(model_name="llama-3.1-8b-instant", temperature=0.1)
    embedding_model = get_embedding()

    # 2. Setup Vector Store (Mode Load Only)
    # Pastikan Anda sudah menjalankan script ingest data sebelumnya
    vector_store = get_or_create_vector_store(
        embedding_model=embedding_model,
        documents=None, 
        force_rebuild=False
    )
    
    retriever = None
    if not vector_store:
        print("⚠️ Vector Store kosong/gagal dimuat. Chatbot hanya bisa menjawab pertanyaan umum.")
    else:
        retriever = vector_store.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 10}
        )

    # 3. Build Graph
    # Memory persistence biasanya ditangani di dalam create_graph menggunakan MemorySaver/Checkpointer
    conn = sqlite3.connect('chat_history.sqlite', check_same_thread=False)
    memory = SqliteSaver(conn)

    graph = create_graph(
        llm=llm,
        retriever=retriever,
        rag_prompt=RAG_PROMPT_TEMPLATE,
        condense_prompt=CONDENS_QUESTION_PROMPT_TEMPLATE,
        classification_prompt=CLASSIFICATION_PROMPT_TEMPLATE,
        general_chat_prompt=GENERAL_CHAT_PROMPT_TEMPLATE,
        memory=memory
    )
    
    print("✅ Chatbot Siap!")
    return graph

# Inisialisasi saat file di-load
try:
    app_graph = initialize_chatbot()
except Exception as e:
    print(f"❌ Gagal inisialisasi app: {e}")

@app.route("/chat", methods=["POST"])
def chat():
    if not app_graph:
        return jsonify({"error": "Chatbot not initialized properly"}), 500
        
    data = request.json
    if not data:
        return jsonify({"error": "Invalid JSON body"}), 400

    user_message = data.get("message")
    thread_id = data.get("thread_id", "default_thread")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # Konfigurasi untuk session/memory per user
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        # Input state: Sesuaikan dengan definisi GraphState Anda
        # Kita perlu mengirimkan 'messages' karena node di graph mengakses state["messages"][-1]
        inputs = {
            "question": user_message,
            "messages": [HumanMessage(content=user_message, additional_kwargs={"timestamp": datetime.now().isoformat()})]
        }
        
        # Gunakan .invoke() untuk mendapatkan hasil akhir secara langsung
        # .stream() lebih cocok jika Anda menggunakan WebSocket atau Server-Sent Events (SSE)
        result = app_graph.invoke(inputs, config=config)
        
        # Logika Ekstraksi Jawaban (Menangani berbagai kemungkinan output state)
        ai_response = ""
        timestamp = datetime.now().isoformat()
        msg_id = None
        
        # Prioritas 1: Jika output ada di key 'generation' (umum untuk RAG graph sederhana)
        if "generation" in result and result["generation"]:
            ai_response = result["generation"]
            
        # Prioritas 2: Jika output ada di key 'answer'
        elif "answer" in result and result["answer"]:
            ai_response = result["answer"]
            
        # Prioritas 3: Jika output ada di list 'messages' (umum untuk Chat graph)
        elif "messages" in result and len(result["messages"]) > 0:
            last_message = result["messages"][-1]
            ai_response = last_message.content
            msg_id = getattr(last_message, "id", None)
            if hasattr(last_message, "additional_kwargs"):
                timestamp = last_message.additional_kwargs.get("timestamp", timestamp)
            
        else:
            ai_response = "Maaf, sistem tidak dapat menghasilkan jawaban (Format output tidak dikenali)."
        
        return jsonify({
            "response": ai_response,
            "thread_id": thread_id,
            "message": {
                "role": "assistant",
                "content": ai_response,
                "timestamp": timestamp
            }
        })

    except Exception as e:
        print(f"Error processing chat: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/history", methods=["GET"])
def get_history():
    if not app_graph:
        return jsonify({"error": "Chatbot not initialized properly"}), 500

    thread_id = request.args.get("thread_id")
    if not thread_id:
        return jsonify({"error": "Missing thread_id parameter"}), 400

    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        # Ambil state terakhir dari graph untuk thread_id tersebut
        state_snapshot = app_graph.get_state(config)
        
        # Cek apakah ada state/values
        if not state_snapshot.values:
             return jsonify({"history": []})

        messages = state_snapshot.values.get("messages", [])
        
        # Format pesan agar mudah dibaca frontend
        formatted_history = []
        for msg in messages:
            # Mapping tipe pesan LangChain ke format umum (user/assistant)
            role = "user"
            if msg.type == "ai":
                role = "assistant"
            elif msg.type == "human":
                role = "user"
            else:
                role = msg.type # Fallback untuk system message dll
            
            formatted_history.append({
                "role": role,
                "content": msg.content,
                "timestamp": msg.additional_kwargs.get("timestamp") if hasattr(msg, "additional_kwargs") else None
            })
            
        return jsonify({"history": formatted_history})

    except Exception as e:
        print(f"Error retrieving history: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
