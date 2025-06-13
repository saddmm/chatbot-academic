import uuid
from flask import Flask, abort, make_response, request, jsonify
from app.chain import create_graph
from app.document_processor import process_document_for_rag
from app.llm_config import get_embedding, get_llm
from app.prompt import CLASSIFICATION_PROMPT_TEMPLATE, CONDENS_QUESTION_PROMPT_TEMPLATE, RAG_PROMPT_TEMPLATE
from app.vectorstore import get_or_create_vector_store
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage
import sqlite3


app = Flask(__name__)

try:
    llm = get_llm()
    embedding_model = get_embedding()

    document_chunks = process_document_for_rag(local_dir="documents",chunk_size=1000, chunk_overlap=150)
    vector_store = get_or_create_vector_store(
        documents=document_chunks,
        embedding_model=embedding_model,
        vector_store_dir="vector_store",
        index_name="index_prodi",
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    sqlite_conn = sqlite3.connect("memory.sqlite", check_same_thread=False)
    memory = SqliteSaver(sqlite_conn)
    print("memory initialized with SqliteSaver.")
    graph = create_graph(
        llm=llm,
        retriever=retriever,
        rag_prompt=RAG_PROMPT_TEMPLATE,
        condense_prompt=CONDENS_QUESTION_PROMPT_TEMPLATE,
        classification_prompt=CLASSIFICATION_PROMPT_TEMPLATE,
        memory=memory
    )
    print("Komponen LLM dan Vector Store berhasil diinisialisasi.")
except Exception as e:
    print(f"Error saat menginisialisasi LLM atau Vector Store: {e}")
    graph = None

@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()
    
    question = data.get("question")
    session_id = request.cookies.get("session_id")


    if not session_id:
        session_id = str(uuid.uuid4()) # 1 day expiration
    
    if not graph:
        abort(500, "Graph is not initialized. Please check the server logs for details.")
    

    try:
        config = {"configurable": {"thread_id": session_id}}

        input_data = {
            "messages": [HumanMessage(content=question)],
            "question": question,
        }

        print(f"Received question: {question} with session_id: {session_id}")

        final_state = graph.invoke(
            input_data,
            config,
        )

        answer = final_state["messages"][-1].content
        response_data = {
            "answer": answer,
            "session_id": session_id
        }
        response = make_response(jsonify(response_data))
        if not request.cookies.get("session_id"):
            request.set_cookie("session_id", session_id, max_age=60*60*24)
        
        return response
    except Exception as e:
        print(f"Error during graph invocation: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(port=5000, debug=True)  # Set debug=True for development purposes
