from typing import Annotated, List, Sequence, TypedDict, Optional, Dict, Any
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import operator
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphState(TypedDict):
    """Enhanced state with better type hints and optional fields"""

    question: str
    messages: Annotated[Sequence[BaseMessage], operator.add]
    documents: List[Document]  # Renamed from 'document' for clarity
    sources: str
    error: Optional[str]  # For error handling
    metadata: Optional[Dict[str, Any]]  # For additional context


class RAGGraphBuilder:
    """Optimized RAG Graph Builder with better structure and error handling"""

    def __init__(
        self,
        llm,
        retriever,
        prompts: Dict[str, Any],
        memory: Optional[MemorySaver] = None,
    ):
        self.llm = llm
        self.retriever = retriever
        self.prompts = prompts
        self.memory = memory or MemorySaver()
        self._validate_prompts()

    def _validate_prompts(self) -> None:
        """Validate required prompts are provided"""
        required_prompts = ["rag_prompt", "condense_prompt"]
        missing = [p for p in required_prompts if p not in self.prompts]
        if missing:
            raise ValueError(f"Missing required prompts: {missing}")

    def _extract_sources(self, documents: List[Document]) -> str:
        """Extract and format unique sources from documents"""
        if not documents:
            return "Tidak ada sumber dokumen yang spesifik."

        unique_sources = set()
        for doc in documents:
            source = doc.metadata.get("source", "Tidak diketahui")
            if source != "Tidak diketahui":
                unique_sources.add(os.path.basename(source))

        if not unique_sources:
            return "Tidak ada sumber dokumen yang spesifik."

        return "\n".join([f"- {s}" for s in sorted(unique_sources)])

    def node_condense_and_retrieve(self, state: GraphState) -> Dict[str, Any]:
        """Combined node: condense question and retrieve documents"""
        try:
            messages = state["messages"]

            # Step 1: Condense question if there's chat history
            if len(messages) <= 1:
                question = messages[-1].content
                logger.info(f"No chat history, using original question: {question}")
            else:
                chat_history = messages[:-1]
                new_question = messages[-1].content

                condense_chain = (
                    self.prompts["condense_prompt"] | self.llm | StrOutputParser()
                )

                question = condense_chain.invoke(
                    {
                        "chat_history": chat_history,
                        "question": new_question,
                    }
                )
                logger.info(f"Question condensed: {question}")

            # Step 2: Retrieve documents
            documents = self.retriever.invoke(question)
            sources = self._extract_sources(documents)

            logger.info(f"Retrieved {len(documents)} documents for: {question}")

            return {
                "question": question,
                "documents": documents,
                "sources": sources,
                "metadata": {"retrieval_count": len(documents)},
            }

        except Exception as e:
            logger.error(f"Error in condense and retrieve: {str(e)}")
            return {
                "question": state["messages"][-1].content,
                "documents": [],
                "sources": "Error retrieving documents",
                "error": f"Condense and retrieve error: {str(e)}",
            }

    def node_answer_rag(self, state: GraphState) -> Dict[str, Any]:
        """Generate RAG answer with enhanced context handling"""
        try:
            question = state["question"]
            documents = state.get("documents", [])
            messages = state["messages"]
            sources = state.get("sources", "")

            if not documents:
                return {
                    "messages": [
                        AIMessage(
                            content="Maaf, tidak ada dokumen yang relevan ditemukan untuk menjawab pertanyaan Anda."
                        )
                    ]
                }

            rag_chain = create_stuff_documents_chain(
                self.llm, self.prompts["rag_prompt"]
            )

            answer = rag_chain.invoke(
                {
                    "question": question,
                    "context": documents,
                    "chat_history": messages[:-1],  # Exclude current question
                    "sources": sources,
                }
            )

            logger.info(f"RAG answer generated for: {question}")
            return {"messages": [AIMessage(content=answer)]}

        except Exception as e:
            logger.error(f"Error in RAG answer generation: {str(e)}")
            return {
                "messages": [
                    AIMessage(
                        content=f"Maaf, terjadi kesalahan saat memproses pertanyaan Anda: {str(e)}"
                    )
                ]
            }

    def node_answer_general_chat(self, state: GraphState) -> Dict[str, Any]:
        """Fallback for when no documents are found"""
        try:
            messages = state["messages"]
            question = state.get("question", messages[-1].content)

            # Simple fallback response when no documents found
            response = f"Maaf, saya tidak menemukan informasi yang relevan untuk menjawab pertanyaan: '{question}'. Silakan coba pertanyaan yang lebih spesifik atau pastikan dokumen yang dibutuhkan tersedia."

            logger.info(f"No documents found, providing fallback response")
            return {"messages": [AIMessage(content=response)]}

        except Exception as e:
            logger.error(f"Error in fallback response: {str(e)}")
            return {
                "messages": [
                    AIMessage(
                        content="Maaf, saya mengalami kesulitan memproses pertanyaan Anda."
                    )
                ]
            }

    def decide_path(self, state: GraphState) -> str:
        """Decide if we have documents to answer or need fallback"""
        documents = state.get("documents", [])

        if state.get("error"):
            logger.warning(f"Error detected, routing to fallback: {state['error']}")
            return "generate_answer_general"

        if not documents:
            logger.info("No documents found, routing to fallback")
            return "generate_answer_general"

        logger.info(f"Found {len(documents)} documents, routing to RAG answer")
        return "generate_answer_rag"

    def create_graph(self) -> StateGraph:
        """Create simplified StateGraph without classification"""
        try:
            workflow = StateGraph(GraphState)

            # Add nodes
            workflow.add_node("condense_and_retrieve", self.node_condense_and_retrieve)
            workflow.add_node("generate_answer_rag", self.node_answer_rag)
            workflow.add_node("generate_answer_general", self.node_answer_general_chat)

            # Define workflow - start with condense and retrieve
            workflow.set_entry_point("condense_and_retrieve")

            # Add conditional edges based on whether documents were found
            workflow.add_conditional_edges(
                "condense_and_retrieve",
                self.decide_path,
                {
                    "generate_answer_rag": "generate_answer_rag",
                    "generate_answer_general": "generate_answer_general",
                },
            )

            # Add edges to END
            workflow.add_edge("generate_answer_rag", END)
            workflow.add_edge("generate_answer_general", END)

            # Compile graph
            graph = workflow.compile(checkpointer=self.memory)

            if not graph:
                raise ValueError("Failed to compile the graph")

            logger.info("Simplified graph compiled successfully")
            return graph

        except Exception as e:
            logger.error(f"Error creating graph: {str(e)}")
            raise


# Convenience function for backward compatibility
def create_graph(llm, retriever, rag_prompt, condense_prompt, memory=None):
    """
    Simplified legacy function for backward compatibility (without classification)
    """
    prompts = {"rag_prompt": rag_prompt, "condense_prompt": condense_prompt}

    builder = RAGGraphBuilder(llm, retriever, prompts, memory)
    return builder.create_graph()


# Example usage
if __name__ == "__main__":
    # Example of how to use the simplified builder
    """
    # Initialize your components
    llm = your_llm_instance
    retriever = your_retriever_instance

    prompts = {
        'rag_prompt': your_rag_prompt,
        'condense_prompt': your_condense_prompt
    }

    # Create builder
    builder = RAGGraphBuilder(llm, retriever, prompts)

    # Create graph
    graph = builder.create_graph()

    # Use the graph
    result = graph.invoke({
        "messages": [HumanMessage(content="Your question here")],
        "question": "",
        "documents": [],
        "sources": "",
        "error": None,
        "metadata": {}
    })
    """
    pass
