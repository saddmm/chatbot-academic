import os
import sqlite3
import warnings
import pandas as pd
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from ragas import evaluate, RunConfig
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

# --- IMPORT MODUL APLIKASI ANDA ---
# Pastikan modul ini ada di struktur project Anda
try:
    from app.graph_builder import create_graph
    from app.llm_config import get_embedding
    from app.prompt import (
        CLASSIFICATION_PROMPT_TEMPLATE,
        CONDENS_QUESTION_PROMPT_TEMPLATE,
        GENERAL_CHAT_PROMPT_TEMPLATE,
        RAG_PROMPT_TEMPLATE,
    )
    from app.vectorstore import get_or_create_vector_store
except ImportError as e:
    print(f"Error Importing App Modules: {e}")
    print("Pastikan Anda menjalankan script ini dari root directory project.")
    exit()

# Filter warnings agar output bersih
warnings.filterwarnings("ignore", category=DeprecationWarning)


# --- 1. CUSTOM WRAPPER UNTUK GROQ (FIX n=1 ISSUE) ---
class SafeChatGroq(ChatGroq):
    """
    Wrapper khusus untuk Groq agar kompatibel dengan Ragas.
    Ragas sering meminta n=2 atau n=3 untuk variasi jawaban,
    tapi Groq API saat ini hanya mendukung n=1.
    """

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if kwargs.get('n', 1) > 1:
            kwargs['n'] = 1
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def run_evaluation():
    print("🚀 Starting Ragas Evaluation (Groq Optimized)...")

    # --- 2. SETUP COMPONENTS ---
    print("Loading components...")

    # Gunakan Wrapper SafeChatGroq
    # Temperature 0.1 agar jawaban konsisten saat evaluasi
    llm_judge = SafeChatGroq(
        model="llama-3.1-8b-instant", temperature=0.1, api_key=os.getenv("GROQ_API_KEY")
    )

    # LLM untuk Graph (Bisa sama atau beda)
    llm_app = SafeChatGroq(
        model="llama-3.1-8b-instant", temperature=0.1, api_key=os.getenv("GROQ_API_KEY")
    )

    embedding_model = get_embedding()

    # Load Vector Store
    vector_store = get_or_create_vector_store(
        embedding_model=embedding_model, documents=None, force_rebuild=False
    )
    if not vector_store:
        print("❌ Vector store not found. Please ingest data first.")
        return

    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 10}
    )

    # Memory
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    memory = SqliteSaver(conn)

    # Create Graph
    graph = create_graph(
        llm=llm_app,
        retriever=retriever,
        rag_prompt=RAG_PROMPT_TEMPLATE,
        condense_prompt=CONDENS_QUESTION_PROMPT_TEMPLATE,
        classification_prompt=CLASSIFICATION_PROMPT_TEMPLATE,
        general_chat_prompt=GENERAL_CHAT_PROMPT_TEMPLATE,
        memory=memory,
    )

    # --- 3. DATASET PENGUJIAN ---
    eval_data = [
        {
            "question": "Apa saja fokus utama dari Laboratorium Sistem Cerdas di Informatika UMSIDA?",
            "ground_truth": "Fokus utama Laboratorium Sistem Cerdas adalah Software Engineering, Programming, dan Computer Science. Aktivitasnya meliputi pengembangan basis kode video game dan alat pengembangan software.",
        },
        {
            "question": "Sebutkan fasilitas umum yang tersedia di kampus UMSIDA untuk mahasiswa.",
            "ground_truth": "Fasilitas umum di UMSIDA meliputi Masjid Kampus, Gedung Perkuliahan, Area Parkir yang luas, Ruang Baca dan Perpustakaan, serta Kantin Universitas.",
        },
        {
            "question": "Berikan link untuk mengunduh Jadwal Praktikum PBO Semester Genap 2024/2025.",
            "ground_truth": "Jadwal Praktikum PBO (Pemrograman Berorientasi Objek) Semester Genap 2024/2025 dapat diunduh melalui tautan ini: https://informatika.umsida.ac.id/wp-content/uploads/2025/03/PRAKTIKUM-PBO-2024-2025.pdf",
        },
        {
            "question": "Dimana saya bisa download jadwal praktikum Jaringan Komputer?",
            "ground_truth": "Jadwal Praktikum Jaringan Komputer (Jarkom) Semester Genap 2024/2025 tersedia di tautan berikut: https://informatika.umsida.ac.id/wp-content/uploads/2025/03/PRAKTIKUM-JARINGAN-KOMPUTER-2024-2025.pdf",
        },
        {
            "question": "Bagaimana cara mengajukan surat keterangan aktif kuliah secara online?",
            "ground_truth": "Mahasiswa dapat mengajukan Surat Aktif Kuliah melalui layanan administrasi online menggunakan formulir berikut: https://docs.google.com/forms/d/1ijKTVs1T546WU__zqqEc9JeSfj2HoGVWFqhcGaJr-bs/viewform?edit_requested=true",
        },
        {
            "question": "Apa visi dari Program Studi Informatika UMSIDA?",
            "ground_truth": "Visi Program Studi Informatika UMSIDA adalah menghasilkan lulusan yang profesional, unggul, inovatif, dan kompetitif dalam rekayasa perangkat lunak dan sistem cerdas yang adaptif terhadap perkembangan IPTEKS berdasarkan nilai-nilai Islam untuk kesejahteraan masyarakat tingkat ASEAN pada tahun 2038.",
        },
        {
            "question": "Mata kuliah apa saja yang dipelajari pada semester 1?",
            "ground_truth": "Mata kuliah pada semester 1 meliputi: Kemanusiaan dan Keimanan, Algoritma dan Pemrograman, Sistem Digital, Arsitektur Komputer, Kalkulus, dan Fisika.",
        },
        {
            "question": "Sebutkan mata kuliah pilihan yang tersedia di semester 7.",
            "ground_truth": "Mata kuliah pilihan di semester 7 meliputi Multimedia, Game Programming (Game Prog), Web Mining, dan Ethical Hacking.",
        },
        {
            "question": "Apa itu program Magang Bersertifikat di UMSIDA?",
            "ground_truth": "Program Magang Bersertifikat adalah program MBKM yang berlangsung selama 1-2 semester untuk memberikan pengalaman industri agar mahasiswa siap kerja, dimana mahasiswa dapat mengaplikasikan ilmunya dan industri mendapatkan talenta.",
        },
        {
            "question": "Saya membutuhkan Sertifikat Akreditasi Informatika, dimana saya bisa mengunduhnya?",
            "ground_truth": "Dokumen Sertifikat Akreditasi Program Studi Informatika UMSIDA dapat diunduh melalui tautan ini: https://informatika.umsida.ac.id/wp-content/uploads/2025/04/file_sertifikat_25011014431107106055201_1742941667.pdf",
        },
        {
            "question": "Apakah ada dokumen SK mengenai alternatif pengganti skripsi?",
            "ground_truth": "Ya, Surat Keputusan (SK) Alternatif Pengganti Skripsi berisi aturan mengenai jalur kelulusan non-skripsi dan dapat diunduh di sini: https://informatika.umsida.ac.id/wp-content/uploads/2024/02/984SK-Penetapan-Kegiatan-Alternatif-sebagai-Pengganti-Tesis-Skripsi-Tugas-Akhir_11zon.pdf",
        },
        {
            "question": "Minta link download untuk Template Proposal Skripsi.",
            "ground_truth": "Template Proposal Skripsi Informatika yang berisi format baku dan aturan penulisan dapat diunduh di: https://informatika.umsida.ac.id/wp-content/uploads/2024/02/template-proposal-skripsi-1.docx",
        },
        {
            "question": "Dimana saya bisa mendapatkan formulir pendaftaran ujian proposal skripsi?",
            "ground_truth": "Form Pendaftaran Ujian Proposal Skripsi dapat diunduh melalui tautan berikut: https://informatika.umsida.ac.id/wp-content/uploads/2024/02/FORM-DAFTAR-UJIAN-PROPOSAL-SKRIPSI-FST.pdf",
        },
        {
            "question": "Bagaimana format penulisan proposal PKL? Apakah ada panduannya?",
            "ground_truth": "Format Penulisan Proposal PKL 2023 tersedia sebagai panduan dan template yang dapat diunduh di sini: https://informatika.umsida.ac.id/wp-content/uploads/2024/02/Format_Penulisan_PROPOSAL_PKL-1-1.docx",
        },
        {
            "question": "Apa fungsi dari surat keterangan lulus praktikum dan dimana downloadnya?",
            "ground_truth": "Surat Keterangan Lulus Praktikum menyatakan bahwa mahasiswa telah menyelesaikan seluruh beban praktikum dan menjadi syarat mendaftar skripsi/yudisium. Dokumennya dapat diunduh di: https://informatika.umsida.ac.id/wp-content/uploads/2024/02/SURAT-PERNYATAAN-LULUS-PRAKTIKUM-1-1.pdf",
        },
        {
            "question": "Saya ingin mengajukan dispensasi SPP, apakah ada formulirnya?",
            "ground_truth": "Ya, Form Surat Permohonan Dispensasi SPP untuk mengajukan keringanan atau penundaan pembayaran dapat diunduh di: https://informatika.umsida.ac.id/wp-content/uploads/2024/02/Format-Surat-Permohonan-Dispensasi-SPP-TA-Ganjil-2023-2024-1.docx",
        },
        {
            "question": "Apa peran HIMATIKA bagi mahasiswa?",
            "ground_truth": "HIMATIKA (Himpunan Mahasiswa Informatika) adalah wadah pengembangan pola pikir, kepribadian, dan potensi intelektual mahasiswa Informatika yang berlandaskan nilai-nilai Islam.",
        },
        {
            "question": "Apa itu ASLAB dan apa saja tugasnya?",
            "ground_truth": "ASLAB adalah Asisten Laboratorium Informatika, sebuah organisasi yang bertugas mewujudkan laboratorium bermutu, menyelenggarakan praktikum, dan menyediakan sarana penelitian.",
        },
        {
            "question": "Apa saja mata kuliah di semester 8?",
            "ground_truth": "Mata kuliah di Semester 8 hanya berfokus pada pengerjaan Skripsi dengan bobot 6 SKS.",
        },
        {
            "question": "Apakah ada form persetujuan dosen pembimbing skripsi?",
            "ground_truth": "Ya, Form Persetujuan Dosen Pembimbing Skripsi sebagai bukti tertulis persetujuan dosbing dapat diunduh di: https://informatika.umsida.ac.id/wp-content/uploads/2024/05/Form-Persetujuan-Dosen-Pembimbing-Skripsi_New.docx",
        },
    ]

    test_questions = [item["question"] for item in eval_data]
    ground_truths = [item["ground_truth"] for item in eval_data]

    # --- 4. RUN INFERENCE (Menjalankan Chatbot Anda) ---
    answers = []
    contexts = []

    print(f"Running inference on {len(test_questions)} questions...")

    for i, question in enumerate(test_questions):
        print(f"Processing {i+1}/{len(test_questions)}: {question}")

        inputs = {"question": question, "messages": [HumanMessage(content=question)]}
        # Gunakan thread_id unik agar memory tidak tercampur antar pertanyaan
        config = {"configurable": {"thread_id": f"eval_user_{i}"}}

        try:
            # Invoke Graph
            result = graph.invoke(inputs, config=config)

            # Extract Answer Logic (Sesuaikan dengan struktur output graph Anda)
            ans = ""
            if "generation" in result and result["generation"]:
                ans = result["generation"]
            elif "messages" in result and len(result["messages"]) > 0:
                # Ambil konten pesan terakhir dari AI
                last_msg = result["messages"][-1]
                ans = (
                    last_msg.content if hasattr(last_msg, "content") else str(last_msg)
                )
            else:
                ans = "No answer generated."

            answers.append(ans)

            # Extract Contexts
            # Pastikan key 'document' atau 'documents' ada di state graph Anda
            docs = result.get("document", [])
            ctx = [d.page_content for d in docs]
            contexts.append(ctx)

        except Exception as e:
            print(f"Error processing question '{question}': {e}")
            answers.append("Error generating answer")
            contexts.append([])

    # --- 5. PREPARE DATASET FOR RAGAS ---
    data = {
        "question": test_questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }
    dataset = Dataset.from_dict(data)

    # --- 6. RUN EVALUATION (THE FIX) ---
    print("\nCalculating metrics with Ragas...")

    # Konfigurasi PENTING untuk Groq:
    # max_workers=1 -> Agar tidak kena Rate Limit (Sequential)
    # timeout=120 -> Memberi waktu lebih lama untuk respon
    run_config = RunConfig(max_workers=1, timeout=180, max_retries=5, max_wait=60)

    try:
        results = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            llm=llm_judge,
            embeddings=embedding_model,
            run_config=run_config,
            raise_exceptions=False,  # PENTING: Jangan crash jika 1 data gagal
        )

        print("\nEvaluation Results:")
        print(results)

        # Save to CSV
        df = results.to_pandas()

        # Cek jika ada yang NaN (gagal dievaluasi)
        failed_rows = df[df["faithfulness"].isna()]
        if not failed_rows.empty:
            print(
                f"\n⚠️ Warning: {len(failed_rows)} rows failed to evaluate (check Groq limits)."
            )

        output_file = "ragas_evaluation_results.csv"
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")

    except Exception as e:
        print(f"\n❌ Fatal Error during evaluation: {e}")


if __name__ == "__main__":
    run_evaluation()
