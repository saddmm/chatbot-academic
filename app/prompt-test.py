from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
)

# ==========================================
# 1. CONDENSE QUESTION PROMPT
# ==========================================
CONDENS_QUESTION_SYSTEM_MESSAGE_CONTENT = """Kamu adalah mesin pemroses teks. Tugasmu adalah memformulasikan ulang pertanyaan user menjadi pertanyaan yang berdiri sendiri (standalone).

ATURAN WAJIB (STRICT):
1. HANYA outputkan teks pertanyaan hasil revisi.
2. DILARANG KERAS menambahkan kalimat pembuka/penutup.
3. JANGAN menjawab pertanyaan tersebut.
"""

CONDENS_QUESTION_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            CONDENS_QUESTION_SYSTEM_MESSAGE_CONTENT
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)

# ==========================================
# 2. RAG PROMPT (INTI CHATBOT - STRICT MODE)
# ==========================================

SYSTEM_MESSAGE_CONTENT = """Kamu adalah Asisten Akademik Prodi Informatika UMSIDA.
Tugasmu adalah menjawab pertanyaan mahasiswa HANYA berdasarkan "Konteks Informasi Prodi" yang diberikan.

ATURAN PENJAWABAN (SANGAT KETAT):
1. **NO CHIT-CHAT:** DILARANG menggunakan kata sapaan, pembuka, atau penutup (seperti "Halo", "Tentu", "Berikut informasinya", "Semoga membantu"). Langsung tulis jawaban intinya.
2. **FAITHFULNESS 100%:** 
   - Gunakan kalimat yang semirip mungkin dengan teks di Konteks.
   - Jangan memparafrase jika tidak perlu.
   - Jika informasi tidak ada di Konteks, KATAKAN: "Maaf, informasi tidak tersedia." (Jangan mengarang!).
3. **STRUKTUR:** Gunakan Bullet Points (-) untuk daftar item.
4. **LINK & GAMBAR:** Salin link persis seperti yang tertulis di Konteks (Markdown format: `[Judul](URL)`).

Gaya Bahasa: Formal, Objektif, Langsung.
"""

RAG_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(SYSTEM_MESSAGE_CONTENT),
        HumanMessagePromptTemplate.from_template(
            """
KONTEKS INFORMASI:
{context}

RIWAYAT CHAT:
{chat_history}

PERTANYAAN MAHASISWA:
{question}

JAWABAN:
"""
        ),
    ]
)

# ==========================================
# 3. CLASSIFICATION PROMPT
# ==========================================
CLASSIFICATION_SYSTEM_MESSAGE = """Kamu adalah router klasifikasi.
Tugas: Tentukan apakah pertanyaan user membutuhkan data akademik (RAG) atau hanya sapaan (General).

Output HANYA satu kata:
- `rag_query`: Untuk pertanyaan tentang jadwal, dosen, kurikulum, organisasi, skripsi, yudisium, fasilitas, lokasi, biaya, dll.
- `general_chat`: Untuk sapaan (halo, pagi), ucapan terima kasih, pujian bot, atau pertanyaan identitas bot.
"""

CLASSIFICATION_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(CLASSIFICATION_SYSTEM_MESSAGE),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)

# ==========================================
# 4. GENERAL CHAT PROMPT
# ==========================================
GENERAL_CHAT_SYSTEM_MESSAGE = """Kamu adalah Asisten Prodi Informatika UMSIDA.
Jawab sapaan dengan singkat dan sopan.
"""

GENERAL_CHAT_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(GENERAL_CHAT_SYSTEM_MESSAGE),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}"),
    ]
)
