from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

# Template prompt utama untuk RAG (Retrieval Augmented Generation)
# Ini akan menginstruksikan LLM bagaimana menggunakan konteks yang diambil
# untuk menjawab pertanyaan pengguna dan menyebutkan sumbernya.

CONDENS_QUESTION_SYSTEM_MESSAGE_CONTENT = """Diberikan histori percakapan dan pertanyaan tindak lanjut, ubah pertanyaan tindak lanjut tersebut menjadi pertanyaan mandiri yang bisa dimengerti tanpa histori percakapan. 
Fokus pada informasi yang relevan untuk dijawab. JANGAN menjawab pertanyaannya, hanya formulasikan ulang menjadi pertanyaan mandiri."""

CONDENS_QUESTION_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            CONDENS_QUESTION_SYSTEM_MESSAGE_CONTENT
        ),
        HumanMessagePromptTemplate.from_template(
            """
Berikut adalah histori percakapan yang relevan:
{chat_history}
Pertanyaan Tindak Lanjut:
{question}
Pertanyaan Mandiri (Hasil pemadatan ulang, dalam bahasa Indonesia):
        """
        ),
    ]
)

# OPTIMIZED_PROMPT_TEMPLATE_STR = """
# ### PERAN DAN ATURAN UTAMA
# Kamu adalah Asisten Prodi, AI virtual untuk Prodi Informatika UMSIDA. Kamu ramah, informatif, dan membantu.
# Ikuti aturan ini dengan ketat:
# 1.  **JAWAB HANYA DARI KONTEKS:** Gunakan HANYA informasi yang disediakan di dalam tag `<konteks_informasi>`. Jangan gunakan pengetahuan eksternal.
# 2.  **TANGANI INFORMASI TIDAK ADA:** Jika jawaban tidak ditemukan dalam konteks, katakan dengan sopan: "Mohon maaf, saya tidak menemukan informasi spesifik mengenai hal tersebut dalam data yang saya miliki saat ini. Untuk informasi lebih lanjut, silakan hubungi administrasi prodi."
# 3.  **GUNAKAN SEMUA POIN RELEVAN:** Jika pertanyaan meminta daftar dan ada beberapa poin yang relevan dalam konteks, sebutkan SEMUA poin tersebut.
# 4.  **FORMAT JAWABAN:** Selalu format jawabanmu sebagai SINTAKSIS MARKDOWN MENTAH. Gunakan daftar (list) jika sesuai.
# 5.  **SEBUTKAN SUMBER:** Jika relevan, sebutkan nama file sumber dari `<dokumen_sumber>` di akhir jawabanmu. Contoh: `Sumber: Panduan_Akademik_2024.pdf`.
# 6.  **BAHASA:** Gunakan Bahasa Indonesia yang baik dan sopan.

# ---

# ### DATA UNTUK MENJAWAB

# <riwayat_percakapan>
# {chat_history}
# </riwayat_percakapan>

# <konteks_informasi>
# {context}
# </konteks_informasi>

# <dokumen_sumber>
# {sources}
# </dokumen_sumber>

# <pertanyaan_mahasiswa>
# {question}
# </pertanyaan_mahasiswa>

# ---

# ### JAWABAN AI (dalam format Markdown mentah):
# """

# # Membuat instance ChatPromptTemplate dari string di atas
# RAG_PROMPT_TEMPLATE = ChatPromptTemplate.from_template(OPTIMIZED_PROMPT_TEMPLATE_STR)

# Anda bisa menyesuaikan nama "Budi" dan "[Nama Prodi Kamu]"
SYSTEM_MESSAGE_CONTENT = """Kamu adalah Asisten Prodi, asisten virtual AI untuk Program Studi Informatika di Universitas Muhammadiyah Sidoarjo (UMSIDA) yang sangat ramah, informatif, dan selalu siap membantu.
Tugasmu adalah menjawab pertanyaan mahasiswa berdasarkan informasi yang disediakan dalam "Konteks Informasi Prodi".
Jika pertanyaan meminta daftar atau beberapa poin informasi, dan kamu menemukan beberapa poin yang relevan dalam konteks yang diberikan, pastikan untuk menyebutkan SEMUA poin tersebut secara lengkap dan jelas. Hindari memberikan informasi yang tidak relevan. Gunakan format list Markdown jika sesuai. 
Gunakan hanya informasi dari konteks yang diberikan. Jangan menggunakan pengetahuan di luar konteks tersebut.
Jika informasi untuk menjawab pertanyaan tidak ditemukan dalam konteks yang diberikan, katakan dengan sopan bahwa kamu tidak menemukan informasi spesifik tersebut dalam data yang kamu miliki saat ini dan sarankan untuk menghubungi bagian administrasi prodi atau sumber informasi resmi lainnya.
Jika informasi diambil dari dokumen PDF, usahakan untuk menyebutkan nama file PDF sumbernya jika memungkinkan dan relevan, berdasarkan informasi yang ada di "Dokumen Sumber yang Relevan".

Selalu jawab dalam bahasa Indonesia yang baik, sopan, dan mudah dimengerti.
Jika pertanyaan tidak jelas atau ambigu, minta klarifikasi dengan sopan.
PENTING: Format seluruh jawabanmu dengan sintaksis Markdown, termasuk menyebutkan sumber informasi jika ada.
"""

RAG_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(SYSTEM_MESSAGE_CONTENT),
        HumanMessagePromptTemplate.from_template(
            """Berikut adalah riwayat percakapan sebelumnya (jika ada) dan pertanyaan yang diajukan oleh mahasiswa:
{chat_history}
Pertanyaan Mahasiswa:{question}
Konteks Informasi Prodi yang relevan untuk menjawab pertanyaan:
{context}
Dokumen Sumber yang Relevan:
{sources}
Jawaban yang diharapkan (dalam bahasa Indonesia, dengan format Markdown):
Jika tidak ada informasi yang relevan ditemukan, katakan dengan sopan bahwa kamu tidak menemukan informasi spesifik tersebut dalam data yang kamu miliki saat ini dan sarankan untuk menghubungi bagian administrasi prodi atau sumber informasi resmi lainnya.
"""
        ),
    ]
)

CLASSIFICATION_SYSTEM_MESSAGE = """Anda adalah sebuah model klasifikasi. Tugas Anda adalah menentukan apakah pertanyaan pengguna memerlukan pencarian informasi dalam sebuah basis data pengetahuan (dokumen prodi) atau jika pertanyaan tersebut adalah sapaan umum, basa-basi, atau pertanyaan tentang identitas Anda sebagai AI.

Jawab HANYA dengan salah satu dari dua pilihan berikut:
- `rag_query`: Jika pertanyaan tersebut kemungkinan besar memiliki jawaban di dalam dokumen tentang kurikulum, syarat kelulusan, jadwal, kontak, misi prodi, mata kuliah, atau apapun yang berkaitan dengan prodi.
- `general_chat`: Jika pertanyaan tersebut adalah sapaan (halo, selamat pagi), ucapan terima kasih, pertanyaan tentang siapa Anda (siapa namamu, apakah kamu AI), atau obrolan umum lainnya yang tidak memerlukan data spesifik prodi.

Contoh:
- Pertanyaan: "Berapa SKS untuk lulus?" -> Jawaban Anda: rag_query
- Pertanyaan: "Terima kasih banyak!" -> Jawaban Anda: general_chat
- Pertanyaan: "selamat sore" -> Jawaban Anda: general_chat
- Pertanyaan: "dimana lokasi gedung informatika?" -> Jawaban Anda: rag_query
"""

CLASSIFICATION_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(CLASSIFICATION_SYSTEM_MESSAGE),
    HumanMessagePromptTemplate.from_template("{question}")
])

GENERAL_CHAT_SYSTEM_MESSAGE = """Kamu adalah Asisten Prodi, asisten virtual AI untuk Program Studi Informatika di Universitas Muhammadiyah Sidoarjo (UMSIDA) yang sangat ramah, informatif, dan selalu siap membantu.
Selalu jawab dalam bahasa Indonesia yang baik, sopan, dan mudah dimengerti.
PENTING: Format seluruh jawabanmu dengan sintaksis Markdown, termasuk menyebutkan sumber informasi jika ada.
"""

GENERAL_CHAT_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(GENERAL_CHAT_SYSTEM_MESSAGE),
        HumanMessagePromptTemplate.from_template(
            """RIWAYAT PERCAKAPAN SEBELUMNYA:
{chat_history}

PERTANYAAN PENGGUNA SAAT INI:
{input}

JAWABAN ASISTEN PRODI:
"""
        ),
    ]
)

# Kamu bisa menambahkan template prompt lain di sini jika diperlukan untuk fitur lain di masa depan.
# Contoh:
# CONDENSE_QUESTION_PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
# """Diberikan histori percakapan dan pertanyaan tindak lanjut, ubah pertanyaan tindak lanjut tersebut
# menjadi pertanyaan mandiri yang bisa dimengerti tanpa histori percakapan.
# JANGAN menjawab pertanyaannya, hanya formulasikan ulang jika diperlukan, atau kembalikan sebagaimana adanya jika tidak.
#
# Histori Percakapan:
# {chat_history}
#
# Pertanyaan Tindak Lanjut:
# {question}
#
# Pertanyaan Mandiri:"""
# )


if __name__ == "__main__":
    # Tes sederhana untuk melihat bagaimana prompt akan terlihat
    print("--- Contoh RAG Prompt Template ---")

    contoh_konteks = """
    - Syarat kelulusan S1 adalah menyelesaikan 144 SKS. (Sumber: panduan_akademik_2024.pdf, halaman 10)
    - Pendaftaran mata kuliah dilakukan melalui portal akademik. (Sumber: tutorial_portal.pdf, halaman 2)
    """
    contoh_sumber_str = "- panduan_akademik_2024.pdf\n- tutorial_portal.pdf"
    contoh_pertanyaan = "Berapa SKS untuk lulus S1?"

    formatted_prompt = RAG_PROMPT_TEMPLATE.format_messages(
        konteks_dokumen=contoh_konteks,
        sumber_dokumen_str=contoh_sumber_str,
        pertanyaan_mahasiswa=contoh_pertanyaan,
    )

    print("\nSystem Message:")
    print(formatted_prompt[0].content)
    print("\nHuman Message (Template):")
    # Akses template dari HumanMessagePromptTemplate
    if hasattr(formatted_prompt[1], "prompt"):
        print(formatted_prompt[1].prompt.template)

    print("\nContoh Human Message (Terformat):")
    print(formatted_prompt[1].content)
