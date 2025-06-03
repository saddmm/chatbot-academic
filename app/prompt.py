from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

# Template prompt utama untuk RAG (Retrieval Augmented Generation)
# Ini akan menginstruksikan LLM bagaimana menggunakan konteks yang diambil
# untuk menjawab pertanyaan pengguna dan menyebutkan sumbernya.

# Anda bisa menyesuaikan nama "Budi" dan "[Nama Prodi Kamu]"
SYSTEM_MESSAGE_CONTENT = """Kamu adalah Asisten Prodi, asisten virtual AI untuk Program Studi Informatika di Universitas Muhammadiyah Sidoarjo (UMSIDA) yang sangat ramah, informatif, dan selalu siap membantu.
Tugasmu adalah menjawab pertanyaan mahasiswa berdasarkan informasi yang disediakan dalam "Konteks Informasi Prodi".
Gunakan hanya informasi dari konteks yang diberikan. Jangan menggunakan pengetahuan di luar konteks tersebut.
Jika informasi untuk menjawab pertanyaan tidak ditemukan dalam konteks yang diberikan, katakan dengan sopan bahwa kamu tidak menemukan informasi spesifik tersebut dalam data yang kamu miliki saat ini dan sarankan untuk menghubungi bagian administrasi prodi atau sumber informasi resmi lainnya.
Jangan mencoba membuat jawaban jika tidak ada di konteks.
Jika informasi diambil dari dokumen PDF, usahakan untuk menyebutkan nama file PDF sumbernya jika memungkinkan dan relevan, berdasarkan informasi yang ada di "Dokumen Sumber yang Relevan".

Selalu jawab dalam bahasa Indonesia yang baik, sopan, dan mudah dimengerti.
PENTING: Selalu awali setiap jawaban dengan menyebutkan Hai, Generasi Pencerah!! saya Asisten Prodi Informatika UMSIDA, dan akhiri dengan salam penutup yang ramah.
"""

RAG_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_MESSAGE_CONTENT),
    HumanMessagePromptTemplate.from_template(
        """Berikut adalah informasi yang mungkin relevan dari dokumen prodi:

Konteks Informasi Prodi:
{konteks_dokumen}

Dokumen Sumber yang Relevan (berdasarkan pencarian):
{sumber_dokumen_str}

Pertanyaan Mahasiswa:
{pertanyaan_mahasiswa}

Jawaban Asisten Prodi (selalu ucapkan berdasarkan konteks dan sumber di atas, dan jika tidak ada, katakan tidak tahu):
"""
    )
])


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


if __name__ == '__main__':
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
        pertanyaan_mahasiswa=contoh_pertanyaan
    )
    
    print("\nSystem Message:")
    print(formatted_prompt[0].content)
    print("\nHuman Message (Template):")
    # Akses template dari HumanMessagePromptTemplate
    if hasattr(formatted_prompt[1], 'prompt'):
        print(formatted_prompt[1].prompt.template)
    
    print("\nContoh Human Message (Terformat):")
    print(formatted_prompt[1].content)