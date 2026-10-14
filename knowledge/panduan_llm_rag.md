# Panduan Belajar LLM, RAG, dan Chatbot

## Halusinasi
Halusinasi adalah kondisi saat LLM menghasilkan jawaban yang terdengar meyakinkan tetapi salah. Salah satu penyebabnya adalah prompt yang tidak memberi konteks cukup jelas. Cara mengurangi: beri konteks (RAG), minta model mengakui ketidaktahuan, dan gunakan teknik prompting yang tepat.

## Teknik Prompting
- Zero-shot: tugas diberikan tanpa contoh.
- Few-shot: model diberi 2-5 contoh sebelum diminta mengerjakan tugas.
- Role-based: model diminta menjawab dengan peran tertentu, misalnya dokter atau mentor.
- Chain-of-Thought (CoT): model diminta menalar langkah demi langkah sebelum menjawab.

## Parameter Model
- Temperature: mengontrol keacakan dan kreativitas. Temperature 0 bersifat deterministik.
- Max tokens: batas panjang output. Jika terlalu kecil, jawaban bisa terpotong.
- Top-p (nucleus sampling): memilih token dari probabilitas tertinggi hingga total mencapai nilai p.
- Top-k: memilih dari k token paling mungkin (tersedia di Gemini).
- Frequency penalty: mengurangi pengulangan kata. Presence penalty: mendorong kata atau ide baru.

## RAG (Retrieval-Augmented Generation)
RAG menggabungkan retrieval (pencarian informasi eksternal) dan generation (pembuatan teks oleh LLM). Alurnya: pertanyaan diubah menjadi vektor, dicari dokumen terdekat di vector database, dokumen dikirim ke LLM sebagai konteks, lalu LLM menjawab berdasarkan referensi tersebut. Manfaat RAG: mengurangi halusinasi, informasi dapat diperbarui tanpa melatih ulang model, dan implementasi chatbot lebih cepat, mudah, serta murah.

## Vector Database
Vector database menyimpan dan mencari data dalam bentuk vektor (deretan angka). Dokumen diubah menjadi vektor oleh embedding model, lalu pencarian dilakukan dengan kemiripan, misalnya cosine similarity, dan mengembalikan top-K vektor terdekat. Contoh: FAISS, Chroma, Qdrant (open source); Pinecone dan Vertex AI Vector Search (enterprise).

## LangChain
LangChain adalah framework Python untuk membangun aplikasi berbasis LLM. Komponen dasarnya: Model, Prompt Template, Chain, dan Memory. Prompt Template menjaga konsistensi instruksi dan memudahkan penggunaan ulang pola prompt dengan placeholder.

## AI Agent dan Function Calling
Function calling memungkinkan LLM memutuskan untuk memanggil fungsi (tool) yang disediakan. AI Agent pada dasarnya adalah sekumpulan tool yang dapat dipilih model secara otomatis untuk mencapai tujuan. Berbeda dengan AI Assistant yang bersifat reaktif dan hanya memberi saran, agent bersifat proaktif dan berorientasi tujuan.

## Streamlit
Streamlit adalah framework open-source berbasis Python untuk membuat aplikasi web interaktif tanpa HTML, CSS, atau JavaScript. Untuk chatbot tersedia widget st.chat_input dan st.chat_message. Aplikasi dapat dideploy gratis lewat Streamlit Community Cloud (share.streamlit.io) dari repositori GitHub.

## Etika AI
Prinsip utama: transparansi, keadilan, akuntabilitas, privasi, dan keamanan. Manusia, bukan mesin, bertanggung jawab atas hasil yang dihasilkan AI.
