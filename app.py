import os
import tempfile

import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
st.set_page_config(page_title="Si Pintar", page_icon="🎓")
KB_DEFAULT = os.path.join(os.path.dirname(__file__), "knowledge", "panduan_llm_rag.md")

# ---------- Prompt Template (LangChain): role-based + few-shot + anti-halusinasi ----------
SYSTEM = """Kamu adalah 'Si Pintar', mentor yang sabar untuk belajar LLM, RAG, dan Data Science.
Gaya bahasa: {gaya}. Level pengguna: {level}.

Aturan:
1. Utamakan KONTEKS di bawah dan sebutkan sumbernya (nama file/halaman). Jika konteks tidak memuat
   jawabannya dan kamu tidak yakin, katakan terus terang. Jangan mengarang (hindari halusinasi).
2. Untuk topik teknis, jelaskan bertahap (chain-of-thought singkat), lalu beri analogi atau contoh.
3. Gunakan tool cari_wikipedia untuk istilah umum di luar konteks. Gunakan tool simpan_catatan
   saat pengguna menyebut tujuan belajar atau preferensinya.
4. Jaga privasi: jangan meminta data pribadi. Akui bahwa kamu AI jika ditanya.
5. Akhiri dengan bagian '📚 Lanjut belajar:' berisi 2 rekomendasi topik berikutnya.

Contoh gaya jawaban:
Pengguna: Apa itu vector database?
Si Pintar: Bayangkan perpustakaan yang menata buku berdasarkan kemiripan topik, bukan abjad. Teks diubah
menjadi vektor (deretan angka), lalu dicari yang terdekat dengan pertanyaan. ...

Catatan tentang pengguna (memory): {catatan}

KONTEKS:
{konteks}"""

PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM), MessagesPlaceholder("riwayat"), ("human", "{pertanyaan}")]
)


def secret(nama):
    try:
        return st.secrets[nama]
    except Exception:
        return os.getenv(nama)


# ---------- Tools (Function Calling / AI Agent) ----------
@tool
def cari_wikipedia(topik: str) -> str:
    """Cari ringkasan artikel Wikipedia Indonesia untuk sebuah istilah atau topik umum."""
    try:
        r = requests.get(
            "https://id.wikipedia.org/w/api.php",
            params={"action": "query", "format": "json", "generator": "search",
                    "gsrsearch": topik, "gsrlimit": 1, "prop": "extracts",
                    "exintro": 1, "explaintext": 1, "exchars": 1000},
            headers={"User-Agent": "SiPintarBot/1.0 (proyek edukasi)"}, timeout=8,
        )
        p = next(iter(r.json().get("query", {}).get("pages", {}).values()), None)
        return f"{p['title']}: {p['extract']}" if p else "Tidak ditemukan."
    except Exception as e:
        return f"Gagal mengakses Wikipedia: {e}"


@tool
def simpan_catatan(catatan: str) -> str:
    """Simpan tujuan belajar atau preferensi pengguna ke memori agar diingat di percakapan berikutnya."""
    st.session_state.catatan.append(catatan)
    return "Catatan tersimpan."


TOOLS = {t.name: t for t in (cari_wikipedia, simpan_catatan)}


# ---------- RAG: dokumen -> chunk -> embedding -> FAISS (vector database) ----------
@st.cache_resource(show_spinner="Membangun vector database...")
def bangun_db(nama: str, data: bytes):
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.embeddings import FastEmbedEmbeddings
    from langchain_community.vectorstores import FAISS

    if nama.lower().endswith(".pdf"):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(data)
        docs = PyPDFLoader(f.name).load()
        for d in docs:
            d.metadata["source"] = nama
    else:
        docs = [Document(page_content=data.decode("utf-8"), metadata={"source": nama})]
    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(docs)
    emb = FastEmbedEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return FAISS.from_documents(chunks, emb)


def buat_llm(provider, model, temp, top_p, top_k, max_tok, fp, pp):
    if provider.startswith("Groq"):
        from langchain_groq import ChatGroq
        kw = {"top_p": top_p}
        if fp:
            kw["frequency_penalty"] = fp
        if pp:
            kw["presence_penalty"] = pp
        return ChatGroq(model=model, api_key=secret("GROQ_API_KEY"), temperature=temp,
                        max_tokens=max_tok, model_kwargs=kw)
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=model, google_api_key=secret("GOOGLE_API_KEY"),
                                  temperature=temp, max_output_tokens=max_tok,
                                  top_p=top_p, top_k=top_k)


def teks(msg):
    c = msg.content
    if isinstance(c, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c)
    return c


def jalankan(llm, msgs, pakai_tools, status):
    """Loop agent: model memutuskan tool mana yang dipanggil, hasilnya dikembalikan ke model."""
    if not pakai_tools:
        return llm.invoke(msgs)
    asli = list(msgs)
    try:
        llm_t = llm.bind_tools(list(TOOLS.values()))
        for _ in range(3):
            ai = llm_t.invoke(msgs)
            if not ai.tool_calls:
                return ai
            msgs.append(ai)
            for tc in ai.tool_calls:
                status.write(f"🔧 Tool `{tc['name']}` dipanggil: {tc['args']}")
                hasil = TOOLS[tc["name"]].invoke(tc["args"])
                msgs.append(ToolMessage(content=str(hasil), tool_call_id=tc["id"]))
    except Exception as e:
        status.write(f"Tool calling gagal ({type(e).__name__}); lanjut tanpa tools.")
    return llm.invoke(asli)


# ---------- Sidebar: konfigurasi ----------
st.session_state.setdefault("messages", [])
st.session_state.setdefault("catatan", [])

with st.sidebar:
    st.header("⚙️ Pengaturan")
    provider = st.selectbox("Model AI", ["Groq (Llama)", "Google Gemini"])
    groq = provider.startswith("Groq")
    model = st.text_input("Nama model", "llama-3.3-70b-versatile" if groq else "gemini-2.5-flash",
                          key=f"model_{provider}", help="Cek daftar model terbaru di dokumentasi provider.")
    with st.expander("🎛️ Parameter model"):
        temp = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1, help="Makin tinggi, makin kreatif.")
        top_p = st.slider("Top-p", 0.1, 1.0, 0.95, 0.05)
        top_k = st.slider("Top-k (Gemini)", 1, 100, 40, disabled=groq)
        max_tok = st.slider("Max tokens", 128, 4096, 1024, 128)
        fp = st.slider("Frequency penalty (Groq)", 0.0, 2.0, 0.0, 0.1, disabled=not groq)
        pp = st.slider("Presence penalty (Groq)", 0.0, 2.0, 0.0, 0.1, disabled=not groq)
    st.subheader("🎭 Persona")
    gaya = st.radio("Gaya bahasa", ["Santai", "Formal"], horizontal=True)
    level = st.selectbox("Level kamu", ["Pemula", "Menengah", "Mahir"])
    st.subheader("📚 Pengetahuan (RAG)")
    pakai_rag = st.toggle("Aktifkan RAG", True)
    upload = st.file_uploader("Unggah PDF (opsional)", type="pdf")
    pakai_tools = st.toggle("Aktifkan tools (agent)", True)
    if st.session_state.catatan:
        st.subheader("🧠 Memory")
        for c in st.session_state.catatan:
            st.caption(f"• {c}")
    if st.button("🗑️ Reset percakapan"):
        st.session_state.clear()
        st.rerun()

st.title("🎓 Si Pintar")
st.caption("Asisten belajar LLM, RAG & Data Science · LangChain + Streamlit")

kunci = "GROQ_API_KEY" if groq else "GOOGLE_API_KEY"
if not secret(kunci):
    st.error(f"{kunci} belum diatur (file .env atau Streamlit Secrets).")
    st.stop()

db = None
if pakai_rag:
    try:
        if upload:
            db = bangun_db(upload.name, upload.getvalue())
        else:
            with open(KB_DEFAULT, "rb") as f:
                db = bangun_db("panduan_llm_rag.md", f.read())
    except Exception as e:
        st.sidebar.warning(f"RAG nonaktif: {e}")

if not st.session_state.messages:
    st.info("Coba tanya: *Apa itu RAG dan kenapa bisa mengurangi halusinasi?*")
for m in st.session_state.messages:
    st.chat_message(m["role"]).write(m["content"])

# ---------- Chat ----------
if pertanyaan := st.chat_input("Ketik pertanyaanmu..."):
    st.chat_message("user").write(pertanyaan)
    konteks, sumber = "(RAG nonaktif)", []
    if db:
        sumber = db.similarity_search(pertanyaan, k=3)
        konteks = "\n\n".join(
            f"[{d.metadata.get('source')}" + (f" hal.{d.metadata['page'] + 1}" if "page" in d.metadata else "")
            + f"] {d.page_content}" for d in sumber)
    riwayat = [(HumanMessage if m["role"] == "user" else AIMessage)(content=m["content"])
               for m in st.session_state.messages[-8:]]
    msgs = PROMPT.invoke({
        "gaya": gaya.lower(), "level": level.lower(), "konteks": konteks,
        "catatan": "; ".join(st.session_state.catatan) or "-",
        "riwayat": riwayat, "pertanyaan": pertanyaan,
    }).to_messages()

    with st.chat_message("assistant"):
        try:
            llm = buat_llm(provider, model, temp, top_p, top_k, max_tok, fp, pp)
            with st.status("Si Pintar sedang berpikir...", expanded=False) as status:
                jawaban = teks(jalankan(llm, msgs, pakai_tools, status))
                status.update(label="Selesai", state="complete")
            st.write(jawaban)
            if sumber:
                with st.expander("📄 Sumber RAG"):
                    for d in sumber:
                        st.caption(d.page_content[:300] + "...")
            st.session_state.messages += [{"role": "user", "content": pertanyaan},
                                          {"role": "assistant", "content": jawaban}]
        except Exception as e:
            st.error(f"Gagal menghubungi model: {e}")

st.caption("Si Pintar adalah AI dan bisa keliru; periksa kembali informasi penting. Jangan memasukkan data pribadi.")
