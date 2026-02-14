import streamlit as st
import os
import time
import chromadb
from chromadb.config import Settings

# PyPDFLoader yerine çok daha başarılı olan PyMuPDFLoader kullanıyoruz
from langchain_community.document_loaders import PyMuPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings

# --- DEĞİŞKENLERİ AYARLAYIN ---
PDF_DOSYA_ADI = "ABAP-1_merged.pdf"
DB_DIZINI = "chroma_db"
KOLEKSIYON_ADI = "gaih-abap-chatbot"

# Deploy için API anahtarını ayarla
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

@st.cache_resource
def load_rag_chain():
    """
    Uygulama açılırken sadece bir kez çalışır.
    PDF'i işler, vektör veritabanını oluşturur (veya yükler) ve RAG zincirini kurar.
    """
    
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", task_type="RETRIEVAL_QUERY")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    if os.path.exists(DB_DIZINI):
        st.info("Mevcut veritabanı yükleniyor...")
        client_settings = Settings(anonymized_telemetry=False, is_persistent=True)
        client = chromadb.PersistentClient(path=DB_DIZINI, settings=client_settings)
        
        vector_store = Chroma(
            client=client,
            collection_name=KOLEKSIYON_ADI,
            embedding_function=embeddings,
        )
        st.info("Veritabanı başarıyla yüklendi.")
    else:
        st.info("Veritabanı bulunamadı. Dökümanlar işleniyor (Bu biraz sürebilir)...")
        
        # GÜNCELLEME: PyMuPDFLoader metinleri, formatları ve tabloları çok daha iyi yakalar.
        loader = PyMuPDFLoader(PDF_DOSYA_ADI)
        documents = loader.load()
        
        # Chunking ayarlarını biraz daha optimize ettik
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250)
        texts = text_splitter.split_documents(documents)
        
        vector_store = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            collection_name=KOLEKSIYON_ADI,
            persist_directory=DB_DIZINI
        )
        st.info("Veritabanı oluşturuldu ve diske kaydedildi.")

    # Arama katsayısını (k) 4'e çıkarmak daha geniş bir bağlam yakalamayı sağlar
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # LLM Modeli
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3) 

    # Geliştirilmiş Prompt Şablonu
    prompt_template = """
    Sen deneyimli bir SAP ABAP danışmanısın. Sana verilen bağlamı (context) kullanarak kullanıcının sorusunu yanıtla.
    Eğer bağlamda kod parçacıkları veya teknik detaylar varsa, bunları formatına uygun şekilde (Markdown kod blokları içinde) ilet.
    Bilgiyi bulduğun sayfa numaralarını referans olarak ekle.
    Cevabı bağlamda bulamazsan, kendi bilgilerini uydurma ve "Üzgünüm, bu bilgi notlarımda yer almıyor." de.

    Bağlam: {context}
    
    Soru: {question}
    Cevap:
    """
    prompt = PromptTemplate.from_template(prompt_template)
    
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# --- STREAMLIT ARAYÜZÜ ---

st.set_page_config(page_title="Kişisel ABAP Asistanı", page_icon="💻", layout="wide")
st.title("Kişisel ABAP Asistanı 💻")
st.markdown("Kendi SAP ABAP notlarınız, kodlarınız ve dökümanlarınız hakkında sorular sorun.")
st.divider()

try:
    rag_chain = load_rag_chain()
    st.success("Asistan hazır! Sorularınızı bekliyor.")
except Exception as e:
    st.error(f"Asistan yüklenirken bir hata oluştu: {e}")
    st.stop()

# Sohbet hafızasını başlat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hafızadaki mesajları göster
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Kullanıcıdan yeni soru al
if prompt := st.chat_input("Örn: ALV Grid oluşturmak için hangi fonksiyon kullanılır?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Notlar taranıyor ve cevap hazırlanıyor..."):
            response = rag_chain.invoke(prompt)
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})








