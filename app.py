import streamlit as st
import os
import chromadb
from chromadb.config import Settings

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- DEĞİŞKENLERİ AYARLAYIN ---
# 1. PDF dosyanızın adını buraya yazın
PDF_DOSYA_ADI = "ABAP-1_merged.pdf" 
# 2. Vektör veritabanının kaydedileceği klasör
DB_DIZINI = "./chroma_db"
# 3. ChromaDB koleksiyon adı
KOLEKSIYON_ADI = "gaih-abap-chatbot"
# --- AYARLAR BİTTİ ---


# API Anahtarını Streamlit Secrets'tan al ve ortam değişkeni olarak ayarla
# Lokal test için: 
os.environ["GOOGLE_API_KEY"] = "key"
# Deploy için:
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]


@st.cache_resource
def load_rag_chain():
    """
    Uygulama açılırken sadece bir kez çalışır.
    PDF'i işler, vektör veritabanını oluşturur (veya yükler) ve RAG zincirini kurar.
    """
    
    # Embedding modelini yükle
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # ChromaDB istemcisini ayarla (Diske kaydeden versiyon)
    client_settings = Settings(
        anonymized_telemetry=False,
        is_persistent=True
    )
    client = chromadb.PersistentClient(
        path=DB_DIZINI, 
        settings=client_settings
    )

    try:
        # 1. Varolan veritabanını yüklemeyi dene
        vector_store = Chroma(
            client=client,
            collection_name=KOLEKSIYON_ADI,
            embedding_function=embeddings,
        )
        print("Mevcut veritabanı yüklendi.")
    
    except Exception as e:
        # 2. Veritabanı yoksa veya yüklenemezse, sıfırdan oluştur
        print(f"Veritabanı yüklenemedi (Hata: {e}). Yeniden oluşturuluyor...")
        
        # PDF'i yükle ve böl
        loader = PyPDFLoader(PDF_DOSYA_ADI)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        
        # Vektör veritabanını oluştur ve diske kaydet
        vector_store = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            client=client,
            collection_name=KOLEKSIYON_ADI
        )
        print("Veritabanı oluşturuldu ve kaydedildi.")

    # 3. Retriever oluştur
    # search_kwargs={"k": 3} -> Soruya en yakın 3 metin parçasını bul
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # 4. LLM'i ayarla (En son düzelttiğimiz model adı)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.57)

    # 5. Prompt şablonunu oluştur
    prompt_template = """
    Sana verilen bağlamı (context) kullanarak soruyu yanıtla. 
    Eğer cevabı bağlamda bulamazsan, "Üzgünüm, bu bilgi notlarımda yer almıyor." de.

    Bağlam: {context}
    Soru: {question}
    Cevap:
    """
    prompt = PromptTemplate.from_template(prompt_template)
    
    # 6. RAG Zincirini kur
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# --- STREAMLIT ARAYÜZÜ ---

st.set_page_config(page_title="Kişisel ABAP Asistanı", page_icon="🤖")
st.title("Kişisel ABAP Asistanı 🤖")
st.write("Kendi SAP ABAP notlarınız hakkında sorular sorun.")

# RAG zincirini yükle (cache'ten gelir)
try:
    with st.spinner("Asistan hazırlanıyor... Notlar yükleniyor..."):
        rag_chain = load_rag_chain()
    st.success("Asistan hazır! Sorularınızı sorabilirsiniz.")
except Exception as e:
    st.error(f"Asistan yüklenirken bir hata oluştu: {e}")
    st.error("Lütfen PDF dosyasının adının doğru olduğundan ve GOOGLE_API_KEY'inizin Streamlit Secrets'a eklendiğinden emin olun.")
    st.stop()


# Sohbet hafızasını başlat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hafızadaki mesajları göster
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Kullanıcıdan yeni soru al
if prompt := st.chat_input("ABAP ile ilgili sorunuzu buraya yazın..."):
    # Kullanıcı mesajını hafızaya ve ekrana ekle
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Asistanın cevabını al
    with st.chat_message("assistant"):
        with st.spinner("Düşünüyorum..."):
            response = rag_chain.invoke(prompt)
            st.markdown(response)
    
    # Asistan mesajını hafızaya ekle

    st.session_state.messages.append({"role": "assistant", "content": response})
