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
# History-Aware Retriever için
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

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
    retriever = vector_store.as_retriever(search_kwargs={"k": 6})

    # LLM Modeli
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3) 

    # History-Aware Retriever -> Sohbet geçmişini de dahil ediyoruz
    # --- 1. ADIM: SORUYU YENİDEN YAZAN PROMPT (Hafıza için) ---
    contextualize_q_system_prompt = """Sohbet geçmişine ve kullanıcının son sorusuna bakarak,
    kullanıcının ne sormak istediğini anla ve vektör veritabanında arama yapmak için 
    tek başına anlamlı (bağımsız) bir soru cümlesi oluştur.
    Soruyu cevaplama, sadece soruyu yeniden formüle et. Eğer soru zaten tek başına anlamlıysa aynen bırak."""
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # Geçmişin farkında olan arayıcıyı oluştur
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # --- 2. ADIM: CEVABI ÜRETEN PROMPT ---
    qa_system_prompt = """Sen deneyimli bir SAP ABAP asistanısın. 
    Aşağıdaki bağlamı (context) kullanarak kullanıcının sorusunu yanıtla.
    Eğer bağlamda kod parçacıkları veya teknik detaylar varsa, formatına uygun şekilde ilet.
    Bilgiyi bulduğun sayfa numaralarını referans olarak ekle.
    Cevabı bağlamda bulamazsan, kendi bilgilerini uydurma ve "Üzgünüm, bu bilgi notlarımda yer almıyor." de.

    Bağlam: 
    {context}"""
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # --- 3. ADIM: ZİNCİRLERİ BİRLEŞTİR ---
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain

    """ # Geliştirilmiş Prompt Şablonu
    prompt_template = """
    """Sen deneyimli bir SAP ABAP danışmanısın. Sana verilen bağlamı (context) kullanarak kullanıcının sorusunu yanıtla.
    Eğer bağlamda kod parçacıkları veya teknik detaylar varsa, bunları formatına uygun şekilde (Markdown kod blokları içinde) ilet.
    Bilgiyi bulduğun sayfa numaralarını referans olarak ekle.
    Cevabı bağlamda bulamazsan, kendi bilgilerini uydurma ve "Üzgünüm, bu bilgi notlarımda yer almıyor." de.

    Bağlam: {context}
    
    Soru: {question}
    Cevap: """
    """
    prompt = PromptTemplate.from_template(prompt_template)
    
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    ) 
    
    return rag_chain """

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
            # 🔴 YENİ EKLENEN KISIM: Streamlit mesajlarını LangChain formatına çevir
            chat_history = []
            for msg in st.session_state.messages[:-1]: # Son mesajı (şu anki soruyu) geçmişe koyma
                if msg["role"] == "user":
                    chat_history.append(HumanMessage(content=msg["content"]))
                else:
                    chat_history.append(AIMessage(content=msg["content"]))

            # RAG zincirine soruyu ve sohbet geçmişini (chat_history) birlikte yolla
            response = rag_chain.invoke({
                "input": prompt, 
                "chat_history": chat_history
            })
            
            # Artık cevap 'answer' anahtarı içinde dönüyor
            ai_response = response["answer"]
            st.markdown(ai_response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})










