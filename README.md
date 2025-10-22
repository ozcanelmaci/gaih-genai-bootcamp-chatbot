# <img width="65" height="36.4" alt="image" src="https://github.com/user-attachments/assets/2138fb1a-a5c5-45d2-b174-068558b5d206" /> ABAP Notlarına Özel Chatbot
## 📋 Proje Hakkında

AKBANK GENERATIVE AI GİRİŞ BOOTCAMP için geliştirilmiş, Türkçe RAG(Retrieval-Augmented Generation) özellikli, ABAP ve genel SAP bilgilerine göre alınmış notlardan oluşan PDF'ten beslenen bir chatbot'tur. İlgili PDF'teki verileri chunk'layarak(anlamlı parçalara bölerek), Chroma Vektör Veritabanına kaydediyor, en alakalı parçaları alıyor ve Google Gemini'nin pro modelini kullanarak faydalı yanıtlar üretiyor.

## ✨ Özellikler
* Basitçe RAG pipeline: **chunk → embed → store → retrieve → generate**
* Response Generation için: **Google Gemini 2.5 Pro**
* Embedding için: **`text-embedding-001`**
* Vektör Database için: **Chroma vector store**
* Arayüz için: **Streamlit**

## 🛠️ Kullanılan Teknolojiler
* Web Deploy: **Streamlit**
* RAG: **LangChain**, **Chroma**
* LLM: **Google Gemini (gemini-2.5-pro)**
* Embeddings: **Google models/text-embedding-001**
* Data: **ABAP-1_merged.pdf dosyası**

## 📚 Gereklilikler
* **requirements.txt** isimli dosyadaki gerekliliklerin yüklenmesi
* Geçerli bir **API KEY**

## 🚀 Kurulum
### 1. Gerekli Paketleri Yükleyin
<pre>
<small># Virtual environment oluşturun (opsiyonel ama önerilir)</small> 
python3 -m venv genai-env
  
<small># source genai-env/bin/activate  # macOS/Linux</small> 
genai-env\Scripts\activate  # Windows

<small># Paketleri yükleyin</small> 
pip install -r requirements.txt
</pre>

### 2. API Anahtarlarını Ayarlayın
Local'de çalışacağınız için Proje kök dizininde .env dosyası oluşturun ve içine api key ve diğer key'lerinizi yazın:
<pre>
GOOGLE_API_KEY=your_google_api_key_here
</pre>

* **Google Api Key**'inizi [Google AI Studio](https://makersuite.google.com/app/apikey) üzerinden alabilirsiniz.

### 3. Uygulamayı Çalıştırın
<pre>
  streamlit run app.py
</pre>

## ⚙️ Default Konfigürasyonlar
* **Generative Model:** gemini-2.5-pro (temperature=0.75)
* **Embedding Model:** models/text-embedding-001
* **Chunking:** 1000 characters, 300 overlap
* **Retrieval k:** 3

## 📁 Proje Yapısı
<pre>
.
├── app.py              # Ana uygulama dosyası
├── requirements.txt    # Python bağımlılıkları
├── ABAP-1_merged.pdf   # Veri
├── .env                # API anahtarları (git'e eklenmez)
└── README.md           # Bu dosya(projenin açıklanması)
</pre>
**NOT:** **ABAP-1_merged.pdf** dosyası kişisel bilgiler içerdiği için github repository'den **KALDIRILMIŞTIR!**

## 💻 Chatbot Örnek Soru ve Yanıtları
Soru: **invoice ile ilgili neler biliyorsun?**

Cevap:
<pre>
<img align="left" width="600" height="308" alt="image" src="https://github.com/user-attachments/assets/0566909c-c495-4438-ad9d-39453bcd95a3" />
<br></br>
<img align="left" width="600" height="308" alt="image" src="https://github.com/user-attachments/assets/5720c1bf-a867-46d5-b490-abdcbbfd6b58" />
</pre>

Soru: **BAPI_SALESORDER_CHANGE ile ilgili bilgi verebilir misin?**

Cevap:
<pre>
<img align="left" width="600" height="308" alt="image" src="https://github.com/user-attachments/assets/709fb1b4-329a-4778-a580-5054121b8835" />
</pre>

Soru: **User exit ile ilgili bilgi verir misin?**

Cevap:
<pre>
<img align="left" width="600" height="308" alt="image" src="https://github.com/user-attachments/assets/7f4fb4da-ed8f-4dfe-922a-05f3cb9d93eb" />
</pre>
