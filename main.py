from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# 1. Dosya Yükleme ve Parçalama (Document Loading & Chunking)
loader = TextLoader("notlar.txt", encoding="utf-8")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = text_splitter.split_documents(documents)

# 2. Vektör Veritabanı Oluşturma (Embeddings & ChromaDB)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

# 3. Alakalı Bilgiyi Çekme (Retrieval)
soru = "Yapay zeka ne üzerinde eğitilir?"
bulunan_sonuclar = db.similarity_search(soru, k=1)
baglam = bulunan_sonuclar[0].page_content

# 4. Foundry Local Yapay Zeka Bağlantısı (LLM)
llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="local",
    model="local-model"
)

# 5. Prompt Tasarımı ve Yanıt Üretme (Generation)
prompt_metni = """Sana verilen bilgiyi kullanarak soruya kısa ve net bir cevap ver.

Bağlam: {baglam}
Soru: {soru}

Cevap:"""

prompt = PromptTemplate.from_template(prompt_metni)
chain = prompt | llm

print("\n=== YAPAY ZEKA CEVABI ===")
try:
    cevap = chain.invoke({"baglam": baglam, "soru": soru})
    print(cevap.content)
except Exception:
    print(f"Bulunan Bağlam: {baglam}")
    print("\n(Not: Foundry Local servis adresi hazır. Yerel servis açık olduğunda yanıt doğrudan modelden üretilecektir.)")
