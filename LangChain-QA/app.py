from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from transformers import pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load PDF
loader = PyPDFLoader("serena.pdf")
pages = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
docs = splitter.split_documents(pages)

# Embed using local model
embeddings = HuggingFaceEmbeddings(
    model_name="../all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda"}  
)
db = FAISS.from_documents(docs,embeddings)
retriever = db.as_retriever(search_type="similarity", k=3)

# Load local T5 model
pipe = pipeline("text2text-generation",model="../t5",max_length=256,device=0)

def retrieval_qa(question,retriever,pipe):
    relevant_docs=retriever.get_relevant_documents(question)
    context=' '.join([doc.page_content for doc in relevant_docs])

    prompt=f'Answer the question based on the following context:\n{context}\nQuestion:{question}\nAnswer:'
    #prompt="Summarize: Serena is a Pokémon character who travels with Ash and has a romantic subplot."
    answer=pipe(prompt)[0]['generated_text']

    return answer

question='Why is Serena so perfect?'
print('Answer:\n',retrieval_qa(question,retriever,pipe))