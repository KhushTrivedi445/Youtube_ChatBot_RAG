import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
# Use the newer, non-deprecated class
from langchain_huggingface import HuggingFaceEndpointEmbeddings

load_dotenv()

# Initialize Embeddings using the non-deprecated class
embeddings = HuggingFaceEndpointEmbeddings(
    model="BAAI/bge-m3",
    huggingfacehub_api_token=os.getenv("HF_TOKEN")
)

# Load the FAISS index
vectorstore = FAISS.load_local(
    "./vector_db", 
    embeddings, 
    allow_dangerous_deserialization=True
)

llm = ChatOpenAI(
    model="qwen/qwen-2.5-7b-instruct", 
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

def ask_rag(question):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    # Modern syntax: use .invoke() instead of get_relevant_documents()
    docs = retriever.invoke(question)
    context = "\n".join([d.page_content for d in docs])
    
    prompt = f"Answer based only on this context: {context}\n\nQuestion: {question}"
    return llm.invoke(prompt).content

print("Bot is ready! Type 'quit' to exit.")
while True:
    q = input("You: ")
    if q.lower() == 'quit': break
    print(f"Bot: {ask_rag(q)}")