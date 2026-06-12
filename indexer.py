import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

# 1. Setup Embeddings (Latest standard)
embeddings = HuggingFaceEndpointEmbeddings(
    model="BAAI/bge-m3",
    huggingfacehub_api_token=os.getenv("HF_TOKEN")
)

def download_transcript(video_id, output_file="transcript.txt"):
    try:
        # 1. Initialize API
        yt_api = YouTubeTranscriptApi()
        
        # 2. Get the transcript for the video
        # The .list() method followed by .find_transcript() is the robust way
        transcript_obj = yt_api.list(video_id).find_transcript(['en'])
        
        # 3. Fetch the data
        data = transcript_obj.fetch()
        
        # 4. Extract text using attribute access (chunk.text)
        # This fixes the "'FetchedTranscriptSnippet' object has no attribute 'get'" error
        transcript = " ".join(chunk.text for chunk in data)
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(transcript)
            
        print(f"✅ Transcript for {video_id} saved.")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading transcript: {e}")
        return False

# 2. Pipeline
VIDEO_ID = "7ARBJQn6QkM"

if download_transcript(VIDEO_ID):
    with open("transcript.txt", "r", encoding="utf-8") as f:
        text = f.read()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = [Document(page_content=chunk) for chunk in splitter.split_text(text)]
    
    print(f"Indexing {len(docs)} chunks...")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("./vector_db")
    print("✅ Indexing complete! FAISS database saved to ./vector_db/")