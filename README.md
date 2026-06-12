# Youtube_ChatBot_RAG
Chat with any YouTube video using AI — paste a link, and ask questions powered by RAG (Retrieval-Augmented Generation).

# 🎥 YouTube RAG Chatbot

A Streamlit app that lets you paste a YouTube video link, index its transcript, and chat with an AI about the video's content using Retrieval-Augmented Generation (RAG).

## Features

- 🔗 Paste any YouTube video URL
- 📝 Automatically fetches and indexes the video transcript
- 💬 Chat interface to ask questions about the video
- 🖼️ Displays video thumbnail, title, and channel
- 💡 Auto-generated suggested starter questions
- 📄 Source citations showing which transcript chunks were used
- ⏱️ Response time shown for each answer
- 🎨 Custom dark theme UI

## Tech Stack

- **Frontend**: Streamlit
- **LLM**: Qwen 2.5 7B Instruct (via OpenRouter)
- **Embeddings**: BAAI/bge-m3 (via Hugging Face Inference Endpoint)
- **Vector Store**: FAISS
- **Transcript Fetching**: youtube-transcript-api
- **Framework**: LangChain

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root with the following keys:

```env
HF_TOKEN=your_huggingface_api_token
OPENROUTER_API_KEY=your_openrouter_api_key
```

- Get a Hugging Face token at: https://huggingface.co/settings/tokens
- Get an OpenRouter API key at: https://openrouter.ai/keys

### 4. Run the app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## How to Use

1. Paste a YouTube video URL into the sidebar input.
2. Click **Start Chat** — the app fetches the transcript and builds a searchable index.
3. Once indexed, try one of the suggested questions or type your own in the chat box.
4. View the sources used for each answer by expanding **📄 Sources used**.
5. To switch videos, open **🔁 Load a different video** in the sidebar and repeat.
6. Use **Clear chat** to reset the conversation for the current video.

## Project Structure

```
.
├── app.py              # Main Streamlit application
├── .env                # API keys (not committed to version control)
├── requirements.txt    # Python dependencies
├── vector_db_<id>/     # Auto-generated FAISS index per video
└── README.md
```

## Notes

- Transcripts are fetched in English (`languages=["en"]`). Videos without English transcripts will fail to index.
- Each indexed video creates its own FAISS index folder (`vector_db_<video_id>`), so previously indexed videos don't need to be re-processed.
- Make sure your `.env` file is added to `.gitignore` to avoid leaking API keys.

## License

This project is for educational/personal use. Add your preferred license here.
