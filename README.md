# 🎬 YouTube Transcript AI

An AI-powered chatbot that allows users to ask questions about a YouTube video's transcript using **Retrieval-Augmented Generation (RAG)**.

Upload a transcript in `.txt` format and ask questions to summarize the video, understand its main topic, and discover key takeaways.

## ✨ Features

- **Transcript Upload:** Upload a YouTube transcript as a text file.
- **Text Chunking:** Split the transcript into smaller, overlapping chunks.
- **Embeddings:** Generate semantic embeddings using OpenAI.
- **Vector Search:** Store and retrieve relevant transcript chunks using FAISS.
- **AI-Powered Answers:** Generate context-based answers using LangChain and an OpenAI chat model.
- **Interactive Chat:** Ask custom questions or use suggested prompts.
- **Custom UI:** A Streamlit interface with a mint-green theme.

## 🛠️ Tech Stack

- Python
- Streamlit
- LangChain
- OpenAI API
- OpenAI Embeddings (`text-embedding-3-small`)
- FAISS
- Retrieval-Augmented Generation (RAG)

## 📂 Project Structure

```text
youtube-transcript-ai/
├── app.py
├── rag_pipeline.py
├── style.css
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/youtube-transcript-ai.git
cd youtube-transcript-ai
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure your OpenAI API key

Create a `.env` file in the project root directory:

```dotenv
OPENAI_API_KEY=your_openai_api_key
```

Replace the placeholder with your own valid API key.

**Security:** Never commit your `.env` file or expose your API key publicly.

### 5. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal.

## 🚀 How It Works

1. Upload a transcript in `.txt` format.
2. Clean the transcript text and split it into overlapping chunks.
3. Generate embeddings for the chunks using OpenAI.
4. Store the embeddings in a FAISS vector store.
5. Retrieve the most relevant chunks for the user's question.
6. Pass the retrieved context and question to the language model.
7. Display the generated answer in the Streamlit chat interface.

## 🧠 RAG Architecture

```text
Transcript (.txt)
       ↓
Text Cleaning
       ↓
Text Chunking
       ↓
OpenAI Embeddings
       ↓
FAISS Vector Store
       ↓
Similarity Search
       ↓
Retrieved Context + Question
       ↓
LangChain + OpenAI Chat Model
       ↓
AI-Generated Answer
```

## 📁 File Responsibilities

- **`app.py`** — Streamlit interface, transcript upload, suggested questions, and chat history.
- **`rag_pipeline.py`** — Transcript processing, chunking, embeddings, FAISS retrieval, prompting, and answer generation.
- **`style.css`** — Custom styling for the application.
- **`requirements.txt`** — Python package dependencies.
- **`.env`** — Local API key configuration; excluded from Git.

## 🔐 Notes

- A valid OpenAI API key is required.
- OpenAI API usage may incur separate charges.
- The current application accepts uploaded `.txt` transcripts; it does not automatically fetch transcripts directly from YouTube.
- Answers depend on the information retrieved from the uploaded transcript.

## 👩‍💻 Future Improvements

- Fetch transcripts directly from YouTube video URLs.
- Persist FAISS indexes to avoid rebuilding them.
- Add support for multiple transcript formats.
- Improve retrieval quality and answer evaluation.
- Deploy the application online.

---

Built with Python, Streamlit, LangChain, OpenAI, and FAISS.
