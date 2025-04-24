# Visual-and-Text-Question-answering
A ChatBot in which we can upload pdf's and ask our queries from the document.

# PDF Chatbot with Summarization & Gemini Web Search

A powerful Streamlit app that lets you chat with your PDFs using LLMs, get instant summaries, and even fetch up-to-date answers from the web using Google Gemini!

---

## ✨ Features
- **PDF Upload & Extraction**: Upload any PDF and extract text and images.
- **LLM-Powered Q&A**: Ask questions about your PDF (text or images) and get answers using local LLMs (Ollama/llama2).
- **Summarization**: Instantly summarize your entire PDF with a single click.
- **Contextual Chat History**: See your previous questions & answers in the sidebar.
- **Always-Visible Chat Bar**: Ask questions any time using the chat input fixed at the bottom.
- **Image Q&A (Llava)**: Ask questions about images in your PDF using the Llava model.
- **Gemini Web Search (Optional)**: Enable Google Gemini to get answers that combine your PDF content with the latest facts from the web.

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd <your-repo-directory>
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Your Environment
- Create a `.env` file in the project root:
  ```
  GEMINI_API_KEY=your-gemini-api-key-here
  ```
- You can get a Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
- (Optional) For Ollama/Llava, ensure you have the appropriate models and local server running.

### 4. Run the App
```bash
streamlit run app.py
```

---

## 🖥️ Usage
- **Upload a PDF** using the uploader at the top.
- **Summarize** the PDF with the "Summarize PDF" button.
- **Ask questions** about the PDF using the chat bar at the bottom.
- **Enable Gemini** (checkbox) to get answers that combine PDF context and the latest web facts.
- **View chat history** in the sidebar.
- **Ask about images**: Select images and ask questions; get answers from the Llava model.

---

## 🔒 Security Notes
- Your Gemini API key is read from `.env` and never exposed in the UI.
- No user authentication is enabled by default.

---

## 🛠️ Tech Stack
- **Streamlit** (UI)
- **LangChain** (text processing, QA chains)
- **Ollama** (local LLMs)
- **Google Gemini API** (web-augmented answers)
- **Llava** (image Q&A)
- **PyMuPDF, pypdf** (PDF parsing)
- **FAISS** (vector search)
- **python-dotenv** (env management)

---

## 📄 Example .env
```
GEMINI_API_KEY=your-gemini-api-key-here
```

---

## 🙏 Credits
- Built by Atishay Jain 
- Inspired by open-source LLM and PDF QA projects.

---

