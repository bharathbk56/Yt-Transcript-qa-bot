# 🎬 YouTube Transcript Q&A Bot

An AI-powered app that transcribes any YouTube video and answers questions based on the actual video content — no hallucinations, only transcript-grounded answers.

---

## 🔗 Live Demo
👉 [Click here to try it live](https://yt-transcript-qa-bot.onrender.com/)

---

## 💡 How It Works

1. Paste any YouTube video URL
2. Click **Process Video** — transcript is fetched and indexed
3. Ask any question about the video
4. Get AI-powered answers grounded in the actual transcript

---

## 🧠 Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| AI Model | Google Gemini 2.5 Flash |
| Transcript | Supadata API |
| Retrieval | TF-IDF + Cosine Similarity |
| Chunking | LangChain Text Splitter |
| Language | Python 3.11 |
| Hosting | Render |

---

## ⚙️ Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/bharathbk56/yt-transcript-qa-bot.git
cd yt-transcript-qa-bot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Create `.env` file
GEMINI_API_KEY=your-gemini-key-here
SUPADATA_API_KEY=your-supadata-key-here

### 4. Run the app
```bash
streamlit run App.py
```

### 5. Open in browser
http://localhost:8501

---

## 📁 Project Structure
yt-transcript-qa-bot/
├── App.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── runtime.txt             # Python version (3.11.6)
├── .streamlit/
│   └── config.toml         # Streamlit server config
└── .gitignore              # Ignores .env and cache files

---

## 🔑 Environment Variables

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Google Gemini API key |
| `SUPADATA_API_KEY` | Supadata transcript API key |

---

## ✨ Features

- ✅ Works with any public YouTube video
- ✅ Context-grounded answers — no hallucinations
- ✅ Fast TF-IDF retrieval
- ✅ Clean minimal UI
- ✅ Handles long videos with chunking

---

## 🚀 Deployment

This app is deployed on **Render** free tier.

To deploy your own:
1. Fork this repo
2. Create account on [render.com](https://render.com)
3. Connect your GitHub repo
4. Add environment variables
5. Deploy ✅

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
