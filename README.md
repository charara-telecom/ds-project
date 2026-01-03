# 🎬 Real-Time Movie Review RAG

A real-time data pipeline and RAG (Retrieval-Augmented Generation) application that ingests movie reviews and details from TMDB, streams them via Apache Kafka, stores embeddings in ChromaDB, and provides an interactive QA interface using Streamlit and Google Gemini.

## 🚀 Features

- **Real-Time Ingestion**: Fetches live movie data (Now Playing, Popular, Top Rated, Upcoming) and reviews from [The Movie Database (TMDB)](https://www.themoviedb.org/).
- **Streaming Pipeline**: Uses **Apache Kafka** to decouple data fetching from processing.
- **Vector Database**: Stores movie reviews and descriptions in **ChromaDB** with semantic embeddings.
- **RAG Architecture**: Retrieves relevant context (reviews + descriptions) to answer user queries accurately.
- **AI Powered**: Uses **Google Gemini** (via `google-genai` SDK) for both embeddings (`text-embedding-004`) and answer generation (`gemini-2.5-flash`).
- **Interactive UI**: Built with **Streamlit** for easy interaction.

## 🛠️ Tech Stack

- **Python 3.10+**
- **Apache Kafka** (Message Broker)
- **ChromaDB** (Vector Store)
- **Google Gemini API** (LLM & Embeddings)
- **Streamlit** (Frontend)
- **LangChain** (RAG Framework)

## 📂 Project Structure

```
.
├── consumer.py          # Kafka consumer: reads messages, embeds text, saves to ChromaDB
├── producer_tmdb.py     # Kafka producer: fetches data from TMDB API, sends to Kafka
├── rag_service.py       # RAG logic: retrieval, filtering, and LLM generation
├── ui.py                # Streamlit application entry point
├── gemini_helper.py     # Helper class for Google GenAI SDK interactions
├── chroma_inspect.py    # Utility to inspect/debug ChromaDB contents
├── requirements.txt     # Python dependencies
└── .env                 # Environment variables (API keys)
```

## ⚙️ Setup & Installation

### 1. Prerequisites

- Python 3.10 or higher
- Apache Kafka running locally (default: `localhost:9092`)
- TMDB API Key
- Google Gemini API Key

### 2. Clone the Repository

```bash
git clone <repository-url>
cd kafka-project
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory:

```ini
# .env
GEMINI_API_KEYS="YOUR_GEMINI_API_KEY_1|||||YOUR_GEMINI_API_KEY_2"
# Note: You can provide multiple keys separated by ||||| for rotation
```


## 🏃‍♂️ Running the Application

You will need to run multiple terminal sessions.

### Step 1: Start Apache Kafka

Ensure your Zookeeper and Kafka server are running.
```bash
# Example (if using standard Kafka scripts)
bin/zookeeper-server-start.sh config/zookeeper.properties
bin/kafka-server-start.sh config/server.properties
```

### Step 2: Start the Consumer

This service listens to Kafka topics, generates embeddings, and saves data to ChromaDB.

```bash
python consumer.py
```

### Step 3: Start the Producer

This script fetches data from TMDB and pushes it to Kafka.

```bash
python producer_tmdb.py
```

### Step 4: Launch the UI

Open the Streamlit app to ask questions about movies.

```bash
streamlit run ui.py
```

## 🔍 Usage

1.  **Ingest Data**: Let the producer and consumer run for a while to populate the database.
2.  **Ask Questions**: Go to the Streamlit UI (usually `http://localhost:8501`).
3.  **Query Examples**:
    - "What do people think about Inception?"
    - "Are there any good horror movies from last year?"
    - "Summarize the reviews for the latest Marvel movie."

## 🐛 Debugging

- Use `chroma_inspect.py` to view stored documents:
  ```bash
  python chroma_inspect.py
  ```
- Check `debug.py` for a quick count of records.


