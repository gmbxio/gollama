# 🦙 Gollama: Local SLM Production & Inference Benchmarking Dashboard

Gollama is a production-grade, privacy-first middleware wrapper and dashboard built around local Small Language Models (SLMs). Running completely offline on Apple Silicon architectures, the system provides structured JSON data extraction with zero stochastic variance alongside real-time inference telemetry tracking.

## 🚀 Key Features
* **Deterministic Structured Extraction:** Implements strict grammar constraints via Pydantic and Ollama schemas to force local models (`gemma3:1b`) to output verified JSON data blocks with zero conversational fluff.
* **Live Telemetry Engine:** Tracks critical production SRE metrics including Time-To-First-Token (TTFT), total processing latency, exact token generation counts, and overall generation throughput (tokens per second).
* **Interactive Control Center:** A beautiful, responsive frontend built with Streamlit allowing seamless model benchmarking and structured profile testing side-by-side.

## 📊 Baseline Performance (Apple M2 Silicon, 8GB RAM)
During localized testing with a `gemma3:1b` model parameter class, Gollama recorded the following execution metrics:
* **Time-to-First-Token (TTFT):** ~681.28 ms (Sub-second interaction response)
* **Inference Throughput:** ~35.66 tokens/sec (Blazing fast local generation)
* **Data Reliability:** 100% compliance with target JSON validation structures.

---

## 🛠️ Tech Stack
* **LLM Engine:** Ollama (`gemma3:1b` / `llama3.2:1b`)
* **Backend:** FastAPI, Uvicorn, Pydantic (v2)
* **Frontend:** Streamlit, Requests
* **Environment:** Python 3.13+

---

## 💻 Installation & Local Setup

### 1. Clone & Set Up Environment
```bash
git clone [https://github.com/YOUR_USERNAME/gollama.git](https://github.com/YOUR_USERNAME/gollama.git)
cd gollama

python3 -m venv slm
source slm/bin/activate
pip install -r requirements.txt