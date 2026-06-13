# 🦙 Gollama: Local SLM Production & Inference Benchmarking Dashboard

Gollama is a production-grade, privacy-first middleware wrapper, self-healing validation engine, and telemetry dashboard built around localized Small Language Models (SLMs). Running completely offline on Apple Silicon architectures, the system bridges the gap between stochastically unpredictable local AI behaviors and strict enterprise data schemas.

The framework provides multi-model comparative benchmarking across parameter scales (`1B` to `1.5B`), real-time streaming SRE telemetry tracking, and an automated exception-gating corrective loop to repair corrupted data payloads dynamically.

---

## 🚀 Key Architectural Pillars

### 1. Performance Benchmarking & Telemetry (Phase 1)

* **Real-Time Streaming Engine:** Uses FastAPI backend utilities to capture token-by-token streaming events directly out of local model contexts.
* **SRE Metrics Harvesting:** Measures and renders system-level telemetry instantly:
* **Time-to-First-Token (TTFT):** Monitors sub-second interactive model reactions.
* **Total Response Latency:** Tracks round-trip compute duration.
* **Generation Throughput:** Calculates tokens generated per second ($\text{tok/s}$).



### 2. Structured Extraction & Self-Healing Architecture (Phase 2)

* **Grammar-Constrained Decoding:** Integrates Pydantic schema contracts into Ollama's structural decoding layer to force models to strip conversational pleasantries and output strict, compliant JSON structures.
* **Defensive Retry Exception Gates:** Intercepts runtime `ValidationError` or structural formatting mutations caused by high-temperature sampling. If the data type fails constraint matching, a defensive loop captures the crash logs, injects a targeted repair prompt context back into the model pipeline, drops the temperature to `0.0` for maximum predictability, and repairs the data packet before it can cause a system crash.

### 3. Multi-Model Edge Matrix (Phase 3)

* **Cross-Model Infrastructure Selector:** Built a decoupled, reactive global sidebar in Streamlit allowing rapid switching between four state-of-the-art edge model nodes: `gemma3:1b`, `llama3.2:1b`, `deepseek-r1:1.5b`, and `qwen2.5:1.5b`.

---

## 📊 Hardware Benchmarking Overview (Apple M2, 8GB Unified Memory)

Comprehensive testing across varying temperatures ($0.0 \le T \le 1.5$) exposed unique token-entropy boundary limits on local consumer hardware. A complete empirical record of these findings can be reviewed in our detailed [Technical Comparison Study](https://www.google.com/search?q=./COMPARISON_STUDY.md).

### Summary of Performance Dynamics:

* **The Sweet Spot (`gemma3:1b`):** Delivered the highest data extraction integrity, recording an immediate reaction threshold of **~1998ms TTFT** and a stable throughput tracking profile at **~25.24 tokens/sec**.
* **The Truncation Anomaly (`llama3.2:1b`):** Exhibited critical token drop behaviors when running between a threshold temperature of `0.6`, dropping secondary data nodes entirely while maintaining structural validities.
* **The Reasoning Penalty (`deepseek-r1:1.5b`):** Successfully passed advanced mathematical logic checks (e.g., system equations), but encountered a 300% operational latency spike (**14s - 33s total processing times**) due to its internal chain-of-thought calculation overhead.
* **The Qwen Contrast (`qwen2.5:1.5b`):** Displayed a highly unusual distribution where data extraction failed due to greedy path truncation at `temp=0.0`, but achieved maximum completion accuracy under chaotic sampling structures (`temp=1.5`).

---

## 🛠️ Tech Stack

* **LLM Core Inferences:** Ollama Engine
* **Model Configurations Deployed:** `gemma3:1b`, `llama3.2:1b`, `deepseek-r1:1.5b`, `qwen2.5:1.5b`
* **Backend Middleware:** FastAPI, Uvicorn Async Server, Pydantic (v2 Model Validation)
* **Frontend UI Matrix:** Streamlit Dashboard Engine, Python Requests HTTP Pipelines
* **Language Runtime:** Python 3.13+

---

## 💻 Installation & Local Execution

### 1. Clone & Initialize Environments

```bash
git clone https://github.com/YOUR_USERNAME/gollama.git
cd gollama

# Instantiate virtual environment
python3 -m venv slm
source slm/bin/activate

# Install dependency dependencies 
pip install -r requirements.txt

```

### 2. Download Core Local Models via Ollama

Ensure you have the Ollama daemon running in the background of your system, then pull down the designated model matrices:

```bash
ollama pull gemma3:1b
ollama pull llama3.2:1b
ollama pull deepseek-r1:1.5b
ollama pull qwen2.5:1.5b

```

### 3. Boot Up the API Server (Terminal 1)

```bash
uvicorn api:app --reload --port 8000

```

> View interactive OpenAPI documentation by steering your browser to: `http://localhost:8000/docs`

### 4. Deploy the UI Control Center (Terminal 2)

```bash
streamlit run app.py

```

> The web browser will spin open automatically to: `http://localhost:8501`

---

## 📂 Repository Structure

```text
├── gollama/
│   ├── api.py                  # FastAPI server containing telemetry loops and self-healing logic
│   ├── app.py                  # Streamlit frontend engine tracking metrics and model changes
│   ├── COMPARISON_STUDY.md     # In-depth technical report profiling empirical model insights
│   ├── requirements.txt        # System library requirements
│   └── README.md               # Product Overview documentation

```