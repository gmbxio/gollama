import streamlit as st
import requests

# Set up page configurations
st.set_page_config(page_title="Gollama Dashboard", page_icon="🦙", layout="wide")

st.title("🦙 Gollama: Local SLM Production Dashboard")
st.markdown("Interact with your offline local AI engine and monitor live inference telemetry.")

# Setup two tabs for the two features we built
tab1, tab2 = st.tabs(["🚀 Performance Benchmarking", "📊 Structured Profile Extraction"])

# Base URL pointing to your FastAPI backend
API_URL = "http://localhost:8000"

# --- TAB 1: BENCHMARKING ---
with tab1:
    st.subheader("Live Inference Speed Test")
    user_prompt = st.text_area(
        "Enter a prompt to test your PC's inference speeds:",
        value="Write a catchy 3-sentence slogan for an offline local AI tool named Gollama.",
        key="benchmark_input"
    )
    
    if st.button("Run Benchmark Test", type="primary"):
        with st.spinner("Gollama is thinking..."):
            try:
                # Call our FastAPI benchmark endpoint
                response = requests.post(f"{API_URL}/benchmark", json={"prompt": user_prompt})
                data = response.json()
                
                # Layout metric cards side by side
                col1, col2, col3, col4 = st.columns(4)
                metrics = data["telemetry"]
                
                col1.metric("TTFT (Reaction)", f"{metrics['time_to_first_token_ms']} ms")
                col2.metric("Total Latency", f"{metrics['total_latency_seconds']} s")
                col3.metric("Tokens Generated", f"{metrics['tokens_count']}")
                col4.metric("Throughput Speed", f"{metrics['throughput_tokens_per_sec']} tok/s")
                
                st.success("Response Received:")
                st.write(data["response"])
                
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to FastAPI! Make sure your Uvicorn server is running on port 8000.")

# --- TAB 2: STRUCTURED EXTRACTION ---
with tab2:
    st.subheader("Deterministic JSON Extraction Engine")
    messy_text = st.text_area(
        "Paste an unformatted bio or raw text paragraphs here:",
        value="Hey there! My name is Alex and I have been writing code for over 6 years now. I mostly live in Python nowadays, but over the years I have picked up a lot of deep familiarity with Docker, AWS Cloud architectures, and PostgreSQL databases.",
        key="extraction_input"
    )
    
    if st.button("Extract Schema Object"):
        with st.spinner("Enforcing structural constraints..."):
            try:
                # Call our FastAPI structural engine endpoint
                response = requests.post(f"{API_URL}/extract-profile", json={"prompt": messy_text})
                data = response.json()
                
                col_left, col_right = st.columns(2)
                
                with col_left:
                    st.success("Validated JSON Data Structure:")
                    st.json(data["structured_data"])
                    
                with col_right:
                    st.info("Extraction Statistics:")
                    st.metric("Extraction Latency", f"{data['total_latency_seconds']} seconds")
                    st.caption("Output actively constrained using Pydantic + Ollama JSON Schema limits.")
                    
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to FastAPI! Make sure your Uvicorn server is running on port 8000.")