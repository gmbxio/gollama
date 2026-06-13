import streamlit as st
import requests

st.set_page_config(page_title="Gollama Dashboard", page_icon="🦙", layout="wide")

st.title("🦙 Gollama: Local SLM Production Dashboard")
st.markdown("Interact with your offline local AI engine and monitor live inference telemetry.")

# SIDEBAR CONTROL ELEMENT: Phase 3 Dropdown Architecture Selection
with st.sidebar:
    st.header("⚙️ Global Engine Model Configuration")
    selected_model = st.selectbox(
        "Choose Local Target Model Architecture:",
        options=["gemma3:1b", "llama3.2:1b", "deepseek-r1:1.5b", "qwen2.5:1.5b"],
        help="Select which localized model weights to deploy and query."
    )
    st.info(f"Active Model Context Pipeline: **{selected_model}**")

tab1, tab2 = st.tabs(["🚀 Performance Benchmarking", "📊 Structured Profile Extraction"])
API_URL = "http://localhost:8000"

# --- TAB 1: BENCHMARKING ---
with tab1:
    st.subheader(f"Live Inference Speed Test — Run Mode: {selected_model}")
    user_prompt = st.text_area(
        "Enter a prompt to test your PC's inference speeds:",
        value="Write a catchy 3 sentence slogan for an offline local AI tool named Gollama.",
        key="benchmark_input"
    )
    
    if st.button("Run Benchmark Test", type="primary"):
        with st.spinner(f"{selected_model} is thinking..."):
            try:
                payload = {"prompt": user_prompt, "model": selected_model}
                response = requests.post(f"{API_URL}/benchmark", json=payload)
                data = response.json()
                
                col1, col2, col3, col4 = st.columns(4)
                metrics = data["telemetry"]
                
                col1.metric("TTFT (Reaction)", f"{metrics['time_to_first_token_ms']} ms")
                col2.metric("Total Latency", f"{metrics['total_latency_seconds']} s")
                col3.metric("Tokens Generated", f"{metrics['tokens_count']}")
                col4.metric("Throughput Speed", f"{metrics['throughput_tokens_per_sec']} tok/s")
                
                st.success(f"Response Received from model pipeline [{data['model']}]:")
                st.write(data["response"])
            except requests.exceptions.ConnectionError:
                st.error("Backend offline! Run 'uvicorn api:app --reload' in your backend terminal.")

# --- TAB 2: STRUCTURED EXTRACTION ---
with tab2:
    st.subheader(f"Deterministic JSON Extraction Engine — Run Mode: {selected_model}")
    
    selected_temp = st.slider(
        "Adjust Model Temperature (Variance Chaos):",
        min_value=0.0, max_value=1.5, value=0.0, step=0.1
    )
    
    messy_text = st.text_area(
        "Paste an unformatted bio or raw text paragraphs here:",
        value="i'm Golam and i code in python and react with 3 years of experience",
        key="extraction_input"
    )
    
    if st.button("Extract Schema Object"):
        with st.spinner("Enforcing structural constraints..."):
            try:
                payload = {"prompt": messy_text, "temperature": selected_temp, "model": selected_model}
                response = requests.post(f"{API_URL}/extract-profile", json=payload)
                data = response.json()
                
                col_left, col_right = st.columns(2)
                
                with col_left:
                    st.success("Validated JSON Data Structure:")
                    st.json(data["structured_data"])
                    
                with col_right:
                    st.info("Extraction Statistics:")
                    st.metric("Extraction Latency", f"{data['total_latency_seconds']} seconds")
                    st.markdown(f"**Model Context Responding:** `{data['model']}`")
                    
                    if "defensive_telemetry" in data:
                        telemetry = data["defensive_telemetry"]
                        if telemetry["self_healing_retry_triggered"]:
                            st.warning("⚠️ Self-Healing Triggered!")
                            st.caption(f"Reason: {telemetry['original_error_exception']}")
                        else:
                            st.success("✅ Clean Execution (First Pass Success)")
            except requests.exceptions.ConnectionError:
                st.error("Backend offline! Run 'uvicorn api:app --reload' in your backend terminal.")