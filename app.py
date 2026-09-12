import streamlit as st
import requests
import pandas as pd

# ----------------- CONFIGURATION -----------------
st.set_page_config(
    page_title="Gollama Dashboard", 
    page_icon="🦙", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State for Benchmarking History
if "benchmark_history" not in st.session_state:
    st.session_state.benchmark_history = []

# ----------------- HELPER FUNCTIONS -----------------
def fetch_available_models():
    try:
        response = requests.get("http://localhost:8000/models", timeout=2)
        if response.status_code == 200:
            return response.json().get("models", [])
    except requests.exceptions.ConnectionError:
        pass
    return ["gemma3:1b", "llama3.2:1b", "deepseek-r1:1.5b", "qwen2.5:1.5b"]

API_URL = "http://localhost:8000"

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.title("🦙 Gollama")
    st.caption("Local SLM Production Engine")
    st.divider()
    
    st.header("⚙️ Engine Configuration")
    available_models = fetch_available_models()
    
    selected_model = st.selectbox(
        "Target Model Architecture:",
        options=available_models,
        help="Dynamically fetched from local Ollama instance."
    )
    
    # Live Status Indicator
    st.success("🟢 Backend API: Online")
    
    st.divider()
    st.markdown("### 📊 Session Stats")
    # Create a placeholder we can update later
    runs_metric = st.empty()
    runs_metric.metric("Total Runs", len(st.session_state.benchmark_history))
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.benchmark_history = []
        st.rerun()

# ----------------- MAIN UI -----------------
st.title("Gollama: Edge Optimization Dashboard")
st.markdown("Monitor sub-second interactive model reactions, throughput, and JSON extraction integrity.")

tab1, tab2 = st.tabs(["🚀 Live Benchmarking", "🧩 Structured Extraction"])

# --- TAB 1: BENCHMARKING ---
with tab1:
    st.subheader("Interactive Speed & Telemetry Test")
    
    # Preset Prompt Buttons for quick testing
    st.markdown("**Quick Test Prompts:**")
    col_a, col_b, col_c = st.columns(3)
    if col_a.button("Slogan Generation", use_container_width=True):
        st.session_state.bench_prompt = "Write a catchy 3 sentence slogan for an offline local AI tool named Gollama."
    if col_b.button("Math Logic", use_container_width=True):
        st.session_state.bench_prompt = "Solve this step-by-step: If I have 3 apples and buy 4 more, but give 2 to a friend, how many do I have?"
    if col_c.button("Code Generation", use_container_width=True):
        st.session_state.bench_prompt = "Write a simple Python function to calculate the Fibonacci sequence."
    
    user_prompt = st.text_area(
        "Enter your prompt:",
        value=st.session_state.get("bench_prompt", "Write a catchy 3 sentence slogan for an offline local AI tool named Gollama."),
        height=100
    )
    
    if st.button("▶ Run Benchmark", type="primary", use_container_width=True):
        with st.spinner(f"Running inference on {selected_model}..."):
            try:
                payload = {"prompt": user_prompt, "model": selected_model}
                response = requests.post(f"{API_URL}/benchmark", json=payload)
                data = response.json()
                metrics = data["telemetry"]
                
                # Save to history
                st.session_state.benchmark_history.append({
                    "Model": data["model"],
                    "TTFT (ms)": metrics['time_to_first_token_ms'],
                    "Throughput (tok/s)": metrics['throughput_tokens_per_sec'],
                    "Latency (s)": metrics['total_latency_seconds'],
                    "Tokens": metrics['tokens_count']
                })

                # UPDATE THE SIDEBAR METRIC DYNAMICALLY
                runs_metric.metric("Total Runs", len(st.session_state.benchmark_history))
                
                # Display Metrics using native styling
                st.markdown("### ⏱️ Telemetry Results")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("TTFT (Reaction)", f"{metrics['time_to_first_token_ms']} ms")
                m2.metric("Total Latency", f"{metrics['total_latency_seconds']} s")
                m3.metric("Tokens Generated", f"{metrics['tokens_count']}")
                m4.metric("Throughput Speed", f"{metrics['throughput_tokens_per_sec']} tok/s")
                
                # Display output in a clean expander
                st.info("Output generated successfully!")
                with st.expander("👁️ View Model Output", expanded=True):
                    st.write(data["response"])
                    
                st.toast(f"Benchmark completed in {metrics['total_latency_seconds']}s!", icon="🚀")
                
            except requests.exceptions.ConnectionError:
                st.error("🔴 Backend offline! Run 'python3 -m uvicorn api:app --reload' in your backend terminal.")

    # Show History Table if runs exist
    if st.session_state.benchmark_history:
        st.divider()
        st.markdown("### 📈 Session History")
        history_df = pd.DataFrame(st.session_state.benchmark_history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)

# --- TAB 2: STRUCTURED EXTRACTION ---
with tab2:
    st.subheader("Deterministic JSON Extraction & Self-Healing")
    
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        st.markdown("**Configuration**")
        selected_temp = st.slider("Temperature (Chaos)", 0.0, 1.5, 0.0, 0.1)
        st.caption("Higher temps risk schema breaks, triggering the self-healing loop.")
        
    with col_right:
        messy_text = st.text_area(
            "Raw Unstructured Text:",
            value="i'm Golam and i code in python and react with 3 years of experience. I also know Docker and SQL.",
            height=120
        )
    
    if st.button("🔨 Extract & Validate JSON", type="primary", use_container_width=True):
        with st.spinner(f"Enforcing Pydantic Schema on {selected_model}..."):
            try:
                payload = {"prompt": messy_text, "temperature": selected_temp, "model": selected_model}
                response = requests.post(f"{API_URL}/extract-profile", json=payload)
                data = response.json()
                
                # Layout results side by side
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    st.success("✅ Validated JSON Object")
                    st.json(data["structured_data"])
                    
                with res_col2:
                    st.markdown("### Execution Telemetry")
                    st.metric("Total Latency", f"{data['total_latency_seconds']} s")
                    
                    if "defensive_telemetry" in data:
                        telemetry = data["defensive_telemetry"]
                        if telemetry["self_healing_retry_triggered"]:
                            st.error("⚠️ Primary extraction failed! Self-healing triggered.")
                            with st.expander("View Crash Logs"):
                                st.code(telemetry['original_error_exception'])
                            st.info("System successfully dropped temperature to 0.0 and repaired the data.")
                        else:
                            st.success("🎯 Clean Execution (First Pass Success)")
            except requests.exceptions.ConnectionError:
                st.error("🔴 Backend offline! Run 'python3 -m uvicorn api:app --reload' in your backend terminal.")