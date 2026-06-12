from fastapi import FastAPI
from pydantic import BaseModel
import ollama
import time

app = FastAPI(title="Gollama API", description="Unified Local SLM Engine")

# --- Data Models ---
class QueryRequest(BaseModel):
    prompt: str

class DeveloperProfile(BaseModel):
    name: str
    primary_language: str
    years_of_experience: int
    skills: list[str]


# --- Endpoint 1: Structured JSON Extraction (From Step 3) ---
@app.post("/extract-profile")
async def extract_profile(request: QueryRequest):
    start_time = time.time()
    
    response = ollama.chat(
        model='gemma3:1b',
        messages=[
            {
                'role': 'user', 
                'content': f"Extract developer profile details from this text.'primary_language' means their main PROGRAMMING language, not spoken language. Text: {request.prompt}"
            }
        ],
        format=DeveloperProfile.model_json_schema(), # Constrains output to Pydantic shape
        options={'temperature': 0} # Eliminates stochastic randomness
    )
    
    elapsed_time = time.time() - start_time
    raw_content = response['message']['content']
    
    # Securely validate the string JSON into a structured Python dictionary
    validated_data = DeveloperProfile.model_validate_json(raw_content)
    
    return {
        "model": "gemma3:1b",
        "structured_data": validated_data,
        "total_latency_seconds": round(elapsed_time, 2)
    }


# --- Endpoint 2: Telemetry Benchmarking (From Step 4) ---
@app.post("/benchmark")
async def benchmark_endpoint(request: QueryRequest):
    start_time = time.time()
    ttft = None
    full_text = ""
    tokens_generated = 0
    
    stream = ollama.chat(
        model='gemma3:1b',
        messages=[{'role': 'user', 'content': request.prompt}],
        stream=True
    )
    
    for chunk in stream:
        if ttft is None:
            ttft = time.time() - start_time

        # Safe parsing for varying library versions
        if isinstance(chunk, dict):
            content = chunk.get('message', {}).get('content', '')
            is_done = chunk.get('done', False)
            eval_count = chunk.get('eval_count', 0)
        else:
            content = chunk.message.content or ""
            is_done = getattr(chunk, 'done', False)
            eval_count = getattr(chunk, 'eval_count', 0)

        full_text += content

        if is_done:
            tokens_generated = eval_count
            
    total_latency = time.time() - start_time
    
    if tokens_generated == 0:
        tokens_generated = int(len(full_text.split()) * 1.3)
        
    generation_time = total_latency - ttft if ttft else total_latency
    tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
    
    return {
        "model": "gemma3:1b",
        "response": full_text.strip(),
        "telemetry": {
            "time_to_first_token_ms": round(ttft * 1000, 2) if ttft else 0,
            "total_latency_seconds": round(total_latency, 2),
            "tokens_count": tokens_generated,
            "throughput_tokens_per_sec": round(tokens_per_second, 2)
        }
    }