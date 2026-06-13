from fastapi import FastAPI
from pydantic import BaseModel, ValidationError
import ollama
import time
import json

app = FastAPI(title="Gollama Core API Engine", description="Dynamic Multi Model SLM Optimization Engine")

# --- Dynamic Request Schema Contracts ---
class QueryRequest(BaseModel):
    prompt: str
    model: str = "gemma3:1b" # Accepts incoming model parameters dynamically

class ExtractionRequest(BaseModel):
    prompt: str
    model: str = "gemma3:1b"
    temperature: float = 0.0

class DeveloperProfile(BaseModel):
    name: str
    primary_language: str
    years_of_experience: int
    skills: list[str]


# --- Dynamic Structured Extraction Endpoint ---
@app.post("/extract-profile")
async def extract_profile(request: ExtractionRequest):
    start_time = time.time()
    retry_triggered = False
    error_log = None
    schema_layout = DeveloperProfile.model_json_schema()

    try:
        response = ollama.chat(
            model=request.model, # Dynamically maps chosen model
            messages=[
                {
                    'role': 'user',
                    'content': (
                        f"Extract developer profile details from this text. 'primary_language' means their main PROGRAMMING language, not spoken language."
                        f" Text: {request.prompt}"
                    )
                }
            ],
            format=schema_layout,
            options={'temperature': request.temperature}
        )
        raw_content = response['message']['content']
        validated_data = DeveloperProfile.model_validate_json(raw_content)
        
    except (ValidationError, ValueError, json.JSONDecodeError) as e:
        retry_triggered = True
        error_log = str(e)
        
        repair_prompt = (
            f"Your previous output failed strict validation rules.\n"
            f"Validation Failure Details: {error_log}\n"
            f"Original Input Source: {request.prompt}\n"
            f"Task: Correct the data types. Output clean JSON matching this schema: {schema_layout}"
        )
        
        # Self-heals using the exact same requested model under zero temperature
        repair_response = ollama.chat(
            model=request.model,
            messages=[{'role': 'user', 'content': repair_prompt}],
            format=schema_layout,
            options={'temperature': 0.0}
        )
        raw_content = repair_response['message']['content']
        validated_data = DeveloperProfile.model_validate_json(raw_content)

    elapsed_time = time.time() - start_time
    
    return {
        "model": request.model,
        "structured_data": validated_data,
        "total_latency_seconds": round(elapsed_time, 2),
        "defensive_telemetry": {
            "requested_temperature": request.temperature,
            "self_healing_retry_triggered": retry_triggered,
            "original_error_exception": error_log
        }
    }


# --- Dynamic Benchmarking Endpoint ---
@app.post("/benchmark")
async def benchmark_endpoint(request: QueryRequest):
    start_time = time.time()
    ttft = None
    full_text = ""
    tokens_generated = 0
    
    stream = ollama.chat(
        model=request.model, # Dynamically maps chosen model
        messages=[{'role': 'user', 'content': request.prompt}],
        stream=True
    )
    
    for chunk in stream:
        if ttft is None:
            ttft = time.time() - start_time

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
        "model": request.model,
        "response": full_text.strip(),
        "telemetry": {
            "time_to_first_token_ms": round(ttft * 1000, 2) if ttft else 0,
            "total_latency_seconds": round(total_latency, 2),
            "tokens_count": tokens_generated,
            "throughput_tokens_per_sec": round(tokens_per_second, 2)
        }
    }