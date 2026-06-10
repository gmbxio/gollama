from fastapi import FastAPI # type: ignore
from pydantic import BaseModel # type: ignore
import ollama # type: ignore
import time

app = FastAPI(title="Gollama API", description="Local SLM Wrapper with Structured JSON")

# --- Step 2 Data Model (Generic Input) ---
class QueryRequest(BaseModel):
    prompt: str

# --- Step 3 Structured Data Model ---
# This defines exactly what fields we expect back from our SLM
class DeveloperProfile(BaseModel):
    name: str
    primary_language: str
    years_of_experience: int
    skills: list[str]

# Step 2 endpoint (Kept for reference)
@app.post("/generate")
async def generate_response(request: QueryRequest):
    start_time = time.time()
    response = ollama.chat(model='gemma3:1b', messages=[
        {'role': 'user', 'content': request.prompt}
    ])
    elapsed_time = time.time() - start_time
    return {
        "model": "gemma3:1b",
        "response": response['message']['content'],
        "total_latency_seconds": round(elapsed_time, 2)
    }

# --- Step 3 New Endpoint ---
@app.post("/extract-profile")
async def extract_profile(request: QueryRequest):
    start_time = time.time()
    
    # Call Ollama and strictly enforce our schema shape and zero temperature
    response = ollama.chat(
        model='gemma3:1b',
        messages=[
            {
                'role': 'user', 
                'content': f"Extract developer profile details from this text.'primary_language' means their main PROGRAMMING language, not spoken language. Text: {request.prompt}"
            }
        ],
        format=DeveloperProfile.model_json_schema(), # Constrains output to our schema shape
        options={'temperature': 0} # Eradicates randomness for deterministic results
    )
    
    elapsed_time = time.time() - start_time
    raw_content = response['message']['content']
    
    # Clean validation check: parses the string JSON into a secure Python object
    validated_data = DeveloperProfile.model_validate_json(raw_content)
    
    return {
        "model": "gemma3:1b",
        "structured_data": validated_data,
        "total_latency_seconds": round(elapsed_time, 2)
    }