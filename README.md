# Step 1:

### Downloading ollama, the model that i want to use in my case gemma3:1b because that was the most capable model and 1b parameter because my ram was 8gb to make it run faster i have chosen that.

I have already downloaded ollama in my device. 

created a dir called gollama, then set up a virtual env with python(alias python) -m venv slm, then activate it using  source slm/bin/activate

then pull the model : ollama pull gemma3:1b

then a sample run to check if successfully downloaded or not , ollama run gemma3:1b and stop it by /bye 

```bash
➜  gollama python -m venv slm
➜  gollama source slm/bin/activate
(slm) ➜  gollama ollama pull gemma3:1b
```

then a sample run :

```bash
(slm) ➜  gollama ollama run gemma3:1b
> how are you doing ?
I’m doing well, thanks for asking! As a large language model, I don’t experience feelings in the same way humans do, but I’m functioning
perfectly and ready to help you with whatever you need.
How are *you* doing today? 😊
Do you want to talk about something in particular, or is there anything I can do for you?
> /bye
```

# Step 2:

### First installing  the Core Dependencies

```bash
pip install fastapi uvicorn pydantic ollama
```

- **FastAPI & Uvicorn:** The web framework and the server to run it.
- **Pydantic:** The data validator we will use heavily in later steps.
- **Ollama:** The official Python client to communicate with your local model.

### Create api.py in gollama dir outside slm

```python
from fastapi import FastAPI
from pydantic import BaseModel  # type: ignore[import-not-found]
import ollama  # type: ignore[import-not-found]
import time

# Initialize the FastAPI app
app = FastAPI(title="Gollama API", description="Local SLM Wrapper")

# Define the data structure for incoming requests
class QueryRequest(BaseModel):
    prompt: str

@app.post("/generate")
async def generate_response(request: QueryRequest):
    start_time = time.time()
    
    # Send the prompt to your local model
    response = ollama.chat(model='gemma3:1b', messages=[
        {
            'role': 'user',
            'content': request.prompt
        }
    ])
    
    # Calculate how long the response took
    elapsed_time = time.time() - start_time
    
    return {
        "model": "gemma3:1b",
        "response": response['message']['content'],
        "total_latency_seconds": round(elapsed_time, 2)
    }
```

### then run the following command

```bash
uvicorn api:app --reload
```

then visit **http://127.0.0.1:8000/docs**  to test the endpoints visually and http://127.0.0.1:8000/openapi.json for raw api spec 

# Step 3:

We are going to add a brand-new endpoint (`/extract-profile`) that takes a messy, unformatted block of text about a developer and forces **Gollama** to extract specific data into a flawless, predictable JSON block. To do this, we are using two advanced production techniques:

1. **Grammar Constraints (`format` schema):** We pass our Pydantic schema directly to Ollama. This actively blocks the model from generating any characters that violate the JSON structure at a token level.
2. **Zero Temperature:** We lock the model's `temperature` down to `0`. This eliminates randomness and forces the model to choose the most mathematically confident tokens, making it highly reliable.

We then update the api.py with our new code

```python
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
```

and test the Fast API endpoint with the given prompt:

Hey there! My name is Alex and I have been writing code for over 6 years now. I mostly live in Python nowadays, but over the years I have picked up a lot of deep familiarity with Docker, AWS Cloud architectures, and PostgreSQL databases.

and we get the following JSON response: 

```json
{
  "model": "gemma3:1b",
  "structured_data": {
    "name": "Alex",
    "primary_language": "Python",
    "years_of_experience": 6,
    "skills": [
      "Docker",
      "AWS Cloud Architectures",
      "PostgreSQL Databases"
    ]
  }
}
```