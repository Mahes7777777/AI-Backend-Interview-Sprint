from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("FastAPIService")

app = FastAPI(title="AI Backend Inference API", version="1.0.0")

# 1. Pydantic Model for Strict Request Validation
class PromptRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=500, description="The user query text")
    max_tokens: int = Field(default=100, ge=10, le=1000)
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)

# 2. Pydantic Model for Response
class InferenceResponse(BaseModel):
    status: str
    generated_text: str
    tokens_used: int

# Dependency simulation for auth or database session
def verify_api_key(api_key: str = "secret-token-123") -> bool:
    logger.info("Authenticating API request via dependency injection...")
    return True

@app.post("/v1/generate", response_model=InferenceResponse)
async def generate_text(payload: PromptRequest, authenticated: bool = Depends(verify_api_key)):
    """
    Production-grade endpoint simulating LLM text generation with input validation.
    """
    logger.info(f"Received valid prompt: '{payload.prompt[:30]}...'")
    
    # Simulating LLM text generation processing
    simulated_output = f"Processed response for query: [{payload.prompt}] using temp {payload.temperature}."
    
    return InferenceResponse(
        status="SUCCESS",
        generated_text=simulated_output,
        tokens_used=len(payload.prompt.split()) + 15
    )

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "ai-inference-backend"}

if __name__ == "__main__":
    import uvicorn
    print("\nStarting FastAPI server simulation. Run using: uvicorn fastapi_llm_endpoint:app --reload")
    # uvicorn.run(app, host="127.0.0.1", port=8000)