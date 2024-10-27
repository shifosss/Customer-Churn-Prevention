# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Optional
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Llama 3.2 3B API",
    description="API for Meta's Llama 3.2 3B-instruct model",
    version="1.0.0"
)


# Define request/response models
class GenerateRequest(BaseModel):
    prompt: str
    max_length: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    num_return_sequences: Optional[int] = 1


class GenerateResponse(BaseModel):
    generated_text: str
    generation_time: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


class ModelService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()

    def load_model(self):
        """Load the Llama 2 model and tokenizer"""
        try:
            logger.info("Loading Llama 3.2 3B model and tokenizer...")
            model_name = "meta-llama/Llama-3.2-3B-Instruct"

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
            )
            self.model.to(self.device)
            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Generate text based on the input prompt"""
        try:
            # Start timing
            start_time = time.time()

            # Prepare the prompt with the instruction format
            formatted_prompt = f"""[INST] {request.prompt} [/INST]"""

            # Tokenize input
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=request.max_length,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    num_return_sequences=request.num_return_sequences,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True
                )

            # Decode output
            generated_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            # Remove the prompt from the generated text
            generated_text = generated_text.replace(formatted_prompt, "").strip()

            # Calculate generation time
            generation_time = time.time() - start_time

            return GenerateResponse(
                generated_text=generated_text,
                generation_time=generation_time
            )

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))


# Initialize model service
model_service = ModelService()


@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """
    Generate text based on the input prompt
    """
    try:
        return model_service.generate(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check the health status of the API
    """
    return HealthResponse(
        status="healthy",
        model_loaded=model_service.model is not None,
        device=model_service.device
    )


# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("API starting up...")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("API shutting down...")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)