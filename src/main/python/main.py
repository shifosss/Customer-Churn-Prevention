from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Optional, List
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
    title="Llama 3.2 API",
    description="""
    API for Meta's Llama 3.2 3B-instruct model.

    ## Available Endpoints
    - `/`: Welcome page
    - `/docs`: Interactive API documentation (Swagger UI)
    - `/redoc`: Alternative API documentation
    - `/generate`: Generate text from prompt
    - `/health`: Check API status

    Made with FastAPI and Hugging Face Transformers
    """,
    version="1.0.0"
)


# Define request/response models
class Message(BaseModel):
    role: str = Field(..., description="Role of the message sender (e.g., 'user')")
    content: str = Field(..., description="Content of the message")


class GenerateRequest(BaseModel):
    messages: List[Message]
    max_length: Optional[int] = Field(default=512, description="Maximum length of generated text")
    temperature: Optional[float] = Field(default=0.7, description="Temperature for text generation")
    top_p: Optional[float] = Field(default=0.9, description="Top p for nucleus sampling")

    class Config:
        schema_extra = {
            "example": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Write a short story about a robot"
                    }
                ],
                "max_length": 512,
                "temperature": 0.7,
                "top_p": 0.9
            }
        }


class GenerateResponse(BaseModel):
    generated_text: str
    generation_time: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    model_name: str


class ModelService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "meta-llama/Llama-3.2-3B-Instruct"
        self.load_model()

    def load_model(self):
        try:
            logger.info(f"Loading {self.model_name} model and tokenizer...")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
            )
            self.model.to(self.device)
            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        try:
            start_time = time.time()

            # Format the conversation
            conversation = []
            for msg in request.messages:
                conversation.append({
                    "role": msg.role,
                    "content": msg.content
                })

            # Convert conversation to model input format
            prompt = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True
            )

            # Tokenize
            inputs = self.tokenizer(
                prompt,
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
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True
                )

            # Decode and clean up response
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remove the input prompt from the generated text
            generated_text = generated_text.replace(prompt, "").strip()

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


# Root route with welcome page
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>Llama 3.2 API</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                    line-height: 1.6;
                }
                .container {
                    background-color: #f5f5f5;
                    padding: 20px;
                    border-radius: 8px;
                }
                code {
                    background-color: #e0e0e0;
                    padding: 2px 5px;
                    border-radius: 3px;
                }
                pre {
                    background-color: #f8f8f8;
                    padding: 15px;
                    border-radius: 5px;
                    overflow-x: auto;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Welcome to Llama 3.2 API</h1>
                <p>This API provides access to Meta's Llama 3.2 3B-instruct model.</p>

                <h2>Available Endpoints:</h2>
                <ul>
                    <li><a href="/docs">/docs</a> - Interactive API documentation (Swagger UI)</li>
                    <li><a href="/redoc">/redoc</a> - Alternative API documentation</li>
                    <li><code>POST /generate</code> - Generate text from prompt</li>
                    <li><a href="/health">/health</a> - Check API status</li>
                </ul>

                <h2>Quick Start:</h2>
                <p>To generate text, send a POST request to <code>/generate</code> with a JSON body:</p>
                <pre>
{
    "messages": [
        {
            "role": "user",
            "content": "Write a short story about a robot"
        }
    ],
    "max_length": 512,
    "temperature": 0.7
}
                </pre>
            </div>
        </body>
    </html>
    """


@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """
    Generate text based on the input messages
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
        device=model_service.device,
        model_name=model_service.model_name
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8008, reload=False)