from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
    description="API for Meta's Llama 3.2 3B-instruct model",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: str = Field(default="user", description="Role of the message sender (e.g., 'user')")
    content: Optional[str] = Field(None, description="Content of the message")
    msg: Optional[str] = Field(None, description="Alternative field for content")
    type: Optional[str] = Field(None, description="Message type")

    def get_content(self) -> str:
        """Get the message content from either content or msg field"""
        return self.content or self.msg or ""


class GenerateRequest(BaseModel):
    messages: Optional[List[Message]] = None
    message: Optional[Message] = None  # Single message support
    msg: Optional[str] = None  # Direct message support
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

    def get_messages(self) -> List[dict]:
        """Get normalized messages list"""
        if self.messages:
            return [{"role": msg.role, "content": msg.get_content()} for msg in self.messages]
        elif self.message:
            return [{"role": self.message.role, "content": self.message.get_content()}]
        elif self.msg:
            return [{"role": "user", "content": self.msg}]
        return []


class GenerateResponse(BaseModel):
    generated_text: str
    generation_time: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    model_name: str

# Add these new classes near your other model definitions
class SystemPromptRequest(BaseModel):
    prompt: str = Field(..., description="System prompt to set")

# Add this new endpoint to your FastAPI app
@app.post("/set_system_prompt")
async def set_system_prompt(request: SystemPromptRequest):
    """
    Update the system prompt used by the model
    """
    try:
        model_service.set_system_prompt(request.prompt)
        return {"status": "success", "message": "System prompt updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ModelService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "AlexZhang05/autotrain-llama3-2-3B-Instruct-FineTune"
        self.system_prompt = \
            """You are a helpful AI assistant. Provide clear and accurate responses. You must generate a <label>
             (Spam or Ham) for the given input_. Your output should be in format: It is a <label>. Please avoid any other unnecessary or unrelated responses.
        """  # Default system prompt
        self.load_model()

    def set_system_prompt(self, prompt: str):
        """Update the system prompt"""
        self.system_prompt = prompt
        logger.info("System prompt updated")

    def load_model(self):
        try:
            logger.info(f"Loading {self.model_name} model and tokenizer...")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
            )
            if self.model.config.pad_token_id is None:
                self.model.config.pad_token_id = self.tokenizer.pad_token_id

            self.model.to(self.device)
            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        try:
            start_time = time.time()

            # Get normalized messages and add system prompt
            conversation = [
                {"role": "system", "content": self.system_prompt}
            ]
            conversation.extend(request.get_messages())

            if len(conversation) < 2:  # Only system prompt exists
                raise HTTPException(status_code=400, detail="No valid message content provided")

            # Convert conversation to model input_ format
            prompt = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True
            )

            # Tokenize with proper padding
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=request.max_length,
                return_attention_mask=True
            ).to(self.device)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=request.max_length,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                    do_sample=True
                )

            # Decode and clean up response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the assistant's response
            try:
                # Find the last assistant response
                if "assistant" in full_response:
                    generated_text = full_response.split("assistant")[-1].strip()
                else:
                    # Fallback: just remove the prompt
                    generated_text = full_response.replace(prompt, "").strip()
            except Exception as e:
                logger.error(f"Error cleaning response: {str(e)}")
                generated_text = full_response

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


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>Fine-tuned Llama 3.2 API</title>
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
                <p>This API provides access to Fine-tuned Llama 3.2 3B-instruct model.</p>

                <h2>Available Endpoints:</h2>
                <ul>
                    <li><a href="/docs">/docs</a> - Interactive API documentation</li>
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
    Generate text based on the input_ messages
    """
    try:
        return model_service.generate(request).model_dump()
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