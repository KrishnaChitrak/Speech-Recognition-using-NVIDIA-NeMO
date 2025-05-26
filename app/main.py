import os
import torch
import numpy as np
import librosa
from typing import Optional, Dict, Any
from pathlib import Path
from fastapi import FastAPI, UploadFile, HTTPException, status
from pydantic import BaseModel
from nemo.collections.asr.models import EncDecCTCModelBPE
from nemo.collections.asr.parts.utils.transcribe_utils import setup_transcribe
from nemo.core.classes import IterableDataset
from nemo.core.neural_types import NeuralType, AudioSignal, LengthsType
import onnxruntime as ort
import logging
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_NAME = "nvidia/stt_hi_conformer_ctc_medium"
MODEL_DIR = os.getenv("MODEL_DIR", "./models")
ONNX_MODEL_PATH = os.path.join(MODEL_DIR, "stt_hi_conformer_ctc_medium.onnx")
MAX_AUDIO_LENGTH = 10  # seconds
MIN_AUDIO_LENGTH = 5   # seconds
SAMPLE_RATE = 16000    # Hz

class ModelManager:
    def __init__(self):
        self.model = None
        self.onnx_session = None
        self.tokenizer = None

    async def initialize(self):
        try:
            # Create model directory
            os.makedirs(MODEL_DIR, exist_ok=True)

            # Load NeMo model
            logger.info(f"Loading model {MODEL_NAME}...")
            self.model = EncDecCTCModelBPE.from_pretrained(MODEL_NAME)
            self.model.eval()
            self.tokenizer = self.model.tokenizer

            # Export to ONNX if needed
            if not os.path.exists(ONNX_MODEL_PATH):
                logger.info("Exporting model to ONNX format...")
                self.model.export(ONNX_MODEL_PATH)

            # Initialize ONNX Runtime session
            logger.info("Initializing ONNX Runtime session...")
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.onnx_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=providers)

            logger.info("Model initialization complete")
        except Exception as e:
            logger.error(f"Error during model initialization: {str(e)}")
            raise

    def validate_audio(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Validate and preprocess audio data."""
        try:
            # Check audio length
            duration = len(audio_data) / sample_rate
            if duration < MIN_AUDIO_LENGTH or duration > MAX_AUDIO_LENGTH:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Audio duration must be between {MIN_AUDIO_LENGTH} and {MAX_AUDIO_LENGTH} seconds"
                )

            # Resample if necessary
            if sample_rate != SAMPLE_RATE:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=SAMPLE_RATE)

            # Convert stereo to mono if necessary
            if len(audio_data.shape) > 1:
                audio_data = librosa.to_mono(audio_data)

            return audio_data

        except Exception as e:
            logger.error(f"Error in audio validation: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Audio validation failed: {str(e)}"
            )

    def process_audio(self, audio_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Process audio data for model input."""
        try:
            # Normalize audio
            audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Prepare input tensor
            audio_signal = torch.tensor(audio_data).unsqueeze(0)
            audio_length = torch.tensor([len(audio_data)]).long()

            return {
                "audio_signal": audio_signal.numpy(),
                "length": audio_length.numpy()
            }

        except Exception as e:
            logger.error(f"Error in audio processing: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Audio processing failed: {str(e)}"
            )

    def decode_output(self, output: np.ndarray) -> str:
        """Decode model output using CTC decoder."""
        try:
            # Get predictions using argmax
            predictions = output.argmax(axis=-1)
            
            # Decode predictions to text using NeMo's tokenizer
            decoded = []
            for prediction in predictions:
                text = self.tokenizer.ids_to_text(prediction.tolist())
                decoded.append(text)
            
            return " ".join(decoded)

        except Exception as e:
            logger.error(f"Error in output decoding: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Output decoding failed: {str(e)}"
            )

class TranscriptionResponse(BaseModel):
    text: str
    duration: float
    sample_rate: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    onnx_session_initialized: bool

model_manager = ModelManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize model on startup
    await model_manager.initialize()
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/transcribe/", response_model=TranscriptionResponse)
async def transcribe_audio(audio_file: UploadFile):
    try:
        # Read and validate audio file
        audio_data, sample_rate = librosa.load(audio_file.file, sr=None)
        
        # Validate and preprocess audio
        audio_data = model_manager.validate_audio(audio_data, sample_rate)
        
        # Process audio for model input
        inputs = model_manager.process_audio(audio_data)
        
        # Run inference
        outputs = model_manager.onnx_session.run(None, inputs)
        
        # Decode output
        text = model_manager.decode_output(outputs[0])
        
        return TranscriptionResponse(
            text=text,
            duration=len(audio_data) / SAMPLE_RATE,
            sample_rate=SAMPLE_RATE
        )

    except Exception as e:
        logger.error(f"Error in transcription endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {str(e)}"
        )

@app.get("/health/", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=model_manager.model is not None,
        onnx_session_initialized=model_manager.onnx_session is not None
    )

if __name__ == "__main__":
    app.run(main)