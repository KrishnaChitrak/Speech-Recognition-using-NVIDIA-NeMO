# Hindi Speech Recognition API

A FastAPI-based service for Hindi Automatic Speech Recognition (ASR) using NVIDIA's NeMo toolkit, optimized with ONNX runtime.

## Features

- Hindi ASR using NeMo's Conformer-CTC Medium model
- ONNX optimization for efficient inference
- FastAPI REST endpoints
- Input validation and error handling
- Containerized deployment
- Secure runtime practices

## Prerequisites

- Docker
- NVIDIA GPU with CUDA support (recommended)

## Quick Start

1. Build the container:
```bash
docker build -t hindi-asr-api .
```

2. Run the service:
```bash
docker run -d -p 8000:8000 --name hindi-asr hindi-asr-api
```

## API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Transcribe Audio
```bash
curl -X POST http://localhost:8000/transcribe \
  -H "Content-Type: multipart/form-data" \
  -F "audio_file=@/path/to/audio.wav"
```

### Input Requirements
- Format: WAV files only
- Duration: 5-10 seconds
- Sampling Rate: 16kHz
- Channels: Mono

## Design Considerations

### Security
- Multi-stage Docker build for minimal attack surface
- Non-root user for container runtime
- No unnecessary system packages

### Performance
- ONNX optimization for faster inference
- Efficient audio file handling
- Input validation to prevent resource waste

### Containerization
- Multi-stage build for smaller image size
- Only essential runtime dependencies
- Python slim base image
- Layer optimization for better caching

### Error Handling
- Comprehensive input validation
- Clear error messages
- Proper cleanup of temporary files
- Graceful handling of invalid requests

## Development

To run locally without Docker:

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start the server:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## License

MIT License