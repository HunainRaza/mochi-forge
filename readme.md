# Mochi Video Generator - Production Ready

A reliable, production-ready implementation of Genmo's Mochi video generation model using HuggingFace Diffusers. Optimized for RTX 3070 laptops with comprehensive memory management and fallback strategies.

## Features

- **Web Interface**: User-friendly Flask web application
- **Asynchronous Processing**: Celery-based task queue for background video generation
- **Memory Optimized**: Aggressive memory management for 8GB VRAM GPUs
- **CPU Fallback**: Automatic fallback to CPU mode when GPU memory is insufficient
- **Progress Tracking**: Real-time generation progress and status updates
- **Multiple Quality Settings**: From quick 8-frame tests to high-quality 31-frame videos

## Benchmark Results (RTX 3070 Laptop)

| Setting | Frames | Steps | CPU Mode | GPU Mode | Recommended |
|---------|--------|-------|----------|----------|-------------|
| Quick Test | 8f | 8s | ~33 min | - | Testing |
| Standard | 16f | 16s | ~131 min | ~214 min | **CPU Mode** |
| High Quality | 24f | 24s | ~295 min | - | Overnight |
| Maximum | 31f | 32s | ~521 min | - | Overnight |

**Key Finding**: CPU mode is 40% faster than GPU mode on RTX 3070 laptops due to memory management overhead.

## System Requirements

### Minimum Requirements
- **OS**: Windows 10/11
- **RAM**: 16GB system RAM
- **Virtual Memory**: 100GB+ (critical for model loading)
- **Storage**: 50GB free space
- **GPU**: NVIDIA RTX 3070 or better (8GB+ VRAM)

### Recommended Setup
- **RAM**: 32GB system RAM
- **Virtual Memory**: 128GB
- **CPU**: 8+ cores for optimal CPU mode performance

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/HunainRaza/mochi-forge.git
cd mochi-video-generator
```

### 2. Create Virtual Environment
```bash
python -m venv env
env\Scripts\activate  # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
pip install diffusers[torch] accelerate transformers
```

### 4. Download Required External Components

**Mochi Repository** (for reference, not included):
```bash
git clone https://github.com/genmoai/mochi.git
```

**Model Weights** (download separately):
- Models are automatically downloaded from HuggingFace on first run
- Requires ~40GB download and storage space
- Cached in `~/.cache/huggingface/`

### 5. Setup Redis
Download and install Redis for Windows or use Docker:
```bash
docker run -d -p 6379:6379 redis:alpine
```

### 6. Configure Virtual Memory (Critical)
1. Windows + R → `sysdm.cpl`
2. Advanced → Performance → Settings
3. Advanced → Virtual Memory → Change
4. Custom size: Initial and Maximum both set to `131072 MB`
5. Restart Windows

## Usage

### Production Start
```bash
start_production.bat
```

This will:
1. Activate virtual environment
2. Start Redis (if running)
3. Launch Celery worker
4. Start Flask web application
5. Open web interface at `http://localhost:5000`

### Manual Start
```bash
# Terminal 1: Start Celery Worker
celery -A app_huggingface.celery worker --loglevel=info --pool=solo --concurrency=1

# Terminal 2: Start Flask App
python app_huggingface.py
```

### Web Interface
1. Navigate to `http://localhost:5000`
2. Enter your video prompt
3. Select quality settings
4. Click "Generate Video"
5. Monitor progress in real-time
6. Download completed videos

## Configuration

### Performance Tuning
Edit `app_huggingface.py` to adjust:

```python
# CPU vs GPU Mode
USE_CPU_MODE = True  # Set to False for GPU mode

# Generation Defaults
DEFAULT_RESOLUTION = (480, 848)  # 480p widescreen
DEFAULT_FRAMES = 16              # ~0.5 seconds
DEFAULT_STEPS = 16               # CPU optimized
```

### Quality vs Speed Settings

**Quick Testing**: 8 frames, 8 steps (~33 min)
**Production Standard**: 16 frames, 16 steps (~131 min)
**High Quality**: 24 frames, 24 steps (~295 min)
**Maximum Quality**: 31 frames, 32 steps (~521 min)

## Troubleshooting

### Common Issues

**Virtual Memory Error**:
```
Solution: Increase Windows virtual memory to 131072 MB
```

**GPU Out of Memory**:
```
The application automatically falls back to CPU mode
CPU mode is actually faster on RTX 3070 laptops
```

**Model Download Fails**:
```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface/
```

**Redis Connection Error**:
```bash
# Check Redis service is running
redis-cli ping
# Should return: PONG
```

### Performance Optimization

1. **Close all unnecessary applications** before generation
2. **Use CPU mode** for RTX 3070 laptops (proven faster)
3. **Start with 8-frame tests** to verify setup
4. **Run overnight** for high-quality videos
5. **Monitor Task Manager** during generation

## File Structure

```
mochi-video-generator/
├── app_huggingface.py      # Main Flask application
├── worker.py               # Celery worker configuration
├── config.py               # Configuration settings
├── start_production.bat    # Production startup script
├── requirements.txt        # Python dependencies
├── static/                 # Web assets (empty)
├── outputs/                # Generated videos
└── checkpoints/            # Processing checkpoints
```

## Technology Stack

- **Backend**: Flask + Celery + Redis
- **ML Framework**: HuggingFace Diffusers + PyTorch
- **Video Generation**: Genmo Mochi-1-Preview model
- **Memory Management**: CPU offloading + attention slicing
- **Frontend**: HTML + JavaScript (vanilla)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Test on RTX 3070 hardware if possible
4. Submit pull request with benchmark results

## License

This project uses the Genmo Mochi model. Please review the original model license at:
https://huggingface.co/genmo/mochi-1-preview

## Credits

- **Genmo**: Original Mochi video generation model
- **HuggingFace**: Diffusers implementation
- **Community**: RTX 3070 optimization insights

## Sample Outputs

See `outputs/` directory for example generated videos demonstrating different quality settings.

---

**Note**: This implementation prioritizes reliability and memory efficiency over raw speed. The CPU mode optimization makes it particularly suitable for consumer GPUs with limited VRAM.