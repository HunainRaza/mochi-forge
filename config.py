# config_cpu_only.py - CPU-Only Configuration for RTX 3070
# This forces everything to run on CPU to avoid memory issues

import torch
import os

# Force CPU mode - bypass GPU entirely
DEVICE = "cpu"
ENABLE_CPU_OFFLOAD = True
FORCE_CPU_ONLY = True  # New flag

# Memory Management (More aggressive for CPU)
MAX_RAM_USAGE = 50.0       # GB - Use up to 50GB RAM
AGGRESSIVE_CLEANUP = True
FORCE_MEMORY_MAPPING = True

# Model Settings (CPU optimized)
DEFAULT_PRECISION = "float32"  # CPU doesn't support bf16
ENABLE_ATTENTION_SLICING = True
ENABLE_VAE_SLICING = True

# CPU-specific settings
CPU_THREADS = 8  # Limit CPU threads
SEQUENTIAL_LOADING = True  # Load models one at a time

# Generation Defaults (Conservative for CPU)
DEFAULT_RESOLUTION = (384, 640)  # Smaller for CPU
DEFAULT_FRAMES = 25              # Shorter videos
DEFAULT_STEPS = 32               # Fewer steps
MAX_FRAMES = 37                  # CPU limit
MAX_STEPS = 64

# Enhanced Retry Settings
MAX_RETRIES = 3
FALLBACK_RESOLUTION = (256, 448)
FALLBACK_FRAMES = 16
FALLBACK_STEPS = 16

# Timeout Settings
MODEL_LOAD_TIMEOUT = 14400  # 4 hours for CPU loading
GENERATION_TIMEOUT = 7200   # 2 hours for generation
TASK_TIMEOUT = 21600        # 6 hours total

# File Paths
MODELS_PATH = os.path.join(os.getcwd(), "models")
OUTPUTS_PATH = os.path.join(os.getcwd(), "outputs")
CHECKPOINTS_PATH = os.path.join(os.getcwd(), "checkpoints")

# Redis Configuration
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0

# Flask Configuration
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
FLASK_DEBUG = True

# CPU-optimized Environment Variables
def setup_cpu_environment():
    """Setup environment variables for CPU-only operation"""
    env_vars = {
        # Force CPU
        'CUDA_VISIBLE_DEVICES': '',  # Hide GPU from PyTorch
        'TORCH_USE_CUDA_DSA': '0',
        
        # Memory Management
        'SAFETENSORS_MMAP': '1',
        'PYTORCH_CUDA_ALLOC_CONF': '',  # Disable CUDA allocator
        
        # CPU Threading
        'OMP_NUM_THREADS': str(CPU_THREADS),
        'MKL_NUM_THREADS': str(CPU_THREADS),
        'NUMEXPR_NUM_THREADS': str(CPU_THREADS),
        
        # HuggingFace Settings
        'TRANSFORMERS_VERBOSITY': 'error',
        'HF_HUB_DISABLE_PROGRESS_BARS': '1',
        'TRANSFORMERS_CACHE': os.path.join(os.getcwd(), 'cache'),
        
        # Memory Optimization
        'PYTHONUNBUFFERED': '1',
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
    
    print(f"✅ CPU-only environment configured")

# Initialize CPU environment
setup_cpu_environment()

print("🔧 CPU-Only Mode Enabled")
print("   This will be slower but more reliable for RTX 3070 Laptop")
print("   Expected generation time: 2-4 hours")

# # config.py - Enhanced RTX 3070 Configuration with Memory Management
# import torch
# import os
# import psutil

# # GPU Configuration
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ENABLE_CPU_OFFLOAD = True  # Essential for 8GB VRAM
# ENABLE_VAE_TILING = True   # Critical for memory efficiency

# # Enhanced Memory Management
# MAX_VRAM_USAGE = 6.5       # GB - More conservative for RTX 3070
# AGGRESSIVE_CLEANUP = True   # Force garbage collection
# FORCE_MEMORY_MAPPING = True # Use memory mapping for large models

# # Model Settings
# DEFAULT_PRECISION = "bf16"  # Balance between quality and memory
# ENABLE_ATTENTION_SLICING = True
# ENABLE_VAE_SLICING = True

# # CPU Offloading Settings (Enhanced)
# ENABLE_SEQUENTIAL_CPU_OFFLOAD = True
# ENABLE_MODEL_CPU_OFFLOAD = True
# LOW_MEM_MODE = True
# T5_CPU_ONLY = True  # Force T5 to CPU to save VRAM

# # Generation Defaults (Conservative for RTX 3070)
# DEFAULT_RESOLUTION = (480, 848)  # 480p widescreen
# DEFAULT_FRAMES = 31              # ~1 second
# DEFAULT_STEPS = 64               # Balanced quality/speed
# MAX_FRAMES = 49                  # Hardware limit for RTX 3070
# MAX_STEPS = 100

# # Enhanced Retry Settings
# MAX_RETRIES = 3
# FALLBACK_RESOLUTION = (384, 640)
# FALLBACK_FRAMES = 25
# FALLBACK_STEPS = 32

# # Timeout Settings (NEW)
# MODEL_LOAD_TIMEOUT = 7200  # 2 hours for model loading
# GENERATION_TIMEOUT = 3600  # 1 hour for generation
# TASK_TIMEOUT = 10800       # 3 hours total

# # File Paths
# MODELS_PATH = os.path.join(os.getcwd(), "models")
# OUTPUTS_PATH = os.path.join(os.getcwd(), "outputs")
# CHECKPOINTS_PATH = os.path.join(os.getcwd(), "checkpoints")

# # Redis Configuration
# REDIS_HOST = 'localhost'
# REDIS_PORT = 6379
# REDIS_DB = 0

# # Flask Configuration
# FLASK_HOST = '0.0.0.0'
# FLASK_PORT = 5000
# FLASK_DEBUG = True

# # Enhanced Environment Variables for Memory Optimization
# def setup_environment():
#     """Setup environment variables for optimal memory usage"""
#     env_vars = {
#         # Memory Management
#         'SAFETENSORS_MMAP': '1',  # Use memory mapping
#         'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:256,expandable_segments:True',
        
#         # CUDA Settings
#         'CUDA_LAUNCH_BLOCKING': '1',  # Better error messages
#         'TORCH_USE_CUDA_DSA': '1',    # Memory debugging
#         'CUDA_VISIBLE_DEVICES': '0',  # Use only first GPU
        
#         # HuggingFace Settings
#         'TRANSFORMERS_VERBOSITY': 'error',  # Reduce noise
#         'HF_HUB_DISABLE_PROGRESS_BARS': '1',
#         'TRANSFORMERS_CACHE': os.path.join(os.getcwd(), 'cache'),
        
#         # Memory Optimization
#         'OMP_NUM_THREADS': '8',  # Limit CPU threads
#         'MKL_NUM_THREADS': '8',
#         'NUMEXPR_NUM_THREADS': '8',
        
#         # Windows Specific
#         'PYTHONUNBUFFERED': '1',  # Better logging
#     }
    
#     for key, value in env_vars.items():
#         os.environ[key] = value
    
#     print(f"✅ Environment configured for memory optimization")

# # Memory Requirements Check
# def check_memory_requirements():
#     """Check if system meets memory requirements"""
#     try:
#         # Get system memory info
#         virtual_memory = psutil.virtual_memory()
#         swap_memory = psutil.swap_memory()
        
#         total_ram_gb = virtual_memory.total / (1024**3)
#         total_virtual_gb = (virtual_memory.total + swap_memory.total) / (1024**3)
#         available_ram_gb = virtual_memory.available / (1024**3)
        
#         print(f"💾 Memory Status:")
#         print(f"   Total RAM: {total_ram_gb:.1f}GB")
#         print(f"   Available RAM: {available_ram_gb:.1f}GB")
#         print(f"   Total Virtual Memory: {total_virtual_gb:.1f}GB")
        
#         # Check requirements
#         requirements_met = True
        
#         if total_virtual_gb < 80:
#             print(f"⚠️  WARNING: Insufficient virtual memory!")
#             print(f"   Current: {total_virtual_gb:.1f}GB")
#             print(f"   Required: 100GB+")
#             print(f"   Please increase Windows virtual memory:")
#             print(f"   1. System Properties → Advanced → Performance → Settings")
#             print(f"   2. Advanced → Virtual Memory → Change")
#             print(f"   3. Custom Size: Initial 131072 MB, Maximum 131072 MB")
#             requirements_met = False
        
#         if available_ram_gb < 8:
#             print(f"⚠️  WARNING: Low available RAM: {available_ram_gb:.1f}GB")
#             print(f"   Close other applications before generating videos")
        
#         # Check GPU
#         if torch.cuda.is_available():
#             gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
#             print(f"🎮 GPU Memory: {gpu_memory:.1f}GB")
            
#             if gpu_memory < 7:
#                 print(f"⚠️  WARNING: GPU has less than 8GB VRAM")
#                 requirements_met = False
#         else:
#             print(f"❌ CUDA not available!")
#             requirements_met = False
        
#         return {
#             'requirements_met': requirements_met,
#             'total_ram_gb': total_ram_gb,
#             'available_ram_gb': available_ram_gb,
#             'total_virtual_gb': total_virtual_gb,
#             'gpu_memory_gb': gpu_memory if torch.cuda.is_available() else 0
#         }
        
#     except Exception as e:
#         print(f"❌ Memory check failed: {e}")
#         return {'requirements_met': False, 'error': str(e)}

# # Model Verification
# def verify_models():
#     """Verify that required model files exist and are valid"""
#     required_files = {
#         'dit.safetensors': 'DIT Model (40GB)',
#         'decoder.safetensors': 'Decoder Model (1.5GB)',
#     }
    
#     print(f"🔍 Checking model files...")
    
#     all_present = True
#     for filename, description in required_files.items():
#         file_path = os.path.join(MODELS_PATH, filename)
        
#         if os.path.exists(file_path):
#             size_gb = os.path.getsize(file_path) / (1024**3)
#             print(f"   ✅ {description}: {size_gb:.1f}GB")
#         else:
#             print(f"   ❌ {description}: Not found at {file_path}")
#             all_present = False
    
#     if not all_present:
#         print(f"\n❌ Missing model files!")
#         print(f"   Please download the required models to: {MODELS_PATH}")
        
#     return all_present

# # GPU Optimization Settings
# def get_gpu_settings():
#     """Get optimized settings based on GPU"""
#     if not torch.cuda.is_available():
#         return {
#             'device': 'cpu',
#             'precision': 'float32',
#             'cpu_offload': True,
#             'low_mem': True
#         }
    
#     gpu_name = torch.cuda.get_device_name(0)
#     gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
#     # RTX 3070 specific optimizations
#     if 'RTX 3070' in gpu_name or gpu_memory <= 8.5:
#         return {
#             'device': 'cuda',
#             'precision': 'bf16',
#             'cpu_offload': True,
#             'low_mem': True,
#             'max_vram_usage': 6.5,
#             'tile_decode': True,
#             'attention_slicing': True,
#             'sequential_offload': True
#         }
    
#     # Higher-end GPU settings
#     elif gpu_memory > 12:
#         return {
#             'device': 'cuda',
#             'precision': 'bf16',
#             'cpu_offload': False,
#             'low_mem': False,
#             'max_vram_usage': gpu_memory * 0.9,
#             'tile_decode': False,
#             'attention_slicing': False,
#             'sequential_offload': False
#         }
    
#     # Mid-range GPU settings
#     else:
#         return {
#             'device': 'cuda',
#             'precision': 'bf16',
#             'cpu_offload': True,
#             'low_mem': True,
#             'max_vram_usage': gpu_memory * 0.8,
#             'tile_decode': True,
#             'attention_slicing': True,
#             'sequential_offload': True
#         }

# # Initialize configuration
# def initialize_config():
#     """Initialize and validate configuration"""
#     print(f"🚀 Initializing Mochi Configuration...")
    
#     # Setup environment
#     setup_environment()
    
#     # Check memory requirements
#     memory_status = check_memory_requirements()
    
#     # Verify models
#     models_present = verify_models()
    
#     # Get GPU settings
#     gpu_settings = get_gpu_settings()
    
#     print(f"\n📋 Configuration Summary:")
#     print(f"   Memory OK: {'✅' if memory_status['requirements_met'] else '❌'}")
#     print(f"   Models Present: {'✅' if models_present else '❌'}")
#     print(f"   GPU: {gpu_settings['device']} ({gpu_settings['precision']})")
#     print(f"   CPU Offload: {'✅' if gpu_settings['cpu_offload'] else '❌'}")
    
#     if not memory_status['requirements_met']:
#         print(f"\n⚠️  Configuration issues detected!")
#         print(f"   Please fix the above issues before continuing.")
#         return False
    
#     print(f"\n✅ Configuration ready!")
#     return True

# # Run initialization when imported
# if __name__ == "__main__":
#     initialize_config()
# else:
#     # Setup environment when imported
#     setup_environment()

# # config.py - RTX 3070 Optimized Configuration
# import torch
# import os

# # GPU Configuration
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ENABLE_CPU_OFFLOAD = True  # Essential for 8GB VRAM
# ENABLE_VAE_TILING = True   # Critical for memory efficiency
# # TORCH_COMPILE = False     # REMOVED: Not supported by current Mochi version

# # T5 Text Encoder Settings (NEW)
# T5_DEVICE = "cpu"
# FORCE_T5_CPU = True  # Force T5 to run on CPU for memory efficiency

# # Memory Management
# MAX_VRAM_USAGE = 7.0       # GB - More conservative for RTX 3070
# AGGRESSIVE_CLEANUP = True   # Force garbage collection

# # Model Settings
# DEFAULT_PRECISION = "bf16"  # Balance between quality and memory
# ENABLE_ATTENTION_SLICING = True
# ENABLE_VAE_SLICING = True

# # CPU Offloading Settings (UPDATED)
# ENABLE_SEQUENTIAL_CPU_OFFLOAD = True
# ENABLE_MODEL_CPU_OFFLOAD = True
# LOW_MEM_MODE = True

# # Generation Defaults (Conservative for RTX 3070)
# DEFAULT_RESOLUTION = (480, 848)  # 480p widescreen
# DEFAULT_FRAMES = 31              # ~1 second
# DEFAULT_STEPS = 64               # Balanced quality/speed
# MAX_FRAMES = 49                  # Hardware limit for RTX 3070
# MAX_STEPS = 100

# # VAE Tiling Configuration (REMOVED - not supported in current version)
# # VAE_TILE_HEIGHT = 80
# # VAE_TILE_WIDTH = 144
# # VAE_TILE_OVERLAP = 0.25

# # Retry Settings
# MAX_RETRIES = 2
# FALLBACK_RESOLUTION = (384, 640)
# FALLBACK_FRAMES = 25
# FALLBACK_STEPS = 32

# # File Paths
# MODELS_PATH = os.path.join(os.getcwd(), "models")
# OUTPUTS_PATH = os.path.join(os.getcwd(), "outputs")
# CHECKPOINTS_PATH = os.path.join(os.getcwd(), "checkpoints")

# # Redis Configuration
# REDIS_HOST = 'localhost'
# REDIS_PORT = 6379
# REDIS_DB = 0

# # Flask Configuration
# FLASK_HOST = '0.0.0.0'
# FLASK_PORT = 5000
# FLASK_DEBUG = True

# # Environment Variables for Memory Optimization (NEW)
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Better error messages
# os.environ['TORCH_USE_CUDA_DSA'] = '1'    # Memory debugging
# os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  # Reduce noise
# os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
# # os.environ['HF_HUB_OFFLINE'] = '1'  # Force offline mode
# # os.environ['TRANSFORMERS_OFFLINE'] = '1'  # Force offline mode
# os.environ['SAFETENSORS_MMAP'] = '1'  # Use memory mapping
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'