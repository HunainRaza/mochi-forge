#!/usr/bin/env python3
"""
Flask Application Using HuggingFace Mochi Implementation
This uses the proven working HuggingFace Diffusers approach
"""

from flask import Flask, request, jsonify, render_template_string, send_file
from celery import Celery
import os
import json
import uuid
from datetime import datetime
import redis
import logging
from pathlib import Path
import torch
import gc
import psutil
import threading
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
FLASK_DEBUG = True

# HuggingFace specific settings
HUGGINGFACE_MODEL = "genmo/mochi-1-preview"
USE_CPU_MODE = True  # Set based on your testing results
OUTPUTS_PATH = "outputs"

# Generation defaults optimized for RTX 3070
DEFAULT_RESOLUTION = (480, 848)
DEFAULT_FRAMES = 16
DEFAULT_STEPS = 16  # CPU optimized
MAX_FRAMES = 31
MAX_STEPS = 32

# Environment setup
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Initialize Flask app
app = Flask(__name__)

# Celery configuration
app.config['CELERY_BROKER_URL'] = f'redis://{REDIS_HOST}:{REDIS_PORT}'
app.config['CELERY_RESULT_BACKEND'] = f'redis://{REDIS_HOST}:{REDIS_PORT}'

# Initialize Celery
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(
    result_backend=app.config['CELERY_RESULT_BACKEND'],
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_max_tasks_per_child=1,
    task_soft_time_limit=7200,  # 2 hours
    task_time_limit=10800,      # 3 hours
)

# Redis client
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

# Create output directory
Path(OUTPUTS_PATH).mkdir(exist_ok=True)

# Global pipeline cache
_pipeline_cache = None
_pipeline_lock = threading.Lock()

# Enhanced HTML template with HuggingFace info
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mochi Video Generator - HuggingFace Edition</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f5f5; }
        .container { max-width: 1000px; margin: 0 auto; padding: 20px; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }
        .success-banner { background: linear-gradient(135deg, #00b894 0%, #00cec9 100%); color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .form-container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 30px; }
        .form-group { margin-bottom: 20px; }
        label { display: block; margin-bottom: 8px; font-weight: 600; color: #333; }
        input, textarea, select { width: 100%; padding: 12px; border: 2px solid #e1e5e9; border-radius: 8px; font-size: 14px; }
        .btn { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px 30px; border: none; border-radius: 8px; cursor: pointer; font-size: 16px; font-weight: 600; }
        .btn:hover { transform: translateY(-2px); }
        .btn:disabled { background: #ccc; cursor: not-allowed; transform: none; }
        .status-container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 30px; }
        .status { padding: 20px; border-radius: 8px; margin: 15px 0; }
        .status.processing { background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%); color: #2d3436; }
        .status.completed { background: linear-gradient(135deg, #00b894 0%, #00cec9 100%); color: white; }
        .status.error { background: linear-gradient(135deg, #e17055 0%, #fd79a8 100%); color: white; }
        .progress-bar { background: #e1e5e9; border-radius: 10px; overflow: hidden; margin: 15px 0; }
        .progress-fill { background: linear-gradient(90deg, #667eea, #764ba2); height: 20px; border-radius: 10px; transition: width 0.3s; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        video { max-width: 100%; border-radius: 8px; }
        .job-card { background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #667eea; }
        .info-box { background: #e8f4f8; padding: 15px; border-radius: 8px; border-left: 4px solid #667eea; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎬 Mochi Video Generator</h1>
            <p>HuggingFace Edition • Reliable & Optimized for RTX 3070</p>
        </div>

        <div class="success-banner">
            <h3>🎉 Success! HuggingFace Implementation Working!</h3>
            <p><strong>Confirmed:</strong> CPU mode successfully generated videos on your system</p>
            <p><strong>Performance:</strong> ~15-45 minutes per video (depending on settings)</p>
            <p><strong>Reliability:</strong> Much more stable than raw Mochi implementation</p>
        </div>

        <div class="info-box">
            <h3>💡 Optimized Settings for Your Hardware</h3>
            <p><strong>CPU Mode:</strong> Reliable, slower generation (~30 minutes)</p>
            <p><strong>Resolution:</strong> 480p recommended for balance of quality/speed</p>
            <p><strong>Frames:</strong> 16 frames (~0.5 seconds) for testing, up to 31 for final videos</p>
            <p><strong>Steps:</strong> 16 steps for CPU (faster), 32 for higher quality</p>
        </div>

        <div class="form-container">
            <h2>Generate New Video</h2>
            <form id="videoForm">
                <div class="grid">
                    <div class="form-group">
                        <label for="prompt">Video Prompt *</label>
                        <textarea id="prompt" name="prompt" rows="4" placeholder="A beautiful cat sitting by a window, cinematic quality" required></textarea>
                    </div>
                    <div class="form-group">
                        <label for="negative_prompt">Negative Prompt</label>
                        <textarea id="negative_prompt" name="negative_prompt" rows="3" placeholder="blurry, low quality, distorted, artifacts"></textarea>
                    </div>
                </div>

                <div class="grid">
                    <div class="form-group">
                        <label for="num_frames">Duration</label>
                        <select id="num_frames" name="num_frames">
                            <option value="8">8 frames (~0.3s) - Quick Test</option>
                            <option value="16" selected>16 frames (~0.5s) - Recommended</option>
                            <option value="24">24 frames (~0.8s) - Good</option>
                            <option value="31">31 frames (~1.0s) - Maximum</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="steps">Quality Level</label>
                        <select id="steps" name="steps">
                            <option value="8">8 steps - Very Fast (~10 min)</option>
                            <option value="16" selected>16 steps - Fast (~20 min)</option>
                            <option value="24">24 steps - Good (~30 min)</option>
                            <option value="32">32 steps - High Quality (~45 min)</option>
                        </select>
                    </div>
                </div>

                <div class="form-group">
                    <label for="seed">Seed (for reproducible results)</label>
                    <input type="number" id="seed" name="seed" value="42" placeholder="42">
                </div>

                <button type="submit" class="btn" id="generateBtn">🚀 Generate Video (HuggingFace)</button>
            </form>
        </div>

        <div id="statusContainer" class="status-container" style="display: none;">
            <h2>Generation Status</h2>
            <div id="status"></div>
        </div>

        <div class="status-container">
            <h2>Recent Jobs</h2>
            <div id="jobs"></div>
        </div>
    </div>

    <script>
        let currentJobId = null;
        let pollInterval = null;

        document.getElementById('videoForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = new FormData(e.target);
            const data = Object.fromEntries(formData.entries());
            
            const generateBtn = document.getElementById('generateBtn');
            generateBtn.disabled = true;
            generateBtn.textContent = '⏳ Submitting...';
            
            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                if (result.success) {
                    currentJobId = result.job_id;
                    document.getElementById('statusContainer').style.display = 'block';
                    document.getElementById('status').innerHTML = `
                        <div class="status processing">
                            <h3>🔄 HuggingFace Generation Started</h3>
                            <p><strong>Job ID:</strong> ${result.job_id}</p>
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: 0%"></div>
                            </div>
                            <p id="progressText">Initializing HuggingFace pipeline...</p>
                            <p><em>First generation: ~5 minutes setup + generation time</em></p>
                            <p><em>Subsequent generations: Much faster (pipeline cached)</em></p>
                        </div>
                    `;
                    
                    startPolling(result.job_id);
                } else {
                    throw new Error(result.error);
                }
            } catch (error) {
                document.getElementById('statusContainer').style.display = 'block';
                document.getElementById('status').innerHTML = `
                    <div class="status error">
                        <h3>❌ Error</h3>
                        <p>${error.message}</p>
                    </div>
                `;
            }
            
            generateBtn.disabled = false;
            generateBtn.textContent = '🚀 Generate Video (HuggingFace)';
        });

        function startPolling(jobId) {
            if (pollInterval) clearInterval(pollInterval);
            
            pollInterval = setInterval(async () => {
                try {
                    const response = await fetch(`/status/${jobId}`);
                    const status = await response.json();
                    
                    updateStatusDisplay(status);
                    
                    if (status.status === 'completed' || status.status === 'failed') {
                        clearInterval(pollInterval);
                        loadRecentJobs();
                    }
                } catch (error) {
                    console.error('Error polling status:', error);
                }
            }, 5000);
        }

        function updateStatusDisplay(status) {
            const progress = parseInt(status.progress || 0);
            const statusDiv = document.getElementById('status');
            let className = 'processing';
            let icon = '🔄';
            
            if (status.status === 'completed') {
                className = 'completed';
                icon = '✅';
            } else if (status.status === 'failed') {
                className = 'error';
                icon = '❌';
            }
            
            statusDiv.innerHTML = `
                <div class="status ${className}">
                    <h3>${icon} ${status.status.charAt(0).toUpperCase() + status.status.slice(1)}</h3>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${progress}%"></div>
                    </div>
                    <p><strong>Progress:</strong> ${progress}%</p>
                    ${status.current_step ? `<p><strong>Current Step:</strong> ${status.current_step}</p>` : ''}
                    ${status.video_path ? `
                        <div style="margin-top: 20px;">
                            <h4>🎉 Video Completed!</h4>
                            <video controls style="max-width: 100%; margin: 10px 0;">
                                <source src="/video/${status.job_id}" type="video/mp4">
                            </video>
                            <br>
                            <a href="/video/${status.job_id}" download class="btn" style="display: inline-block; text-decoration: none;">📥 Download Video</a>
                        </div>
                    ` : ''}
                    ${status.error ? `<p style="color: #ff4757; margin-top: 10px;"><strong>Error:</strong> ${status.error}</p>` : ''}
                </div>
            `;
        }

        async function loadRecentJobs() {
            try {
                const response = await fetch('/jobs');
                const jobs = await response.json();
                
                const jobsDiv = document.getElementById('jobs');
                if (jobs.length === 0) {
                    jobsDiv.innerHTML = '<p>No recent jobs found.</p>';
                    return;
                }
                
                jobsDiv.innerHTML = jobs.map(job => `
                    <div class="job-card">
                        <div class="grid">
                            <div>
                                <p><strong>Prompt:</strong> ${job.prompt.substring(0, 100)}${job.prompt.length > 100 ? '...' : ''}</p>
                                <p><strong>Status:</strong> ${job.status}</p>
                                <p><strong>Created:</strong> ${new Date(job.created_at).toLocaleString()}</p>
                            </div>
                            <div style="text-align: right;">
                                <button onclick="checkJobStatus('${job.job_id}')" class="btn" style="margin: 5px;">📊 Check Status</button>
                                ${job.status === 'completed' ? `
                                    <br><a href="/video/${job.job_id}" download class="btn" style="margin: 5px; text-decoration: none;">📥 Download</a>
                                ` : ''}
                            </div>
                        </div>
                    </div>
                `).join('');
            } catch (error) {
                console.error('Error loading jobs:', error);
            }
        }

        async function checkJobStatus(jobId) {
            try {
                const response = await fetch(`/status/${jobId}`);
                const status = await response.json();
                
                currentJobId = jobId;
                document.getElementById('statusContainer').style.display = 'block';
                updateStatusDisplay(status);
                
                if (status.status === 'processing') {
                    startPolling(jobId);
                }
            } catch (error) {
                console.error('Error checking status:', error);
            }
        }

        // Load recent jobs on page load
        window.addEventListener('load', loadRecentJobs);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/generate', methods=['POST'])
def generate_video():
    """Submit a new video generation job using HuggingFace"""
    try:
        data = request.get_json()
        
        if not data.get('prompt'):
            return jsonify({'success': False, 'error': 'Prompt is required'}), 400
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Process seed
        seed_input = data.get('seed')
        if seed_input and str(seed_input).strip():
            try:
                processed_seed = int(seed_input)
            except ValueError:
                processed_seed = 42
        else:
            processed_seed = 42
        
        # Prepare job data
        job_data = {
            'job_id': job_id,
            'prompt': str(data.get('prompt')),
            'negative_prompt': str(data.get('negative_prompt', '')),
            'num_frames': str(min(int(data.get('num_frames', 16)), MAX_FRAMES)),
            'steps': str(min(int(data.get('steps', 16)), MAX_STEPS)),
            'seed': str(processed_seed),
            'status': 'queued',
            'created_at': datetime.now().isoformat(),
            'progress': '0'
        }
        
        # Store in Redis
        redis_client.hset(f"job:{job_id}", mapping=job_data)
        
        # Queue the task
        task = generate_video_hf_task.delay(job_id, job_data)
        
        logger.info(f"HuggingFace video generation job {job_id} queued")
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'task_id': task.id
        })
        
    except Exception as e:
        logger.error(f"Error submitting job: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/status/<job_id>')
def get_status(job_id):
    """Get job status"""
    try:
        job_data = redis_client.hgetall(f"job:{job_id}")
        if not job_data:
            return jsonify({'error': 'Job not found'}), 404
        
        status = {k.decode(): v.decode() for k, v in job_data.items()}
        return jsonify(status)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/video/<job_id>')
def get_video(job_id):
    """Download video"""
    try:
        video_path = os.path.join(OUTPUTS_PATH, f"{job_id}.mp4")
        if os.path.exists(video_path):
            return send_file(video_path, as_attachment=True)
        else:
            return jsonify({'error': 'Video not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/jobs')
def get_jobs():
    """Get recent jobs"""
    try:
        job_keys = redis_client.keys("job:*")
        jobs = []
        
        for key in job_keys:
            job_data = redis_client.hgetall(key)
            if job_data:
                job = {k.decode(): v.decode() for k, v in job_data.items()}
                jobs.append(job)
        
        jobs.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        return jsonify(jobs[:10])
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# HuggingFace Celery task
@celery.task(bind=True)
def generate_video_hf_task(self, job_id, job_data):
    """Generate video using HuggingFace Diffusers (proven working approach)"""
    
    logger.info(f"Starting HuggingFace video generation for job {job_id}")
    
    try:
        # Update status
        redis_client.hset(f"job:{job_id}", "status", "processing")
        redis_client.hset(f"job:{job_id}", "progress", "5")
        redis_client.hset(f"job:{job_id}", "current_step", "Initializing HuggingFace pipeline...")
        
        # Import HuggingFace components
        from diffusers import MochiPipeline
        from diffusers.utils import export_to_video
        import numpy as np
        
        redis_client.hset(f"job:{job_id}", "progress", "10")
        redis_client.hset(f"job:{job_id}", "current_step", "Loading pipeline (cached after first run)...")
        
        # Create or get cached pipeline
        global _pipeline_cache
        with _pipeline_lock:
            if _pipeline_cache is None:
                logger.info("Creating new HuggingFace pipeline...")
                
                if USE_CPU_MODE:
                    # CPU mode (confirmed working)
                    _pipeline_cache = MochiPipeline.from_pretrained(
                        HUGGINGFACE_MODEL,
                        torch_dtype=torch.float32
                    )
                    _pipeline_cache = _pipeline_cache.to("cpu")
                    redis_client.hset(f"job:{job_id}", "current_step", "CPU pipeline loaded...")
                else:
                    # GPU mode 
                    _pipeline_cache = MochiPipeline.from_pretrained(
                        HUGGINGFACE_MODEL,
                        torch_dtype=torch.bfloat16
                    )
                    _pipeline_cache.enable_model_cpu_offload()
                    _pipeline_cache.enable_attention_slicing()
                    redis_client.hset(f"job:{job_id}", "current_step", "GPU pipeline loaded...")
                
                logger.info("Pipeline cached for future use")
            else:
                logger.info("Using cached pipeline")
                redis_client.hset(f"job:{job_id}", "current_step", "Using cached pipeline...")
        
        pipeline = _pipeline_cache
        
        redis_client.hset(f"job:{job_id}", "progress", "20")
        redis_client.hset(f"job:{job_id}", "current_step", "Starting video generation...")
        
        # Parse parameters
        prompt = job_data['prompt']
        negative_prompt = job_data['negative_prompt']
        num_frames = int(job_data['num_frames'])
        num_steps = int(job_data['steps'])
        seed = int(job_data['seed'])
        height, width = DEFAULT_RESOLUTION
        
        logger.info(f"Generating: {num_frames}f, {num_steps}s, {width}x{height}, seed={seed}")
        
        # Generate video
        redis_client.hset(f"job:{job_id}", "current_step", f"Generating {num_frames} frames with {num_steps} steps...")
        
        start_time = time.time()
        video_frames = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_steps,
            guidance_scale=6.0,
            generator=torch.Generator().manual_seed(seed)
        ).frames[0]
        
        generation_time = time.time() - start_time
        logger.info(f"Generation completed in {generation_time:.1f} seconds")
        
        redis_client.hset(f"job:{job_id}", "progress", "85")
        redis_client.hset(f"job:{job_id}", "current_step", "Saving video...")
        
        # Save video
        output_path = os.path.join(OUTPUTS_PATH, f"{job_id}.mp4")
        export_to_video(video_frames, output_path, fps=8)
        
        file_size = Path(output_path).stat().st_size / (1024*1024)
        logger.info(f"Video saved: {output_path} ({file_size:.1f}MB)")
        
        # Cleanup
        del video_frames
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Update final status
        redis_client.hset(f"job:{job_id}", "status", "completed")
        redis_client.hset(f"job:{job_id}", "progress", "100")
        redis_client.hset(f"job:{job_id}", "current_step", "Video generation completed!")
        redis_client.hset(f"job:{job_id}", "video_path", output_path)
        redis_client.hset(f"job:{job_id}", "completed_at", datetime.now().isoformat())
        
        logger.info(f"HuggingFace generation completed successfully for job {job_id}")
        return {'status': 'completed', 'output_path': output_path}
        
    except Exception as e:
        error_msg = f"HuggingFace generation failed: {str(e)}"
        logger.error(error_msg)
        
        redis_client.hset(f"job:{job_id}", "status", "failed")
        redis_client.hset(f"job:{job_id}", "error", error_msg)
        redis_client.hset(f"job:{job_id}", "failed_at", datetime.now().isoformat())
        
        # Cleanup on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return {'status': 'failed', 'error': error_msg}

if __name__ == '__main__':
    logger.info("Starting HuggingFace Mochi Flask Application")
    app.run(debug=FLASK_DEBUG, host=FLASK_HOST, port=FLASK_PORT)