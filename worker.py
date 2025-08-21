# worker.py - Celery Worker Configuration for Mochi 1
import os
import sys

# Add mochi to Python path
sys.path.append(os.path.join(os.getcwd(), 'mochi'))

# Import the Celery app and task from app.py
from app_huggingface import celery, generate_video_hf_task

# Use the same Celery instance from app.py
app = celery

# Configuration for RTX 3070 optimization
app.conf.update(
    # Worker optimization for single GPU
    worker_prefetch_multiplier=1,  # Process one task at a time
    task_acks_late=True,
    worker_max_tasks_per_child=1,  # Restart worker after each task (memory cleanup)
    
    # Memory optimization
    worker_disable_rate_limits=True,
    task_compression='gzip',
    
    # Timeout settings (video generation can take long)
    task_soft_time_limit=3600,  # 1 hour soft limit
    task_time_limit=7200,       # 2 hour hard limit
)

if __name__ == '__main__':
    app.start()

# # worker.py - Celery Worker Configuration for Mochi 1
# import os
# import sys
# from celery import Celery

# # Add mochi to Python path
# sys.path.append(os.path.join(os.getcwd(), 'mochi'))

# # Create Celery instance
# app = Celery('mochi_worker')

# # Configuration for RTX 3070 optimization
# app.conf.update(
#     broker_url='redis://localhost:6379',
#     result_backend='redis://localhost:6379',
#     task_serializer='json',
#     accept_content=['json'],
#     result_serializer='json',
#     timezone='UTC',
#     enable_utc=True,
    
#     # Worker optimization for single GPU
#     worker_prefetch_multiplier=1,  # Process one task at a time
#     task_acks_late=True,
#     worker_max_tasks_per_child=1,  # Restart worker after each task (memory cleanup)
    
#     # Task routing
#     task_routes={
#         'app.generate_video_task': {'queue': 'video_generation'},
#     },
    
#     # Memory optimization
#     worker_disable_rate_limits=True,
#     task_compression='gzip',
    
#     # Timeout settings (video generation can take long)
#     task_soft_time_limit=3600,  # 1 hour soft limit
#     task_time_limit=7200,       # 2 hour hard limit
# )

# if __name__ == '__main__':
#     app.start()