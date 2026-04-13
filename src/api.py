"""
FastAPI web server for object detection UI.

Provides HTTP endpoints for video upload, processing status tracking,
and result file downloads. Processing runs asynchronously in the background
to avoid blocking the web server.
"""

from fastapi import FastAPI, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uuid
import os
import shutil

from src.pipeline import process_video


app = FastAPI(
    title="Object Detector API",
    description="YOLOv8x object detection with ByteTrack tracking",
    version="1.0.0"
)

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
app.mount('/static', StaticFiles(directory='frontend'), name='static')

# In-memory job store
# In production, use a database (Redis, PostgreSQL, etc.)
# Structure: {job_id: {status, progress, output_video, output_csv, error, filename}}
JOBS: dict = {}


@app.get('/')
def serve_ui():
    """Serve the main web interface."""
    return FileResponse('frontend/index.html')


@app.post('/upload')
async def upload_video(file: UploadFile, background_tasks: BackgroundTasks):
    """
    Receive video file and queue it for processing.

    The video is saved to a temporary location, then processing starts
    in the background. This endpoint returns immediately with a job_id
    that can be used to poll for status.

    Args:
        file: Uploaded video file
        background_tasks: FastAPI background task manager

    Returns:
        JSON with job_id and original filename
    """
    # Generate unique job ID
    job_id = str(uuid.uuid4())

    # Save uploaded file to temp directory
    os.makedirs('temp', exist_ok=True)
    temp_path = f'temp/{job_id}_{file.filename}'

    with open(temp_path, 'wb') as f:
        shutil.copyfileobj(file.file, f)

    # Register job as queued
    JOBS[job_id] = {
        'status': 'queued',
        'progress': 0,
        'output_video': None,
        'output_csv': None,
        'error': None,
        'filename': file.filename
    }

    # Add processing task to background queue
    background_tasks.add_task(run_pipeline, job_id, temp_path)

    return {
        'job_id': job_id,
        'filename': file.filename,
        'message': 'Video uploaded successfully. Processing will begin shortly.'
    }


def run_pipeline(job_id: str, input_path: str):
    """
    Background task that executes the video processing pipeline.

    Updates the job status in JOBS dict as processing progresses.
    Cleans up temporary input file when done (success or failure).

    Args:
        job_id: Unique identifier for this job
        input_path: Path to uploaded video file
    """
    try:
        # Update status to processing
        JOBS[job_id]['status'] = 'processing'
        JOBS[job_id]['progress'] = 10

        # Run the pipeline
        output_video, output_csv = process_video(
            input_path,
            output_dir=f'outputs/{job_id}'
        )

        # Mark job as complete
        JOBS[job_id].update({
            'status': 'done',
            'progress': 100,
            'output_video': output_video,
            'output_csv': output_csv
        })

    except Exception as e:
        # Log error and update job status
        print(f"ERROR processing job {job_id}: {str(e)}")
        JOBS[job_id].update({
            'status': 'error',
            'error': str(e),
            'progress': 0
        })

    finally:
        # Clean up temporary input file
        if os.path.exists(input_path):
            os.remove(input_path)


@app.get('/status/{job_id}')
def get_status(job_id: str):
    """
    Check the processing status of a job.

    Args:
        job_id: Unique job identifier

    Returns:
        JSON with job status, progress, and output file paths (if done)
    """
    if job_id not in JOBS:
        return JSONResponse(
            {'error': 'Job not found'},
            status_code=404
        )

    return JOBS[job_id]


@app.get('/download/{job_id}/video')
def download_video(job_id: str):
    """
    Download the annotated output video.

    Args:
        job_id: Unique job identifier

    Returns:
        Video file download
    """
    job = JOBS.get(job_id)

    if not job:
        return JSONResponse({'error': 'Job not found'}, status_code=404)

    if job['status'] != 'done':
        return JSONResponse({'error': 'Job not ready'}, status_code=400)

    if not os.path.exists(job['output_video']):
        return JSONResponse({'error': 'Output file not found'}, status_code=404)

    return FileResponse(
        job['output_video'],
        media_type='video/mp4',
        filename=f"annotated_{job['filename']}"
    )


@app.get('/download/{job_id}/csv')
def download_csv(job_id: str):
    """
    Download the detection log CSV.

    Args:
        job_id: Unique job identifier

    Returns:
        CSV file download
    """
    job = JOBS.get(job_id)

    if not job:
        return JSONResponse({'error': 'Job not found'}, status_code=404)

    if job['status'] != 'done':
        return JSONResponse({'error': 'Job not ready'}, status_code=400)

    if not os.path.exists(job['output_csv']):
        return JSONResponse({'error': 'Output file not found'}, status_code=404)

    base_name = job['filename'].rsplit('.', 1)[0]
    return FileResponse(
        job['output_csv'],
        media_type='text/csv',
        filename=f"detections_{base_name}.csv"
    )


@app.get('/health')
def health_check():
    """Health check endpoint for monitoring."""
    return {
        'status': 'healthy',
        'active_jobs': len([j for j in JOBS.values() if j['status'] == 'processing']),
        'total_jobs': len(JOBS)
    }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
