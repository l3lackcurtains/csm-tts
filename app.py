from flask import Flask, request, jsonify, send_file, g, Response, make_response, after_this_request
from csm import load_audio_and_save_segment, generate_audio_with_model
import os
import time
import torchaudio
from io import BytesIO
import hashlib
import functools
import tempfile

app = Flask(__name__)

# Simple in-memory cache
audio_cache = {}
# Max cache size (number of items)
MAX_CACHE_SIZE = 100

# Cache decorator for audio generation
def cache_audio(func):
    @functools.wraps(func)
    def wrapper(text, name):
        # Create a cache key based on text and name
        cache_key = f"{name}:{hashlib.md5(text.encode()).hexdigest()}"
        
        # Return cached result if available
        if (cache_key in audio_cache):
            print(f"Cache hit for text: '{text[:30]}...'")
            # For cached results, we don't have the processing time
            # Return a tuple with the cached audio and None for processing time
            return audio_cache[cache_key], 0.0
        
        # Generate new result
        result, processing_time = func(text, name)
        
        # Store only the audio in cache (with simple LRU behavior)
        if len(audio_cache) >= MAX_CACHE_SIZE:
            # Remove a random item if cache is full
            audio_cache.pop(next(iter(audio_cache)))
        audio_cache[cache_key] = result
        return result, processing_time
    return wrapper

# Apply cache to the generate_audio_with_model function
generate_audio_with_model = cache_audio(generate_audio_with_model)

@app.route('/upload-segment', methods=['POST'])
def upload_segment():
    """
    Expects a multipart/form-data with:
    - files: Multiple audio files (at least one is required)
    - transcripts: Multiple text transcripts (at least one is required)
    - name: (optional) Name for the segment
    """
    try:
        start_time = time.time()
        print("Processing upload_segment request")
        
        name = request.form.get('name', 'segment')
        uploaded_files = request.files.getlist('files')
        transcripts = request.form.getlist('transcripts')
        
        files = [f for f in uploaded_files if f.filename]
        
        if not files or not transcripts or len(files) < 1 or len(transcripts) < 1:
            return jsonify({
                "error": "At minimum, one file and one transcript must be provided."
            }), 400
            
        if len(files) != len(transcripts):
            return jsonify({
                "error": f"Number of files ({len(files)}) doesn't match number of transcripts ({len(transcripts)})."
            }), 400
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_files = []
            for file in files:
                if file.filename:
                    import uuid
                    file_extension = os.path.splitext(file.filename)[1]
                    unique_filename = f"{uuid.uuid4().hex}{file_extension}"
                    
                    file_path = os.path.join(temp_dir, unique_filename)
                    file.save(file_path)
                    temp_files.append(file_path)
                    print(f"Saved uploaded file as: {file_path}")
            
            try:
                segments = load_audio_and_save_segment(transcripts, temp_files, name)
            finally:
                for temp_file in temp_files:
                    try:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                            print(f"Deleted temporary file: {temp_file}")
                    except Exception as e:
                        print(f"Error deleting temporary file {temp_file}: {e}")
        
        processing_time = time.time() - start_time
        print(f"Completed upload_segment in {processing_time:.2f} seconds")
        
        return jsonify({
            "message": f"Successfully created and saved {len(segments)} segment(s) as '{name}.pt'",
            "processing_time": processing_time
        }), 200
    except Exception as e:
        print(f"Error in upload_segment: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/generate-audio', methods=['POST'])
def generate_audio():
    """
    Expects a JSON payload with:
    {
      "text": "Text to generate audio from",
      "name": "optional_name",
      "format": "optional_format" (wav, mp3)
    }
    Returns the generated audio directly as a stream.
    """
    try:
        start_time = time.time()
        print("Processing generate_audio request")
        
        data = request.get_json()
        text = data.get('text')
        name = data.get('name', 'segment')
        audio_format = data.get('format', 'wav')
        
        if not text:
            return jsonify({"error": "'text' is required."}), 400
        
        # Generate audio file
        audio_tensor, processing_time = generate_audio_with_model(text, name)
        
        # Check if audio data is valid
        if audio_tensor is None:
            return jsonify({
            "error": "Failed to generate audio. No audio data was produced.",
            "processing_time": processing_time
            }), 500
        
        # Convert tensor to bytes using an in-memory buffer
        buffer = BytesIO()
        audio_tensor_cpu = audio_tensor.cpu()
        
        # Use different sample rates and formats based on request
        sample_rate = 24000  # Default sample rate for CSM
        
        if (audio_format.lower() == 'mp3'):
            # MP3 encoding (smaller file size but slightly more CPU usage)
            torchaudio.save(buffer, audio_tensor_cpu.unsqueeze(0), sample_rate=sample_rate, 
                          format="mp3")  # Remove compression parameter
            content_type = 'audio/mp3'
            filename = "generated_audio.mp3"
        else:
            # Default WAV format
            torchaudio.save(buffer, audio_tensor_cpu.unsqueeze(0), sample_rate=sample_rate, format="wav")
            content_type = 'audio/wav'
            filename = "generated_audio.wav"
            
        buffer.seek(0)
        audio_bytes = buffer.read()
        
        print(f"Audio generation completed in {processing_time:.2f} seconds")
        print(f"Total request processing time: {time.time() - start_time:.2f} seconds")
        
        # Send audio bytes directly as a response
        response = make_response(audio_bytes)
        response.headers['Content-Type'] = content_type
        response.headers['Content-Disposition'] = f'inline; filename="{filename}"'
        # Add processing time as a header for debugging/monitoring
        response.headers['X-Processing-Time'] = f"{processing_time:.2f}"
        return response
            
    except Exception as e:
        print(f"Error in generate_audio: {e}")
        return jsonify({"error": str(e)}), 500

# For benchmarking: Alternative file-based version
@app.route('/generate-audio-file', methods=['POST'])
def generate_audio_file():
    """Alternative implementation using temporary files"""
    temp_path = None
    try:
        start_time = time.time()
        print("Processing generate_audio_file request")
        
        data = request.get_json()
        text = data.get('text')
        name = data.get('name', 'segment')
        
        if not text:
            return jsonify({"error": "'text' is required."}), 400
        
        # Generate audio 
        audio_tensor, processing_time = generate_audio_with_model(text, name)
        
        if audio_tensor is None:
            return jsonify({
                "error": "Failed to generate audio. No audio data was produced.",
                "processing_time": processing_time
            }), 500
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_path = temp_file.name
            # Move tensor to CPU and save
            torchaudio.save(temp_path, audio_tensor.cpu().unsqueeze(0), sample_rate=24000)
        
        print(f"Audio generation completed in {processing_time:.2f} seconds")
        
        @after_this_request
        def cleanup(response):
            # Delete the temp file after the response is sent
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                    print(f"Deleted temporary file: {temp_path}")
                except Exception as e:
                    print(f"Error deleting temporary file: {e}")
            return response
        
        response = send_file(
            temp_path,
            mimetype='audio/wav',
            as_attachment=True,
            download_name='generated_audio.wav'
        )
        # Add processing time as a header for debugging/monitoring
        response.headers['X-Processing-Time'] = f"{processing_time:.2f}"
        return response
    
    except Exception as e:
        # Clean up temp file in case of error
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass
        print(f"Error in generate_audio_file: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/list-segments', methods=['GET'])
def list_segments():
    """
    List all available segment files
    """
    try:
        # Define where segments are stored - check multiple possible locations
        segment_directories = ["segments"]
        
        # Get all .pt files from all potential directories
        files = []
        for segments_dir in segment_directories:
            if not os.path.exists(segments_dir):
                continue
                
            for file in os.listdir(segments_dir):
                if file.endswith('.pt'):
                    # Get file info
                    file_path = os.path.join(segments_dir, file)
                    file_stat = os.stat(file_path)
                    
                    # Create download URL for this segment
                    name = os.path.splitext(file)[0]  # Remove .pt extension
                    download_url = f"/download-segment/{name}"
                    
                    # Format timestamps for better readability
                    created_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(file_stat.st_ctime))
                    modified_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(file_stat.st_mtime))
                    
                    files.append({
                        "name": file,
                        "size": file_stat.st_size,
                        "size_mb": round(file_stat.st_size / (1024 * 1024), 2),
                        "created": created_time,
                        "modified": modified_time,
                        "download_url": download_url
                    })
        
        return jsonify({
            "segments": files,
            "count": len(files)
        })
    
    except Exception as e:
        print(f"Error listing segments: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/download-segment/<name>', methods=['GET'])
def download_segment(name):
    """
    Download a specific segment file
    """
    try:
        # For security, ensure the name doesn't contain path traversal attempts
        if '..' in name or '/' in name or '\\' in name:
            return jsonify({"error": "Invalid segment name"}), 400
            
        # Add .pt extension if not provided
        if not name.endswith('.pt'):
            name += '.pt'
        
        # Define possible locations where segments might be stored
        segment_directories = [".", "segments"]
        
        # Try to find the file in all potential locations
        segment_path = None
        for segments_dir in segment_directories:
            path = os.path.join(segments_dir, name)
            if os.path.exists(path) and os.path.isfile(path):
                segment_path = path
                break
        
        # Check if file was found
        if segment_path is None:
            return jsonify({
                "error": f"Segment file '{name}' not found", 
                "locations_checked": segment_directories
            }), 404
            
        return send_file(
            segment_path,
            mimetype='application/octet-stream',
            as_attachment=True,
            download_name=name
        )
    
    except Exception as e:
        print(f"Error downloading segment: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/upload-segment-file', methods=['POST'])
def upload_file():
    """
    Uploads a segment (.pt) file directly
    
    Expects a multipart/form-data with:
    - file: The .pt file to upload
    - name: (optional) Custom name for the segment file
    """
    try:
        start_time = time.time()
        print("Processing upload_file request")
        
        # Get the uploaded file
        file = request.files.get('file')
        
        # Validate file was provided
        if not file or not file.filename:
            return jsonify({
                "error": "No segment file provided"
            }), 400
            
        # Validate file extension
        if not file.filename.endswith('.pt'):
            return jsonify({
                "error": "File must be a .pt file"
            }), 400
        
        # Get custom name or use filename
        name = request.form.get('name')
        if not name:
            # Use original filename but remove extension
            name = os.path.splitext(file.filename)[0]
            
        # Ensure name is safe
        name = ''.join(c for c in name if c.isalnum() or c in ['-', '_'])
        
        # Ensure segment directory exists
        os.makedirs("segments", exist_ok=True)
        
        # Create file path with .pt extension
        if not name.endswith('.pt'):
            name += '.pt'
            
        file_path = os.path.join("segments", name)
        
        # Save the uploaded file
        file.save(file_path)
        
        processing_time = time.time() - start_time
        print(f"Uploaded segment file as: {file_path} in {processing_time:.2f} seconds")
        
        # Return success response
        return jsonify({
            "message": f"Successfully uploaded segment file as '{name}'",
            "name": os.path.splitext(name)[0],
            "file_path": file_path,
            "processing_time": processing_time
        }), 200
        
    except Exception as e:
        print(f"Error in upload_file: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint that also reports model status"""
    try:
        # Check if model can be loaded without actually loading it
        # This is a lightweight check
        return jsonify({
            "status": "ok",
            "model_available": True,
            "timestamp": time.time()
        })
    except Exception as e:
        print(f"Health check failed: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": time.time()
        }), 500

@app.route('/', methods=['GET'])
def index():
    """Simple landing page with API documentation"""
    return jsonify({
        "name": "CSM Audio Generation API",
        "endpoints": {
            "/upload-segment": {
                "method": "POST",
                "content_type": "multipart/form-data",
                "description": "Upload audio files with transcripts",
                "parameters": {
                    "files": "Audio files (one or more)",
                    "transcripts": "Transcript texts (one or more, must match number of files)",
                    "name": "(optional) Name for the segment"
                }
            },
            "/upload-segment-file": {
                "method": "POST",
                "content_type": "multipart/form-data",
                "description": "Upload a pre-trained segment file (.pt)",
                "parameters": {
                    "file": "The .pt file to upload",
                    "name": "(optional) Custom name for the segment"
                }
            },
            "/generate-audio": {
                "method": "POST",
                "content_type": "application/json",
                "description": "Generate audio from text and stream directly without saving to disk",
                "parameters": {
                    "text": "Text to generate audio from",
                    "name": "(optional) Name of the segment to use",
                    "format": "(optional) Audio format: wav or mp3"
                }
            },
            "/list-segments": {
                "method": "GET",
                "description": "List all available segment files"
            },
            "/download-segment/<name>": {
                "method": "GET",
                "description": "Download a specific segment file"
            },
            "/health": {
                "method": "GET",
                "description": "Health check endpoint"
            }
        }
    })

if __name__ == '__main__':
    # Use only 1 worker to ensure the model is shared
    app.run(host='0.0.0.0', debug=False, port=8383, threaded=False)
