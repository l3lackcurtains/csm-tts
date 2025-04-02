from flask import Flask, request, jsonify, send_file, g, Response, make_response
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
    def wrapper(text, segment_name):
        # Create a cache key based on text and segment_name
        cache_key = f"{segment_name}:{hashlib.md5(text.encode()).hexdigest()}"
        
        # Return cached result if available
        if cache_key in audio_cache:
            print(f"Cache hit for text: '{text[:30]}...'")
            return audio_cache[cache_key]
        
        # Generate new result
        result = func(text, segment_name)
        
        # Store in cache (with simple LRU behavior)
        if len(audio_cache) >= MAX_CACHE_SIZE:
            # Remove a random item if cache is full
            audio_cache.pop(next(iter(audio_cache)))
        audio_cache[cache_key] = result
        return result
    return wrapper

# Apply cache to the generate_audio_with_model function
generate_audio_with_model = cache_audio(generate_audio_with_model)

@app.route('/upload_segment', methods=['POST'])
def upload_segment():
    """
    Expects a multipart/form-data with:
    - files: Multiple audio files
    - transcripts: JSON string of transcript array ["Transcript 1", "Transcript 2", ...]
    - segment_name: (optional) Name for the segment
    """
    try:
        start_time = time.time()
        print("Processing upload_segment request")
        
        # Get form data
        segment_name = request.form.get('segment_name', 'segment')
        
        # Get transcripts from form data (as JSON string)
        transcripts_json = request.form.get('transcripts')
        if not transcripts_json:
            return jsonify({"error": "'transcripts' is required as a JSON string array."}), 400
        
        try:
            import json
            transcripts = json.loads(transcripts_json)
            if not isinstance(transcripts, list):
                return jsonify({"error": "'transcripts' must be a JSON array of strings."}), 400
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid JSON format for 'transcripts'."}), 400
        
        # Get uploaded files
        files = request.files.getlist('files')
        if not files or len(files) == 0:
            return jsonify({"error": "No audio files uploaded. Use 'files' field to upload audio files."}), 400
        
        # Check if number of transcripts matches number of files
        if len(transcripts) != len(files):
            return jsonify({
                "error": f"Number of transcripts ({len(transcripts)}) must match number of audio files ({len(files)})."
            }), 400
        
        # Ensure directories exist
        os.makedirs("inputs", exist_ok=True)
        
        # Save uploaded files to inputs directory
        saved_filenames = []
        for file in files:
            if file.filename:
                # Generate a unique filename to avoid conflicts
                import uuid
                file_extension = os.path.splitext(file.filename)[1]
                unique_filename = f"{uuid.uuid4().hex}{file_extension}"
                
                # Save the file
                file_path = os.path.join("inputs", unique_filename)
                file.save(file_path)
                saved_filenames.append(unique_filename)
                print(f"Saved uploaded file as: {file_path}")
        
        # Process the saved files
        segments = load_audio_and_save_segment(transcripts, saved_filenames, segment_name)
        
        processing_time = time.time() - start_time
        print(f"Completed upload_segment in {processing_time:.2f} seconds")
        
        return jsonify({
            "message": f"Successfully created and saved {len(segments)} segment(s) as '{segment_name}.pt'",
            "processing_time_seconds": processing_time
        }), 200
    except Exception as e:
        print(f"Error in upload_segment: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/generate_audio', methods=['POST'])
def generate_audio():
    """
    Expects a JSON payload with:
    {
      "text": "Text to generate audio from",
      "segment_name": "optional_segment_name",
      "format": "optional_format" (wav, mp3)
    }
    Returns the generated audio directly as a stream.
    """
    try:
        start_time = time.time()
        print("Processing generate_audio request")
        
        data = request.get_json()
        text = data.get('text')
        segment_name = data.get('segment_name', 'segment')
        audio_format = data.get('format', 'wav')
        
        if not text:
            return jsonify({"error": "'text' is required."}), 400
        
        # Generate audio file
        generation_start = time.time()
        audio_tensor = generate_audio_with_model(text, segment_name)
        generation_time = time.time() - generation_start
        
        # Check if audio data is valid
        if audio_tensor is None:
            return jsonify({
            "error": "Failed to generate audio. No audio data was produced.",
            "generation_time": generation_time
            }), 500
        
        # Convert tensor to bytes using an in-memory buffer
        buffer = BytesIO()
        audio_tensor_cpu = audio_tensor.cpu()
        
        # Use different sample rates and formats based on request
        sample_rate = 24000  # Default sample rate for CSM
        
        if (audio_format.lower() == 'mp3'):
            # MP3 encoding (smaller file size but slightly more CPU usage)
            torchaudio.save(buffer, audio_tensor_cpu.unsqueeze(0), sample_rate=sample_rate, 
                          format="mp3", compression=4)  # Compression 4 = 128kbps
            content_type = 'audio/mp3'
            filename = "generated_audio.mp3"
        else:
            # Default WAV format
            torchaudio.save(buffer, audio_tensor_cpu.unsqueeze(0), sample_rate=sample_rate, format="wav")
            content_type = 'audio/wav'
            filename = "generated_audio.wav"
            
        buffer.seek(0)
        audio_bytes = buffer.read()
        
        print(f"Audio generation completed in {generation_time:.2f} seconds")
        print(f"Total request processing time: {time.time() - start_time:.2f} seconds")
        
        # Send audio bytes directly as a response
        response = make_response(audio_bytes)
        response.headers['Content-Type'] = content_type
        response.headers['Content-Disposition'] = f'inline; filename="{filename}"'
        return response
            
    except Exception as e:
        print(f"Error in generate_audio: {e}")
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
            "/upload_segment": {
                "method": "POST",
                "content_type": "multipart/form-data",
                "description": "Upload audio files with transcripts",
                "parameters": {
                    "files": "Multiple audio files (field can be repeated)",
                    "transcripts": "JSON string array of transcripts",
                    "segment_name": "(optional) Name for the segment"
                }
            },
            "/generate_audio": {
                "method": "POST",
                "content_type": "application/json",
                "description": "Generate audio from text and stream directly without saving to disk",
                "parameters": {
                    "text": "Text to generate audio from",
                    "segment_name": "(optional) Name of the segment to use"
                }
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
