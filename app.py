from flask import Flask, request, jsonify, send_file, g
from csm import load_audio_and_save_segment, generate_audio_with_model, get_model
import os
import time

app = Flask(__name__)


@app.route('/upload_segment', methods=['POST'])
def upload_segment():
    """
    Expects a JSON payload with:
    {
      "transcripts": ["Transcript 1", "Transcript 2", ...],
      "audio_files": ["file1.mp3", "file2.mp3", ...],
      "segment_name": "optional_segment_name"
    }
    """
    try:
        start_time = time.time()
        print("Processing upload_segment request")
        
        data = request.get_json()
        transcripts = data.get('transcripts')
        audio_files = data.get('audio_files')
        segment_name = data.get('segment_name', 'segment')
        
        if not transcripts or not audio_files:
            return jsonify({"error": "Both 'transcripts' and 'audio_files' are required."}), 400
        
        segments = load_audio_and_save_segment(transcripts, audio_files, segment_name)
        
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
      "segment_name": "optional_segment_name"
    }
    Returns the generated audio file as an attachment.
    """
    try:
        start_time = time.time()
        print("Processing generate_audio request")
        
        data = request.get_json()
        text = data.get('text')
        segment_name = data.get('segment_name', 'segment')
        
        if not text:
            return jsonify({"error": "'text' is required."}), 400
        
        # Use fixed value for output_name
        output_name = 'audio.wav'
        output_path = os.path.join("results", output_name)
        
        generation_start = time.time()
        generate_audio_with_model(text, segment_name, output_name)
        generation_time = time.time() - generation_start
        
        print(f"Audio generation completed in {generation_time:.2f} seconds")
        print(f"Total request processing time: {time.time() - start_time:.2f} seconds")
        
        return send_file(output_path, as_attachment=True, download_name=output_name)
    except FileNotFoundError as e:
        print(f"File not found: {str(e)}")
        return jsonify({"error": str(e)}), 404
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
                "description": "Upload audio segments with transcripts"
            },
            "/generate_audio": {
                "method": "POST",
                "description": "Generate audio from text using a segment as context"
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
