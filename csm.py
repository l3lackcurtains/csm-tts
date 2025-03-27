from generator import load_csm_1b, Segment
import torchaudio
import torch
import os
import logging
import time
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variable to store the model (singleton pattern)
_model = None
_model_lock = False

def get_model():
    """Get or initialize the model (singleton pattern)"""
    global _model, _model_lock
    
    # If model is already loaded, return it
    if _model is not None:
        return _model
    
    # Simple locking mechanism to prevent concurrent model loading
    if _model_lock:
        logger.info("Waiting for model to be loaded by another process...")
        while _model_lock and _model is None:
            time.sleep(0.5)
        return _model
    
    try:
        _model_lock = True
        logger.info("Loading model (singleton)...")
        start_time = time.time()
        _model = load_model()
        logger.info(f"Model loaded successfully in {time.time() - start_time:.2f} seconds")
        return _model
    finally:
        _model_lock = False

def load_model():
    """Load the CSM model with appropriate device detection"""
    # Determine the best available device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    logger.info(f"Using device: {device}")
    
    # Clear memory before loading model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        model = load_csm_1b(device=device)
        return model
    except Exception as e:
        logger.exception(f"Error loading model: {str(e)}")
        raise

def ensure_directories_exist():
    """Create necessary directories if they don't exist."""
    os.makedirs("segments", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("inputs", exist_ok=True)
    logger.info("Ensured that 'segments', 'results', and 'inputs' directories exist")

def load_audio_and_save_segment(transcripts, audio_files, segment_name="segment"):
    """
    Load audio files, create segments with transcripts, and save the segment.
    
    Args:
        transcripts (list): List of text transcripts
        audio_files (list): List of audio filenames (without path) located in the inputs folder
        segment_name (str): Name of the segment without file extension
    
    Returns:
        list: List of created segments
    """
    start_time = time.time()
    logger.info(f"Starting to load audio and create segments for '{segment_name}'")
    
    # Ensure directories exist
    ensure_directories_exist()
    
    # Validate input lists have the same length
    if len(transcripts) != len(audio_files):
        raise ValueError(f"Number of transcripts ({len(transcripts)}) must match number of audio files ({len(audio_files)})")
    
    # Default all speakers to 1
    speakers = [1] * len(transcripts)
    
    # Prepare full path for segment file
    segment_path = os.path.join("segments", f"{segment_name}.pt")
    
    def load_audio(audio_file):
        audio_path = os.path.join("inputs", audio_file)
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file {audio_path} not found")
        
        logger.info(f"Loading audio file: {audio_path}")
        audio_tensor, sample_rate = torchaudio.load(audio_path)
        
        # Use the singleton model
        model = get_model()
        audio_tensor = torchaudio.functional.resample(
            audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=model.sample_rate
        )
        return audio_tensor
    
    logger.info(f"Loading {len(audio_files)} audio files from 'inputs' folder")
    
    segments = []
    for transcript, speaker, audio_file in zip(transcripts, speakers, audio_files):
        try:
            audio = load_audio(audio_file)
            segment = Segment(text=transcript, speaker=speaker, audio=audio)
            segments.append(segment)
            logger.info(f"Created segment for '{audio_file}' with transcript: '{transcript[:50]}...' if len(transcript) > 50 else transcript")
        except Exception as e:
            logger.error(f"Error processing {audio_file}: {str(e)}")
            raise
    
    # Save the segments
    logger.info(f"Saving {len(segments)} segments to {segment_path}")
    torch.save(segments, segment_path, pickle_protocol=4)
    
    logger.info(f"Successfully created and saved {len(segments)} segment(s) in {time.time() - start_time:.2f} seconds")
    return segments

def generate_audio_with_model(generator, text, segment_name="segment", output_name="audio.wav", max_audio_length_ms=10000):
    """
    Generate audio using the specified segment as context and save it.
    
    Args:
        generator: The loaded CSM model
        text (str): Text to generate audio for
        segment_name (str): Name of the segment without file extension
        output_name (str): Name of the output audio file
        max_audio_length_ms (int): Maximum audio length in milliseconds
    
    Returns:
        torch.Tensor: Generated audio
    """
    start_time = time.time()
    logger.info(f"Starting audio generation for text: '{text[:50]}...' if len(text) > 50 else text")
    
    # Ensure directories exist
    ensure_directories_exist()
    
    # Prepare full paths
    segment_path = os.path.join("segments", f"{segment_name}.pt")
    output_path = os.path.join("results", output_name)
    
    # Check if segment file exists
    if not os.path.exists(segment_path):
        raise FileNotFoundError(f"Segment file {segment_path} not found")
    
    # Load the segment from the file
    logger.info(f"Loading segments from {segment_path}")
    loaded_segments = torch.load(segment_path)
    if not isinstance(loaded_segments, list):
        loaded_segments = [loaded_segments]
    
    logger.info(f"Loaded {len(loaded_segments)} segment(s) from {segment_path}")
    
    # Always use speaker 1
    speaker = 1
    
    try:
        # Generate audio
        logger.info("Generating audio...")
        generation_start = time.time()
        audio = generator.generate(
            text=text,
            speaker=speaker,
            context=loaded_segments,
            max_audio_length_ms=max_audio_length_ms,
        )
        generation_time = time.time() - generation_start
        logger.info(f"Audio generation completed in {generation_time:.2f} seconds")
        
        # Save the generated audio
        logger.info(f"Saving audio to {output_path}")
        torchaudio.save(output_path, audio.unsqueeze(0).cpu(), generator.sample_rate)
        
        # Clear CUDA cache after generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"Total processing time: {time.time() - start_time:.2f} seconds")
        return audio
    except Exception as e:
        logger.exception(f"Error generating audio: {str(e)}")
        # Clear CUDA cache on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise e

# Keep the original function for backward compatibility
def generate_audio_stream(text, segment_name="segment", output_name="audio.wav", max_audio_length_ms=10000):
    """Legacy function that loads the model each time - not recommended for web service"""
    logger.warning("Using legacy generate_audio_stream function - not recommended for web service")
    generator = load_model()
    return generate_audio_with_model(generator, text, segment_name, output_name, max_audio_length_ms)

