from generator import load_csm_1b, Segment
import torchaudio
import torch
import os
import logging
import time
import gc
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load model directly when module is imported (singleton pattern)
_model = None
try:
    # Check if CUDA is available
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This application requires a CUDA-enabled GPU.")
        exit(1)
        
    logger.info("Using device: cuda")
    
    # Clear memory before loading model
    gc.collect()
    torch.cuda.empty_cache()
    
    logger.info("Loading model (singleton)...")
    start_time = time.time()
    _model = load_csm_1b(device="cuda")
    logger.info(f"Model loaded successfully in {time.time() - start_time:.2f} seconds")
except Exception as e:
    logger.exception(f"Error loading model: {str(e)}")
    raise

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
    logger.info(f"Creating segments for '{segment_name}' from {len(audio_files)} audio files")
    
    # Validate input lists have the same length
    if len(transcripts) != len(audio_files):
        raise ValueError(f"Number of transcripts ({len(transcripts)}) must match number of audio files ({len(audio_files)})")
    
    segment_path = os.path.join("segments", f"{segment_name}.pt")
    
    segments = []
    for transcript, audio_file in zip(transcripts, audio_files):
        # Load and process audio
        audio_path = os.path.join("inputs", audio_file)
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file {audio_path} not found")
        
        # Load and resample audio
        audio_tensor, sample_rate = torchaudio.load(audio_path)
        audio_tensor = torchaudio.functional.resample(
            audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=_model.sample_rate
        )
        
        # Create segment (always use speaker 1)
        segment = Segment(text=transcript, speaker=1, audio=audio_tensor)
        segments.append(segment)
    
    # Save the segments
    torch.save(segments, segment_path, pickle_protocol=4)
    logger.info(f"Created and saved {len(segments)} segment(s) in {time.time() - start_time:.2f} seconds")
    
    return segments

def generate_audio_with_model(text, segment_name="segment"):
    """
    Generate audio using the specified segment as context.
    
    Args:
        text (str): Text to generate audio for
        segment_name (str): Name of the segment without file extension
    
    Returns:
        tuple: (torch.Tensor, float) - Generated audio tensor and processing time in seconds
    """
    # Validate input parameters
    if text is None:
        raise ValueError("Text parameter cannot be None")
    if segment_name is None:
        segment_name = "segment"  # Use default if None is passed explicitly
        
    start_time = time.time()
    logger.info(f"Starting audio generation for text: '{text[:50]}...")
    
    # Prepare full paths for segment only
    segment_path = os.path.join("segments", f"{segment_name}.pt")
    
    # Check if segment file exists
    if not os.path.exists(segment_path):
        raise FileNotFoundError(f"Segment file {segment_path} not found")
    
    # Load the segment from the file
    loaded_segments = torch.load(segment_path, weights_only=False)
    if not isinstance(loaded_segments, list):
        loaded_segments = [loaded_segments]
    
    logger.info(f"Loaded {len(loaded_segments)} segment(s) from {segment_path}")
    
    # Always use speaker 1
    speaker = 1
    
    try:
        # Generate audio
        logger.info("Generating audio...")
        generation_start = time.time()
        
        # Generate with max_audio_length_ms parameter to limit initial generation
        max_audio_length_ms = len(text) * 120  # ~120ms per character as a heuristic
        
        generate_kwargs = {
            "text": text,
            "speaker": speaker,
            "context": loaded_segments,
            "max_audio_length_ms": max_audio_length_ms
        }
        
        logger.info(f"Setting maximum audio length to {max_audio_length_ms} ms based on text length")
        audio = _model.generate(**generate_kwargs)
        
        generation_time = time.time() - generation_start
        total_time = time.time() - start_time
        
        logger.info(f"Audio generation completed in {generation_time:.2f} seconds")
        logger.info(f"Total processing time: {total_time:.2f} seconds")
        
        # Clear CUDA cache after generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return audio, total_time
    except Exception as e:
        logger.exception(f"Error generating audio: {str(e)}")
        # Clear CUDA cache on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise e

