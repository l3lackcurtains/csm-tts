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

# Global variable to store the model (singleton pattern)
_model = None

def get_model():
    """Get or initialize the model (singleton pattern)"""
    global _model
    
    # If model is already loaded, return it
    if _model is not None:
        return _model
    
    try:
        logger.info("Loading model (singleton)...")
        start_time = time.time()
        _model = load_model()
        logger.info(f"Model loaded successfully in {time.time() - start_time:.2f} seconds")
        return _model
    except Exception as e:
        logger.exception(f"Error loading model: {str(e)}")
        raise

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

def generate_audio_with_model(text, segment_name="segment", output_name=None):
    """
    Generate audio using the specified segment as context and save it.
    
    Args:
        text (str): Text to generate audio for
        segment_name (str): Name of the segment without file extension
        output_name (str, optional): Name of the output audio file. If None, a random name will be generated.
    
    Returns:
        tuple: (audio_tensor, output_filename)
    """
    start_time = time.time()
    logger.info(f"Starting audio generation for text: '{text[:50]}...' if len(text) > 50 else text")
    
    # Get the model
    generator = get_model()
    
    # Ensure directories exist
    ensure_directories_exist()
    
    # If no output_name is provided, generate a random one
    if not output_name:
        output_name = f"audio_{uuid.uuid4().hex[:8]}.wav"
        logger.info(f"Generated random output filename: {output_name}")
    
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
        
        # Generate with max_audio_length_ms parameter to limit initial generation
        # This is a reasonable default that should work for most texts
        max_audio_length_ms = len(text) * 120  # ~120ms per character as a heuristic
        
        generate_kwargs = {
            "text": text,
            "speaker": speaker,
            "context": loaded_segments,
            "max_audio_length_ms": max_audio_length_ms
        }
        
        logger.info(f"Setting maximum audio length to {max_audio_length_ms} ms based on text length")
        audio = generator.generate(**generate_kwargs)
        
        # Trim silence at the end of the audio
        # Find the last non-silent part (where amplitude is above threshold)
        threshold = 0.01  # Adjust this threshold as needed
        amplitude = torch.abs(audio)
        non_silent = amplitude > threshold
        
        if torch.any(non_silent):
            # Find the last index where audio is above threshold
            last_sound_idx = torch.where(non_silent)[0][-1].item()
            # Add a small buffer (e.g., 0.5 seconds worth of samples)
            buffer_samples = int(0.5 * generator.sample_rate)
            end_idx = min(last_sound_idx + buffer_samples, audio.shape[0])
            
            # Trim the audio
            original_length = audio.shape[0]
            audio = audio[:end_idx]
            trimmed_length = audio.shape[0]
            
            logger.info(f"Trimmed audio from {original_length} to {trimmed_length} samples " +
                       f"({original_length/generator.sample_rate:.2f}s to {trimmed_length/generator.sample_rate:.2f}s)")
        
        generation_time = time.time() - generation_start
        logger.info(f"Audio generation completed in {generation_time:.2f} seconds")
        
        # Save the generated audio
        logger.info(f"Saving audio to {output_path}")
        torchaudio.save(output_path, audio.unsqueeze(0).cpu(), generator.sample_rate)
        
        # Clear CUDA cache after generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"Total processing time: {time.time() - start_time:.2f} seconds")
        return audio, output_name  # Return both the audio and the filename
    except Exception as e:
        logger.exception(f"Error generating audio: {str(e)}")
        # Clear CUDA cache on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise e

