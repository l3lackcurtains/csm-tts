from generator import load_csm_1b, Segment
import torchaudio
import torch
import os
import logging
import time
import gc
import uuid
from functools import lru_cache
import queue
import threading

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

# Add a request queue for batching
request_queue = queue.Queue()
BATCH_SIZE = 4  # Adjust based on your GPU memory
BATCH_TIMEOUT = 0.1  # seconds

@lru_cache(maxsize=32)
def _load_segment(segment_name):
    segment_path = os.path.join("segments", f"{segment_name}.pt")
    return torch.load(segment_path, weights_only=False)

def _process_batch():
    while True:
        batch = []
        try:
            # Get first request
            batch.append(request_queue.get_nowait())
            # Try to fill batch
            timeout = BATCH_TIMEOUT
            while len(batch) < BATCH_SIZE:
                batch.append(request_queue.get(timeout=timeout))
                timeout = 0  # No timeout for subsequent items
        except (queue.Empty, TimeoutError):
            pass
        
        if batch:
            try:
                # Prepare batch data
                texts = [item['text'] for item in batch]
                segments_list = [_load_segment(item['segment_name']) for item in batch]
                
                # Generate all audio in batch
                audios = _model.generate_batch(texts, segments_list)
                
                # Distribute results
                for item, audio in zip(batch, audios):
                    item['result'].put(audio)
            except Exception as e:
                logger.exception(f"Error processing batch: {str(e)}")
                # Put error result for each item in batch
                for item in batch:
                    item['result'].put(None)

# Start batch processing thread
threading.Thread(target=_process_batch, daemon=True).start()

def generate_audio_with_model(text, segment_name="segment"):
    """
    Generate audio using the specified segment as context.
    """
    if text is None:
        raise ValueError("Text parameter cannot be None")
    if segment_name is None:
        segment_name = "segment"
        
    start_time = time.time()
    logger.info(f"Starting audio generation for text: '{text[:50]}...")
    
    try:
        # Create result queue
        result_queue = queue.Queue()
        
        # Submit request
        request_queue.put({
            'text': text,
            'segment_name': segment_name,
            'result': result_queue
        })
        
        # Wait for result
        audio = result_queue.get(timeout=30)  # 30 second timeout
        
        total_time = time.time() - start_time
        logger.info(f"Total processing time: {total_time:.2f} seconds")
        
        return audio, total_time
        
    except Exception as e:
        logger.exception(f"Error generating audio: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise e

