from huggingface_hub import hf_hub_download
from generator import load_csm_1b, Segment
import torchaudio
import torch
import os

# Load the model
model_path = hf_hub_download(repo_id="sesame/csm-1b", filename="ckpt.pt")
print("Loading model from", model_path)
generator = load_csm_1b(model_path, "cuda")

def ensure_directories_exist():
    """Create necessary directories if they don't exist."""
    os.makedirs("segments", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("inputs", exist_ok=True)
    print("Ensured that 'segments', 'results', and 'inputs' directories exist")

def load_audio_and_save_segment(transcripts, audio_files, segment_name="segment"):
    """
    Load audio files, create segments with transcripts and speakers, and save the segment.
    
    Args:
        transcripts (list): List of text transcripts
        audio_files (list): List of audio filenames (without path) located in the inputs folder
        segment_name (str): Name of the segment without file extension
    
    Returns:
        list: List of created segments
    """
    # Ensure directories exist
    ensure_directories_exist()
    
    # Validate input lists have the same length
    if len(transcripts) != len(audio_files):
        raise ValueError(f"Number of transcripts ({len(transcripts)}) must match number of audio files ({len(audio_files)})")
    
    # Prepare full path for segment file
    segment_path = os.path.join("segments", f"{segment_name}.pt")
    
    def load_audio(audio_file):
        audio_path = os.path.join("inputs", audio_file)
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file {audio_path} not found")
        
        audio_tensor, sample_rate = torchaudio.load(audio_path)
        audio_tensor = torchaudio.functional.resample(
            audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=generator.sample_rate
        )
        return audio_tensor
    
    print(f"Loading audio files from 'inputs' folder and creating segments to save as '{segment_path}'...")
    
    segments = [
        Segment(text=transcript, speaker=1, audio=load_audio(audio_file))
        for transcript, audio_file in zip(transcripts, audio_files)
    ]
    
    # Save the segments
    # Note: We're saving custom objects that require pickle functionality
    torch.save(segments, segment_path, pickle_protocol=4)
    print(f"Successfully created and saved {len(segments)} segment(s) to '{segment_path}'")
    
    return segments

def generate_audio_stream(text, segment_name="segment", output_name="audio.wav"):
    """
    Generate audio using the specified segment as context and save it.
    
    Args:
        text (str): Text to generate audio for
        segment_name (str): Name of the segment without file extension
        output_name (str): Name of the output audio file
    
    Returns:
        torch.Tensor: Generated audio
    """
    # Ensure directories exist
    ensure_directories_exist()
    
    # Prepare full paths
    segment_path = os.path.join("segments", f"{segment_name}.pt")
    output_path = os.path.join("results", output_name)
    
    # Check if segment file exists
    if not os.path.exists(segment_path):
        raise FileNotFoundError(f"Segment file {segment_path} not found")
    
    # Load the segment from the file
    # Note: We need weights_only=False to load custom Segment objects
    # Only use this with segment files from trusted sources
    loaded_segments = torch.load(segment_path)
    if not isinstance(loaded_segments, list):
        loaded_segments = [loaded_segments]
    
    print(f"Loaded {len(loaded_segments)} segment(s) from {segment_path}")
    
    # Get the speaker from the first segment
    speaker = loaded_segments[0].speaker
    
    # Generate audio
    audio = generator.generate(
        text=text,
        speaker=speaker,
        context=loaded_segments,
    )
    
    # Save the generated audio
    torchaudio.save(output_path, audio.unsqueeze(0).cpu(), generator.sample_rate)
    print(f"Generated audio saved to {output_path}")
    
    return audio

# Example usage:
if __name__ == "__main__":
    # Example for loading audio and saving segment
    transcripts = ["coolio's mansion is a pretty cool place. It's got everything from a pool to a movie theater.", "coolio's mansion is a pretty cool place. It's got everything from a pool to a movie theater."]
    audio_files = ["coolio1.mp3", "coolio1.mp3"]  # These files should be in the ./inputs folder
    load_audio_and_save_segment(transcripts, audio_files, "coolio_segment")
    
    # Example for generating audio
    generate_audio_stream(
        text="Hello sir, im good and whaaaaass up!!",
        segment_name="coolio_segment",
        output_name="audio.wav"
    )