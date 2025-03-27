from huggingface_hub import hf_hub_download
from generator import load_csm_1b, Segment
import torchaudio
import torch

model_path = hf_hub_download(repo_id="sesame/csm-1b", filename="ckpt.pt")
generator = load_csm_1b(model_path, "cuda")
speakers = [1]
# transcripts = [
#     "coolio 's mansion is a pretty cool place. It's got everything from a pool to a movie theater. You should totally come check it out."
# ]
# audio_paths = [
#     "coolio1.mp3",
# ]

# def load_audio(audio_path):
#     audio_tensor, sample_rate = torchaudio.load(audio_path)
#     audio_tensor = torchaudio.functional.resample(
#         audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=generator.sample_rate
#     )
#     return audio_tensor

# segments = [
#     Segment(text=transcript, speaker=speaker, audio=load_audio(audio_path))
#     for transcript, speaker, audio_path in zip(transcripts, speakers, audio_paths)
# ]

print("Generating audio...")
# print(segments)
# torch.save(segments[0], 'segment.pt')

# Load the segment from the file
loaded_segment = torch.load('segment.pt')
# Verify the loaded segment
print(f"Loaded segment: {loaded_segment}")
segments = [loaded_segment]

# Generate audio
audio = generator.generate(
    text="this is a generated audio sample",
    speaker=1,
    context=segments,
)

torchaudio.save("audio.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
print("Generated audio saved to audio.wav")