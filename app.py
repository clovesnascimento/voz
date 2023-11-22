import subprocess

# Run the setup.py install command
try:
    subprocess.run(['python', 'setup.py', 'install'], check=True)
    print("Installation successful.")
except subprocess.CalledProcessError as e:
    print(f"Installation failed with error: {e}")

import gradio as gr
import torch
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Init TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def voice_clone(text: str, speaker_wav: str, language: str):
    # Run TTS
    print("Speaker wav:", speaker_wav)
    tts.tts_to_file(text=text, speaker_wav=speaker_wav, language=language, file_path="output.wav")
    return "output.wav"

iface = gr.Interface(fn=voice_clone, 
                     inputs=["text", gr.Audio(type="filepath"), gr.Radio(['ko', 'en'], label="Language")], 
                     outputs=gr.Audio(type="filepath"), 
                     title="IMCapusle Voice Clone")

iface.launch()