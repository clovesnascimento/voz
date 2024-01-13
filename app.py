import subprocess

# Run the setup.py install command
try:
    subprocess.run(['python', 'setup.py', 'install', '--user'], check=True)
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
                     inputs=[gr.Textbox(label="Введите текст", max_lines=3), gr.Audio(type="filepath", label="Загрузить аудиофайл")), gr.Radio(label="Выбор языка", choices=["ru", "en", "zh-cn", "ja", "de", "fr", "it", "pt", "pl", "tr", "ko", "nl", "cs", "ar", "es", "hu"], value="ru")], 
                     outputs=gr.Audio(type="filepath", label="Вывод"), 
                     title="Клонирование голоса")

iface.launch()