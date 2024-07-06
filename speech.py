import ipywidgets as widgets
from IPython.display import display
from threading import Thread
from queue import Queue
import pyaudio
import json
import time
from vosk import Model, KaldiRecognizer
from transformers import T5Tokenizer, T5ForConditionalGeneration

CHANNELS = 1
FRAME_RATE = 16000
RECORD_SECONDS = 2
AUDIO_FORMAT = pyaudio.paInt16
SAMPLE_SIZE = 2

messages = Queue()
recordings = Queue()

record_button = widgets.Button(description = "Record", disabled = False, button_style = "success", icon = "microphone")
stop_button = widgets.Button(description = "Stop", disabled = False, button_style = "warning", icon = "stop")

output = widgets.Output()

def start_recording(data):
    messages.put(True)

    with output:
        display("Starting...")
        record = Thread(target=record_microphone)
        record.start()

        transcribe = Thread(target=speech_recognition, args=(output,))
        transcribe.start()
def stop_recording(data):
    with output:
        messages.get()
        display("Stopped.")

record_button.on_click(start_recording)
stop_button.on_click(stop_recording)
display(record_button, stop_button, output)

def record_microphone(chunk=1024):
    p = pyaudio.PyAudio()

    stream = p.open(format = AUDIO_FORMAT,
                    channels = CHANNELS,
                    rate = FRAME_RATE,
                    input = True,
                    input_device_index = 1,
                    frames_per_buffer = chunk)
    
    frames = []

    while not messages.empty():
        data = stream.read(chunk)
        frames.append(data)

        if len(frames) >= (FRAME_RATE * RECORD_SECONDS) / chunk:
            recordings.put(frames.copy())
            frames = []
    
    stream.stop_stream()
    stream.close()
    p.terminate

model = Model(model_name = "vosk-model-en-us-0.22")
rec = KaldiRecognizer(model, FRAME_RATE)
rec.SetWords(True)
model_translation = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
def speech_recognition(output):
    while not messages.empty():
        frames = recordings.get()
        task_prefix = "translate English to French: "
        rec.AcceptWaveform(b''.join(frames))
        result = rec.Result()
        text = json.loads(result)["text"]
        input_ids = tokenizer(task_prefix+text,return_tensors="pt").input_ids #only works for one audio snippet
        outputs = model_translation.generate(input_ids)
        time.sleep(0.5)
        print(tokenizer.decode(outputs[0]))
        
"""
def translation(output):
    while not messages.empty():
        frames = recordings.get()

        
        text = json.loads(result)["text"]
        print(text)

"""
