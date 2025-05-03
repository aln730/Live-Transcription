#!/usr/bin/env python3

import argparse
import os
import numpy as np
import speech_recognition as sr
import whisper
import torch
import asyncio
from datetime import datetime, timedelta
from queue import Queue
from sys import platform

# OLED (SH1106) imports
from luma.core.interface.serial import i2c
from luma.oled.device import sh1106
from PIL import Image, ImageDraw, ImageFont

# Initialize SH1106 OLED
try:
    serial = i2c(port=1, address=0x3C)
    device = sh1106(serial, width=128, height=64)
    font = ImageFont.load_default()
    oled_enabled = True
    line_height = 8
    max_lines = device.height // line_height
except Exception as e:
    print(f"Error initializing OLED: {e}")
    oled_enabled = False
    device = None
    font = None
    line_height = 8
    max_lines = 8


def display_text_oled(lines_to_display):
    if not oled_enabled or device is None or font is None:
        return
    image = Image.new("1", (device.width, device.height))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, device.width, device.height), outline=0, fill=0)

    for i, line in enumerate(lines_to_display):
        draw.text((0, i * line_height), line, font=font, fill=255)

    try:
        device.display(image)
    except Exception as e:
        print(f"Error displaying on OLED: {e}")


async def process_audio_chunk(audio_data, audio_model):
    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    try:
        result = await asyncio.to_thread(audio_model.transcribe, audio_np, fp16=False,
                                        beam_size=1, word_timestamps=True)  # Explicit fp16
        return result
    except Exception as e:
        print(f"Error in process_audio_chunk: {e}")
        return None # Important: handle the error.  Returning None is better than crashing.


def word_wrap_oled(text):
    if not oled_enabled or device is None or font is None:
        return [text]
    lines = []
    words = text.split()
    line = ""
    for word in words:
        test_line = f"{line} {word}".strip()
        if ImageDraw.Draw(Image.new("1", (device.width, device.height))).textlength(test_line,
                                                                                 font=font) <= device.width:
            line = test_line
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines[:max_lines]  # Limit to max displayable lines



async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="tiny",  # Default to "tiny" for speed
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model to use."
    )
    parser.add_argument("--non_english", action="store_true")
    parser.add_argument("--energy_threshold", default=1000, type=int)
    parser.add_argument("--record_timeout", default=0.5,
                        type=float)  # Reduced timeout for faster processing
    parser.add_argument("--phrase_timeout", default=1.0, type=float)
    if "linux" in platform:
        parser.add_argument("--default_microphone", default="pulse", type=str)
    args = parser.parse_args()

    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    recorder.dynamic_energy_threshold = False

    if "linux" in platform:
        mic_name = args.default_microphone
        if mic_name == "list":
            print("Available microphones:")
            for idx, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"{idx}: {name}")
            return
        else:
            found_idx = None
            for idx, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    found_idx = idx
                    break
            if found_idx is None:
                print(f"ERROR: Microphone '{mic_name}' not found.")
                return
            source = sr.Microphone(sample_rate=16000, device_index=found_idx)
    else:
        source = sr.Microphone(sample_rate=16000)

    model_str = args.model
    if args.model != "large" and not args.non_english:
        model_str += ".en"
    print(f"Loading Whisper model: {model_str} ...")
    try:
        audio_model = whisper.load_model(model_str)
    except Exception as e:
        print(f"Error loading whisper model: {e}")
        return  # Exit if the model fails to load
    print("Whisper model loaded.\n")

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout

    lines = [[]]
    phrase_time = None
    data_queue = Queue()
    transcription_queue = asyncio.Queue()
    display_queue = asyncio.Queue()  # Queue for lines to display on OLED

    def record_callback(_, audio: sr.AudioData):
        data = audio.get_raw_data()
        data_queue.put(data)

    with source:
        print("Calibrating mic...")
        recorder.adjust_for_ambient_noise(source, duration=1)
        print("Calibration done.")

    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)
    print("Recording... Speak!\n")



    async def process_queue():
        while True:
            audio_data = await asyncio.to_thread(data_queue.get)
            result = await process_audio_chunk(audio_data, audio_model)
            if result is not None: # Check if result is not None
                await transcription_queue.put(result)
            data_queue.task_done()



    async def update_transcript():
        nonlocal phrase_time, lines
        current_display_line = ""
        while True:
            result = await transcription_queue.get()
            now = datetime.utcnow()
            if phrase_time and (now - phrase_time) > timedelta(seconds=phrase_timeout):
                lines.append([])
                current_display_line = ""  # Reset display line on new phrase
                await display_queue.put(
                    [])  # Clear OLED at the start of a new phrase, use empty list to clear
            phrase_time = now

            if result is not None: # Check if result is not None
                for seg in result["segments"]:
                    # Check if "words" key exists before iterating
                    if "words" in seg:
                        for w in seg["words"]:
                            word = w["word"]
                            print(word, end=" ", flush=True)
                            lines[-1].append(word)
                            current_display_line += f"{word} "
                            await display_queue.put(
                                word_wrap_oled(current_display_line.strip()))  # Send wrapped lines for immediate display
                    elif "text" in seg:
                        text = seg["text"]
                        print(text, end=" ", flush=True)
                        lines[-1].append(text)
                        current_display_line += f"{text} "
                        await display_queue.put(word_wrap_oled(current_display_line.strip()))

            transcription_queue.task_done()

            os.system('cls' if os.name == 'nt' else 'clear')
            for word_list in lines:
                print(" ".join(word_list))



    async def display_task():
        while True:
            lines_to_display = await display_queue.get()
            display_text_oled(lines_to_display)
            display_queue.task_done()



    processing_task = asyncio.create_task(process_queue())
    updating_task = asyncio.create_task(update_transcript())
    displaying_task = asyncio.create_task(display_task())

    try:
        await asyncio.gather(processing_task, updating_task, displaying_task)
    except KeyboardInterrupt:
        pass
    finally:
        processing_task.cancel()
        updating_task.cancel()
        displaying_task.cancel()
        await asyncio.gather(processing_task, updating_task, displaying_task,
                             return_exceptions=True)  # Ensure all tasks are awaited

    print("\nFinal Transcript:")
    for i, word_list in enumerate(lines):
        print(f"Line {i + 1}: {' '.join(word_list)}")



if __name__ == "__main__":
    asyncio.run(main())
