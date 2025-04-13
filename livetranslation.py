#!/usr/bin/env python3

import argparse
import os
import numpy as np
import speech_recognition as sr
import whisper
import torch

from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform

def main():
    parser = argparse.ArgumentParser(
        description="Real-time microphone transcription using SpeechRecognition + whisper."
    )
    parser.add_argument(
        "--model",
        default="medium",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model to use for transcription.",
    )
    parser.add_argument(
        "--non_english",
        action="store_true",
        help="If set, don't force the English model (e.g. use the multi-language version)."
    )
    parser.add_argument(
        "--energy_threshold",
        default=1000,
        type=int,
        help="Energy level threshold for mic detection (SpeechRecognition)."
    )
    parser.add_argument(
        "--record_timeout",
        default=2.0,
        type=float,
        help="Recording phrase time limit in seconds before the callback is triggered."
    )
    parser.add_argument(
        "--phrase_timeout",
        default=3.0,
        type=float,
        help="If there's a pause longer than this, we consider it a new phrase."
    )
    if "linux" in platform:
        parser.add_argument(
            "--default_microphone",
            default="pulse",
            type=str,
            help=(
                "Default microphone name for SpeechRecognition on Linux. "
                "Use 'list' to list available devices, or set to a valid device name."
            )
        )
    args = parser.parse_args()

    # ----------------------------------------------------------------
    # SETUP SPEECHRECOGNITION & MICROPHONE
    # ----------------------------------------------------------------
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    recorder.dynamic_energy_threshold = False  # recommended for stable detection

    # On Linux, pick the mic by name or list them if requested
    if "linux" in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == "list":
            print("Available microphone devices:")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"  {index}: \"{name}\"")
            return
        else:
            # Attempt to find the mic that matches the given name
            found_idx = None
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    found_idx = index
                    break
            if found_idx is None:
                print(
                    f"ERROR: Microphone '{mic_name}' not found among devices. "
                    "Use --default_microphone list to see available devices."
                )
                return
            source = sr.Microphone(sample_rate=16000, device_index=found_idx)
    else:
        # On other OSes, just pick the default with 16k sample rate
        source = sr.Microphone(sample_rate=16000)

    # ----------------------------------------------------------------
    # LOAD WHISPER MODEL
    # ----------------------------------------------------------------
    model_str = args.model
    # If not large and not multi-language forced, use the .en variant
    if args.model != "large" and not args.non_english:
        model_str += ".en"
    print(f"Loading whisper model: {model_str} ...")
    audio_model = whisper.load_model(model_str)
    print("Model loaded.\n")

    # ----------------------------------------------------------------
    # PHRASE HANDLING CONFIG
    # ----------------------------------------------------------------
    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout

    # We'll store separate phrases in a list
    transcription = ['']

    # The time we last received a chunk from the callback
    phrase_time = None
    data_queue = Queue()

    def record_callback(_, audio: sr.AudioData):
        """
        A threaded callback function to receive audio data when recordings finish.
        Called by SpeechRecognition when phrase_time_limit = record_timeout is reached.
        """
        data = audio.get_raw_data()
        data_queue.put(data)

    # Adjust for ambient noise (before we start capturing)
    with source:
        print("Calibrating microphone for ambient noise...")
        recorder.adjust_for_ambient_noise(source, duration=1)
        print("Done calibrating. Starting background listening...")

    # Start a background thread to gather audio data
    recorder.listen_in_background(
        source,
        record_callback,
        phrase_time_limit=record_timeout
    )

    print("Recording started. Speak!\n")

    while True:
        try:
            now = datetime.utcnow()
            # If we have data in the queue, process it
            if not data_queue.empty():
                # If enough time has passed since our last chunk, consider the old phrase closed
                phrase_complete = False
                if phrase_time and (now - phrase_time) > timedelta(seconds=phrase_timeout):
                    phrase_complete = True

                # Update phrase time
                phrase_time = now

                # Gather all raw data from the queue at once
                audio_data = b"".join(list(data_queue.queue))
                data_queue.queue.clear()

                # Convert raw 16-bit PCM -> float32 array
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                # Transcribe with GPU if available
                use_fp16 = torch.cuda.is_available()
                result = audio_model.transcribe(audio_np, fp16=use_fp16)
                text = result["text"].strip()

                # If we had a big gap, treat this as a new phrase
                if phrase_complete:
                    transcription.append(text)
                else:
                    # Otherwise, update the last item in the list
                    transcription[-1] = text

                # Clear screen to reprint
                os.system('cls' if os.name == 'nt' else 'clear')
                for line in transcription:
                    print(line)
                print("", end="", flush=True)

            else:
                sleep(0.25)

        except KeyboardInterrupt:
            break

    # ----------------------------------------------------------------
    # FINISHED -> PRINT RESULTS
    # ----------------------------------------------------------------
    print("\n\nFinal Transcription:")
    for line in transcription:
        print(line)

if __name__ == "__main__":
    main()
