import pyaudio

audio = pyaudio.PyAudio()

mic_index = 22  # Change this to your microphone index

device_info = audio.get_device_info_by_index(mic_index)
print(f"\n🎤 Microphone: {device_info['name']} (Index {mic_index})")
print(f"🔍 Supported Sample Rate: {device_info['defaultSampleRate']} Hz")

audio.terminate()
