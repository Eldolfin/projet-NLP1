import numpy as np
import whisper
import scipy.signal
import soundcard as sc

def select_best_input_device():
    mics = sc.all_microphones()
    for mic in mics:
        if 'airpods' in mic.name.lower():
            print(f"Airpods selected: {mic.name}")
            return mic
    default = sc.default_microphone()
    print(f"Miro par defaut : {default.name}")
    return default

def select_best_output_device():
    speakers = sc.all_speakers()
    for sp in speakers:
        if 'airpods' in sp.name.lower():
            print(f"Airpods output: {sp.name}")
            return sp
    default = sc.default_speaker()
    print(f"Output par defaut: {default.name}")
    return default

def record_until_silence(mic, sample_rate=44100, channels=1,
                         block_duration=0.1, silence_threshold=0.01, silence_duration=1.0):
    frames = []
    silence_time = 0.0
    recording_started = False
    block_size = int(sample_rate * block_duration)

    with mic.recorder(samplerate=sample_rate, blocksize=block_size) as recorder:
        print("Recording... (silence detecte automatiquement)")
        while True:
            data = recorder.record(numframes=block_size)
            frames.append(data.copy())
            rms = np.sqrt(np.mean(np.square(data)))
            print(f"{rms}    {silence_threshold}    {silence_time}")
            if rms > silence_threshold:
                recording_started = True
                silence_time = 0.0
            else:
                if recording_started:
                    silence_time += block_duration
                    if silence_time >= silence_duration:
                        print(f"Silence detecte")
                        break
    return np.concatenate(frames, axis=0)

def main():
    mic = select_best_input_device()
    speaker = select_best_output_device()

    sample_rate = 44100
    channels = 1

    recording = record_until_silence(
        mic,
        sample_rate=sample_rate,
        channels=channels,
        block_duration=0.1,
        silence_threshold=0.01,
        silence_duration=1.0
    )
    print("Enregistrement fini")

    print("Lecture audio... (si t'entends rien c'est que soit mauvais innput ou mauvais output ou auter probleme)")
    speaker.play(recording, samplerate=sample_rate)
    print("LEcture fini")

    print("Transcription avec whisper...")
    audio_data = np.squeeze(recording)
    
    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0]

    target_sr = 16000
    if sample_rate != target_sr:
        num_samples = int(len(audio_data) * target_sr / sample_rate)
        audio_data = scipy.signal.resample(audio_data, num_samples)
    audio_data = audio_data.astype(np.float32)

    model = whisper.load_model("small")
    result = model.transcribe(audio_data)
    
    print("Transcription:")
    print(result["text"])

if __name__ == "__main__":
    main()
