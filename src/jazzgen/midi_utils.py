from pathlib import Path
import tempfile
import subprocess

import librosa
import numpy as np
import soundfile as sf
import pretty_midi

TARGET_SR = 16000

FLUIDSYNTH_EXE = "fluidsynth"
SOUNDFONT_PATH = r""

def render_midi_to_audio_array(
    midi_input,
    soundfont_path=SOUNDFONT_PATH,
    fluidsynth_exe=FLUIDSYNTH_EXE,
    sample_rate=44100,
    target_sample_rate=TARGET_SR,
    normalize=True,
    save_path=None,
):
    """
    Render MIDI to a mono float32 audio array.

    If save_path is provided, also saves the final processed audio to disk.
    """

    soundfont_path = Path(soundfont_path)

    if not soundfont_path.exists():
        raise FileNotFoundError(f"Soundfont not found: {soundfont_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        if isinstance(midi_input, (str, Path)):
            midi_path = Path(midi_input)
            if not midi_path.exists():
                raise FileNotFoundError(f"MIDI file not found: {midi_path}")
        else:
            raise TypeError("midi_input must be a file path")

        wav_path = tmpdir / "rendered.wav"

        cmd = [
            str(fluidsynth_exe),
            "-ni",
            "-T", "wav",
            "-F", str(wav_path),
            "-r", str(sample_rate),
            str(soundfont_path),
            str(midi_path),
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "FluidSynth failed.\n"
                f"Command: {' '.join(cmd)}\n\n"
                f"STDOUT:\n{result.stdout}\n\n"
                f"STDERR:\n{result.stderr}"
            )

        audio, sr = sf.read(wav_path, always_2d=False)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    audio = audio.astype(np.float32)

    # remove DC offset
    audio = audio - float(np.mean(audio))

    # resample for classifier
    if sr != target_sample_rate:
        audio = librosa.resample(
            audio,
            orig_sr=sr,
            target_sr=target_sample_rate,
        )

    # peak normalize
    if normalize:
        peak = float(np.max(np.abs(audio)))
        if peak > 0:
            audio = audio / peak * 0.95

    audio = audio.astype(np.float32)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(save_path, audio, target_sample_rate)

    return audio

def save_first_notes(midi_path, save_path, n=10):
    pm = pretty_midi.PrettyMIDI(midi_path)
    del pm.instruments[0].notes[n:]
    pm.write(save_path)