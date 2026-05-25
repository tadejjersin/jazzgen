from jazzgen.midi_utils import render_midi_to_audio_array
from jazzgen.clf_utils import load_discogs_maest_pipeline, evaluate_audio
from jazzgen.modeling_utils import generate_midi
import os
import numpy as np

clf = load_discogs_maest_pipeline()

def generate_and_evaluate(input_midi_path, model, tokenizer):
    # generate a temporary MIDI file
    generate_midi(input_midi_path, "tmp.midi", tokenizer, model)

    # generate audio from MIDI
    audio = render_midi_to_audio_array("tmp.midi")

    # remove temporary MIDI
    os.remove("tmp.midi")

    # run classifier on the audio
    return evaluate_audio(audio, clf)

def run_dataset_eval(folder_path, model, tokenizer, n_repeats=1):
    results = {}

    midi_extensions = [".mid", ".midi"]

    for filename in os.listdir(folder_path):
        if os.path.splitext(filename)[1].lower() not in midi_extensions:
            continue

        input_midi_path = os.path.join(folder_path, filename)

        file_results = []
        for _ in range(n_repeats):
            score = generate_and_evaluate(input_midi_path, model, tokenizer)
            file_results.append(score)

        results[input_midi_path] = file_results

    return results

def compute_genre_stats(scores, genre):
    """
    Compute overall statistics for one genre from the output of run_dataset_eval.
    """

    values = []

    for midi_path, repeats in scores.items():
        for result in repeats:
            values.append(result[genre])

    if len(values) == 0:
        raise ValueError("No scores found. The input scores dictionary is empty.")

    values = np.array(values, dtype=float)

    return {
        "genre": genre,
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "n": int(values.size),
    }