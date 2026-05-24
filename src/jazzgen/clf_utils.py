from pathlib import Path
from transformers import pipeline
from .midi_utils import render_midi_to_audio_array

MODEL_ID = "mtg-upf/discogs-maest-10s-pw-129e"
TARGET_SR = 16000

GENRES = [
    "Jazz",
    "Non-Music",
    "Blues",
    "Pop",
    "Stage & Screen",
    "Classical",
    "Electronic",
    "Folk, World, & Country",
    "Funk / Soul",
    "Latin",
    "Rock",
    "Hip Hop",
    "Children's",
    "Reggae",
    "Brass & Military",
]

def load_discogs_maest_pipeline(device=-1):
    """Load the Discogs MAEST audio-classification pipeline."""

    return pipeline(
        task="audio-classification",
        model=MODEL_ID,
        trust_remote_code=True,
        device=device,
    )


def classify_audio_style(
    audio,
    clf,
    top_k=400,
):
    """Classify the musical style of an audio array."""

    return clf(
        {
            "array": audio,
            "sampling_rate": TARGET_SR,
        },
        top_k=top_k,
    )


def prediction_score(
    predictions,
):
    """Sum predictions by genre groups."""

    genre_sums = {genre.lower(): 0.0 for genre in GENRES}

    for pred in predictions:
        label = pred["label"].lower()
        pred_score = float(pred["score"])

        for genre in GENRES:
            if label.startswith(genre.lower()):
                genre_sums[genre.lower()] += pred_score

    return genre_sums

def evaluate_audio(
    audio,
    clf,
    top_k=400,
):
    """Classify audio style and group predictions by genre."""

    predictions = classify_audio_style(
        audio=audio,
        clf=clf,
        top_k=top_k,
    )

    return prediction_score(predictions)