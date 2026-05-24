from pathlib import Path
from transformers import pipeline
from .midi_utils import render_midi_to_audio_array

MODEL_ID = "mtg-upf/discogs-maest-10s-pw-129e"
TARGET_SR = 16000


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
    top_k: int = 100,
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
    """Compute a heuristic jazz-family and classical-family score from classifier predictions."""

    jazz_sum = 0.0
    jazz_best_score = 0.0
    jazz_best = None

    classical_sum = 0.0 
    classical_best_score = 0.0
    classical_best = None

    for pred in predictions:
        label = pred["label"].lower()
        pred_score = float(pred["score"])

        # jazz label
        if label.startswith("jazz"):
            jazz_sum += pred_score
            if pred_score > jazz_best_score:
                jazz_best = label
                jazz_best_score = pred_score

        # classical label
        if label.startswith("classical"):
            classical_sum += pred_score
            if pred_score > classical_best_score:
                classical_best = label
                classical_best_score = pred_score

    return {
        "jazz": (jazz_sum, jazz_best_score, jazz_best),
        "classical": (classical_sum, classical_best_score, classical_best)
    }


def evaluate_audio(
    audio,
    clf,
    top_k=100,
):
    """Classify audio style and process predictions for a jazz vs. classical comparison."""

    predictions = classify_audio_style(
        audio=audio,
        clf=clf,
        top_k=top_k,
    )

    return prediction_score(predictions)