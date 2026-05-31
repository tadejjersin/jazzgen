from jazzgen.dataset_utils import get_midi_paths, create_split, tokenize_midi_paths
from jazzgen.midi_utils import save_first_notes
import os

root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

midi_paths = get_midi_paths()
split = create_split(midi_paths)

train_items = tokenize_midi_paths(
    split["train"],
    out_dir="../aria_tokens/train"
)

val_items = tokenize_midi_paths(
    split["val"],
    out_dir="../aria_tokens/val"
)

test_items = tokenize_midi_paths(
    split["test"],
    out_dir="../aria_tokens/test"
)

# make new midis with fist 10 nodes of test files as other examples of generation
save_dir = os.path.join(root_dir, "data", "test")
os.makedirs(save_dir, exist_ok=True)
for i, item in enumerate(test_items):
    path = os.path.join(root_dir, item["midi_path"])
    save_path = os.path.join(save_dir, f"s-{i:02}.mid")
    save_first_notes(path, save_path, n=10)
