# Jazz Piano Generation with a Fine-Tuned Symbolic Music Model

This repository contains code and experiments for fine-tuning a pretrained symbolic music generation model for solo jazz piano generation. The project uses a pretrained [Aria](https://huggingface.co/loubb/aria-medium-base) model as the base model and adapts it to the [PiJAMA](https://zenodo.org/records/8354955) solo jazz piano dataset using LoRA fine-tuning.

The goal of the project is to test whether parameter-efficient fine-tuning can shift a general symbolic piano generation model toward jazz-style outputs. Several LoRA configurations are compared, using different LoRA ranks and target modules. The generated MIDI outputs are evaluated using an audio-based music style classifier, with a focus on jazz and classical music scores.

## Repository Structure

```text
.
├── data/                  # Dataset files or dataset processing outputs
├── notebooks/             # Analysis and visualization notebooks
├── src/jazzgen            # Utils for training, working with MIDIs, ...
├── src/hpc                # Code for training on the HPC cluster
├── requirements.txt       # Python dependencies
└── README.md
```

To reproduce the results, you need to download the `midi_kong.zip` file from the [PiJAMA](https://zenodo.org/records/8354955) dataset website and put the extracted files in the `data` folder. 

## Environment Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Or with conda: 

```bash
conda create -n jazzgen python=3.12
conda activate jazzgen
```

Install the required dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

For GPU support, make sure that the installed PyTorch version matches your CUDA version. 

To allow for MIDI to audio conversion via command line, you must also install [FluidSynth](https://www.fluidsynth.org/), add it to PATH and download a soundfont, for example from [this](https://ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General) website. In `src/jazzgen/midi_utils.py` change the `SOUNDFONT_PATH` to be the path to this file.

## Usage 

To ensure notebooks run as intended, run the following command first:

```bash
python src/prepare_dataset.py
```

Some notebooks may require training the models first, which you can do with:

```bash
python src/hpc/train.py
```

## Generation