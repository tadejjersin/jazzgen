from jazzgen.dataset_utils import get_midi_paths, create_split, tokenize_midi_paths, AriaChunkDataset
import torch
from jazzgen.modeling_utils import load_model
from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments, Trainer
from transformers.trainer_utils import get_last_checkpoint
import os

root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# define lora configs
lora_configs = {
    ("att", 16): LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "mixed_qkv",
        "att_proj_linear",
    ],
    bias="none",
    ),
    ("att", 32): LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=[
        "mixed_qkv",
        "att_proj_linear",
    ],
    bias="none",
    ),
    ("all", 16): LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "mixed_qkv",
        "att_proj_linear",
        "ff_gate_proj",
        "ff_up_proj",
        "ff_down_proj"
    ],
    bias="none",
    ),
    ("all", 32): LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=[
        "mixed_qkv",
        "att_proj_linear",
        "ff_gate_proj",
        "ff_up_proj",
        "ff_down_proj"
    ],
    bias="none",
    )
}

def train(train_items, val_items, config=("att", 16)):
    # create dataset
    train_dataset = AriaChunkDataset(
        train_items,
        block_size=2048,
        stride=1024,
    )

    val_dataset = AriaChunkDataset(
        val_items,
        block_size=2048,
        stride=1024,
    )

    # load base model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, base_model = load_model()

    model = get_peft_model(base_model, lora_configs[config])

    # perform training
    output_dir = os.path.join(root_dir, "fine_tuning", f"aria_lora_{config[0]}_{config[1]}")

    training_args = TrainingArguments(
        output_dir=output_dir,

        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,

        num_train_epochs=5,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",

        logging_steps=50,
        eval_strategy="steps",
        eval_steps=250,
        save_steps=250,

        bf16=False,
        fp16=True,

        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    last_checkpoint = None
    if os.path.isdir(output_dir):
        last_checkpoint = get_last_checkpoint(output_dir)

    trainer.train(resume_from_checkpoint=last_checkpoint)

    model.save_pretrained(f"{output_dir}/final_adapter")

def main():
    # get splits
    midi_paths = get_midi_paths()
    split = create_split(midi_paths)

    train_items = tokenize_midi_paths(
        split["train"],
        out_dir=os.path.join(root_dir, "aria_tokens", "train")
    )

    val_items = tokenize_midi_paths(
        split["val"],
        out_dir=os.path.join(root_dir, "aria_tokens", "val")
    )
    # train
    print(f"Training {('att', 32)}")
    train(train_items, val_items, config=("att", 32))
    print(f"Training {('all', 16)}")
    train(train_items, val_items, config=("all", 16))
    print(f"Training {('all', 32)}")
    train(train_items, val_items, config=("all", 32))
    print(f"Training {('att', 16)}")
    train(train_items, val_items, config=("att", 16))

if __name__ == "__main__":
    main()
