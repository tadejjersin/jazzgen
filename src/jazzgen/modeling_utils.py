import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_ID = "loubb/aria-medium-base"
ADAPTER_PATH = "../aria_lora_pijama/final_adapter"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

def load_model(adapter_path=None):
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    if adapter_path is not None:
        model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
        )
        model.to(device)
        return tokenizer, model 
    else:
        base_model.to(device)
        return tokenizer, base_model
    
def generate_midi(base_midi_path, save_midi_path, tokenizer, model, max_length=1000, temperature=1.1, top_p=0.95):
    model.eval()
    prompt = tokenizer.encode_from_file(
        base_midi_path,
        return_tensors="pt",
    )

    input_ids = prompt.input_ids.to(device)

    with torch.inference_mode():
        continuation = model.generate(
            input_ids=input_ids,
            max_length=max_length, # max_new_tokens ?
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            use_cache=True,
        )

    midi_dict = tokenizer.decode(continuation[0].detach().cpu().tolist())
    midi_dict.to_midi().save(save_midi_path)