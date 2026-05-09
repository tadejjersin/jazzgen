import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class MIDIDataset(Dataset):
    def __init__(self, token_sequences, seq_len=512, stride=256):
        self.seq_len = seq_len
        self.samples = []

        for tokens in token_sequences:
            if len(tokens) < seq_len + 1:
                continue

            for i in range(0, len(tokens) - seq_len, stride):
                chunk = tokens[i:i + seq_len + 1]

                x = chunk[:-1]
                y = chunk[1:]

                self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

class MIDIDataset(Dataset):
    def __init__(self, token_sequences, seq_len=512, stride=256, include_final_chunk=True):
        self.seq_len = seq_len
        self.stride = stride
        self.samples = []

        for tokens in token_sequences:
            if len(tokens) < seq_len + 1:
                continue

            last_start = len(tokens) - (seq_len + 1)
            starts = list(range(0, last_start + 1, stride))

            if include_final_chunk and starts[-1] != last_start:
                starts.append(last_start)

            for start in starts:
                chunk = tokens[start:start + seq_len + 1]

                x = chunk[:-1]
                y = chunk[1:]

                self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

    
def create_dataset_and_loader(token_sequences, seq_len=512, stride=256, batch_size=16):
    dataset = MIDIDataset(
        token_sequences=token_sequences,
        seq_len=seq_len,
        stride=stride 
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return dataset, dataloader