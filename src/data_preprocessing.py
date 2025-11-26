import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from datasets import load_dataset

class EnglishTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        for text in texts:
            # Tokenize English text
            tokens = tokenizer.encode(text, max_length=max_length, truncation=True)
            if len(tokens) > 1:
                self.examples.append(tokens)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        tokens = self.examples[idx]
        # Input: all tokens except last
        # Target: all tokens except first
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        return input_ids, target_ids

def prepare_english_data():
    # Load English dataset (WikiText-2)
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
    
    # Use GPT-2 tokenizer for English
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare training data
    train_texts = [text for text in dataset['train']['text'] if len(text.strip()) > 0]
    train_dataset = EnglishTextDataset(train_texts, tokenizer)
    
    # Create DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        collate_fn=lambda x: pad_sequence(x)
    )
    
    return train_loader, tokenizer

def pad_sequence(batch):
    # Pad sequences to same length in batch
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    inputs_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_padded = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
    
    return inputs_padded, targets_padded