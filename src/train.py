import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from model import SmallLanguageModel
from data_preprocessing import prepare_english_data
import os

def train_model(model, train_loader, epochs=10, device='cuda'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch} completed. Average Loss: {avg_loss:.4f}')
        
        # Save checkpoint
        os.makedirs('models/checkpoints', exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, f'models/checkpoints/checkpoint_epoch_{epoch}.pt')

if __name__ == "__main__":
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Prepare English data
    print("Loading English dataset...")
    train_loader, tokenizer = prepare_english_data()
    
    # Create model
    vocab_size = tokenizer.vocab_size
    model = SmallLanguageModel(vocab_size)
    
    print(f"Model created with vocab size: {vocab_size}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train
    train_model(model, train_loader, epochs=10, device=device)