import torch
from transformers import GPT2Tokenizer
from model import SmallLanguageModel

def generate_english_text(model, tokenizer, prompt, max_length=100, device='cuda', temperature=1.0):
    model.eval()
    model = model.to(device)
    
    # Encode English prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            output = model(input_ids)
            
            # Apply temperature
            logits = output[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Stop if EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Load model
    model = SmallLanguageModel(vocab_size=tokenizer.vocab_size)
    checkpoint = torch.load('models/checkpoints/checkpoint_epoch_9.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Generate English text
    prompts = [
        "Once upon a time",
        "The future of artificial intelligence",
        "In a world where"
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        generated = generate_english_text(model, tokenizer, prompt, max_length=50, device=device)
        print(f"Generated: {generated}\n")