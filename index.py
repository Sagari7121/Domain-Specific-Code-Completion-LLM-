import torch
from lib.bigramModel import BigramModel

block_size = 128
batch_size = 64
eval_iters = 200
n_emb = 256
n_heads = 8
n_layers = 6
max_iter = 5000
learning_rate = 1e-3

with open('data/input.txt', 'r') as file:
    text = file.read()

torch.manual_seed(1337)

chars = sorted(list(set(text)))
vocal_size = len(chars)
print(vocal_size)
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for ch, i in stoi.items()}

encoder = lambda s: [stoi[c] for c in s]
decoder = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encoder(text), dtype=torch.long)

# SPLIT DATA TO TRAINING AND TEST DATA

n = int(0.9*len(data))

train_data = data[:n]
test_data = data[n:]

def get_batch(split):
    data = train_data if split == "train" else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x ,y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "test"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        
        out[split] = losses.mean()
    model.train()
    return out

model = BigramModel(vocal_size, block_size, n_emb, n_heads, n_layers)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for i in range(max_iter):
    if i % eval_iters == 0:
        losses = estimate_loss()
        print(f"step {i} train loss: {losses['train']:.4f}, val loss: {losses['test']:.4f}")
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(losses)

idx = torch.zeros((1, 1), dtype=torch.long)
res = model.generate(idx, block_size, max_new_tokens=400)
print(decoder(res[0].tolist()))