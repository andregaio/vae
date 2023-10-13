import torch
from torchvision.utils import save_image
from torch.optim import Adam
from data import load_dataset
from loss import criterion
from model import VAE
from utils import DEVICE


batch_size = 100
lr = 1e-3
epochs = 30
x_dim = 784
hidden_dim = 400
latent_dim = 200

train_loader, val_loader = load_dataset(batch_size)

model = VAE(x_dim, hidden_dim, latent_dim).to(DEVICE)

optimizer = Adam(model.parameters(), lr=lr)

print("Start training VAE...")
model.train()

for epoch in range(epochs):
    overall_loss = 0
    for batch_idx, (x, _) in enumerate(train_loader):
        x = x.view(batch_size, x_dim)
        x = x.to(DEVICE)

        optimizer.zero_grad()

        x_hat, mean, log_var = model(x)
        loss = criterion(x, x_hat, mean, log_var)
        
        overall_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
        torch.save(model.state_dict(), 'weights/last.pt')

        with torch.no_grad():
            noise = torch.randn(1, latent_dim).to(DEVICE)
            out = model.decoder(noise)
        save_image(out.view(1, 1, 28, 28), 'test/output_image.jpg')

        
    print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx*batch_size))
    
print("Finish!!")

