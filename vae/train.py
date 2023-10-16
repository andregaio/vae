import torch
from torchvision.utils import save_image
from torch.optim import Adam
from data import load_dataset
from loss import criterion
from model import VAE
import wandb
import argparse


def train(args):

    if args.wandb:
        run = wandb.init(
            project='vae',            
            config={
                'model': 'vae',
                'dataset': 'MNIST',
                'epochs': args.epochs,
                'batch' : args.batch,
                'learning_rate': args.learning_rate,
                'x_dim' : args.x_dim,
                'hidden_dim' : args.hidden_dim,
                'latent_dim' : args.latent_dim
            })



    DEVICE = torch.device('cuda')

    train_loader, _ = load_dataset(args.batch)

    model = VAE(args.x_dim, args.hidden_dim, args.latent_dim).to(DEVICE)

    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    model.train()

    for epoch in range(args.epochs):
        overall_loss = 0
        for i, (x, _) in enumerate(train_loader, 0):
            x = x.view(args.batch, args.x_dim)
            x = x.to(DEVICE)

            optimizer.zero_grad()

            x_hat, mean, std = model(x)
            loss, rep, kld = criterion(x, x_hat, mean, std)
            
            overall_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
            torch.save(model.state_dict(), 'weights/last.pt')

            with torch.no_grad():
                model.generate_image()
                            
        print(f'\tEpoch: {epoch + 1} / {args.epochs} \t Loss: {loss:.2f}, \t Rep. loss: {rep:.2f}, \t KLD loss: {kld:.2f}')
    
        if args.wandb:
            wandb.log({'loss': loss,
                        'rep': rep,
                        'kld': kld,
                        'epoch': epoch + 1,
                    })



if __name__ == "__main__":

    parser = argparse.ArgumentParser("Argument parser")
    parser.add_argument("--batch", type = int, default = 100)
    parser.add_argument("--epochs", type = int, default = 30)
    parser.add_argument("--learning_rate", type = float, default = 1e-3)
    parser.add_argument("--x_dim", type = int, default = 784)
    parser.add_argument("--hidden_dim", type = int, default = 400)
    parser.add_argument("--latent_dim", type = int, default = 200)
    parser.add_argument("--wandb", action="store_true", default = False)
    args = parser.parse_args()

    train(args)