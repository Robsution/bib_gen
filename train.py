import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
import yaml

from src.dataset import BibDataset
from src.models.model import MuonGenerator, DeepSetsDiscriminator, CategoricalOracle
from src.utils import TrainingMonitor  

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def pretrain_oracle(config, dataset, device):
    """ Rapidly trains the Stage 1 Oracle with Early Stopping. """
    print("\n" + "="*50)
    print("STAGE 1: PRE-TRAINING CATEGORICAL ORACLE")
    print("="*50)
    
    max_daughters = config.get("max_daughters", 50)
    oracle = CategoricalOracle(parent_dim=3, max_daughters=max_daughters).to(device)
    optimizer = optim.Adam(oracle.parameters(), lr=1e-3)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    max_epochs = 500
    patience = 10
    best_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(max_epochs):
        epoch_loss = []
        for parent, _, _, scaled_n in dataloader:
            parent = parent.to(device)
            # Revert scaling to get absolute integer classes
            true_n = (scaled_n.to(device) * max_daughters).round().long().squeeze()
            
            optimizer.zero_grad()
            logits = oracle(parent)
            loss = nn.CrossEntropyLoss()(logits, true_n)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
            
        avg_loss = np.mean(epoch_loss)
        print(f"Oracle Epoch [{epoch+1}/{max_epochs}] - Loss: {avg_loss:.4f}")
        
        # --- Early Stopping Check ---
        if avg_loss < best_loss - 1e-4: # Added tiny margin to prevent noisy micro-improvements
            best_loss = avg_loss
            epochs_no_improve = 0
            # Save the best weights in memory temporarily
            best_weights = oracle.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n--> Early stopping triggered! Loss flattened out at {best_loss:.4f}.")
                oracle.load_state_dict(best_weights) # Restore best version
                break
                
    save_path = os.path.join(config["save_dir"], "oracle_weights.pth")
    torch.save(oracle.state_dict(), save_path)
    print(f"Oracle locked and saved to {save_path}\n")
    return oracle



def train(config, resume_epoch=0, retrain_oracle=False):
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(config["save_dir"], exist_ok=True)
    os.makedirs(config["plot_dir"], exist_ok=True)
    
    # 1. Load Data
    print("Loading Dataset...")
    dataset = BibDataset(
        config["data_dir"], 
        min_n=config.get("min_daughters", 1), 
        max_n=config.get("max_daughters", 50)
    )
    dataloader = DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=True, 
        drop_last=True, num_workers=config.get("num_workers", 2)
    )
    
    sample_parent, sample_features, _, _ = dataset[0]
    parent_dim = sample_parent.shape[0]
    feature_dim = sample_features.shape[1]
    max_daughters = config.get("max_daughters", 50)
    kin_dim = 8
    pdg_dim = feature_dim - kin_dim

    # 2. Stage 1: Load or Train the Oracle
    oracle_path = os.path.join(config["save_dir"], "oracle_weights.pth")
    oracle = CategoricalOracle(parent_dim, max_daughters).to(device)
    
    if os.path.exists(oracle_path) and not retrain_oracle:
        print(f"Loading pre-trained Oracle from {oracle_path}...")
        oracle.load_state_dict(torch.load(oracle_path, map_location=device))
    else:
        oracle = pretrain_oracle(config, dataset, device)
        
    oracle.eval() # Freeze the Oracle completely
    oracle.requires_grad_(False)

    # 3. Stage 2: Initialize GAN
    print("\n" + "="*50)
    print(f"STAGE 2: TRAINING CONDITIONED SN-GAN on {device}")
    print("="*50)
    
    generator = MuonGenerator(
        noise_dim=config["z_dim"], parent_dim=parent_dim, max_daughters=max_daughters,
        kin_dim=kin_dim, pdg_dim=pdg_dim, hidden_dim=config.get("g_hidden_dim", 256)
    ).to(device)

    discriminator = DeepSetsDiscriminator(
        feature_dim=feature_dim, parent_dim=parent_dim, hidden_dim=config.get("d_latent_dim", 128)
    ).to(device)

    opt_g = optim.Adam(generator.parameters(), lr=config["lr_g"], betas=(0.0, 0.9))
    opt_d = optim.Adam(discriminator.parameters(), lr=config["lr_d"], betas=(0.0, 0.9))
    monitor = TrainingMonitor(config["plot_dir"])

    if resume_epoch > 0:
        generator.load_state_dict(torch.load(os.path.join(config['save_dir'], f"generator_epoch_{resume_epoch}.pth"), map_location=device))
        discriminator.load_state_dict(torch.load(os.path.join(config['save_dir'], f"discriminator_epoch_{resume_epoch}.pth"), map_location=device))
        print(f"--> Resumed GAN from Epoch {resume_epoch}")

    global_step = resume_epoch * len(dataloader)
    
    for epoch in range(resume_epoch, config["epochs"]):
        epoch_d_loss, epoch_g_loss, epoch_w_dist = [], [], []
        epoch_real_scores, epoch_fake_scores = [], []
        
        for i, (real_parent, real_features, real_mask, real_n) in enumerate(dataloader):
            real_parent = real_parent.to(device)
            real_features = real_features.to(device)
            real_mask = real_mask.to(device)
            real_n = real_n.to(device).float().unsqueeze(1) # [0, 1] scaled
            batch_size = real_parent.size(0)

            # --- THE ORACLE HAND-OFF ---
            # Oracle predicts logits, we softmax to get probs, then roll a loaded dice
            with torch.no_grad():
                oracle_logits = oracle(real_parent)
                oracle_probs = torch.softmax(oracle_logits, dim=-1)
                # target_n_int will correctly be ~50/50 for 2 and 4 based on learned physics
                target_n_int = torch.multinomial(oracle_probs, num_samples=1).float()
                target_n = target_n_int / max_daughters # Scale to [0, 1]

            # --- 1. Train Discriminator ---
            z = torch.randn(batch_size, config["z_dim"]).to(device)
            with torch.no_grad():
                fake_features, fake_mask = generator(z, real_parent, target_n)

            # D evaluates Real (with Real N) and Fake (with Target N)
            d_real = discriminator(real_features, real_parent, real_mask, real_n)
            d_fake = discriminator(fake_features, real_parent, fake_mask, target_n)
            
            # Hinge Loss
            loss_d = F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()

            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()
            
            epoch_d_loss.append(loss_d.item())
            epoch_w_dist.append((d_real.mean() - d_fake.mean()).item()) 
            epoch_real_scores.append(d_real.mean().item())
            epoch_fake_scores.append(d_fake.mean().item())

            # --- 2. Train Generator ---
            if global_step % config.get("n_critic", 1) == 0:
                z = torch.randn(batch_size, config["z_dim"]).to(device)
                fake_features, fake_mask = generator(z, real_parent, target_n)
                d_fake = discriminator(fake_features, real_parent, fake_mask, target_n)
                
                loss_g = -d_fake.mean()
                
                opt_g.zero_grad()
                loss_g.backward()
                opt_g.step()
                epoch_g_loss.append(loss_g.item())

            global_step += 1
            
        # Logging & Saving...
        avg_d_loss = np.mean(epoch_d_loss)
        avg_g_loss = np.mean(epoch_g_loss) if epoch_g_loss else 0
        
        print(f"[Epoch {epoch+1}/{config['epochs']}] D Loss: {avg_d_loss:.4f} | G Loss: {avg_g_loss:.4f} | Pseudo W-Dist: {np.mean(epoch_w_dist):.4f}")
        
        monitor.update(d_loss=avg_d_loss, g_loss=avg_g_loss, w_dist=np.mean(epoch_w_dist), real_score=np.mean(epoch_real_scores), fake_score=np.mean(epoch_fake_scores))
        monitor.plot_stats(epoch+1)
        
        if (epoch + 1) % 50 == 0:
            torch.save(generator.state_dict(), os.path.join(config['save_dir'], f"generator_epoch_{epoch+1}.pth"))
            torch.save(discriminator.state_dict(), os.path.join(config['save_dir'], f"discriminator_epoch_{epoch+1}.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--resume_epoch', type=int, default=0)
    parser.add_argument('--retrain_oracle', action='store_true', help='Force the Oracle to re-train from scratch')
    args = parser.parse_args()
    
    config = load_config(args.config)
    train(config, resume_epoch=args.resume_epoch, retrain_oracle=args.retrain_oracle)