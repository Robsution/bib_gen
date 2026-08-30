import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
import yaml

# --- Import Project Modules ---
from src.dataset import BibDataset
from src.models.generator import MuonGenerator
from src.models.discriminator import DeepSetsDiscriminator
from src.utils import TrainingMonitor  

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def train(config, resume_epoch=0):
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Training on {device} | Experiment: {config['experiment_name']}")
    
    os.makedirs(config["save_dir"], exist_ok=True)
    os.makedirs(config["plot_dir"], exist_ok=True)
    
    monitor = TrainingMonitor(config["plot_dir"])

    # 1. Load Data
    # Note: Removed the hardcoded min_n=2, max_n=2 so the GAN actually sees a distribution
    print("Loading Dataset...")
    dataset = BibDataset(
        config["data_dir"], 
        min_n=config.get("min_daughters", 2), 
        max_n=config.get("max_daughters", 2)
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=config["batch_size"], 
        shuffle=True, 
        drop_last=True,
        num_workers=config.get("num_workers", 2)
    )
    
    sample_parent, sample_features, _, _ = dataset[0]
    parent_dim = sample_parent.shape[0]
    feature_dim = sample_features.shape[1]
    max_daughters = sample_features.shape[0]
    
    print(f"Dimensions: Parent={parent_dim}, Features={feature_dim}, Max Daughters={max_daughters}")

    # 2. Initialize Models
    kin_dim = 8
    pdg_dim = feature_dim - kin_dim  # Should be 16 based on your logs

    generator = MuonGenerator(
        noise_dim=config["z_dim"],
        parent_dim=parent_dim,
        max_daughters=max_daughters,
        kin_dim=kin_dim,
        pdg_dim=pdg_dim,
        hidden_dim=config.get("g_hidden_dim", 256) # Passed as integer, not a list
    ).to(device)

    discriminator = DeepSetsDiscriminator(
        feature_dim=feature_dim,
        parent_dim=parent_dim,
        hidden_dim=config.get("d_latent_dim", 128)
    ).to(device)

    # 3. Optimizers
    # CRITICAL: betas=(0.0, 0.9) is required for Spectral Normalization GANs
    opt_g = optim.Adam(generator.parameters(), lr=config["lr_g"], betas=(0.0, 0.9))
    opt_d = optim.Adam(discriminator.parameters(), lr=config["lr_d"], betas=(0.0, 0.9))

    # ==================================================================
    # QUICK RESUME INJECTION
    # ==================================================================
    if resume_epoch > 0:
        gen_path = os.path.join(config['save_dir'], f"generator_epoch_{resume_epoch}.pth")
        disc_path = os.path.join(config['save_dir'], f"discriminator_epoch_{resume_epoch}.pth")
        
        if os.path.exists(gen_path) and os.path.exists(disc_path):
            print(f"--> Resuming seamlessly from weights at Epoch {resume_epoch}...")
            generator.load_state_dict(torch.load(gen_path, map_location=device))
            discriminator.load_state_dict(torch.load(disc_path, map_location=device))
        else:
            raise FileNotFoundError(f"Could not find checkpoint files at {gen_path} or {disc_path}!")

    global_step = resume_epoch * len(dataloader)
    
    for epoch in range(resume_epoch, config["epochs"]):
        epoch_d_loss = []
        epoch_g_loss = []
        epoch_w_dist = []
        epoch_real_scores = []
        epoch_fake_scores = []
        
        for i, (real_parent, real_features, real_mask, real_n) in enumerate(dataloader):
            real_parent = real_parent.to(device)
            real_features = real_features.to(device)
            real_mask = real_mask.to(device)
            
            batch_size = real_parent.size(0)

            # --- 1. Train Discriminator (Critic) ---
            z = torch.randn(batch_size, config["z_dim"]).to(device)
            with torch.no_grad():
                fake_features, fake_mask, _ = generator(z, real_parent)

            d_real = discriminator(real_features, real_parent, real_mask)
            d_fake = discriminator(fake_features, real_parent, fake_mask)
            
            # HINGE LOSS for SN-GAN (Replaces WGAN-GP completely)
            loss_d = F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()

            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()
            
            epoch_d_loss.append(loss_d.item())
            epoch_w_dist.append((d_real.mean() - d_fake.mean()).item()) # Pseudo W-Dist tracking
            
            epoch_real_scores.append(d_real.mean().item())
            epoch_fake_scores.append(d_fake.mean().item())

            # --- 2. Train Generator ---
            if global_step % config.get("n_critic", 1) == 0:
                z = torch.randn(batch_size, config["z_dim"]).to(device)
                fake_features, fake_mask, _ = generator(z, real_parent)
                d_fake = discriminator(fake_features, real_parent, fake_mask)
                
                # Pure Adversarial Hinge Generator Loss
                loss_g = -d_fake.mean()
                
                opt_g.zero_grad()
                loss_g.backward()
                opt_g.step()
                
                epoch_g_loss.append(loss_g.item())

            global_step += 1
            
        # --- End of Epoch Logging ---
        avg_d_loss = np.mean(epoch_d_loss)
        avg_g_loss = np.mean(epoch_g_loss) if epoch_g_loss else 0
        avg_w_dist = np.mean(epoch_w_dist)
        avg_real_score = np.mean(epoch_real_scores)
        avg_fake_score = np.mean(epoch_fake_scores)
        
        print(
            f"[Epoch {epoch+1}/{config['epochs']}] "
            f"D Loss: {avg_d_loss:.4f} | G Loss: {avg_g_loss:.4f} | "
            f"Pseudo W-Dist: {avg_w_dist:.4f}"
        )
        
        # Zeroing out grad_norm and aux_loss since they are mathematically obsolete here
        monitor.update(
            d_loss=avg_d_loss, 
            g_loss=avg_g_loss, 
            w_dist=avg_w_dist, 
            real_score=avg_real_score, 
            fake_score=avg_fake_score
        )

        monitor.plot_stats(epoch+1)
        monitor.plot_roc(generator, discriminator, dataloader, config, epoch+1, device)
        
        if (epoch + 1) % 50 == 0:
            torch.save(generator.state_dict(), os.path.join(config['save_dir'], f"generator_epoch_{epoch+1}.pth"))
            torch.save(discriminator.state_dict(), os.path.join(config['save_dir'], f"discriminator_epoch_{epoch+1}.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--resume_epoch', type=int, default=0, help='Epoch checkpoint to resume from')
    args = parser.parse_args()
    
    config = load_config(args.config)
    train(config, resume_epoch=args.resume_epoch)