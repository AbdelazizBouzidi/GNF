import torch
import argparse
import json
import time
import os
import yaml
from models.network import NeuralFieldNetwork
from models.encoder import get_encoder
from models.decoder import get_decoder
from utils import seed_everything
from dataLoader.datasets import ImageDataset
from trainer import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--config', type=str, default='configs/rgb.yaml', help="path to config file")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--encoder', type=str, default='dif_grid', choices=['dif_grid', 'hashgrid', 'kplanes', 'hybrid-kplanes'], help="encoder type")
    parser.add_argument('--decoder', type=str, default='rbf', choices=['rbf', 'mlp'], help="decoder type")

    opt = parser.parse_args()
    print(opt)

    seed_everything(opt.seed)

    # Load config (YAML with comments)
    with open(opt.config, 'r') as f:
        config = yaml.safe_load(f)

    train_cfg = config.get('training', {})
    train_size = train_cfg.get('train_size', 15000)
    train_samples = train_cfg.get('train_num_samples', 262144)
    valid_size = train_cfg.get('valid_size', 1)
    valid_samples = train_cfg.get('valid_num_samples', 32768)
    n_iters = train_cfg.get('n_iters', 1)
    vis_interval = train_cfg.get('vis_interval', 1)

    workspace = os.path.join('trials', opt.workspace)
    print(config)

    train_dataset = ImageDataset(opt.path, size=train_size, num_samples=train_samples)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False,
                                            num_workers=8,
                                            persistent_workers=True,
                                            pin_memory=True)
    
    valid_dataset = ImageDataset(opt.path, size=valid_size, num_samples=valid_samples)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1,
                                            num_workers=8,
                                            persistent_workers=True,
                                            pin_memory=True)

    # Initialize encoder
    encoder = get_encoder(opt.encoder, config['model'])
    
    # Initialize decoder
    decoder = get_decoder(opt.decoder, config['model'], encoder.output_dim)
    
    # Initialize model
    model = NeuralFieldNetwork(encoder, decoder)
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {total_params}")

    if opt.test:
        t0 = time.time()
        trainer = Trainer(t0,'ngp', model, workspace=workspace, fp16=opt.fp16, use_checkpoint='best', eval_interval=1)
        trainer.save_image(os.path.join(opt.workspace, 'results', 'output.png'), 
                         resolution_x=train_dataset.w, resolution_y=train_dataset.h)
        t1 = time.time()
        print(t1-t0)
    else:
        criterion = torch.nn.L1Loss()
        
        scheduler = lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1 ** (1 / 15000))
        optimizer = lambda model: torch.optim.Adam([
            {'name': 'encoding', 'params': model.parameters()}], 
            lr=1e-2)

        t0 = time.time()
        trainer = Trainer(t0,'ngp', model, workspace=workspace, criterion=criterion, optimizer=optimizer,
                        ema_decay=None, fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint='latest', 
                        eval_interval=vis_interval, scheduler_update_every_step=True)

        trainer.train(train_loader, valid_loader, n_iters)

        t1 = time.time()
        print(t1-t0)
        
        trainer.save_image(os.path.join(workspace, 'results', 'output.png'), 
                         resolution_x=train_dataset.w, resolution_y=train_dataset.h)

        config_save_path = os.path.join(workspace, 'config.json')
        with open(config_save_path, 'w') as f:
            json.dump(config, f, indent=4) 