import os
import glob
import tqdm
import random
import warnings
import tensorboardX
from PIL import Image

import numpy as np
import pandas as pd

import time
from datetime import datetime

import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

import trimesh
import mcubes
from rich.console import Console
from torch_ema import ExponentialMovingAverage

import packaging
from skimage import color

from torchmetrics import PeakSignalNoiseRatio
from loss import cie94_loss

from models.utils import extract_fields, extract_geometry, custom_meshgrid

class Trainer(object):
    def __init__(self, 
                 t0,
                 name, # name of this experiment
                 model, # network 
                 criterion=None, # loss function, if None, assume inline implementation in train_step
                 optimizer=None, # optimizer
                 ema_decay=None, # if use EMA, set the decay
                 lr_scheduler=None, # scheduler
                 metrics=[], # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0, # which GPU am I
                 world_size=1, # total num of GPUs
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 mute=False, # whether to mute all print
                 fp16=False, # amp optimize level
                 eval_interval=1, # eval once every $ epoch
                 max_keep_ckpt=2, # max num of saved ckpts in disk
                 workspace='workspace', # workspace to save logs & ckpts
                 best_mode='min', # the smaller/larger result, the better
                 use_loss_as_metric=True, # use loss as the first metirc
                 report_metric_at_train=False, # also report metrics at training
                 use_checkpoint="latest", # which ckpt to use at init time
                 use_tensorboardX=True, # whether to use tensorboard for logging
                 scheduler_update_every_step=False, # whether to call scheduler.step() after every train step
                 ):
        self.t0 = t0
        self.name = name
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()
        self.densification_points = []

        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        if optimizer is None:
            self.optimizer = optim.Adam(model.parameters(), lr=1e-3)
        else:
            self.optimizer = optimizer(self.model)
        self.lr_scheduler_setter = lr_scheduler

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1)
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [], # metrics[0], or valid_loss
            "checkpoints": [], # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)        
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth.tar"
            os.makedirs(self.ckpt_path, exist_ok=True)
            
        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else: # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)

    def __del__(self):
        if self.log_ptr: 
            self.log_ptr.close()

    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute: 
                self.console.print(*args, **kwargs)
            if self.log_ptr: 
                print(*args, file=self.log_ptr)
                self.log_ptr.flush()

    def train_step(self, data):
        # Check if this is an SDF dataset or Image dataset
        if 'points' in data and 'sdfs' in data:
            # SDF case
            X = data["points"][0] # [B, 3]
            y = data["sdfs"][0] # [B]

            pred = self.model(X)
            
            mse = torch.mean((pred - y) ** 2)
            psnr = -10 * torch.log(mse) / torch.log(torch.tensor(10.0))       
            
            data_term = self.criterion(pred, y) 
            loss = data_term

        else:
            # Image case
            X = data["coords"][0] # [B, 2]
            y = data["rgbs"][0] # [B, 3]

            pred = self.model(X)
            
            mse = torch.mean((pred - y) ** 2)
            psnr = -10 * torch.log(mse) / torch.log(torch.tensor(10.0))       
            
            data_term = self.criterion(pred, y) 
            loss = data_term

        return pred, y, loss, 0, data_term, psnr

    def eval_step(self, data):
        return self.train_step(data)

    def test_step(self, points):  
        X = points
        pred = self.model(X)
        return pred

    def save_image(self, save_path=None, resolution_x=1024, resolution_y=1024, batch_size=10240, ground_truth_path=None):
        if save_path is None:
            save_path = os.path.join(self.workspace, 'validation', f'{self.name}_{self.epoch}.png')
        
        self.log(f"==> Saving image to {save_path}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        def query_func(pts):
            pts = pts.to(self.device)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    labs = self.model(pts)
            return labs
        
        def mse(image_pred, image_gt, valid_mask=None, reduction='mean'):
            value = (image_pred - image_gt) ** 2
            if valid_mask is not None:
                value = value[valid_mask]
            if reduction == 'mean':
                return value.mean()
            return value

        def psnr(image_pred, image_gt, valid_mask=None, reduction='mean', vmin=0, vmax=1):
            if torch.is_tensor(image_pred):
                return -10 * torch.log10(mse(image_pred, image_gt, valid_mask, reduction))
            elif isinstance(image_pred, np.ndarray):
                return -10 * np.log10(mse(image_pred, image_gt, valid_mask, reduction))

        # Define bounds for the image coordinates
        bounds_min = torch.FloatTensor([-1.0, -1.0])
        bounds_max = torch.FloatTensor([1.0, 1.0])
        
        # Create a grid of points over the image plane
        x = torch.linspace(bounds_min[0], bounds_max[0], resolution_x)
        y = torch.linspace(bounds_min[1], bounds_max[1], resolution_y)
        xs, ys = custom_meshgrid(x, y)
        pts = torch.stack([xs, ys], dim=-1).reshape(-1, 2)
        
        # Initialize an array to hold LAB values
        rgbs = np.zeros((pts.shape[0], 3), dtype=np.float32)
        start_time = time.time()

        # Process points in batches
        for start in range(0, pts.shape[0], batch_size):
            end = min(start + batch_size, pts.shape[0])
            batch_pts = pts[start:end]
            
            # Query the model at these points
            rgb_batch = query_func(batch_pts)
            rgbs[start:end] = rgb_batch.cpu().numpy()

        end_time = time.time()
        total_time = end_time - start_time

        print(f"Total time taken: {total_time:.2f} seconds")
        rgb_values = rgbs * 255
        rgb_image = rgb_values.reshape(resolution_x, resolution_y, 3)
        rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)

        generated_image = Image.fromarray(rgb_image)
        generated_image.save(save_path)
        self.log(f"==> Finished saving image.")

    def save_mesh(self, save_path=None, train_dataset=None, resolution=256, thres=0.5, ground_truth_occ=None):
        if save_path is None:
            save_path = os.path.join(self.workspace, 'validation', f'{self.name}_{self.epoch}.ply')
        
        self.log(f"==> Saving mesh to {save_path}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        def query_func(pts):
            pts = pts.to(self.device)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    sdfs = self.model(pts)
            return sdfs

        bounds_min = torch.FloatTensor([-0.9, -0.9, -0.9])
        bounds_max = torch.FloatTensor([0.9, 0.9, 0.9])

        # Generate the mesh
        t0 = time.time()
        vertices, triangles = extract_geometry(bounds_min, bounds_max, resolution=resolution, threshold=0, query_func=query_func)
        t1 = time.time()
        print("inference_time:", t1-t0)

        # Save the mesh
        mesh = trimesh.Trimesh(vertices, triangles, process=False)
        mesh.export(save_path)
        self.log(f"==> Finished saving mesh.")

        if train_dataset is not None:
            # Calculate IoU by generating an occupancy grid
            grid_pts = torch.stack(torch.meshgrid(
                torch.linspace(-0.9, 0.9, 256),
                torch.linspace(-0.9, 0.9, 256),
                torch.linspace(-0.9, 0.9, 256)
            ), -1).reshape(-1, 3)

            # Get SDF values for the grid points
            sdfs = query_func(grid_pts)
            predicted_occ = sdfs <= 0

            def iou(occ1, occ2):
                occ1 = occ1.reshape(occ2.shape)
                area_union = (occ1 | occ2).sum()
                area_intersect = (occ1 & occ2).sum()
                return area_intersect/(area_union+1e-8)

        # Generate and save contour plots
        x = torch.linspace(bounds_min[0].item(), bounds_max[0].item(), 128)
        y = torch.linspace(bounds_min[1].item(), bounds_max[1].item(), 128)
        z = torch.linspace(bounds_min[2].item(), bounds_max[2].item(), 128)
        
        X, Y, Z = custom_meshgrid(x, y, z)
        grid_points = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3).to(self.device)

        # Sample SDF values
        sdf_values = self.model(grid_points).view(128, 128, 128).detach().cpu().numpy()

        # Generate and save contour plots
        save_dir = os.path.join(save_path[:-10], 'sdf_contours')
        os.makedirs(save_dir, exist_ok=True)

        def save_contour(image, axis_name, levels):
            X_cpu = X[:, 0, 0].cpu().numpy()
            Y_cpu = Y[0, :, 0].cpu().numpy()
            image_cpu = image.cpu().numpy()

            plt.figure(figsize=(9, 9))
            plt.contour(X_cpu, Y_cpu, image_cpu.T, levels=levels, cmap='viridis')
            plt.colorbar()
            plt.tight_layout()
            plt.xlabel(axis_name[0])
            plt.ylabel(axis_name[1])
            plt.savefig(os.path.join(save_dir, f'{axis_name}_contour_{self.epoch:08d}.png'))
            plt.close()

        # Define the levels to visualize
        levels = [-0.1, -0.05, -0.02, -0.01, -0.001, 0, 0.001, 0.01, 0.1, 0.2, 0.3]

        # XY Contour
        xy_contour = sdf_values[:, :, 128 // 2]
        save_contour(torch.tensor(xy_contour), 'xy', levels)

        # XZ Contour
        xz_contour = sdf_values[:, 128 // 2, :]
        save_contour(torch.tensor(xz_contour), 'xz', levels)

        # YZ Contour
        yz_contour = sdf_values[128 // 2, :, :]
        save_contour(torch.tensor(yz_contour), 'yz', levels)

        self.log(f"==> Finished saving mesh and contours.")

    def train(self, train_loader, valid_loader, max_epochs):
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))
        self.densification_points = []
        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch

            self.train_one_epoch(train_loader)

            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)

            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                self.save_mesh()
                self.save_checkpoint(full=False, best=True)

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader)
        self.use_tensorboardX = use_tensorboardX

    def prepare_data(self, data):
        if isinstance(data, list):
            for i, v in enumerate(data):
                if isinstance(v, np.ndarray):
                    data[i] = torch.from_numpy(v).to(self.device, non_blocking=True)
                if torch.is_tensor(v):
                    data[i] = v.to(self.device, non_blocking=True)
        elif isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    data[k] = torch.from_numpy(v).to(self.device, non_blocking=True)
                if torch.is_tensor(v):
                    data[k] = v.to(self.device, non_blocking=True)
        elif isinstance(data, np.ndarray):
            data = torch.from_numpy(data).to(self.device, non_blocking=True)
        else: # is_tensor, or other similar objects that has `to`
            data = data.to(self.device, non_blocking=True)
        return data

    def train_one_epoch(self, loader):
        self.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        total_psnr = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)
        
        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        for data in loader:
            self.local_step += 1
            self.global_step += 1
            
            data = self.prepare_data(data)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, truths, loss, _, _, psnr = self.train_step(data)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad() 
            if self.ema is not None:
                self.ema.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val
            total_psnr += psnr.item()

            if self.local_rank == 0:
                if self.report_metric_at_train:
                    for metric in self.metrics:
                        metric.update(preds, truths)
                        
                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/psnr", psnr, self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

                if self.scheduler_update_every_step:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                pbar.update(loader.batch_size)

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")

    def evaluate_one_epoch(self, loader):
        self.log(f"++> Evaluate at epoch {self.epoch} ...")

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0
        for data in loader:    
            self.local_step += 1
            
            data = self.prepare_data(data)

            if self.ema is not None:
                self.ema.store()
                self.ema.copy_to()
        
            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, truths, loss, eik_loss, _, _= self.eval_step(data)

            if self.ema is not None:
                self.ema.restore()
            
            if self.world_size > 1:
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                loss = loss / self.world_size
                
                preds_list = [torch.zeros_like(preds).to(self.device) for _ in range(self.world_size)]
                dist.all_gather(preds_list, preds)
                preds = torch.cat(preds_list, dim=0)

                truths_list = [torch.zeros_like(truths).to(self.device) for _ in range(self.world_size)]
                dist.all_gather(truths_list, truths)
                truths = torch.cat(truths_list, dim=0)

            loss_val = loss.item()
            total_loss += loss_val

            if self.local_rank == 0:
                for metric in self.metrics:
                    metric.update(preds, truths)

                pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}) ({eik_loss/self.local_step:.4f})")
                pbar.update(loader.batch_size)

        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric:
                for metric in self.metrics:
                    self.log(metric.report(), style="blue")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="evaluate")
                    metric.clear()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    def save_checkpoint(self, full=False, best=False):
        if self.local_rank == 0:
            if full:
                save_path = os.path.join(self.ckpt_path, f'{self.name}_ep{self.epoch:04d}.pth.tar')
            else:
                save_path = os.path.join(self.ckpt_path, f'{self.name}_latest.pth.tar')

            if best:
                save_path = self.best_path

            self.stats["checkpoints"].append(save_path)

            if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                old_ckpt = self.stats["checkpoints"].pop(0)
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)

            torch.save({
                'epoch': self.epoch,
                'global_step': self.global_step,
                'stats': self.stats,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                'scaler': self.scaler.state_dict(),
            }, save_path)

            self.log(f"[INFO] Saved checkpoint: {save_path}")

    def load_checkpoint(self, checkpoint=None):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/{self.name}_ep*.pth.tar'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)
        
        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")   

        if self.ema is not None and 'ema' in checkpoint_dict:
            self.ema.load_state_dict(checkpoint_dict['ema'])

        if self.optimizer and 'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")

        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")

        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}") 