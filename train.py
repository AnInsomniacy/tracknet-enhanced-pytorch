"""
TrackNet Training Script

Usage Examples:
python train.py --train train_data/train.h5 --val train_data/val.h5
python train.py --resume best.pth --train train_data/train.h5 --val train_data/val.h5 --optimizer Adam --epochs 100
python train.py --train train_data/train.h5 --val train_data/val.h5 --batch 8 --epochs 50 --lr 0.001 --wd 0.0001 --optimizer Adam --scheduler ReduceLROnPlateau --factor 0.5 --patience 3 --min_lr 1e-6 --plot 5 --out outputs --name experiment

Parameters:
--train: Training dataset HDF5 file path (required)
--val: Validation dataset HDF5 file path (required)
--resume: Checkpoint path for resuming
--batch: Batch size (default: 3)
--epochs: Training epochs (default: 30)
--workers: Data loader workers (default: 0)
--device: Device auto/cpu/cuda/mps (default: auto)
--optimizer: Adadelta/Adam/AdamW/SGD (default: Adadelta)
--lr: Learning rate (default: auto per optimizer)
--wd: Weight decay (default: 0)
--scheduler: ReduceLROnPlateau/None (default: ReduceLROnPlateau)
--factor: LR reduction factor (default: 0.5)
--patience: LR reduction patience (default: 3)
--min_lr: Minimum learning rate (default: 1e-6)
--plot: Loss plot interval (default: 1)
--out: Output directory (default: outputs)
--name: Experiment name (default: exp)
"""

import argparse
import json
import signal
import time
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.loss import WeightedBinaryCrossEntropy
from preprocess.hdf5_dataset import HDF5FrameHeatmapDataset
from model.tracknet_enhanced import TrackNet


def parse_args():
    parser = argparse.ArgumentParser(description="TrackNet Training")
    parser.add_argument('--train', type=str, required=True)
    parser.add_argument('--val', type=str, required=True)
    parser.add_argument('--resume', type=str)
    parser.add_argument('--batch', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--optimizer', type=str, default='Adadelta',
                        choices=['Adadelta', 'Adam', 'AdamW', 'SGD'])
    parser.add_argument('--lr', type=float)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--scheduler', type=str, default='ReduceLROnPlateau',
                        choices=['ReduceLROnPlateau', 'None'])
    parser.add_argument('--factor', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--plot', type=int, default=1)
    parser.add_argument('--out', type=str, default='outputs')
    parser.add_argument('--name', type=str, default='exp')

    args = parser.parse_args()

    if args.lr is None:
        lr_defaults = {'Adadelta': 1.0, 'Adam': 0.001, 'AdamW': 0.001, 'SGD': 0.01}
        args.lr = lr_defaults[args.optimizer]

    return args


class Trainer:
    def __init__(self, args):
        self.args = args
        self.start_epoch = 0
        self.interrupted = False
        self.best_loss = float('inf')
        self.device = self._get_device()
        self._setup_dirs()
        self.checkpoint = None
        self.recovery_mode = None
        self._load_checkpoint()
        self.losses = {'batch': [], 'steps': [], 'lrs': [], 'train': [], 'val': []}
        self.step = 0
        signal.signal(signal.SIGINT, self._interrupt)
        signal.signal(signal.SIGTERM, self._interrupt)

    def _get_device(self):
        if self.args.device == 'auto':
            if torch.backends.mps.is_available():
                return torch.device('mps')
            elif torch.cuda.is_available():
                return torch.device('cuda')
            else:
                return torch.device('cpu')
        return torch.device(self.args.device)

    def _setup_dirs(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "_resumed" if self.args.resume else ""
        self.save_dir = Path(self.args.out) / f"{self.args.name}{suffix}_{timestamp}"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / "checkpoints").mkdir(exist_ok=True)
        (self.save_dir / "plots").mkdir(exist_ok=True)
        with open(self.save_dir / "config.json", 'w') as f:
            json.dump(vars(self.args), f, indent=2)

    def _load_checkpoint(self):
        if not self.args.resume:
            return

        path = Path(self.args.resume)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        self.checkpoint = torch.load(path, map_location='cpu')
        self.start_epoch = self.checkpoint['epoch'] + (0 if self.checkpoint.get('is_emergency', False) else 1)

        checkpoint_optimizer = self.checkpoint.get('optimizer_type', 'Unknown')
        if checkpoint_optimizer == self.args.optimizer:
            self.recovery_mode = 'strict'
        else:
            self.recovery_mode = 'weights_only'

    def _interrupt(self, signum, frame):
        print("\n\033[91mInterrupt detected\033[0m, saving emergency checkpoint...")
        self.interrupted = True

    def _calculate_effective_lr(self):
        if self.args.optimizer == 'Adadelta':
            if not hasattr(self.optimizer, 'state') or not self.optimizer.state:
                return self.args.lr

            effective_lrs = []
            eps = self.optimizer.param_groups[0].get('eps', 1e-6)

            for group in self.optimizer.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    state = self.optimizer.state[p]
                    if len(state) == 0:
                        continue

                    square_avg = state.get('square_avg')
                    acc_delta = state.get('acc_delta')

                    if square_avg is not None and acc_delta is not None:
                        if torch.is_tensor(square_avg) and torch.is_tensor(acc_delta):
                            rms_delta = (acc_delta + eps).sqrt().mean()
                            rms_grad = (square_avg + eps).sqrt().mean()
                            if rms_grad > eps:
                                effective_lr = self.args.lr * rms_delta / rms_grad
                                effective_lrs.append(effective_lr.item())

            if effective_lrs:
                avg_lr = sum(effective_lrs) / len(effective_lrs)
                return max(avg_lr, eps)
            else:
                return self.args.lr
        else:
            return self.optimizer.param_groups[0]['lr']

    def _display_init_info(self):
        print("\nTrackNet Training Initialized")

        device_name = self.device.type.upper()
        if self.device.type == 'cuda':
            device_info = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"Device: \033[93m{device_name}\033[0m ({device_info}, \033[94m{memory_gb:.0f}GB\033[0m)")
        else:
            print(f"Device: \033[93m{device_name}\033[0m")

        total_params = sum(p.numel() for p in self.model.parameters())
        params_str = f"{total_params / 1e6:.1f}M" if total_params >= 1e6 else f"{total_params / 1e3:.1f}K"
        print(f"Model: TrackNet (\033[94m{params_str}\033[0m params)")

        print(
            f"Optimizer: \033[93m{self.args.optimizer}\033[0m (lr=\033[94m{self.args.lr}\033[0m, wd=\033[94m{self.args.wd}\033[0m)")

        train_size = len(self.train_loader.dataset)
        val_size = len(self.val_loader.dataset)
        print(f"Data: Train \033[94m{train_size:,}\033[0m | Val \033[94m{val_size:,}\033[0m samples")

        remaining_epochs = self.args.epochs - self.start_epoch
        total_steps = remaining_epochs * len(self.train_loader)
        print(
            f"Training: \033[94m{self.args.epochs}\033[0m epochs × \033[94m{len(self.train_loader)}\033[0m batches → \033[94m{total_steps:,}\033[0m steps")

        print(f"Output: {self.save_dir}")

        if not self.args.resume:
            print(f"Status: \033[92mNew Training\033[0m\n")
        else:
            if self.recovery_mode == 'strict':
                print(f"Status: Resume epoch \033[93m{self.start_epoch + 1}\033[0m (\033[92mFull Recovery\033[0m)\n")
            else:
                checkpoint_optimizer = self.checkpoint.get('optimizer_type', 'Unknown')
                print(
                    f"Status: Resume epoch \033[93m{self.start_epoch + 1}\033[0m (\033[91mWeights Only\033[0m - Optimizer Changed: \033[93m{checkpoint_optimizer}\033[0m→\033[93m{self.args.optimizer}\033[0m)\n")

    def setup_data(self):
        train_dataset = HDF5FrameHeatmapDataset(self.args.train)
        val_dataset = HDF5FrameHeatmapDataset(self.args.val)

        self.train_loader = DataLoader(train_dataset, batch_size=self.args.batch, shuffle=True,
                                       num_workers=self.args.workers, pin_memory=self.device.type == 'cuda')
        self.val_loader = DataLoader(val_dataset, batch_size=self.args.batch, shuffle=False,
                                     num_workers=self.args.workers, pin_memory=self.device.type == 'cuda')

    def _create_optimizer(self):
        optimizers = {
            'Adadelta': lambda: torch.optim.Adadelta(self.model.parameters(), lr=self.args.lr,
                                                     weight_decay=self.args.wd),
            'Adam': lambda: torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd),
            'AdamW': lambda: torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd),
            'SGD': lambda: torch.optim.SGD(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd,
                                           momentum=0.9)
        }
        return optimizers[self.args.optimizer]()

    def setup_model(self):
        self.model = TrackNet().to(self.device)
        self.criterion = WeightedBinaryCrossEntropy()
        self.optimizer = self._create_optimizer()

        if self.args.scheduler == "ReduceLROnPlateau":
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=self.args.factor,
                                               patience=self.args.patience, min_lr=self.args.min_lr)
        else:
            self.scheduler = None

        if self.checkpoint:
            self.model.load_state_dict(self.checkpoint['model_state_dict'])

            if self.recovery_mode == 'strict':
                if 'optimizer_state_dict' in self.checkpoint:
                    self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])

                if self.scheduler and 'scheduler_state_dict' in self.checkpoint:
                    self.scheduler.load_state_dict(self.checkpoint['scheduler_state_dict'])

                if 'best_loss' in self.checkpoint:
                    self.best_loss = self.checkpoint['best_loss']

    def save_checkpoint(self, epoch, train_loss, val_loss, is_emergency=False):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'optimizer_type': self.args.optimizer,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'is_emergency': is_emergency,
            'history': self.losses.copy(),
            'step': self.step,
            'best_loss': self.best_loss,
            'timestamp': timestamp
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        prefix = "emergency_" if is_emergency else "checkpoint_"
        filename = f"{prefix}epoch_{epoch + 1}_{timestamp}.pth"
        filepath = self.save_dir / "checkpoints" / filename
        torch.save(checkpoint, filepath)

        if not is_emergency and val_loss < self.best_loss:
            self.best_loss = val_loss
            torch.save(checkpoint, self.save_dir / "checkpoints" / "best_model.pth")
            print(f"Checkpoint saved: {filename} (\033[92mBest model updated\033[0m)")
            return filepath, True

        print(f"Checkpoint saved: {filename}")
        return filepath, False

    def plot_curves(self, epoch):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        if self.losses['batch']:
            ax1.plot(self.losses['steps'], self.losses['batch'], 'b-', alpha=0.3, label='Batch Loss')
        if self.losses['train']:
            epochs = list(range(1, len(self.losses['train']) + 1))
            ax1.plot(epochs, self.losses['train'], 'bo-', label='Train')
            ax1.plot(epochs, self.losses['val'], 'ro-', label='Val')

        ax1.set_xlabel('Batch/Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        if self.losses['lrs']:
            ax2.plot(self.losses['steps'], self.losses['lrs'], 'g-')
            ax2.set_xlabel('Batch')
            ax2.set_ylabel('Learning Rate')
            ax2.set_title('Learning Rate')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.save_dir / "plots" / f"epoch_{epoch + 1}.png", dpi=150, bbox_inches='tight')
        plt.close()

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            val_pbar = tqdm(total=len(self.val_loader), desc="Validation", ncols=100)
            for batch_idx, (inputs, targets) in enumerate(self.val_loader):
                if self.interrupted:
                    val_pbar.close()
                    break
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

                val_pbar.update(1)
                val_pbar.set_postfix({'loss': f'{loss.item():.6f}'})
            val_pbar.close()

        avg_loss = total_loss / len(self.val_loader)
        return avg_loss

    def train(self):
        self.setup_data()
        self.setup_model()
        self._display_init_info()

        for epoch in range(self.start_epoch, self.args.epochs):
            if self.interrupted: break

            print(f"Epoch \033[95m{epoch + 1}\033[0m/\033[95m{self.args.epochs}\033[0m")
            start_time = time.time()
            self.model.train()
            total_loss = 0.0

            train_pbar = tqdm(total=len(self.train_loader), desc=f"Training", ncols=100)
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                if self.interrupted:
                    train_pbar.close()
                    print("Emergency save triggered...")
                    val_loss = self.validate()
                    self.save_checkpoint(epoch, total_loss / (batch_idx + 1), val_loss, True)
                    self.plot_curves(epoch)
                    return

                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                batch_loss = loss.item()
                total_loss += batch_loss
                self.step += 1

                current_lr = self._calculate_effective_lr()

                if self.step % self.args.plot == 0:
                    self.losses['batch'].append(batch_loss)
                    self.losses['steps'].append(self.step)
                    self.losses['lrs'].append(current_lr)

                train_pbar.update(1)
                train_pbar.set_postfix({'loss': f'{batch_loss:.6f}', 'lr': f'{current_lr:.2e}'})
            train_pbar.close()

            train_loss = total_loss / len(self.train_loader)
            val_loss = self.validate()

            self.losses['train'].append(train_loss)
            self.losses['val'].append(val_loss)

            current_lr = self.optimizer.param_groups[0]['lr']
            elapsed = time.time() - start_time

            print(
                f"Epoch [\033[95m{epoch + 1}\033[0m/\033[95m{self.args.epochs}\033[0m] Train: \033[94m{train_loss:.6f}\033[0m Val: \033[94m{val_loss:.6f}\033[0m "
                f"LR: \033[94m{current_lr:.6e}\033[0m Time: \033[94m{elapsed:.1f}s\033[0m")

            if self.scheduler:
                self.scheduler.step(val_loss)

            _, is_best = self.save_checkpoint(epoch, train_loss, val_loss)
            if is_best:
                print(f"\033[92mNew best model! Val Loss: {val_loss:.6f}\033[0m")

            self.plot_curves(epoch)

        if not self.interrupted:
            print("\n\033[92mTraining completed successfully!\033[0m")
            print(f"\033[92mAll results saved to: {self.save_dir}\033[0m")


if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()
