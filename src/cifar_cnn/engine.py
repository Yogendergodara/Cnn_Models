import logging
import math
import time
from pathlib import Path

import torch
import torch.nn as nn

from .config import Config
from .data import mixup_cutmix  # optional, see Data pipeline

log = logging.getLogger(__name__)


def param_groups(model: nn.Module, weight_decay: float):
    """No weight decay on BatchNorm parameters and biases."""
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def build_optimizer(model, cfg):
    groups = param_groups(model, cfg.weight_decay)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(groups, lr=cfg.lr, momentum=cfg.momentum, nesterov=True)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(groups, lr=cfg.lr)
    raise ValueError(f"Unknown optimizer {cfg.optimizer}")


class Trainer:
    def __init__(self, model, train_loader, val_loader, cfg: Config, device, run_dir: Path, tracker=None):
        self.cfg, self.tc = cfg, cfg.train
        self.device = device
        self.model = model.to(device, memory_format=torch.channels_last)
        if self.tc.compile:
            self.model = torch.compile(self.model)
        self.train_loader, self.val_loader = train_loader, val_loader
        self.run_dir = run_dir
        self.tracker = tracker

        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.tc.label_smoothing)
        self.optimizer = build_optimizer(self.model, self.tc)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.tc.lr, epochs=self.tc.epochs,
            steps_per_epoch=len(train_loader), pct_start=0.15,
        )
        self.use_amp = self.tc.amp and device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if self.use_amp and torch.cuda.is_bf16_supported() else torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp and self.amp_dtype == torch.float16)

        self.start_epoch, self.best_acc, self.bad_epochs = 0, 0.0, 0
        self.history = []

    # ---------- one epoch ----------
    def _train_one_epoch(self, epoch: int):
        self.model.train()
        total_loss, correct, seen = 0.0, 0, 0
        alpha = self.cfg.data.mixup_alpha
        for images, labels in self.train_loader:
            images = images.to(self.device, non_blocking=True, memory_format=torch.channels_last)
            labels = labels.to(self.device, non_blocking=True)
            targets = labels
            if alpha > 0:
                images, targets = mixup_cutmix(images, labels, alpha, self.cfg.model.num_classes)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, targets)

            if not math.isfinite(loss.item()):
                raise RuntimeError(f"Loss is {loss.item()} at epoch {epoch}; stopping")

            self.scaler.scale(loss).backward()
            if self.tc.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.tc.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()                    # OneCycle steps every batch

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            seen += labels.size(0)
        return total_loss / seen, correct / seen

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        total_loss, correct, seen = 0.0, 0, 0
        plain_ce = nn.CrossEntropyLoss()
        for images, labels in loader:
            images = images.to(self.device, non_blocking=True, memory_format=torch.channels_last)
            labels = labels.to(self.device, non_blocking=True)
            with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                logits = self.model(images)
            total_loss += plain_ce(logits.float(), labels).item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            seen += labels.size(0)
        return total_loss / seen, correct / seen

    # ---------- checkpoints ----------
    def _state(self, epoch):
        model = getattr(self.model, "_orig_mod", self.model)   # unwrap torch.compile
        return {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "best_acc": self.best_acc,
            "config": self.cfg.to_dict(),
        }

    def save(self, epoch, name):
        path = self.run_dir / name
        tmp = path.with_suffix(".tmp")
        torch.save(self._state(epoch), tmp)
        tmp.replace(path)          # atomic: a crash mid-save never corrupts the file

    def resume(self, path: Path):
        ckpt = torch.load(path, map_location=self.device)
        getattr(self.model, "_orig_mod", self.model).load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.start_epoch, self.best_acc = ckpt["epoch"] + 1, ckpt["best_acc"]
        log.info("Resumed from %s at epoch %d (best val acc %.4f)", path, self.start_epoch, self.best_acc)

    # ---------- main loop ----------
    def fit(self):
        for epoch in range(self.start_epoch, self.tc.epochs):
            t0 = time.perf_counter()
            train_loss, train_acc = self._train_one_epoch(epoch)
            val_loss, val_acc = self.evaluate(self.val_loader)
            lr = self.scheduler.get_last_lr()[0]
            row = dict(epoch=epoch + 1, train_loss=train_loss, train_acc=train_acc,
                       val_loss=val_loss, val_acc=val_acc, lr=lr, secs=time.perf_counter() - t0)
            self.history.append(row)
            log.info("epoch %(epoch)d | train %(train_loss).4f / %(train_acc).4f | "
                     "val %(val_loss).4f / %(val_acc).4f | lr %(lr).5f | %(secs).1fs", row)
            if self.tracker:
                self.tracker.log_metrics(row, step=epoch + 1)

            self.save(epoch, "last.pt")
            if val_acc > self.best_acc:
                self.best_acc, self.bad_epochs = val_acc, 0
                self.save(epoch, "best.pt")
            else:
                self.bad_epochs += 1
                if self.bad_epochs >= self.tc.early_stop_patience:
                    log.info("Early stopping: no val improvement for %d epochs", self.bad_epochs)
                    break
        return self.history