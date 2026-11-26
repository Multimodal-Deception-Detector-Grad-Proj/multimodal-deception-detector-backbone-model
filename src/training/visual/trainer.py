from __future__ import annotations

import copy
import time
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader


class Trainer:
    """AdamW + (optionally class-weighted) CE, early stopping on validation macro-F1."""

    def __init__(self, model: nn.Module, cfg, class_weights: Optional[torch.Tensor] = None):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = model.to(self.device)
        w = class_weights.to(self.device) if (class_weights is not None and cfg.class_weighted_loss) else None
        self.criterion = nn.CrossEntropyLoss(weight=w)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, mode="max", factor=0.5, patience=5)
        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.best_state, self.best_epoch = None, -1
        self.history = []

    def _run_epoch(self, loader: DataLoader, train: bool):
        self.model.train(train)
        total, losses, preds, ys = 0.0, 0, [], []
        with torch.set_grad_enabled(train):
            for x, y, _ in loader:
                x, y = x.to(self.device), y.to(self.device)
                with torch.autocast(self.device.type, enabled=self.use_amp):
                    logits = self.model(x)
                    loss = self.criterion(logits.float(), y)
                if train:
                    self.opt.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.opt)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.scaler.step(self.opt)
                    self.scaler.update()
                total += loss.item() * len(y)
                losses += len(y)
                preds.append(logits.argmax(1).cpu().numpy())
                ys.append(y.cpu().numpy())
        preds, ys = np.concatenate(preds), np.concatenate(ys)
        return total / losses, float((preds == ys).mean()), float(f1_score(ys, preds, average="macro"))

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> pd.DataFrame:
        best, bad = -1.0, 0
        t0 = time.time()
        for epoch in range(1, self.cfg.epochs + 1):
            tr_loss, tr_acc, tr_f1 = self._run_epoch(train_loader, True)
            va_loss, va_acc, va_f1 = self._run_epoch(val_loader, False)
            self.sched.step(va_f1)
            self.history.append(dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc, train_f1=tr_f1,
                                     val_loss=va_loss, val_acc=va_acc, val_f1=va_f1, lr=self.opt.param_groups[0]["lr"]))
            if va_f1 > best:
                best, bad, self.best_epoch = va_f1, 0, epoch
                self.best_state = copy.deepcopy(self.model.state_dict())
            else:
                bad += 1
            if epoch == 1 or epoch % 5 == 0:
                print(f"ep {epoch:3d} | loss {tr_loss:.3f}/{va_loss:.3f} | acc {tr_acc:.3f}/{va_acc:.3f} | "
                      f"val F1 {va_f1:.3f} (best {best:.3f} @ {self.best_epoch})")
            if bad >= self.cfg.patience:
                print(f"early stop at epoch {epoch}")
                break
        self.model.load_state_dict(self.best_state)
        print(f"best epoch {self.best_epoch} | val F1 {best:.3f} | {time.time() - t0:.0f}s")
        return pd.DataFrame(self.history)
