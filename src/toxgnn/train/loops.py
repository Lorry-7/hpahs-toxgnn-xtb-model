"""Unified training loop with early stopping and checkpointing."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_


def train_regressor(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    loss_fn,
    device,
    epochs: int,
    patience: int,
    out_dir: Path,
    grad_clip_norm: float = 5.0,
) -> dict:
    """Train a regression model with early stopping.

    Returns dict with best_val_loss, epochs_ran, best_ckpt path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    bad_epochs = 0
    history = []
    best_path = out_dir / "best.ckpt"

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch)
            y = batch.y.view(-1).to(device)
            loss = loss_fn(pred, y)
            loss.backward()
            clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        # Validation
        val_loss = evaluate_loss(model, val_loader, loss_fn, device)

        # Scheduler step
        if scheduler is not None:
            if scheduler.__class__.__name__ == "ReduceLROnPlateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Log history
        row = {
            "epoch": epoch,
            "train_loss": float(sum(train_losses) / len(train_losses)),
            "val_loss": float(val_loss),
        }
        history.append(row)

        # Checkpointing
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_val,
                },
                best_path,
            )
        else:
            bad_epochs += 1

        # Early stopping
        if bad_epochs >= patience:
            break

    # Save last checkpoint
    torch.save(
        {"model_state_dict": model.state_dict(), "epoch": epoch},
        out_dir / "last.ckpt",
    )

    # Save learning curve
    (out_dir / "learning_curve.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )

    return {
        "best_val_loss": best_val,
        "epochs_ran": epoch,
        "best_ckpt": str(best_path),
    }


@torch.no_grad()
def evaluate_loss(model, loader, loss_fn, device) -> float:
    """Evaluate average loss on a data loader."""
    model.eval()
    losses = []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        y = batch.y.view(-1).to(device)
        losses.append(float(loss_fn(pred, y).detach().cpu()))
    return sum(losses) / max(1, len(losses))


@torch.no_grad()
def predict(model, loader, device) -> tuple:
    """Generate predictions and collect targets from a data loader."""
    model.eval()
    all_preds = []
    all_targets = []
    all_ids = []

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        all_preds.append(pred.detach().cpu())

        if hasattr(batch, "y") and batch.y is not None:
            all_targets.append(batch.y.view(-1).cpu())

        if hasattr(batch, "mol_id"):
            if isinstance(batch.mol_id, list):
                all_ids.extend(batch.mol_id)
            else:
                all_ids.append(batch.mol_id)

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy() if all_targets else None

    return preds, targets, all_ids
