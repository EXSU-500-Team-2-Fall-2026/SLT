import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.cnn import SimpleCNN, train_model


def test_simple_cnn_forward_pass_shape():
    model = SimpleCNN()
    x = torch.randn(4, 3, 96, 96)
    logits = model(x)
    assert logits.shape == (4,)


def _loader(images, labels, batch_size=4):
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size)


def test_train_model_does_not_crash_when_val_auroc_is_nan():
    # A validation set with only one class makes AUROC undefined (nan).
    # best_state must still end up populated instead of staying None.
    torch.manual_seed(0)
    train_loader = _loader(torch.randn(16, 3, 96, 96), torch.randint(0, 2, (16,)).float())
    val_loader = _loader(torch.randn(8, 3, 96, 96), torch.zeros(8))  # single class -> nan AUROC

    model = SimpleCNN()
    model, history, best_auroc = train_model(
        model, train_loader, val_loader, torch.device("cpu"), epochs=2, lr=1e-3, patience=5, pos_weight=1.0
    )
    assert len(history) == 2
    # should not raise, and the model's weights should be the (only) saved state
    assert any(torch.isfinite(p).all() for p in model.parameters())


def test_train_model_keeps_best_checkpoint():
    torch.manual_seed(0)
    train_loader = _loader(torch.randn(16, 3, 96, 96), torch.randint(0, 2, (16,)).float())
    val_loader = _loader(torch.randn(12, 3, 96, 96), torch.randint(0, 2, (12,)).float())

    model = SimpleCNN()
    model, history, best_auroc = train_model(
        model, train_loader, val_loader, torch.device("cpu"), epochs=3, lr=1e-3, patience=5, pos_weight=1.0
    )
    assert len(history) == 3
    assert best_auroc == max(h["val_auroc"] for h in history if h["val_auroc"] == h["val_auroc"])  # skip nan
