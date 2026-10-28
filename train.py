"""Train on MNIST with separate validation and untouched official test set."""
from pathlib import Path
import argparse
import hashlib
import json
import random
import urllib.request
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from model import DigitCNN

BASE = Path(__file__).resolve().parent
URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
SHA256 = '731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be at least 1')
    torch.set_num_threads(4)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    out, cache = BASE / 'results', BASE / 'data'
    out.mkdir(exist_ok=True); cache.mkdir(exist_ok=True)
    path = cache / 'mnist.npz'
    if not path.exists():
        print('Downloading official MNIST archive...', flush=True)
        urllib.request.urlretrieve(URL, path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
        raise ValueError('MNIST checksum mismatch. Remove data/mnist.npz and download again.')
    with np.load(path) as data:
        X, y, Xt, yt = [data[k].copy() for k in ['x_train', 'y_train', 'x_test', 'y_test']]
    train, valid = train_test_split(np.arange(len(y)), test_size=0.1, stratify=y, random_state=42)
    def loader(images, labels, shuffle=False):
        ds = TensorDataset(torch.tensor(images[:, None], dtype=torch.float32) / 255,
                           torch.tensor(labels, dtype=torch.long))
        return DataLoader(ds, batch_size=128, shuffle=shuffle,
                          generator=torch.Generator().manual_seed(42), num_workers=0)
    training, validation, testing = loader(X[train], y[train], True), loader(X[valid], y[valid]), loader(Xt, yt)
    model = DigitCNN()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    history, best = [], float('inf')
    def evaluate(batches):
        model.eval(); loss, count, correct = 0.0, 0, 0
        predictions = []
        with torch.no_grad():
            for xb, yb in batches:
                logits = model(xb)
                loss += loss_fn(logits, yb).item() * len(yb)
                pred = logits.argmax(1)
                correct += (pred == yb).sum().item(); count += len(yb)
                predictions.extend(pred.tolist())
        return loss / count, correct / count, np.array(predictions)
    for epoch in range(1, args.epochs + 1):
        model.train(); total, correct, count = 0.0, 0, 0
        for xb, yb in training:
            opt.zero_grad(); logits = model(xb); loss = loss_fn(logits, yb)
            loss.backward(); opt.step()
            total += loss.item() * len(yb); count += len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
        vl, va, _ = evaluate(validation)
        history.append(dict(epoch=epoch, train_loss=total/count, train_accuracy=correct/count,
                            validation_loss=vl, validation_accuracy=va))
        if vl < best:
            best = vl
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch,
                        'validation_loss': vl}, out / 'model.pt')
        print(json.dumps(history[-1]), flush=True)
    checkpoint = torch.load(out / 'model.pt', weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    loss, accuracy, pred = evaluate(testing)
    metrics = {'seed': 42, 'train_samples': len(train), 'validation_samples': len(valid),
               'test_samples': len(yt), 'selected_epoch': checkpoint['epoch'],
               'selection_rule': 'minimum validation loss', 'test_accuracy': accuracy, 'test_loss': loss,
               'majority_baseline_test_accuracy': float(accuracy_score(yt, np.full_like(yt, np.bincount(y[train]).argmax()))),
               'classification_report': classification_report(yt, pred, output_dict=True),
               'dataset_url': URL, 'dataset_sha256': SHA256,
               'versions': {'torch': torch.__version__, 'numpy': np.__version__}, 'history': history}
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(out / 'split_indices.npz', train=train, validation=valid)
    np.savetxt(out / 'test_predictions.csv', np.c_[np.arange(len(yt)), yt, pred],
               delimiter=',', fmt='%d', header='test_row,actual,predicted', comments='')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for key, label in [('train_loss', 'Training'), ('validation_loss', 'Validation')]:
        axes[0].plot([r['epoch'] for r in history], [r[key] for r in history], label=label)
    for key, label in [('train_accuracy', 'Training'), ('validation_accuracy', 'Validation')]:
        axes[1].plot([r['epoch'] for r in history], [r[key] for r in history], label=label)
    for ax, title in zip(axes, ['Cross-entropy loss', 'Accuracy']):
        ax.set_title(title); ax.set_xlabel('Epoch'); ax.legend()
    fig.tight_layout(); fig.savefig(out / 'training_curves.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(yt, pred, ax=ax, colorbar=False)
    ax.set_title('MNIST: official test set'); fig.tight_layout()
    fig.savefig(out / 'confusion_matrix.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(3, 6, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(Xt[i], cmap='gray'); ax.set_title(f'True {yt[i]} / Pred {pred[i]}', fontsize=9); ax.axis('off')
    fig.tight_layout(); fig.savefig(out / 'sample_predictions.png', dpi=160); plt.close(fig)
    errors = np.flatnonzero(pred != yt)
    fig, axes = plt.subplots(2, 6, figsize=(10, 4))
    for ax, i in zip(axes.flat, errors[:12]):
        ax.imshow(Xt[i], cmap='gray'); ax.set_title(f'True {yt[i]} / Pred {pred[i]}', fontsize=9)
    for ax in axes.flat: ax.axis('off')
    fig.tight_layout(); fig.savefig(out / 'error_examples.png', dpi=160); plt.close(fig)
    Image.fromarray(Xt[0]).save(BASE / 'example_digit.png')
    print(f'Official test accuracy: {accuracy:.4%}', flush=True)

if __name__ == '__main__':
    main()
