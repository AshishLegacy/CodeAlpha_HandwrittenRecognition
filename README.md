# CodeAlpha - Handwritten Character Recognition (Task 3)

A convolutional neural network recognizes isolated handwritten digits 0-9 using MNIST, a dataset option explicitly allowed by the brief. Alphabet, word and sentence recognition are possible extensions, not implemented features.

## Run (Python 3.12 recommended)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py --epochs 5
.\.venv\Scripts\python.exe predict.py example_digit.png
```

The first training run downloads MNIST (about 11 MB). A SHA-256 check validates the download. Alternatively use walkthrough.ipynb from this directory.

## Method

- Official MNIST: 60,000 training and 10,000 test images, each 28x28 grayscale. Source: https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data and https://www.tensorflow.org/datasets/catalog/mnist
- Attribution: Yann LeCun, Corinna Cortes and Christopher J. C. Burges, MNIST handwritten digit database. Dataset terms: CC BY-SA 3.0, as documented by TensorFlow. Dataset excluded from Git by default; preserve attribution if redistributing it.
- Reserve a stratified 6,000-image validation set from official training images, leaving 54,000 for learning. Seed 42; pixels scaled to [0,1].
- CNN: convolution 1->16, ReLU, max pooling; convolution 16->32, ReLU, max pooling; dense 64, ReLU, dropout 0.2; 10 output logits. Cross-entropy loss with Adam (learning rate 0.001), batch size 128, CPU training.
- Save checkpoint with lowest validation loss. The official 10,000-image test set is used only after training and checkpoint selection.
- Report test accuracy, per-digit precision/recall/F1, majority baseline, training curves, confusion matrix, sample predictions and misclassified examples. Saved split indices and predictions support review.

## Files and demo

`model.py`: architecture. `train.py`: download, training and evaluation. `predict.py`: custom-image inference. `results/`: checkpoint, measured metrics and plots. `example_digit.png`: first official test image for a simple demo. Only load model checkpoints you trust.

For custom input, supply one clear digit with a plain background. Inference handles light/dark backgrounds, crops the foreground, fits it in a 20x20 box and centers it on a 28x28 canvas. Test metrics use the original MNIST preprocessing; they do not measure the custom-image pipeline. Softmax output is a model score, not a guarantee of correctness.

## Limitations

MNIST has centered, standardized digits. Photos, unusual handwriting, noise and multi-digit images can fail. Inspect error_examples.png to explain actual failures. Letter recognition would require EMNIST and retraining; whole words require a sequence model and different evaluation.

## Recorded result

Test accuracy: **98.70% on the official 10,000-image MNIST test set**. Full metrics and evaluation artifacts are in [RESULTS.md](RESULTS.md) and `results/`. Both training and saved-model inference were run locally.
