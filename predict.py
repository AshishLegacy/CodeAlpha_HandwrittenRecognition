"""Normalize one isolated digit, then predict using the saved CNN."""
from pathlib import Path
import argparse
import json
import numpy as np
from PIL import Image, ImageOps
import torch
from model import DigitCNN

def prepare(path):
    image = Image.open(path).convert('RGBA')
    background = Image.new('RGBA', image.size, 'white')
    background.alpha_composite(image)
    image = background.convert('L')
    pixels = np.array(image)
    border = np.concatenate([pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]])
    if border.mean() > 127:
        image = ImageOps.invert(image)
    image = ImageOps.autocontrast(image)
    box = image.point(lambda value: 255 if value > 30 else 0).getbbox()
    if box is None:
        raise ValueError('Image is blank; provide one clearly visible digit.')
    image = image.crop(box)
    image.thumbnail((20, 20), Image.Resampling.LANCZOS)
    canvas = Image.new('L', (28, 28), 0)
    canvas.paste(image, ((28-image.width)//2, (28-image.height)//2))
    return torch.tensor(np.array(canvas), dtype=torch.float32)[None, None] / 255

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    model = DigitCNN()
    checkpoint = torch.load(Path(__file__).resolve().parent / 'results/model.pt', weights_only=True)
    model.load_state_dict(checkpoint['state_dict']); model.eval()
    with torch.no_grad():
        probabilities = model(prepare(args.image)).softmax(1)[0]
    print(json.dumps({'digit': int(probabilities.argmax()),
                      'model_probability': float(probabilities.max()),
                      'note': 'Supports isolated digits 0-9; custom handwriting may differ from MNIST.'}, indent=2))

if __name__ == '__main__':
    main()
