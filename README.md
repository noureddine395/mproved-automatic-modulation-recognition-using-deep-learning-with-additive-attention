# Improved Automatic Modulation Recognition Using Deep Learning with Additive Attention

[![DOI](https://zenodo.org/badge/1328115675.svg)](https://doi.org/10.5281/zenodo.21854712)

Official code for the paper:

> El-Haryqy, N., Kharbouche, A., Ouamna, H., Madini, Z. & Zouine, Y.
> **"Improved automatic modulation recognition using deep learning with
> additive attention."** *Results in Engineering*, 26, 104783 (2025).
> https://doi.org/10.1016/j.rineng.2025.104783

Reference implementation of the CNN + Bidirectional LSTM + additive-attention
model for automatic modulation recognition (AMR), evaluated on the
RadioML **2016.10a** and **2016.10b** datasets. This repository accompanies
the paper and is provided so that the reported results can be reproduced.

## Repository contents

| File | Description |
|---|---|
| `rml2016_10a.py` | Data loading, model definition, training, and SNR-wise evaluation on RadioML 2016.10a |
| `rml2016_10b.py` | Same pipeline for RadioML 2016.10b, with an added exponential learning-rate schedule |
| `requirements.txt` | Python dependencies |
| `LICENSE` | MIT license |

## Model

Both scripts build the same architecture (`build_improved_model` /
`ImprovedAdditiveAttention`):

1. Two `Conv1D` blocks (64, 128 filters) with batch normalization, max
   pooling, and dropout.
2. Two stacked `Bidirectional LSTM` layers (128 units each) with batch
   normalization and dropout.
3. A custom additive-attention layer (`ImprovedAdditiveAttention`) with a
   learnable scale, dropout on the attention weights, and layer
   normalization on the context vector.
4. Two `Dense` blocks (128, 64 units) with batch normalization and dropout,
   followed by a softmax output layer.

The model is trained with Adam (gradient clipping, `clipnorm=1.0`),
`categorical_crossentropy` loss, early stopping on validation loss, and
model checkpointing on validation accuracy. The 2016.10b script additionally
applies a custom callback that exponentially decays the learning rate each
epoch.

## Datasets

- **RadioML 2016.10a**: [DeepSig RadioML 2016.10a](https://www.deepsig.ai/datasets/)
- **RadioML 2016.10b**: [DeepSig RadioML 2016.10b](https://www.deepsig.ai/datasets/)

Both are third-party datasets released by DeepSig and are not redistributed
in this repository. Download them from the link above and update the
`FILE_PATH` constant at the top of each script to point to the local file
(the scripts default to a Google Drive path used in the original Colab
environment: `/content/drive/MyDrive/RML2016.10a/RML2016.10a_dict.pkl` and
`/content/drive/MyDrive/RML2016.10b/RML2016.10b.dat`).

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Reproducing the results

```bash
# RadioML 2016.10a
python rml2016_10a.py

# RadioML 2016.10b
python rml2016_10b.py
```

Each script will:

1. Load and normalize the I/Q data, shuffle it with a fixed random seed
   (`RANDOM_STATE = 2016`), and split it 70/15/15 into train/validation/test
   sets.
2. Train the model with early stopping and checkpointing, saving
   `best_model.keras` and `final_model.keras`.
3. Plot training/validation loss and accuracy curves.
4. Evaluate the trained model per-SNR: confusion matrices, accuracy vs. SNR,
   F1 score vs. SNR, per-modulation accuracy/F1, and per-modulation accuracy
   broken down by SNR range and by individual SNR value (printed as a
   table).

All random seeds are fixed (`RANDOM_STATE = 2016`) for the data shuffle and
train/val/test split so that the splits are reproducible; exact numerical
results may still vary slightly across TensorFlow/hardware versions due to
GPU non-determinism.

## Citation

If you use this code, please cite the paper and/or the software release:

```bibtex
@article{ELHARYQY2025104783,
  title    = {Improved automatic modulation recognition using deep learning with additive attention},
  journal  = {Results in Engineering},
  volume   = {26},
  pages    = {104783},
  year     = {2025},
  issn     = {2590-1230},
  doi      = {10.1016/j.rineng.2025.104783},
  url      = {https://www.sciencedirect.com/science/article/pii/S2590123025008606},
  author   = {Noureddine El-Haryqy and Anass Kharbouche and Hamza Ouamna and Zhour Madini and Younes Zouine},
  keywords = {Automatic modulation recognition, Bidirectional long short-term memory networks, Convolutional neural networks, Deep learning, Enhanced attention mechanism, Signal-to-noise ratio}
}

@software{elharyqy2025code,
  author  = {El-Haryqy, Noureddine and Kharbouche, Anass and Ouamna, Hamza and Madini, Zhour and Zouine, Younes},
  title   = {Improved Automatic Modulation Recognition Using Deep Learning with Additive Attention (Code)},
  year    = {2025},
  doi     = {10.5281/zenodo.21854712},
  url     = {https://doi.org/10.5281/zenodo.21854712}
}
```

## License

Released under the MIT License — see [LICENSE](LICENSE).
