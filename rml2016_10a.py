"""
Automatic Modulation Classification on RadioML 2016.10a
=========================================================

CNN + Bi-LSTM + Additive Attention model for AMC on the RadioML 2016.10a
dataset. See README.md for usage instructions.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import (
    BatchNormalization, Dropout, Conv1D, Dense, Bidirectional, LSTM,
    Input, Lambda, Layer, LayerNormalization
)
from tensorflow.keras.regularizers import l2
import tensorflow as tf

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
RANDOM_STATE = 2016
FILE_PATH = '/content/drive/MyDrive/RML2016.10a/RML2016.10a_dict.pkl'


# ----------------------------------------------------------------------------
# Data loading / preprocessing
# ----------------------------------------------------------------------------
def load_data(file_path: str) -> dict:
    """Load dataset from pickle file."""
    try:
        with open(file_path, 'rb') as file:
            dataset = pickle.load(file, encoding='latin1')
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        raise
    return dataset


def preprocess_data(dataset: dict) -> tuple:
    """Preprocess dataset."""
    signal_types = set(key[0] for key in dataset.keys())
    snr_values = set(key[1] for key in dataset.keys())
    signals_of_interest = list(signal_types)

    data = []
    labels = []
    snrs = []
    for signal in signals_of_interest:
        for snr in snr_values:
            if (signal, snr) in dataset:
                data.extend(dataset[(signal, snr)])
                labels.extend([signal] * len(dataset[(signal, snr)]))
                snrs.extend([snr] * len(dataset[(signal, snr)]))

    data = np.array(data)
    labels = np.array(labels)
    snrs = np.array(snrs)

    data_normalized = normalize_data(data)
    data_normalized, labels, snrs = shuffle_data(data_normalized, labels, snrs)
    labels_encoded, classes = encode_labels(labels)
    labels_onehot = tf.keras.utils.to_categorical(labels_encoded)

    return data_normalized, labels_onehot, classes, snrs


def normalize_data(data: np.ndarray) -> np.ndarray:
    """Normalize the data."""
    scaler = tf.keras.layers.Normalization(axis=-1)
    scaler.adapt(data.reshape(-1, data.shape[-1]))
    return scaler(data.reshape(-1, data.shape[1], data.shape[2])).numpy()


def shuffle_data(data: np.ndarray, labels: np.ndarray, snrs: np.ndarray) -> tuple:
    """Shuffle the data."""
    np.random.seed(RANDOM_STATE)
    shuffle_indices = np.random.permutation(np.arange(len(data)))
    return data[shuffle_indices], labels[shuffle_indices], snrs[shuffle_indices]


def encode_labels(labels: np.ndarray) -> tuple:
    """Encode labels."""
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    return labels_encoded, label_encoder.classes_


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class ImprovedAdditiveAttention(Layer):
    def __init__(self, units, use_scale=True, dropout_rate=0.1):
        super(ImprovedAdditiveAttention, self).__init__()
        self.units = units
        self.use_scale = use_scale
        self.dropout_rate = dropout_rate

        self.W1 = Dense(units, use_bias=False)
        self.W2 = Dense(units, use_bias=False)
        self.V = Dense(1, use_bias=False)

        if self.use_scale:
            self.scale = self.add_weight(
                name='scale', shape=(), initializer='ones', trainable=True
            )

        self.dropout = Dropout(dropout_rate)
        self.layer_norm = LayerNormalization(epsilon=1e-6)

    def call(self, query, value, training=False):
        # query shape == (batch_size, query_len, hidden size)
        # value shape == (batch_size, value_len, hidden size)
        score = self.V(tf.nn.tanh(self.W1(query) + self.W2(value)))

        if self.use_scale:
            score = score * self.scale

        attention_weights = tf.nn.softmax(score, axis=1)
        attention_weights = self.dropout(attention_weights, training=training)

        context_vector = attention_weights * value
        context_vector = tf.reduce_sum(context_vector, axis=1)
        context_vector = self.layer_norm(context_vector)

        return context_vector


def build_improved_model(num_classes, input_shape=(2, 128)):
    inputs = Input(shape=input_shape)

    # Transpose the input to (batch_size, 128, 2)
    x = Lambda(lambda t: tf.transpose(t, perm=[0, 2, 1]))(inputs)

    # Convolutional layers
    x = Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Conv1D(128, 3, activation='relu', padding='same', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    # LSTM layers
    x = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(1e-4)))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    lstm_out = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(1e-4)))(x)
    x = BatchNormalization()(lstm_out)
    x = Dropout(0.3)(x)

    # Improved Additive Attention mechanism
    attention = ImprovedAdditiveAttention(128, use_scale=True, dropout_rate=0.1)
    x = attention(x, lstm_out)

    # Dense layers
    x = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# ----------------------------------------------------------------------------
# Evaluation / plotting
# ----------------------------------------------------------------------------
def evaluate_at_snrs(model, X_test, Y_test, snrs_test, classes, snr_values):
    predictions = model.predict(X_test)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(Y_test, axis=1)

    confusion_matrices = {}
    accuracies = {}
    f1_scores = {}

    for snr in snr_values:
        indices = np.where(snrs_test == snr)[0]
        if len(indices) > 0:
            cm = confusion_matrix(true_labels[indices], predicted_labels[indices])
            confusion_matrices[snr] = cm
            accuracy = np.sum(predicted_labels[indices] == true_labels[indices]) / len(indices)
            accuracies[snr] = accuracy
            f1 = f1_score(true_labels[indices], predicted_labels[indices], average='weighted')
            f1_scores[snr] = f1

    plot_confusion_matrices(confusion_matrices, classes)
    plot_accuracies(list(accuracies.keys()), list(accuracies.values()))
    plot_f1_scores(list(f1_scores.keys()), list(f1_scores.values()))

    print("\nAccuracies and F1 scores for each modulation:")
    for i, modulation in enumerate(classes):
        mod_indices = np.where(true_labels == i)[0]
        mod_accuracy = np.sum(predicted_labels[mod_indices] == true_labels[mod_indices]) / len(mod_indices)
        mod_f1 = f1_score(true_labels[mod_indices], predicted_labels[mod_indices], average='weighted')
        print(f"{modulation}: Accuracy = {mod_accuracy:.4f}, F1 Score = {mod_f1:.4f}")

    overall_accuracy = np.sum(predicted_labels == true_labels) / len(true_labels)
    overall_f1 = f1_score(true_labels, predicted_labels, average='weighted')
    print(f"\nOverall Accuracy: {overall_accuracy:.4f}")
    print(f"Overall F1 Score: {overall_f1:.4f}")

    snr_ranges = [(-20, -10), (-10, 0), (0, 10), (10, 20)]
    plot_modulation_accuracies(model, X_test, Y_test, snrs_test, classes, snr_ranges)


def plot_confusion_matrices(confusion_matrices, classes):
    num_plots = len(confusion_matrices)
    cols = 4
    rows = (num_plots + cols - 1) // cols
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(20, 5 * rows))
    axes = axes.flatten()

    for ax, (snr, cm) in zip(axes, confusion_matrices.items()):
        cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_percentage, display_labels=classes)
        disp.plot(ax=ax, values_format='.1f', cmap='Blues')
        accuracy = np.trace(cm) / np.sum(cm)
        ax.set_title(f'SNR = {snr} dB, Acc = {accuracy * 100:.2f}%')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')

    for ax in axes[num_plots:]:
        ax.remove()

    plt.tight_layout()
    plt.show()


def plot_accuracies(snrs, accuracies):
    plt.figure(figsize=(10, 6))
    plt.plot(snrs, accuracies, marker='o')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Accuracy')
    plt.title('Classification Accuracy vs. SNR')
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1.05, 0.05))
    plt.grid(True)
    plt.show()


def plot_f1_scores(snrs, f1_scores):
    plt.figure(figsize=(10, 6))
    plt.plot(snrs, f1_scores, marker='o')
    plt.xlabel('SNR (dB)')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs. SNR')
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1.05, 0.05))
    plt.grid(True)
    plt.show()


def plot_modulation_accuracies(model, X_test, Y_test, snrs_test, classes, snr_ranges):
    predictions = model.predict(X_test)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(Y_test, axis=1)

    modulation_accuracies = {mod: [] for mod in classes}

    for snr_min, snr_max in snr_ranges:
        indices = np.where((snrs_test >= snr_min) & (snrs_test < snr_max))[0]
        if len(indices) > 0:
            for i, modulation in enumerate(classes):
                mod_indices = np.where((true_labels == i) & np.isin(np.arange(len(true_labels)), indices))[0]
                if len(mod_indices) > 0:
                    mod_accuracy = np.sum(predicted_labels[mod_indices] == true_labels[mod_indices]) / len(mod_indices)
                    modulation_accuracies[modulation].append(mod_accuracy)
                else:
                    modulation_accuracies[modulation].append(0)

    plt.figure(figsize=(12, 6))
    x = range(len(snr_ranges))
    for modulation in classes:
        plt.plot(x, modulation_accuracies[modulation], marker='o', label=modulation)

    plt.xlabel('SNR Range')
    plt.ylabel('Accuracy')
    plt.title('Modulation Accuracy vs. SNR Range')
    plt.xticks(x, [f'{snr_min} to {snr_max}' for snr_min, snr_max in snr_ranges], rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(True)
    plt.show()


def plot_modulation_accuracies_per_snr(model, X_test, Y_test, snrs_test, classes, snr_values):
    predictions = model.predict(X_test)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(Y_test, axis=1)

    modulation_accuracies = {mod: [] for mod in classes}

    for snr in snr_values:
        indices = np.where(snrs_test == snr)[0]
        if len(indices) > 0:
            for i, modulation in enumerate(classes):
                mod_indices = np.where((true_labels == i) & np.isin(np.arange(len(true_labels)), indices))[0]
                if len(mod_indices) > 0:
                    mod_accuracy = np.sum(predicted_labels[mod_indices] == true_labels[mod_indices]) / len(mod_indices)
                    modulation_accuracies[modulation].append(mod_accuracy)
                else:
                    modulation_accuracies[modulation].append(0)

    plt.figure(figsize=(12, 6))
    for modulation in classes:
        plt.plot(snr_values, modulation_accuracies[modulation], marker='o', label=modulation)

    plt.xlabel('SNR (dB)')
    plt.ylabel('Accuracy')
    plt.title('Modulation Accuracy vs. SNR')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(True)
    plt.show()

    print("\nModulation Accuracies at Different SNR Levels:")
    print("SNR (dB) |", end="")
    for modulation in classes:
        print(f" {modulation:8} |", end="")
    print("\n" + "-" * (10 + 11 * len(classes)))

    for i, snr in enumerate(snr_values):
        print(f"{snr:8d} |", end="")
        for modulation in classes:
            print(f" {modulation_accuracies[modulation][i]:.6f} |", end="")
        print()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    dataset = load_data(FILE_PATH)
    X, Y, classes, snrs = preprocess_data(dataset)

    if X.shape[1] != 2 or X.shape[2] != 128:
        X = X.transpose(0, 2, 1)

    X_train, X_temp, Y_train, Y_temp, snrs_train, snrs_temp = train_test_split(
        X, Y, snrs, test_size=0.3, random_state=RANDOM_STATE
    )
    X_val, X_test, Y_val, Y_test, snrs_val, snrs_test = train_test_split(
        X_temp, Y_temp, snrs_temp, test_size=0.5, random_state=RANDOM_STATE
    )

    model = build_improved_model(len(classes), input_shape=(2, 128))

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=6, mode='min', restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6),
        ModelCheckpoint('best_model.keras', monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)
    ]

    history = model.fit(
        X_train, Y_train, epochs=100, batch_size=32,
        validation_data=(X_val, Y_val), callbacks=callbacks
    )

    model.save('final_model.keras')

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss Over Epochs')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy Over Epochs')

    plt.tight_layout()
    plt.show()

    snr_values = range(-20, 20, 2)
    evaluate_at_snrs(model, X_test, Y_test, snrs_test, classes, snr_values)
    plot_modulation_accuracies_per_snr(model, X_test, Y_test, snrs_test, classes, snr_values)
