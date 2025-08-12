
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import EfficientNetB0

class FramePresence(layers.Layer):
    """Compute boolean mask (batch, time) where a frame is 'present' if any pixel is non-zero."""
    def __init__(self, threshold=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.threshold = float(threshold)

    def call(self, x):
        # x: (B, T, H, W, C) in [0,1] (assumed). Zero-padded frames are all zeros.
        # max over H, W, C -> (B, T)
        maxval = tf.reduce_max(tf.abs(x), axis=[-1, -2, -3])
        mask = maxval > self.threshold
        # Important: cast to bool
        return tf.cast(mask, dtype=tf.bool)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"threshold": self.threshold})
        return cfg

def create_cnn_rnn_model_varlen(
    input_shape=(None, 224, 224, 3),
    base_trainable=False,
    l2_reg=1e-4,
    dropout=0.5,
):
    """
    CNN + BiLSTM video model that supports variable-length sequences.
    - Time dimension is None.
    - We compute a frame-level mask from raw pixels (zero frames = padding).
    - RNN layers receive the mask so they ignore padding.
    - The final BiLSTM returns the last valid timestep (return_sequences=False),
      so we avoid manual temporal pooling.
    """
    video_in = layers.Input(shape=input_shape, name="video_in")

    # Compute mask from raw frames BEFORE feature extractor
    mask = FramePresence(name="frame_presence")(video_in)  # (B, T)

    # Lightweight normalization (assumes inputs in [0,1])
    # If you prefer EfficientNet preprocessing, add it in the data pipeline.
    x = layers.TimeDistributed(layers.Rescaling(scale=255.0), name="rescale_255")(video_in)

    # Per-frame CNN backbone
    base = EfficientNetB0(include_top=False, pooling="avg", weights="imagenet")
    base.trainable = base_trainable
    feat = layers.TimeDistributed(base, name="td_effnet")(x)  # (B, T, D)

    # Temporal modeling with mask
    x = layers.Bidirectional(
        layers.LSTM(
            256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2,
            kernel_regularizer=regularizers.l2(l2_reg),
        ),
        name="bilstm_1",
    )(feat, mask=mask)

    x = layers.Bidirectional(
        layers.LSTM(
            256, return_sequences=False, dropout=0.2, recurrent_dropout=0.2,
            kernel_regularizer=regularizers.l2(l2_reg),
        ),
        name="bilstm_2",
    )(x, mask=mask)  # returns last VALID timestep thanks to mask

    # Head
    x = layers.Dropout(dropout, name="head_dropout_1")(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="fc1")(x)
    x = layers.Dropout(dropout, name="head_dropout_2")(x)
    out = layers.Dense(1, activation="sigmoid", name="prob")(x)

    model = models.Model(video_in, out, name="cnn_rnn_varlen")
    return model
