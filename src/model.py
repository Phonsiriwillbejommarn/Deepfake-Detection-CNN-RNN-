import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Input, TimeDistributed, LSTM, Dense, GlobalAveragePooling2D, Dropout, Bidirectional, Attention, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2 # <-- เพิ่ม regularizer

def create_cnn_rnn_model(input_shape=(100, 224, 224, 3)):
    """
    ฟังก์ชันสำหรับสร้างโมเดล (ฉบับปรับปรุงประสิทธิภาพ)
    """
    # --- ส่วนของ CNN ---
    cnn_base = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape[1:]
    )
    cnn_base.trainable = True
    for layer in cnn_base.layers[:-40]: # <-- Freeze น้อยลง ให้โมเดลเรียนรู้ได้มากขึ้น
        layer.trainable = False
    cnn = Model(inputs=cnn_base.input, outputs=GlobalAveragePooling2D()(cnn_base.output), name="feature_extractor")

    # --- ส่วนของ RNN + Attention ---
    video_input = Input(shape=input_shape)
    encoded_frames = TimeDistributed(cnn)(video_input)
    
    # --- เพิ่มความลึกของ LSTM ---
    lstm_out_1 = Bidirectional(LSTM(units=128, return_sequences=True))(encoded_frames)
    lstm_out_2 = Bidirectional(LSTM(units=128, return_sequences=True))(lstm_out_1) # <-- เพิ่มชั้นที่ 2
    
    attention_out = Attention()([lstm_out_2, lstm_out_2])

    context_vector = Lambda(
        lambda x: tf.reduce_sum(x, axis=1),
        output_shape=(256,)
    )(attention_out)

    # --- เพิ่ม Regularization และปรับ Dropout ---
    x = Dropout(0.6)(context_vector) # <-- เพิ่ม Dropout
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x) # <-- เพิ่มขนาดและ L2
    x = Dropout(0.6)(x) # <-- เพิ่ม Dropout
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=video_input, outputs=output)
    return model