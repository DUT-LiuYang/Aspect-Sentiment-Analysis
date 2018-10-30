import keras
from keras.layers import Dense, Embedding, Bidirectional, GRU, TimeDistributed, Input, Dropout, Lambda
from keras.models import Model
from keras import backend as K
import numpy as np

EMBEDDING_DIM = 300
POSITION_EMBEDDING_DIM = 50
MAX_LEN = 78


def reduce_dimension(x, length, mask):
    res = K.reshape(x, [-1, length])  # (?, 78)
    res = K.softmax(res)
    res = res * K.cast(mask, dtype='float32')  # (?, 78)
    temp = K.sum(res, axis=1, keepdims=True)  # (?, 1)
    temp = K.repeat_elements(temp, rep=length, axis=1)  # (?, 78)
    return res / temp


def reduce_dimension_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 3  # only valid for 3D tensors
    return [shape[0], shape[1]]


def attention(x, dim):
    res = K.batch_dot(x[0], x[1], axes=[1, 1])
    return K.reshape(res, [-1, dim])


def attention_output_shape(input_shape):
    shape = list(input_shape[1])
    assert len(shape) == 3
    return [shape[0], shape[2]]


def no_change(input_shape):
    return input_shape


def liter(x, length):
    res = K.repeat(x, length)  # (?, 82, 300)
    return res


def liter_output_shape(input_shape):
    shape = list(input_shape)
    return [shape[0], MAX_LEN, shape[1]]


def build_model(max_len=200, aspect_max_len=9, embedding_matrix=[],
                position_embedding_matrix=[], class_num=3, num_words=10000):

    MAX_LEN = max_len
    # ============================= Input =================================================
    sentence_input = Input(shape=(max_len,), dtype='int32', name='sentence_input')  # (?, 78)
    position_input = Input(shape=(max_len,), dtype='int32', name='position_input')  # (?, 78)
    aspect_input = Input(shape=(aspect_max_len,), dtype='int32', name='aspect_input')  # (?, 78)

    sentence_embedding_layer = Embedding(num_words + 1, EMBEDDING_DIM, weights=[embedding_matrix],
                                         input_length=max_len, trainable=False, mask_zero=True)
    sentence_embedding = sentence_embedding_layer(sentence_input)  # (?, 78, 300)

    position_embedding = Embedding(max_len * 2, POSITION_EMBEDDING_DIM, weights=[position_embedding_matrix],
                                   input_length=max_len, trainable=True, mask_zero=True)(position_input)  # (?, 78, 50)

    aspect_embedding_layer = Embedding(num_words + 1, EMBEDDING_DIM, weights=[embedding_matrix],
                                       input_length=aspect_max_len, trainable=False, mask_zero=True)
    aspect_embedding = aspect_embedding_layer(aspect_input)  # (?, 9, 300)
    # ============================= Input =================================================

    # =========================== GRU Layer ===============================================
    input_embedding = keras.layers.concatenate([sentence_embedding, position_embedding])  # (?, 78, 350)
    encode_x = Bidirectional(GRU(300, activation="relu", return_sequences=True, recurrent_dropout=0.5, dropout=0.5))(input_embedding)  # (?, 78, 600)
    # =========================== GRU Layer ===============================================

    # =========================== attention ===============================================

    # --------------------------- source2aspect attention ----------------------------------------
    aspect_attention = TimeDistributed(Dense(1, activation='tanh'))(aspect_embedding)  # (?, 9, 1)
    aspect_attention = Lambda(reduce_dimension,
                              output_shape=reduce_dimension_output_shape,
                              arguments={'length': aspect_max_len},
                              mask=aspect_embedding_layer.get_output_mask_at(0),
                              name='aspect_attention')(aspect_attention)  # (?, 9)
    aspect_embedding = Lambda(attention,
                              output_shape=attention_output_shape,
                              arguments={'dim': 300})([aspect_attention, aspect_embedding])  # (?, 300)
    # --------------------------- aspect attention ----------------------------------------

    aspect_embedding = Lambda(liter,
                              output_shape=liter_output_shape,
                              arguments={'length': max_len})(aspect_embedding)  # (?, 82, 300)
    x = keras.layers.concatenate([aspect_embedding, encode_x])  # (?, 82, 900)
    x = TimeDistributed(Dense(300, activation='tanh'))(x)   # (?, 82, 300)
    x = keras.layers.concatenate([x, encode_x])  # (?, 82, 900)

    # --------------------------- source2token attention ----------------------------------------
    x = TimeDistributed(Dense(1, activation='tanh'))(x)  # (?, 78, 1)
    attention_x = Lambda(reduce_dimension,
                         output_shape=reduce_dimension_output_shape,
                         arguments={'length': max_len},
                         mask=sentence_embedding_layer.get_output_mask_at(0),
                         name='attention_x')(x)  # (?, 78)
    # --------------------------- source2token attention ----------------------------------

    # =========================== attention ===============================================

    x = Lambda(attention, output_shape=attention_output_shape, arguments={'dim': 600})([attention_x, encode_x])  # (?, 600)

    x = Dropout(rate=0.5)(x)
    predictions = Dense(class_num, activation='softmax')(x)  # (?, 3)

    model = Model(inputs=[sentence_input, position_input, aspect_input], outputs=predictions)
    model.compile(loss=['categorical_crossentropy'], optimizer='rmsprop', metrics=['accuracy'])

    print(model.summary())
    return model


def train_model(sentence_inputs=[], position_inputs=[], aspect_input=[], labels=[], model=None):
    model.fit({'sentence_input': sentence_inputs, 'position_input': position_inputs, 'aspect_input': aspect_input}, labels, epochs=1, batch_size=64, verbose=2)
    return model


def get_predict(sentence_inputs=[], position_inputs=[], aspect_input=[], model=None):
    results = model.predict({'sentence_input': sentence_inputs, 'position_input': position_inputs, 'aspect_input': aspect_input}, batch_size=64, verbose=0)
    return results
