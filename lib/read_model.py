import h5py
from lib.ExampleReader import ExampleReader
from lib.Evaluator import Evaluator
import lib.Model as m
from keras.models import Model
import numpy as np


def write_attention(file="", attention=None):
    wf = open(file, 'w')
    for line in attention:
        for value in line:
            wf.write(str(value) + " ")
        wf.write("\n")
    wf.close()


if __name__ == '__main__':
    model_path = "../laptop_models/m__acc_76.48902821316614_F_73.73757341046482_30"

    example_reader = ExampleReader()
    position_matrix = example_reader.load_position_matrix()

    train_aspect_labels, train_aspect_text_inputs, train_sentence_inputs, _ = example_reader.load_inputs_and_label(name='train')
    test_aspect_labels, test_aspect_text_inputs, test_sentence_inputs, test_true_labels = example_reader.load_inputs_and_label(name='test')

    train_sentence_inputs, train_aspect_text_inputs, train_positions, _ = example_reader.get_position_input(train_sentence_inputs,
                                                                                                            train_aspect_text_inputs)
    test_sentence_inputs, test_aspect_text_inputs, test_positions, _ = example_reader.get_position_input(test_sentence_inputs,
                                                                                                         test_aspect_text_inputs)

    embedding_matrix = example_reader.get_embedding_matrix()
    position_ids = example_reader.get_position_ids(max_len=82)
    example_reader.convert_position(position_inputs=train_positions, position_ids=position_ids)
    example_reader.convert_position(position_inputs=test_positions, position_ids=position_ids)

    train_aspects = example_reader.pad_aspect_index(train_aspect_text_inputs.tolist(), max_length=9)
    test_aspects = example_reader.pad_aspect_index(test_aspect_text_inputs.tolist(), max_length=9)

    model = m.build_model(max_len=82,
                          aspect_max_len=9,
                          embedding_matrix=embedding_matrix,
                          position_embedding_matrix=position_matrix,
                          class_num=3,
                          num_words=4582)  # 5144 4582 //   # 1523 // 1172
    evaluator = Evaluator(true_labels=test_true_labels, sentences=test_sentence_inputs, aspects=test_aspect_text_inputs)

    model.load_weights(model_path)

    results = m.get_predict(sentence_inputs=test_sentence_inputs,
                            position_inputs=test_positions,
                            aspect_input=test_aspects,
                            model=model)

    F, acc = evaluator.get_macro_f1(predictions=results, epoch=2000)

    new_model = Model(inputs=model.input, outputs=model.get_layer('attention_y').output)
    attention = new_model.predict({'sentence_input': test_sentence_inputs, 'position_input': test_positions, 'aspect_input': test_aspects},
                                  batch_size=64,
                                  verbose=0)

    # print(str(np.shape(attention)))
    # print(str(attention[0]))
    write_attention(file="../attention_76.49", attention=attention)
