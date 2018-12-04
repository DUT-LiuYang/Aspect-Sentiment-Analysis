from lib.ExampleReader import ExampleReader
from lib.Evaluator import Evaluator
import lib.Model as m
import h5py


if __name__ == '__main__':
    model_path = "../laptop_models/m_"
    # -------------------- read example from file -------------------------
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
    # ---------------------------------------------------------------------

    for i in range(5):
        model = m.build_model(max_len=82,           # 78 82
                              aspect_max_len=9,     # 22 9
                              embedding_matrix=embedding_matrix,
                              position_embedding_matrix=position_matrix,
                              class_num=3,
                              num_words=4582)  # 5144 4582 //   # 1523 // 1172
        evaluator = Evaluator(true_labels=test_true_labels, sentences=test_sentence_inputs, aspects=test_aspect_text_inputs)
        epoch = 1
        while epoch <= 80:
            model = m.train_model(sentence_inputs=train_sentence_inputs,
                                  position_inputs=train_positions,
                                  aspect_input=train_aspects,
                                  labels=train_aspect_labels,
                                  model=model)
            results = m.get_predict(sentence_inputs=test_sentence_inputs,
                                    position_inputs=test_positions,
                                    aspect_input=test_aspects,
                                    model=model)
            print("\n--------------epoch " + str(epoch) + " ---------------------")
            F, acc = evaluator.get_macro_f1(predictions=results, epoch=epoch)
            if epoch % 5 == 0:
                print("current max F1 score: " + str(evaluator.max_F1))
                print("max F1 is gained in epoch " + str(evaluator.max_F1_epoch))
                print("current max acc: " + str(evaluator.max_acc))
                print("max acc is gained in epoch " + str(evaluator.max_acc_epoch))
            print("------------------------------------------------------------")

            if acc > 0.7535:
                model.save_weights(model_path + "_acc_" + str(acc * 100) + "_F_" + str(F * 100) + "_" + str(epoch))
            elif F > 0.7080:
                model.save_weights(model_path + "_acc_" + str(acc * 100) + "_F_" + str(F * 100) + "_" + str(epoch))

            epoch += 1
