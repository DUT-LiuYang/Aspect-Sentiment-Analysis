import numpy as np
from keras.preprocessing.sequence import pad_sequences


class ExampleReader:

    def __init__(self):
        self.dir = "../data/"
        self.max_sentence_length = 78
        self.EMBEDDING_DIM = 300

    def load_position_matrix(self):
        position_matrix_file = self.dir + "position_matrix.txt"
        rf = open(position_matrix_file, 'r')
        position_matrix = []
        while True:
            line = rf.readline()
            if line == "":
                break
            temp = line.strip().split()
            for i in range(len(temp)):
                temp[i] = float(temp[i])
            position_matrix.append(temp)
        rf.close()
        position_matrix = np.array(position_matrix)
        return position_matrix

    def load_inputs_and_label(self, name=''):
        sentence_inputs_file = self.dir + name + "_text_index.txt"
        aspect_text_index_file = self.dir + name + "_aspects_text_index.txt"
        aspect_label_index_file = self.dir + name + "_aspects_label_index.txt"

        sentence_inputs = []
        rf1 = open(sentence_inputs_file, 'r')
        aspect_text_inputs = []
        rf2 = open(aspect_text_index_file, 'r')
        aspect_labels = []
        true_labels = []
        rf3 = open(aspect_label_index_file, 'r')

        instance_num = 0

        while True:
            label = rf3.readline()
            if label == "":
                break
            label = int(label)
            sentence_input = rf1.readline()
            aspect_text = rf2.readline()
            if label == 3:  # we removed all the examples having the "conflict" label
                continue

            true_labels.append(label)

            aspect_labels.append([0] * 3)
            aspect_labels[instance_num][label] = 1

            sentence_input = sentence_input.strip()
            sentence_inputs.append(sentence_input[:])

            aspect_text = aspect_text.strip()
            aspect_text_inputs.append(aspect_text[:])

            instance_num += 1

        rf1.close()
        rf2.close()
        rf3.close()

        return np.asarray(aspect_labels), aspect_text_inputs, sentence_inputs, true_labels

    def get_position_input(self, sentences=[], aspects=[]):
        positions = []
        count = 0

        max_length = 0
        sentences_length = []

        for sentence, aspect in zip(sentences, aspects):
            if aspect in sentence:
                positions.append([])

                index = sentence.find(aspect)
                temp = sentence[0:index].count(" ")
                for i in range(temp):
                    positions[count].append(temp * -1 + i)

                for i in range(aspect.count(" ") + 1):
                    positions[count].append(0)

                index = sentence.find(" 0")
                if index == -1:  # if the length of the sentence is max_length
                    sentences_length.append(self.max_sentence_length)
                    temp = sentence.count(" ") - aspect.count(" ") - temp
                    for i in range(temp):
                        positions[count].append(i + 1)
                else:
                    sentences_length.append(sentence[0:index + 1].count(" "))
                    temp = sentence[0:index + 1].count(" ") - (aspect.count(" ") + 1) - temp
                    for i in range(temp):
                        positions[count].append(i + 1)
                    temp = sentence.count(" 0")
                    for i in range(temp):
                        positions[count].append(-255)
            else:
                index = sentence.find(" 0")
                if index == -1:  # if the length of the sentence is max_length
                    sentences_length.append(self.max_sentence_length)
                else:
                    sentences_length.append(sentence[0:index + 1].count(" "))
                print(sentence)
                print(aspect)
                positions.append([0] * self.max_sentence_length)

            sentences[count] = [int(x) for x in sentence.split()]
            aspects[count] = [int(x) for x in aspect.split()]
            if len(aspects[count]) > max_length:
                max_length = len(aspects[count])
                # print(aspects[count])
            count += 1

        print("max length of aspects is " + str(max_length))
        return np.array(sentences), np.array(aspects), np.array(positions), sentences_length

    def get_embedding_matrix(self):
        embedding_matrix_file = self.dir + 'embedding_matrix.txt'
        embedding_matrix = []
        rf = open(embedding_matrix_file, 'r')
        while True:
            line = rf.readline()
            if line == "":
                break
            embedding_matrix.append([float(x) for x in line.split()])
        rf.close()
        return np.array(embedding_matrix)

    @staticmethod
    def get_position_ids(max_len=78):
        position_ids = {}
        position = (max_len - 1) * -1
        position_id = 1
        while position <= max_len - 1:
            position_ids[position] = position_id
            position_id += 1
            position += 1
        position_ids[-255] = 0
        return position_ids

    def convert_position(self, position_inputs, position_ids={}):
        for line in position_inputs:
            for index, position in enumerate(line):
                line[index] = position_ids[position]

    @staticmethod
    def convert_position_weighted(sentences_length=[], position_inputs=None):
        """
        Get a weight vector only according to the position of the word.
        This operation can replace position embeddings.
        :return: The position weight vector of each instance.
        """
        position_inputs = position_inputs.tolist()
        for length, position_input in zip(sentences_length, position_inputs):

            for index, position in enumerate(position_input):
                if position != -255:
                    position_input[index] = 1 - abs(position) * 1.0 / length
                else:
                    position_input[index] = 0.0

        return np.array(position_inputs, dtype='float32')

    # @staticmethod
    # def convert_position_weighted(max_len, position_inputs=None):
    #     position_inputs = position_inputs.tolist()
    #     for position_input in position_inputs:
    #
    #         for index, position in enumerate(position_input):
    #             if position != -255:
    #                 position_input[index] = 1 - abs(position) * 1.0 / max_len
    #             else:
    #                 position_input[index] = 0.0
    #
    #     return np.array(position_inputs, dtype='float32')

    def get_aspect_pooling(self, aspects_index=np.array([0]),
                           aspect_index={}, aspect_embeddings=[],
                           embedding_matrix=np.array([0])):

        aspects = []
        aspect_count = 0

        aspect_embedding_count = 0

        for i, aspect in enumerate(aspects_index):
            aspects.append([])
            key = ""
            length = len(aspect)

            if length == 0:
                print("###########")
                print(str(aspect))
                print(str(i))

            pooling = np.array([0.0] * self.EMBEDDING_DIM, dtype='float32')

            for index in aspect:
                key += str(index) + " "
                pooling += embedding_matrix[index]

            key = key.strip()

            if key in aspect_index:
                aspects[aspect_count].append(aspect_index.get(key))
            else:
                pooling = pooling / length
                aspect_index[key] = aspect_embedding_count
                aspect_embeddings.append(pooling)
                aspects[aspect_count].append(aspect_embedding_count)
                aspect_embedding_count += 1
            aspect_count += 1

            if length == 0:
                print(str(pooling))

        print("Now there're " + str(aspect_embedding_count) + " aspects in the aspect_embeddings.")

        return np.array(aspects)

    @staticmethod
    def pad_aspect_index(aspect_inputs=[], max_length=9):
        return pad_sequences(aspect_inputs, maxlen=max_length, padding='post')

    def get_aspect_mask(self):
        pass


if __name__ == '__main__':
    example_reader = ExampleReader()
    position_matrix = example_reader.load_position_matrix()

    train_aspect_labels, train_aspect_text_inputs, train_sentence_inputs, _ = example_reader.load_inputs_and_label(name='train')
    test_aspect_labels, test_aspect_text_inputs, test_sentence_inputs, test_true_labels = example_reader.load_inputs_and_label(name='test')

    print(train_aspect_text_inputs[639])
    print(train_aspect_text_inputs[638])
    print(train_aspect_text_inputs[637])

    train_sentence_inputs, train_aspect_text_inputs, train_positions, train_sentences_length = example_reader.get_position_input(train_sentence_inputs,
                                                                                                                                 train_aspect_text_inputs)
    test_sentence_inputs, test_aspect_text_inputs, test_positions, test_sentences_length = example_reader.get_position_input(test_sentence_inputs,
                                                                                                                             test_aspect_text_inputs)

    embedding_matrix = example_reader.get_embedding_matrix()
    embedding_matrix[0] = np.array([-float('inf')] * 300, dtype='float32')
    print(embedding_matrix[0])
    print(np.shape(train_sentence_inputs))
    print(np.shape(train_aspect_text_inputs))
    print(np.shape(train_positions))
    print(np.shape(train_aspect_labels))
    print(np.shape(train_sentences_length))
    print("------------------------------------------")
    print(str(train_sentence_inputs[0]))
    print(str(train_aspect_text_inputs[0]))
    print(str(train_positions[0]))
    print(str(train_aspect_labels[0]))
    print("------------------------------------------")

    print(np.shape(test_sentence_inputs))
    print(np.shape(test_aspect_text_inputs))
    print(np.shape(test_positions))
    print(np.shape(test_aspect_labels))
    print(np.shape(test_true_labels))
    print("------------------------------------------")
    print(str(test_sentence_inputs[0]))
    print(str(test_aspect_text_inputs[0]))
    print(str(test_positions[0]))
    print(str(test_aspect_labels[0]))
    print("------------------------------------------")

    # aspect_index = {}
    # aspect_embeddings = []
    # train_aspects = example_reader.get_aspect_pooling(aspects_index=train_aspect_text_inputs, aspect_embeddings=aspect_embeddings,
    #                                                   aspect_index=aspect_index, embedding_matrix=embedding_matrix)
    # test_aspects = example_reader.get_aspect_pooling(aspects_index=test_aspect_text_inputs, aspect_embeddings=aspect_embeddings,
    #                                                  aspect_index=aspect_index, embedding_matrix=embedding_matrix)
    # aspect_embeddings = np.array(aspect_embeddings, dtype='float32')
    print("==========================================")
    train_aspects = example_reader.pad_aspect_index(train_aspect_text_inputs.tolist(), max_length=22)
    test_aspects = example_reader.pad_aspect_index(test_aspect_text_inputs.tolist(), max_length=22)
    # print(str(train_aspect_text_inputs[1]))
    # print(str(train_aspect_text_inputs[2]))
    print(np.shape(train_aspects))
    print(np.shape(test_aspects))
    # print(np.shape(aspect_embeddings))
    print(str(train_aspects[0]))
    # print(str(aspect_embeddings[1]))
    # print(str(embedding_matrix[14]))
    print("==========================================")
    position_ids = example_reader.get_position_ids(max_len=82)
    example_reader.convert_position(position_inputs=train_positions, position_ids=position_ids)
    example_reader.convert_position(position_inputs=test_positions, position_ids=position_ids)
    print(np.shape(train_positions))
    print(str(train_positions[0]))
    print(str(test_positions[0]))

    # example_reader.convert_position_weighted(sentences_length=train_sentences_length, position_inputs=train_positions)
    # example_reader.convert_position_weighted(sentences_length=test_sentences_length, position_inputs=test_positions)
    # print(str(train_positions[0]))
    # print(str(test_positions[0]))
