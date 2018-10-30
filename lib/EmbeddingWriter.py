import numpy as np


class EmbeddingWriter(object):

    def __init__(self):
        self.dir = '../laptop_data/'
        self.EMBEDDING_DIM = 300

    def read_word_index(self):
        word_index_file = self.dir + "word_index.txt"
        word_index = {}
        rf = open(word_index_file, 'r')
        while True:
            line = rf.readline()
            if line == "":
                break
            line = line.split()
            word_index[line[0]] = int(line[1])
        rf.close()
        return word_index

    def read_aspect_index(self, name=''):
        aspect_text_index_file = self.dir + name + "_aspects_text_index.txt"
        rf = open(aspect_text_index_file, 'r')

        while True:
            line = rf.readline()
            if line == "":
                break
            if " " in line:
                pass
        rf.close()

    def convert_embedding_file(self, word_num=10000, word_index={}):

        embedding_file = "../resource/glove.42B.300d.txt"
        rf = open(embedding_file, 'r', encoding='utf-8')
        embeddings_index = {}
        print("reading embedding from " + embedding_file)
        count = 0
        for line in rf:
            count += 1
            if count % 100000 == 0:
                print(str(count))
            values = line.split()
            index = len(values) - self.EMBEDDING_DIM
            if len(values) > (self.EMBEDDING_DIM + 1):
                word = ""
                for i in range(len(values) - self.EMBEDDING_DIM):
                    word += values[i] + " "
                word = word.strip()
            else:
                word = values[0]

            coefs = np.asarray(values[index:], dtype='float32')
            embeddings_index[word] = coefs
        rf.close()
        print("finish.")

        num_words = min(word_num, len(word_index))
        embedding_matrix = np.zeros((num_words + 1, self.EMBEDDING_DIM))
        for word, i in word_index.items():
            if i >= word_num:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
            else:
                print(word)

        embedding_matrix_file = self.dir + "embedding_matrix.txt"
        print("writing embedding matrix to " + embedding_matrix_file)
        wf = open(embedding_matrix_file, 'w')
        for embedding in embedding_matrix:
            for num in embedding:
                wf.write(str(num) + " ")
            wf.write("\n")
        print("finish.")
        wf.close()

    def write_position_embedding(self, vec_size=20, max_len=100):
        length = max_len * 2
        entity_type_matrix = np.random.rand(length, vec_size)
        wf = open(self.dir + "position_matrix.txt", 'w')
        for i in range(len(entity_type_matrix)):
            for j in range(len(entity_type_matrix[i])):
                wf.write(str(entity_type_matrix[i][j]) + " ")
            wf.write("\n")
        wf.close()


if __name__ == '__main__':
    embedding_writer = EmbeddingWriter()
    word_index = embedding_writer.read_word_index()
    embedding_writer.convert_embedding_file(word_num=10000, word_index=word_index)
    embedding_writer.write_position_embedding(vec_size=50, max_len=82)
