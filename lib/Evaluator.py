class Evaluator(object):

    def __init__(self, true_labels=[], sentences=[], aspects=[]):
        self.max_F1 = -1
        self.max_F1_epoch = -1
        self.max_acc = -1
        self.max_acc_epoch = -1
        self.true_labels = true_labels
        self.num0 = 0
        self.num1 = 0
        self.num2 = 0
        self.dir = "../results/"
        self.word_index_dir = "../data/word_index.txt"
        self.get_positive_example_num(self.true_labels)
        self.save_results(true_labels, self.dir + "true_label.txt")
        self.index_word = self.load_index_word()
        self.sentences = sentences
        self.aspects = aspects

    def get_positive_example_num(self, true_labels=[]):
        temp = [0] * 3
        for label in true_labels:
            temp[label] += 1
        self.num0 = temp[0]
        self.num1 = temp[1]
        self.num2 = temp[2]

    def get_macro_f1(self, predictions=[], epoch=-1):

        predictions = self.get_predicted_label(predictions)

        p_temp = [0] * 3
        pr_temp = [0] * 3
        for label, true_label in zip(predictions, self.true_labels):
            p_temp[label] += 1
            if label == true_label:
                pr_temp[label] += 1

        p_p1 = p_temp[0]
        p_p2 = p_temp[1]
        p_p3 = p_temp[2]

        pr_p1 = pr_temp[0]
        pr_p2 = pr_temp[1]
        pr_p3 = pr_temp[2]

        p1, r1, f1 = self.calculate_f_score(p_p1, pr_p1, self.num0)
        p2, r2, f2 = self.calculate_f_score(p_p2, pr_p2, self.num1)
        p3, r3, f3 = self.calculate_f_score(p_p3, pr_p3, self.num2)

        print(str(p_p1) + " : " + str(p1) + " " + str(r1) + " " + str(f1))
        print(str(p_p2) + " : " + str(p2) + " " + str(r2) + " " + str(f2))
        print(str(p_p3) + " : " + str(p3) + " " + str(r3) + " " + str(f3))

        F = (f1 + f2 + f3) / 3
        print(str(F))
        signal = True
        if F > self.max_F1:
            self.max_F1 = F
            self.max_F1_epoch = epoch
            self.save_results(predictions=predictions, file=self.dir + str(epoch) + ".txt")
            signal = False

        acc = self.calculate_acc(p_temp, pr_temp)
        print(str(acc))
        if acc > self.max_acc:
            self.max_acc = acc
            self.max_acc_epoch = epoch
            if signal:
                self.save_results(predictions=predictions, file=self.dir + str(epoch) + ".txt")
            self.error_analysis(true_labels=self.true_labels, predicted_labels=predictions,
                                sentences=self.sentences, aspects=self.aspects,
                                error_analysis_file=self.dir + "e_a_" + str(epoch) + ".txt")

        return F, acc

    @staticmethod
    def save_results(predictions=[], file=''):
        wf = open(file, 'w')
        for label in predictions:
            wf.write(str(label) + "\n")
        wf.close()

    @staticmethod
    def calculate_f_score(p_p=0, pr_p=0, num=0):
        if p_p == 0:
            return 0, 0, 0

        P = float(pr_p) / p_p
        R = float(pr_p) / num
        if P == 0 or R == 0:
            return 0, 0, 0
        F = 2 * P * R / (P + R)

        return P, R, F

    @staticmethod
    def get_predicted_label(predictions=[]):
        prediction_labels = []

        for scores in predictions:
            index = 0
            max = -100
            for i, score in enumerate(scores):
                if score > max:
                    max = score
                    index = i
            prediction_labels.append(index)

        return prediction_labels

    @staticmethod
    def calculate_acc(predicted_positives=[], right_predicted_positives=[]):
        temp1 = temp2 = 0
        for num in predicted_positives:
            temp1 += num
        for num in right_predicted_positives:
            temp2 += num
        if temp1 == 0 or temp2 == 0:
            return 0
        acc = float(temp2) / temp1
        return acc

    def error_analysis(self, true_labels=[], predicted_labels=[], sentences=[], aspects=[], error_analysis_file=""):

        wf = open(error_analysis_file, 'w')

        for label, true_label, sentence, aspects in zip(predicted_labels, true_labels, sentences, aspects):
            if label != true_label:
                wf.write(str(true_label) + "-" + str(label) + "#")
                temp = ""
                for index in sentence:
                    if index == 0:
                        break
                    else:
                        temp += self.index_word[index] + " "
                wf.write(temp + "###")
                temp = ""
                for index in aspects:
                    temp += self.index_word[index] + " "
                wf.write(temp + "\n")

        wf.close()

    def load_index_word(self):
        index_word = {}
        rf = open(self.word_index_dir, 'r')
        while True:
            line = rf.readline()
            if line == "":
                break
            line = line.split()
            index_word[int(line[1])] = line[0]
        rf.close()
        return index_word
