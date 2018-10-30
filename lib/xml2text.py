try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


class xml2text(object):

    def __init__(self, file=""):
        self.file = file
        self.dir = "../laptop_data/"

    def convert_xml_to_text(self, name=""):
        tree = ET.parse(self.file)
        root = tree.getroot()

        sen_num = 0
        aspect_num = 0
        sen_without_aspect_num = 0

        text_file = self.dir + name + "_text.txt"
        aspect_text_file = self.dir + name + "_aspects_text.txt"
        aspect_label_file = self.dir + name + "_aspects_label.txt"
        wf1 = open(text_file, 'w')
        wf2 = open(aspect_text_file, 'w')
        wf3 = open(aspect_label_file, 'w')

        # ============ read xml file ===================
        for sentence in root:
            sen_num += 1
            text = sentence.find("text")
            temp = text.text.strip()

            while '  ' in temp:
                temp = temp.replace('  ', ' ')
            temp = temp.replace(u'\xa0', ' ').lower()
            # print(temp)
            wf1.write(temp + "\n")

            aspect_terms = sentence.find("aspectTerms")
            if aspect_terms is not None:
                for aspect in aspect_terms:
                    aspect_num += 1
                    wf2.write(aspect.get("term").replace(u'\xa0', ' ').lower() + "\n")
                    wf3.write(aspect.get("polarity") + "#" + str(sen_num - 1) + "\n")
            else:
                sen_without_aspect_num += 1
        # ==============================================

        wf1.close()
        wf2.close()
        wf3.close()
        print("There are " + str(sen_num) + " sentences.")  # 3041 800 // 3045 800
        print("There are " + str(sen_without_aspect_num) + " sentences without aspects.")  # 1020 194 // 1557 378
        print("There are " + str(aspect_num) + " aspects.")  # 3693 1134 // 2358 654


if __name__ == '__main__':
    train_corpus = "../resource/Laptop_Train_v2.xml"
    test_corpus = "../resource/Laptops_Test_Gold.xml"

    train_pre = xml2text(train_corpus)
    train_pre.convert_xml_to_text(name="train")

    test_pre = xml2text(test_corpus)
    test_pre.convert_xml_to_text(name="test")
