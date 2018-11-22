# Aspect-Sentiment-Analysis
We will upload our source code for our paper in CONLL 2018 soon.

1. 运行xml2text.py, 读取xml文件。
2. 运行data2inputs.py, 将数据转化为网络可用输入。在这之前，我使用分词工具对文本进行了分词。
3. 运行EmbeddingWriter.py, 生成必要的embedding矩阵。
4. 训练模型，可以使用ExampleReader.py读取数据，示例见该文件。

整个模型的训练和read model类似，我们也提供了一个Main文件，方便运行各个模块。

训练好的模型会存在models文件夹下，预测的结果以及预测错误会报错在results文件夹。

词向量使用如论文所说，来自glove官网。

这个任务仍有很多挑战，比如对于复杂句子（一句话存在相反的情感、双重否定）的识别效果并不理想。
我们也会持续关注最新进展。
