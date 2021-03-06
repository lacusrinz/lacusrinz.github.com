---
title: 使用hanLP进行自定义NER训练
date: 2021-03-06 20:31:55
tags: [tensorflow, NLP,人工智能]
categories: 机器学习应用
---

# 目的

为了将非结构化的简历文本提取出有效的工作经历信息，尝试使用各类NLP框架进行文本实体识别，目标是分解出时间，工作地点，任职单位，职务等有效信息。

在尝试了一些分词和NLP框架后，选用[hanLP](https://github.com/hankcs/HanLP)作为训练工具，在自己制作的数据集（人物简历信息）上训练自定义的NER标签（职务），实现特定标签的文本实体识别。
<!--more-->

# 环境

tensorflow 2.x

Colaboratory with GPU

hanLP 2.1

# 准备数据标注

要制作自定义数据集，就离不开标注工具，这里选用著名的开源标注工具[doccano](https://github.com/doccano/doccano)

在服务器部署好doccano，导入数据

![图片](1.png)

添加标签

![图片](2.png)

开始标注

![图片](3.png)

标注完成后导出数据

![图片](4.png)

注意这里选择第二种格式，这种格式会带着你的标签名字一起导出。

# 转化标注数据集格式

上面导出的JSONL数据并不能直接作为训练集放入NLP框架中训练，通用的训练集一般都使用BIO，BIOES，BMES标注

>一、BMES  四位序列标注法
>B表示一个词的词首位值，M表示一个词的中间位置，E表示一个词的末尾位置，S表示一个单独的字词。
>二、BIO三位标注  (B-begin，I-inside，O-outside)
>B-X 代表实体X的开头，I-X代表实体的结尾O代表不属于任何类型的
>三、BIOES (B-begin，I-inside，O-outside，E-end，S-single)
>B 表示开始，I表示内部， O表示非实体 ，E实体尾部，S表示改词本身就是一个实体。

>示例：
>因 O
>有 O
>关 O
>日 S-NS
>寇 O
>在 O
>京 S-NS
>掠 O
>夺 O
>文 O
>物 O
>详 O
>情 O
>， O
>藏 O
>界 O
>较 O
>为 O
>重 O
>视 O
>， O
>也 O
>是 O
>我 O
>们 O
>收 O
>藏 O
>北 B-NS
>京 E-NS
>史 O
>料 O
>中 O
>的 O
>要 O
>件 O
>之 O
>一 O
>。 O

同时NER训练集一般有“CoNLL 2003” “MSRA”两种dataset形式，doccano为我们提供了doccano-transformer这个转化工具，由于该工具面向是英文文本，所以在转化中文文本是有一些小问题，这里我fork了一份doccano-transformer并进行了修正，可以下载我的这份[代码](https://github.com/lacusrinz/doccano-transformer)

使用

```python
from doccano_transformer.datasets import NERDataset
from doccano_transformer.utils import read_jsonl
dataset = read_jsonl(filepath='example.jsonl', dataset=NERDataset, encoding='utf-8')
dataset.to_conll2003(tokenizer=str.split)
这里注意to_conll2003(tokenizer)方法的参数tokenizer是一个文本分割方法，当传入的是str.split则是将string使用所有的空字符，包括空格、换行(\n)、制表符(\t)等进行分割。这种分割方式对英文句子里每个英文单词使用空字符分割的情况完美适配，但是含有中文文本的句子如果按照这种方式，连续的汉字无法被分割，导致无法正常标注。

```
所以我们需要编写能将文本中的数字英文按空字符分割，中文按汉字分割的方法：
```python
import re
def get_word_list(s1):
    # 把句子按字分开，中文按字分，英文按单词，数字按空格
    res = re.compile(r"([\u4e00-\u9fa5])")    #  [\u4e00-\u9fa5]中文范围
    p1 = res.split(s1)
    # print(p1)
    str1_list = []
    for str in p1:
        if res.split(str) == None:
            str1_list.append(str)
        else:
            ret = res.split(str)
            for ch in ret:
                str1_list.append(ch)
    list_words = [w for w in str1_list if len(w.strip()) > 0]  # 去掉为空的字符
    lists=[]
    for list_word in list_words:
      splits = list_word.split()
      # print(split)
      for split in splits:
        lists.append(split)
    return  lists
```
应用
```plain
dataset = dataset.to_conll2003(tokenizer=get_word_list)
```
然后将dataset输出成tsv，即完成了训练集的制作。
# 开始训练

参考hanLP的文档，训练就非常简单了

```python
import hanlp
from hanlp.components.ner_tf import TransformerNamedEntityRecognizerTF
recognizer = TransformerNamedEntityRecognizerTF()
save_dir = 'data/model/ner/finetune_ner_albert_base_zh_msra'
CONLL03_RESUME_TRAIN="Output_train.tsv"
CONLL03_RESUME_TEST="Output_test.tsv"
recognizer.fit(CONLL03_RESUME_TRAIN, CONLL03_RESUME_TEST, save_dir, epochs=20, transformer='albert_base_zh',
               finetune=hanlp.pretrained.ner.MSRA_NER_ALBERT_BASE_ZH)
recognizer.load(save_dir)
print(recognizer.predict(list('2020年6月23日上午，吴江区第十六届人大常委会第二十九次会议召开，审议和通过有关人事任免，同意李铭同志因工作变动辞去吴江区区长职务')))
# recognizer.evaluate(CONLL03_RESUME_TEST, save_dir=save_dir)
print(f'Model saved in {save_dir}')
```
这里使用MSRA_NER_ALBERT_BASE_ZH预训练模型进行finetune操作，训练结果如下
```plain
2021-03-04 10:18:23 INFO Hyperparameter:
{
  "batch_size": 32,
  "epochs": 20,
  "run_eagerly": false,
  "finetune": "https://file.hankcs.com/hanlp/ner/ner_albert_base_zh_msra_20200111_202919.zip",
  "transformer": "albert_base_zh",
  "optimizer": "adamw",
  "learning_rate": 5e-05,
  "weight_decay_rate": 0,
  "epsilon": 1e-08,
  "clipnorm": 1.0,
  "warmup_steps_ratio": 0,
  "use_amp": false,
  "max_seq_length": 128,
  "metrics": "f1"
}
2021-03-04 10:18:23 INFO Vocab summary:
tag_vocab[4] = ['<pad>', 'O', 'B-job', 'I-job']
2021-03-04 10:18:23 INFO Building...
2021-03-04 10:18:25 INFO Model built:
Model: "model_4"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_ids (InputLayer)          [(None, 128)]        0                                            
__________________________________________________________________________________________________
token_type_ids (InputLayer)     [(None, 128)]        0                                            
__________________________________________________________________________________________________
mask_ids (InputLayer)           [(None, 128)]        0                                            
__________________________________________________________________________________________________
albert (BertModelLayer)         (None, 128, 768)     9957376     input_ids[0][0]                  
                                                                 token_type_ids[0][0]             
                                                                 mask_ids[0][0]                   
__________________________________________________________________________________________________
dense_4 (Dense)                 (None, 128, 4)       3076        albert[0][0]                     
==================================================================================================
Total params: 9,960,452
Trainable params: 9,960,452
Non-trainable params: 0
__________________________________________________________________________________________________
2021-03-04 10:18:25 INFO Loaded pretrained weights from /root/.hanlp/ner/ner_albert_base_zh_msra_20200111_202919/model.h5 for finetuning
Epoch 1/20
15/15 [==============================] - 70s 5s/step - loss: 13.7708 - f1: 0.0012 - val_loss: 2.4522 - val_f1: 0.0946
Epoch 2/20
15/15 [==============================] - 69s 5s/step - loss: 2.0932 - f1: 0.1956 - val_loss: 1.2397 - val_f1: 0.4171
Epoch 3/20
15/15 [==============================] - 69s 5s/step - loss: 1.0135 - f1: 0.3879 - val_loss: 0.9292 - val_f1: 0.4541
Epoch 4/20
15/15 [==============================] - 68s 5s/step - loss: 0.7765 - f1: 0.3948 - val_loss: 1.0566 - val_f1: 0.4069
Epoch 5/20
15/15 [==============================] - 69s 5s/step - loss: 0.6658 - f1: 0.4120 - val_loss: 0.5538 - val_f1: 0.5824
Epoch 6/20
15/15 [==============================] - 68s 5s/step - loss: 0.6176 - f1: 0.5337 - val_loss: 0.4626 - val_f1: 0.5601
Epoch 7/20
15/15 [==============================] - 68s 5s/step - loss: 0.6748 - f1: 0.4999 - val_loss: 0.4283 - val_f1: 0.6131
Epoch 8/20
15/15 [==============================] - 68s 5s/step - loss: 0.3727 - f1: 0.6239 - val_loss: 0.3824 - val_f1: 0.6786
Epoch 9/20
15/15 [==============================] - 68s 5s/step - loss: 0.5163 - f1: 0.6342 - val_loss: 0.2894 - val_f1: 0.6708
Epoch 10/20
15/15 [==============================] - 67s 5s/step - loss: 0.2572 - f1: 0.6855 - val_loss: 0.3856 - val_f1: 0.6812
Epoch 11/20
15/15 [==============================] - 67s 5s/step - loss: 0.3237 - f1: 0.6667 - val_loss: 0.2107 - val_f1: 0.7319
Epoch 12/20
15/15 [==============================] - 68s 5s/step - loss: 0.2778 - f1: 0.7104 - val_loss: 0.2025 - val_f1: 0.7081
Epoch 13/20
15/15 [==============================] - 66s 5s/step - loss: 0.1871 - f1: 0.7224 - val_loss: 0.1639 - val_f1: 0.8040
Epoch 14/20
15/15 [==============================] - 67s 5s/step - loss: 0.1868 - f1: 0.7405 - val_loss: 0.1240 - val_f1: 0.8250
Epoch 15/20
15/15 [==============================] - 67s 5s/step - loss: 0.1251 - f1: 0.7685 - val_loss: 0.1278 - val_f1: 0.8143
Epoch 16/20
15/15 [==============================] - 66s 5s/step - loss: 0.1123 - f1: 0.8196 - val_loss: 0.0957 - val_f1: 0.8591
Epoch 17/20
15/15 [==============================] - 67s 5s/step - loss: 0.1021 - f1: 0.8548 - val_loss: 0.0814 - val_f1: 0.8973
Epoch 18/20
15/15 [==============================] - 66s 5s/step - loss: 0.0803 - f1: 0.8640 - val_loss: 0.0737 - val_f1: 0.8945
Epoch 19/20
15/15 [==============================] - 67s 5s/step - loss: 0.0879 - f1: 0.8620 - val_loss: 0.0679 - val_f1: 0.9035
Epoch 20/20
15/15 [==============================] - 67s 5s/step - loss: 0.0670 - f1: 0.9267 - val_loss: 0.0639 - val_f1: 0.9114
2021-03-04 10:40:57 INFO Trained 20 epochs in 22 m 32 s, each epoch takes 1 m 8 s
[('6', 'ad>', 5, 6), ('日上午', 'job', 9, 12), ('吴江区第十', 'job', 13, 18), ('大', 'job', 21, 22), ('会', 'job', 30, 31)]
Model saved in data/model/ner/finetune_ner_albert_base_zh_msra
```
训练过程没有问题，但是训练出来的模型效果一般，问题可能出在训练集数据太少，后面会做进一步探索。
