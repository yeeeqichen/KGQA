# KGQA:基于知识图谱的问答系统

![](https://img.shields.io/badge/language-Python-brightgreen) ![](https://img.shields.io/badge/language-JavaScript-brightgreen)  
## 项目说明

本次项目中设计并实现了一个基于知识图谱的问答系统，结合MetaQA知识图谱，用于解决电影领域的问答问题

此项目为作者的本科毕业设计项目，完成于2021年5月，引用请注明出处

## 数据集
[数据集下载](https://github.com/yuyuz/MetaQA)

数据集中包含：
- 知识图谱 (kg.txt)
- 问答数据 (1-hop/vanilla)

后续代码中使用到的entity_dict 请自行根据数据集构建，其中每一行包含 entity 和 entity_id 用 '\t' 隔开

MetaQA知识图谱中包含四万多个电影领域相关实体及18种关系  

问答数据中有1-hop、2-hop、3-hop三种类型，本项目中使用 **1-hop** 问答数据

知识图谱中三元组及问答数据示例：
- 三元组
  ```angular2html
  Kismet | directed_by | William Dieterle
  ```
- 问答
  ```angular2html
  what films are on [ginger rogers]   Top Hat|Kitty Foyle|The Barkleys of Broadway
  ```

## 项目目录结构： 
1、train_KGE目录下为训练知识图谱嵌入模型的相关代码，包含以下知识图谱嵌入方法：
- RotatE
- TransE
- DistMult
- ComplEx

2、KGQA_system目录下包含了问答系统的实现代码，包含以下几个模块：
- 知识图谱嵌入模块
- 问题嵌入模块
- 关系预测模块
- 答案生成模块

3、demo目录下为本问答系统的展示demo，是一个基于react开发的网页


****

## 安装依赖  

python 相关依赖： 

- python == 3.7
- pytorch == 1.8.1
- openke == 0.93
- pytorch_transformers == 1.2.0
- py2neo == 4.3.0

命令行中执行：
```angular2html
pip install -r requirements.txt
```

前端demo展示相关依赖(若不需要网页展示此部分可以跳过)：

进入前端demo目录:
```angular2html
cd demo
```
安装react:
```angular2html
npm install react
```
安装echarts:
```angular2html
npm install echarts
```
安装neo4j，请自行前往[neo4j官网](https://neo4j.com)安装，注意要安装对应版本的Java

## 训练知识图谱嵌入模型 

进入train_KGE:

```angular2html
cd train_KGE
```

将知识图谱训练数据放于 data目录下，需包含以下几个文件：
- train2id.txt
- entity2id.txt
- relation2id.txt
- test2id.txt
- valid2id.txt

关于这些文件的格式请参考 [这里](https://github.com/thunlp/OpenKE) 的benchmarks

下一步进行知识图谱嵌入模型的训练，执行对应嵌入模型的训练脚本，训练完成的模型存放于 checkpoint目录中

```angular2html
python3 train_rotate.py
```

## 训练问答系统模型

进入KGQA_system：
```angular2html
cd KGQA_system
```

将问答系统训练数据放于data目录下，需包含以下几个文件：
- train.txt
- valid.txt
- test.txt

数据格式示例如下，每一行为问题及对应答案，若有多个答案则用 | 隔开：
```angular2html
what movies does [Jack Ma] act in?  Alibaba|Taobao
```
执行训练脚本，训练问答系统：
```bash
python3 train.py \
  --train_file **训练文件路径** \
  --valid_file **验证文件路径** \
  --test_file **测试文件路径** \
  --bert_name **使用的bert模型名称** \
  --bert_path **bert模型存放路径** \
  --relation_file **relation2id.txt文件路径** \
  --KGE_method **使用的知识图谱嵌入方法**
```
其中使用上面的参数需要根据实际使用情况进行填写

## Demo展示

问答系统demo主要由前后端两部分构成：
- 后端：
    - neo4j图数据库，用于存储和查询知识图谱
    - 训练完成的问答系统，用于对输入的问题进行回答
- 前端：
    - 前端是基于react开发的网页
    
后端部分，首先启动neo4j图数据库服务，在neo4j安装目录下执行:
```shell
./bin/neo4j console
```
随后执行KGQA_system 目录下的脚本将知识图谱存入图数据库当中:
```bash
python3 create_neo4j.py \
  --data_path **知识图谱存放路径,内有entity2id.txt, relation2id.txt, train2id.txt**
```
注意到，有别于训练数据，我们在实际的使用场景中输入的问题  **不会**  对头实体进行特殊标注：
```angular2html
what movies does [Jack Ma] act in?   <-->  what movies does Jack Ma act in?
```
因此在最终使用时，我们需要使用一个NER系统来检测出输入问题的头实体，NER系统可以使用现成的轮子，详见 [这里](https://github.com/kamalkraj/BERT-NER)

使用手上的问答数据来构造出NER训练数据，用于检测出问题中的头实体

最后，执行KGQA_system/router.py，启动问答服务：
```shell
python3 router.py \
  --ner_model **填写前一步训练好的NER模型路径** \
  --bert_path **填写预训练bert文件夹路径,与下面的bert_name一起构成完成路径** \
  --bert_name **填写使用的bert模型名称,例如bert-base-uncased** \
  --name **填写neo4j数据库用户名** \
  --password **填写neo4j数据库密码** \
  --pretrained_model_path **填写预训练的问答模型路径** \
  --dict_path **填写实体字典路径** \
  --embed_model_path **填写预训练的知识图谱嵌入模型路径** \
  --embed_method **填写知识图谱嵌入方法,与上面的预训练模型保持一致**
```

启动前端网页服务：
```shell
cd demo/
npm start
```
网页展示效果：
![](https://github.com/yeeeqichen/Pictures/blob/master/591627631284_.pic_hd.jpg?raw=true)
![](https://github.com/yeeeqichen/Pictures/blob/master/601627631349_.pic_hd.jpg?raw=true)
![](https://github.com/yeeeqichen/Pictures/blob/master/611627631366_.pic_hd.jpg?raw=true)








