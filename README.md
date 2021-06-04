# KGQA:基于知识图谱的问答系统
基于MetaQA知识图谱及其对应问答数据  

![](https://img.shields.io/badge/language-Python-brightgreen) ![](https://img.shields.io/badge/language-JavaScript-brightgreen)  

##项目目录结构： 
1、train_KGE目录下为训练知识图谱嵌入模型的相关代码  
2、KGQA_system目录下包含了问答系统的实现代码  
3、demo目录下为本问答系统的展示demo，是一个基于react开发的网页
```
.
├── KGQA_system
│   ├── QA_model.py
│   ├── create_neo4j.py
│   ├── dataloader.py
│   ├── graph_manager.py
│   ├── negative_manager.py
│   ├── predict.py
│   ├── prepare_data.py
│   ├── router.py
│   └── train.py
├── README.md
├── __pycache__
├── demo
│   ├── README.md
│   ├── package-lock.json
│   ├── package.json
│   ├── public
│   │   ├── favicon.ico
│   │   ├── imgs
│   │   │   ├── ModelDiagram.png
│   │   │   ├── logo192.png
│   │   │   └── logo512.png
│   │   ├── index.html
│   │   ├── logo192.png
│   │   ├── logo512.png
│   │   ├── manifest.json
│   │   └── robots.txt
│   └── src
│       ├── AnswerList
│       │   ├── index.css
│       │   └── index.js
│       ├── App.css
│       ├── App.js
│       ├── App.test.js
│       ├── SearchInput
│       │   ├── index.css
│       │   └── index.js
│       ├── index.css
│       ├── index.js
│       ├── logo.svg
│       ├── reportWebVitals.js
│       └── setupTests.js
├── notes.txt
└── train_KGE
    ├── train_KGE.py
    ├── train_complex.py
    ├── train_distmult.py
    ├── train_rotate.py
    └── train_transe.py
```

****

## 安装依赖  

python 相关依赖： 

- python == 
- pytorch ==   
- openke ==
- pytorch_transformers ==
- py2neo ==  

命令行中执行：
```angular2html
pip install -r requirement.txt
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
安装neo4j === 3.5




