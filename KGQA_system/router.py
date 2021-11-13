import torch
from QA_model import QuestionAnswerModel
from pytorch_transformers import BertTokenizer
from NER_model.bert import Ner
from py2neo import Graph
import nltk
import re
from flask import Flask, request
import json
from flask_cors import CORS
import argparse

app = Flask(__name__)
CORS(app)

parser = argparse.ArgumentParser()
parser.add_argument("--bert_path", type=str, default="C:/Users/yeeeqichen/Desktop/语言模型/")
parser.add_argument("--bert_name", type=str, default="bert-base-uncased")
parser.add_argument("--name", type=str, default='neo4j')
parser.add_argument("--password", type=str, default='neo4j')
parser.add_argument("--pretrained_model_path", type=str, default='model/2021-04-18__09-36-13')
parser.add_argument("--dict_path", type=str, default='MetaQA/qa_data/entities.dict')
parser.add_argument("--seq_length", type=int, default=20)
parser.add_argument("--embed_model_path", type=str, default="checkpoint/rotatE.ckpt", help="预训练的知识图谱嵌入模型路径")
parser.add_argument("--embed_method", type=str, default='rotatE')
parser.add_argument("--ner_model", type=str, default='NER_model/out_base/', help="NER模型路径")
parser.add_argument("--attn_method", type=str, default="mine", help="attn method for relation prediction model")

args = parser.parse_args()

graph = Graph("bolt://localhost:7687", auth=(args.name, args.password))

tokenizer = BertTokenizer.from_pretrained(args.bert_path + args.bert_name)

model = QuestionAnswerModel(
    embed_model_path=args.embed_model_path,
    embed_method=args.embed_method,
    attention=True,
    fine_tune=True,
    use_lstm=False,
    use_dnn=True,
    bert_name=args.bert_name,
    bert_path=args.bert_path,
    attention_method=args.attn_method,
    n_clusters=8
)
model.to(model.device)
model.load_state_dict(torch.load(args.pretrained_model_path + '/model.pkl'))

NER_model = Ner(args.ner_model)

ent_dict = {}
reverse_dict = {}
with open(args.dict_path) as f:
    for line in f:
        entity, entity_id = line.strip('\n').split('\t')
        ent_dict[entity] = int(entity_id)
        reverse_dict[entity_id] = entity

def preprocess_question(question):
    legal = 0
    NER_results = NER_model.predict(question)
    new_questions = ''
    flag = False
    for item in NER_results:
        if item['tag'] == 'O':
            if flag:
                flag = False
                new_questions += '] '
                legal += 1
            new_questions += item['word'] + ' '
        elif item['tag'] == 'B-MISC':
            new_questions += '[' + item['word'] + ' '
            flag = True
            legal += 1
        elif item['tag'] == 'I-MISC':
            new_questions += item['word'] + ' '
    new_questions = new_questions[:-1]
    if flag:
        new_questions += ']'
        legal += 1
    return legal == 2, new_questions


def query_graph(target):
    sub_graph1 = []
    result = graph.run("match (head:None {name:'" + target + "'})-[r]->(tail) return head, r, tail").data()
    for pair in result:
        sub_graph1.append([dict(pair['head'])['name'], list(pair['r'].types())[0], dict(pair['tail'])['name']])
    sub_graph2 = []
    result = graph.run("match (head:None)-[r]->(tail:None {name:'" + target + "'}) return head, r, tail").data()
    for pair in result:
        sub_graph2.append([dict(pair['head'])['name'], list(pair['r'].types())[0], dict(pair['tail'])['name']])
    return sub_graph1, sub_graph2


def predict(question):
    legal, question = preprocess_question(question)
    if not legal:
        return False, None
    head = re.match('(.*)\[(.*)\](.*)', question).groups()[1].strip(' ')
    if head not in ent_dict.keys():
        return False, None
    new_question = re.sub('\[.*\]', 'xxx', question)
    token_ids = tokenizer.encode(new_question, add_special_tokens=True)
    mask = [1] * len(token_ids)
    if len(token_ids) < args.seq_length:
        mask += [0] * (args.seq_length - len(token_ids))
        token_ids += [1] * (args.seq_length - len(token_ids))
    else:
        token_ids = token_ids[: args.seq_length]
        mask = mask[: args.seq_length - 1] + [2]
    head_id = [ent_dict[head]]
    predicts = model.predict([token_ids], [mask], [head_id])
    answers = []
    answer_query = []
    for index in predicts[0][:3]:
        answers.append(reverse_dict[str(index.item())])
        answer_query.append(query_graph(answers[-1]))
    query_results = query_graph(head)
    return True, [answers, query_results, head, answer_query]

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    res = {}
    question = str(request.args.get("question"))
    print(question)
    status, answer_list = predict(question)
    if not status:
        res['status'] = '500'
        res['data'] = 'illegal input'
    else:
        [answer, sub_graph, head, answer_subgraph] = answer_list
        res['status'] = '200 OK'
        res['data'] = {
            'answer': answer,
            'sub_graph': sub_graph,
            'head': head,
            'answer_subgraph': answer_subgraph
            }
    return json.dumps(res, ensure_ascii=False)


def main():
    app.run(debug=True, threaded=True, host='0.0.0.0', port=2333)


if __name__ == '__main__':
    main()
