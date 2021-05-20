import torch
from QA_model import QuestionAnswerModel
from pytorch_transformers import BertTokenizer
from NER_model.bert import Ner
from py2neo import Graph
import nltk
import re

# nltk.download('punkt')
PRE_TRAINED_MODEL_PATH = 'model/2021-04-18__09-36-13'
SEQ_LENGTH = 20
DICT_PATH = 'MetaQA/qa_data/entities.dict'

graph = Graph("bolt://localhost:7687", auth=('neo4j', 'neo4j'))

tokenizer = BertTokenizer.from_pretrained('C:/Users/yeeeqichen/Desktop/语言模型/bert-base-uncased')

model = QuestionAnswerModel(
    embed_model_path='checkpoint/rotatE.ckpt',
    embed_method='rotatE',
    attention=True,
    fine_tune=True,
    use_lstm=False,
    use_dnn=True,
    bert_name='bert-base-uncased',
    bert_path='C:/Users/yeeeqichen/Desktop/语言模型/',
    attention_method='self-attention',
    n_clusters=8
)
model.to(model.device)
model.load_state_dict(torch.load(PRE_TRAINED_MODEL_PATH + '/model.pkl'))

NER_model = Ner('NER_model/out_base/')

ent_dict = {}
reverse_dict = {}
with open(DICT_PATH) as f:
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
        return False, None, None
    head = re.match('(.*)\[(.*)\](.*)', question).groups()[1].strip(' ')
    new_question = re.sub('\[.*\]', 'xxx', question)
    token_ids = tokenizer.encode(new_question, add_special_tokens=True)
    mask = [1] * len(token_ids)
    if len(token_ids) < SEQ_LENGTH:
        mask += [0] * (SEQ_LENGTH - len(token_ids))
        token_ids += [1] * (SEQ_LENGTH - len(token_ids))
    else:
        token_ids = token_ids[: SEQ_LENGTH]
        mask = mask[: SEQ_LENGTH - 1] + [2]
    head_id = [ent_dict[head]]
    predicts = model.predict([token_ids], [mask], [head_id])
    answers = []
    for index in predicts[0][:3]:
        answers.append(reverse_dict[str(index.item())])
    query_results = query_graph(head)
    return True, answers, query_results


def test():
    test_question = 'what movies can be described with sidney franklin'
    test_answer = "The Good Earth|The Barretts of Wimpole Street|Private Lives|Smilin' Through|The Dark Angel"
    answer, sub_graph = predict(test_question)
    print('input question is: ', test_question)
    print('the related facts in KG is: ', sub_graph)
    print('the answer to this question is: ', test_answer)
    print('the predicted answer is: ', answer)


def main():
    test()
    # print(query_graph('Big'))
# print(model)


if __name__ == '__main__':
    main()

