
def write_dict(in_file_path, out_file_path):
    dic = {}
    with open(in_file_path) as f:
        for line in f:
            a, b = line.strip('\n').split('\t')
            dic[a] = b
    with open(out_file_path, 'w') as f:
        f.write(str(len(dic)) + '\n')
        for key in dic.keys():
            f.write(key + '\t' + str(dic[key]) + '\n')
    return dic


def write_data(in_file_path, out_file_path, entity_dict, relation_dict):
    cnt = 0
    triples = []
    with open(in_file_path) as f:
        for line in f:
            head, rel, tail = line.strip('\n').split('\t')
            cnt += 1
            triples.append([entity_dict[head], entity_dict[tail], relation_dict[rel]])
    with open(out_file_path, 'w') as f:
        f.write(str(cnt) + '\n')
        for triple in triples:
            f.write(' '.join(triple) + '\n')


def construct_data_for_NER(file_path, output_dir, purpose='train'):
    with open(file_path) as f:
        cnt = 0
        data = []
        for line in f:
            question, _ = line.split('\t')
            question = question.replace('[', 'NER_TAG ')
            question = question.replace(']', ' NER_TAG')
            tokens = question.split(' ')
            words = []
            labels = []
            flag = False
            start = False
            for token in tokens:
                if token == 'NER_TAG':
                    if not flag:
                        start = True
                    flag = not flag
                    continue
                words.append(token)
                if flag:
                    if start:
                        labels.append('B-MISC')
                        start = False
                    else:
                        labels.append('I-MISC')
                else:
                    labels.append('O')
            assert len(words) == len(labels)
            data.append([words, labels])
            cnt += 1
            if cnt == 96106 and purpose == 'train':
                break
    with open(output_dir + 'ner_' + purpose + '.txt', 'w') as f:
        f.write('-DOCSTART- -X- -X- O\n')
        f.write('\n')
        for words, labels in data:
            for word, label in zip(words, labels):
                f.write(word + ' ' + label + '\n')
            f.write('\n')


def main():
    folder = './data/MetaQA/'
    entity_dict = write_dict(folder + 'entities.dict', folder + 'entity2id.txt')
    relation_dict = write_dict(folder + 'relations.dict', folder + 'relation2id.txt')
    write_data(folder + 'train.txt', folder + 'train2id.txt', entity_dict, relation_dict)
    write_data(folder + 'valid.txt', folder + 'valid2id.txt', entity_dict, relation_dict)
    write_data(folder + 'test.txt', folder + 'test2id.txt', entity_dict, relation_dict)


def test():
    construct_data_for_NER('MetaQA/qa_data/qa_train_1hop.txt', 'MetaQA/NER_data/', purpose='train')
    construct_data_for_NER('MetaQA/qa_data/qa_dev_1hop.txt', 'MetaQA/NER_data/', purpose='valid')
    construct_data_for_NER('MetaQA/qa_data/qa_test_1hop.txt', 'MetaQA/NER_data/', purpose='test')


if __name__ == '__main__':
    test()