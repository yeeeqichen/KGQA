
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


def main():
    folder = './data/MetaQA/'
    entity_dict = write_dict(folder + 'entities.dict', folder + 'entity2id.txt')
    relation_dict = write_dict(folder + 'relations.dict', folder + 'relation2id.txt')
    write_data(folder + 'train.txt', folder + 'train2id.txt', entity_dict, relation_dict)
    write_data(folder + 'valid.txt', folder + 'valid2id.txt', entity_dict, relation_dict)
    write_data(folder + 'test.txt', folder + 'test2id.txt', entity_dict, relation_dict)


if __name__ == '__main__':
    main()