import numpy as np


class DataLoader:
    def __init__(self, data_directory, batch_size=32, negative_sample_size=8):
        self.train_file = data_directory + '/train.txt'
        self.valid_file = data_directory + '/valid.txt'
        self.test_file = data_directory + '/test.txt'
        self.entity_file = data_directory + '/entities.dict'
        self.relation_file = data_directory + '/relations.dict'
        self.entity_to_idx = {}
        self.relation_to_idx = {}
        self.data = {'train': [], 'valid': [], 'test': []}
        self.negative_sample_size = negative_sample_size
        self.batch_size = batch_size
        self.num_entity = 0
        self.num_relation = 0

    def load_files(self):
        with open(self.entity_file) as f:
            for line in f:
                entity, idx = line.strip('\n').split('\t')
                self.entity_to_idx[entity] = int(idx)
        self.num_entity = len(self.entity_to_idx)
        with open(self.relation_file) as f:
            for line in f:
                relation, idx = line.strip('\n').split('\t')
                self.relation_to_idx[relation] = int(idx)
        self.num_relation = len(self.relation_to_idx)
        with open(self.train_file) as f:
            for line in f:
                entity1, relation, entity2 = line.strip('\n').split('\t')
                self.data['train'].append(
                    [self.entity_to_idx[entity1], self.relation_to_idx[relation], self.entity_to_idx[entity2]])
        with open(self.valid_file) as f:
            for line in f:
                entity1, relation, entity2 = line.strip('\n').split('\t')
                self.data['valid'].append(
                    [self.entity_to_idx[entity1], self.relation_to_idx[relation], self.entity_to_idx[entity2]])
        with open(self.test_file) as f:
            for line in f:
                entity1, relation, entity2 = line.strip('\n').split('\t')
                self.data['test'].append(
                    [self.entity_to_idx[entity1], self.relation_to_idx[relation], self.entity_to_idx[entity2]])
        print("there are{}triplets in train file, {}triples in valid file, {}triples in test file".
              format(len(self.data['train']), len(self.data['valid']), len(self.data['test'])))

    def batch_generator(self, purpose='train'):
        length = len(self.data[purpose])
        idx = 0
        while idx < length:
            next_idx = idx + self.batch_size if idx + self.batch_size < length else length
            batch = [_ for _ in self.data[purpose][idx: next_idx]]
            yield np.array(batch)
            idx = next_idx
            # if purpose == 'train':
            #     # negative_sample = []
            #     # for sample in positive_sample:
            #     #     neg_sample = []
            #     #     for i in range(self.negative_sample_size):
            #     #         neg_sample.append([sample[0], sample[1], random.randint(0, len(self.entity_to_idx) - 1)])
            #     #     negative_sample.append(neg_sample)
            #
            # elif purpose == 'valid':
            #     yield np.array(positive_sample)


# if __name__ == '__main__':
#     loader = DataLoader('../data/MetaQA')
#     loader.load_files()
#     for positive_sample, negative_sample in tqdm.tqdm(loader.batch_generator()):
#         print(positive_sample, negative_sample)
