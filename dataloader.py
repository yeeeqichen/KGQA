from pytorch_transformers import RobertaTokenizer
import re
import tqdm
import logging
BERT_PATH = "C:/Users/yeeeqichen/Desktop/语言模型/roberta-base"
logger = logging.getLogger(__name__)

# train 一共96106句问答
class DataLoader:
    def __init__(self, train_file, valid_file, test_file, dict_path,
                 batch_size=32, seq_length=20):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.ent_dict = {}
        self.tokenizer = RobertaTokenizer.from_pretrained(BERT_PATH)
        logger.info('reading entity dict...')
        with open(dict_path) as f:
            for line in f:
                entity, entity_id = line.strip('\n').split('\t')
                self.ent_dict[entity] = int(entity_id)
        logger.info('there are {} entities'.format(len(self.ent_dict)))
        self.train_corpus = self.read_file(train_file)
        self.total_train_instances = len(self.train_corpus)
        logger.info('there are {} instances in {}'.format(self.total_train_instances, train_file))
        self.valid_corpus = self.read_file(valid_file)
        self.total_valid_instances = len(self.valid_corpus)
        logger.info('there are {} instances in {}'.format(self.total_valid_instances, valid_file))
        self.test_corpus = self.read_file(test_file)
        self.total_test_instances = len(self.test_corpus)
        logger.info('there are {} instances in {}'.format(self.total_test_instances, test_file))

    def read_file(self, file_path):
        corpus = []
        logger.debug('reading data from {}...'.format(file_path))
        with open(file_path) as f:
            for line in tqdm.tqdm(f):
                question, answer = line.strip('\n').split('\t')
                token_ids = self.tokenizer.encode(question, add_special_tokens=True)
                mask = [1] * len(token_ids)
                if len(token_ids) < self.seq_length:
                    mask += [0] * (self.seq_length - len(token_ids))
                    token_ids += [1] * (self.seq_length - len(token_ids))
                else:
                    token_ids = token_ids[: self.seq_length]
                    mask = mask[: self.seq_length - 1] + [2]
                head = re.match('(.*)\[(.*)\](.*)', question).groups()[1]
                answers = answer.split('|')
                head_id = [self.ent_dict[head]]
                answers_id = [self.ent_dict[answer] for answer in answers]
                corpus.append([question, [token_ids, mask], head_id, answers_id])
        return corpus

    def train_batch_generator(self):
        steps = self.total_train_instances // self.batch_size
        for i in range(steps):
            temp = self.train_corpus[i * self.batch_size: (i + 1) * self.batch_size]
            question_token_ids = [_[1][0] for _ in temp]
            question_masks = [_[1][1] for _ in temp]
            head_id = [_[2] for _ in temp]
            answers_id = [_[3] for _ in temp]
            yield question_token_ids, question_masks, head_id, answers_id


def test():
    a = DataLoader('./MetaQA/QA_data/qa_train_1hop.txt', './MetaQA/QA_data/qa_dev_1hop.txt',
                   './MetaQA/QA_data/qa_test_1hop.txt')
    for batch in a.train_batch_generator():
        print(batch)
        break


if __name__ == '__main__':
    test()
