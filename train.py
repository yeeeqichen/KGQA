from dataloader import DataLoader
from QA_model import QuestionAnswerModel
import torch
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--train_file", type=str, default="./MetaQA/QA_data/qa_train_1hop.txt")
parser.add_argument("--valid_file", type=str, default="./MetaQA/QA_data/qa_dev_1hop.txt")
parser.add_argument("--test_file", type=str, default="./MetaQA/QA_data/qa_test_1hop.txt")
parser.add_argument("--dict_path", type=str, default="./MetaQA/QA_data/entities.dict")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--seq_length", type=int, default=26)


args = parser.parse_args()


def train(model, data_generator):
    for question_token_ids, question_masks, head_id, answers_id in data_generator:
        scores = model(question_token_ids, question_masks, head_id).cpu()
        print(scores)


def main():
    data_loader = DataLoader(
        train_file=args.train_file,
        valid_file=args.valid_file,
        test_file=args.test_file,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        dict_path=args.dict_path
    )
    model = QuestionAnswerModel()
    print(model)
    model.to(model.device)
    train(model, data_loader.train_batch_generator())


if __name__ == '__main__':
    main()
