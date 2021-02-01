from dataloader import DataLoader
from QA_model import QuestionAnswerModel
import torch


def train(model, data_generator):
    for question_token_ids, question_masks, head_id, answers_id in data_generator:
        scores = model(question_token_ids, question_masks, head_id).cpu()
        print(scores)


def main():
    data_loader = DataLoader(
        train_file='./data/QA_data/MetaQA/qa_train_1hop.txt',
        valid_file='./data/QA_data/MetaQA/qa_dev_1hop.txt',
        test_file='./data/QA_data/MetaQA/qa_test_1hop.txt'
    )
    model = QuestionAnswerModel()
    model.to(model.device)
    train(model, data_loader.train_batch_generator())


if __name__ == '__main__':
    main()
