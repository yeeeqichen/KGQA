from dataloader import DataLoader
from QA_model import QuestionAnswerModel
import torch
import argparse
import logging

parser = argparse.ArgumentParser()

parser.add_argument("--train_file", type=str, default="./MetaQA/QA_data/qa_train_1hop.txt")
parser.add_argument("--valid_file", type=str, default="./MetaQA/QA_data/qa_dev_1hop.txt")
parser.add_argument("--test_file", type=str, default="./MetaQA/QA_data/qa_test_1hop.txt")
parser.add_argument("--dict_path", type=str, default="./MetaQA/QA_data/entities.dict")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--seq_length", type=int, default=26)
parser.add_argument("--EPOCH", type=int, default=10)
parser.add_argument("--valid_steps", type=int, default=500)
parser.add_argument("--log_level", type=str, default="DEBUG")
parser.add_argument("--require_improvement", type=int, default=5)
parser.add_argument("--save_path", type=str, default="./model.pkl")


args = parser.parse_args()

logging.basicConfig(format='%(asctime)s -- %(levelname)s - %(name)s - %(message)s', level=args.log_level)
logger = logging.getLogger(__name__)


def train(model, data_generator):
    # todo: 使用更加强大的optimizer，增加例如梯度裁剪、warm-up等技术
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    steps = 0
    best_performance = None
    require_improvement = 0
    total_loss = None
    for i in range(args.EPOCH):
        for question_token_ids, question_masks, head_id, answers_id in data_generator:
            scores = model(question_token_ids, question_masks, head_id).cpu()
            loss = []
            for score, answers in zip(scores, answers_id):
                target_scores = torch.index_select(score, 0, torch.tensor(answers))
                loss.append(torch.sum(-torch.log(target_scores)))
            loss = torch.stack(loss).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if steps % args.valid_steps == 0:
                # todo：增加测试部分，计算准确率
                logger.info('EPOCH: {}, STEP: {}, Precision: {}, Average Loss: {}'.format(i, steps, None, None))
                cur_performance = None
                if cur_performance > best_performance:
                    torch.save(model.state_dict(), args.save_path)
                    require_improvement = 0
                else:
                    require_improvement += 1
                    if require_improvement == args.require_improvement:
                        logger.warning('EXIT: training finished because of no improvement')
                        exit(-1)
    logger.info('finish training')


def main():
    exit('training finished because of no improvement')
    data_loader = DataLoader(
        train_file=args.train_file,
        valid_file=args.valid_file,
        test_file=args.test_file,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        dict_path=args.dict_path
    )
    model = QuestionAnswerModel(embed_model_path='./model_Tue Jan 26 19_24_46 2021.ckpt')
    model.to(model.device)
    train(model, data_loader.train_batch_generator())


if __name__ == '__main__':
    main()
