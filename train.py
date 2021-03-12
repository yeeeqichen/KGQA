from dataloader import DataLoader
from QA_model import QuestionAnswerModel
import torch
import argparse
import logging
import tqdm
from pytorch_transformers import AdamW

parser = argparse.ArgumentParser()

parser.add_argument("--train_file", type=str, default="./MetaQA/QA_data/qa_train_1hop.txt")
parser.add_argument("--valid_file", type=str, default="./MetaQA/QA_data/qa_dev_1hop.txt")
parser.add_argument("--test_file", type=str, default="./MetaQA/QA_data/qa_test_1hop.txt")
parser.add_argument("--dict_path", type=str, default="./MetaQA/QA_data/entities.dict")
parser.add_argument("--relation_file", type=str, default='./MetaQA/KGE_data/relation2id.txt')
parser.add_argument("--batch_size", type=int, default=6)
parser.add_argument("--seq_length", type=int, default=20)
parser.add_argument("--EPOCH", type=int, default=10)
parser.add_argument("--valid_steps", type=int, default=1000)
parser.add_argument("--log_level", type=str, default="DEBUG")
parser.add_argument("--require_improvement", type=int, default=100)
parser.add_argument("--save_path", type=str, default="./model/")
parser.add_argument("--require_save", action='store_true', default=True)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--weight_decay", default=0.0, type=float)
parser.add_argument("--gamma", type=float, default=0.95)
parser.add_argument("--adam_epsilon", default=1e-8, type=float)
parser.add_argument("--max_gradient_norm", type=int, default=10)
parser.add_argument("--report_loss_steps", type=int, default=100)


args = parser.parse_args()

logging.basicConfig(format='%(asctime)s -- %(levelname)s - %(name)s - %(message)s', level=args.log_level)
logger = logging.getLogger(__name__)


# todo: self.constrain - scores(done);
#       loss = -log(sigmoid(margin - distance)), minimize the distance;
#       predict = -scores;
#       sort升序还是降序;
def train(model, data_loader):
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=args.gamma)
    # Prepare optimizer and schedule (linear warmup and decay)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.85, patience=0)
    best_performance = {'hits_1': 0, 'hits_3': 0, 'hits_10': 0}
    require_improvement = 0
    total_loss = None
    model.train()
    for i in range(args.EPOCH):
        steps = 1
        for question_token_ids, question_masks, head_id, answers_id in tqdm.tqdm(data_loader.batch_generator('train')):
            model.zero_grad()
            scores = model(question_token_ids, question_masks, head_id).cpu()
            loss = []
            for score, answers in zip(scores, answers_id):
                target_scores = torch.index_select(score, 0, torch.tensor(answers))
                # ****** warning! ****** 这里我之前似乎搞错了,score是l2距离,越小越好
                loss.append(torch.sum(-torch.log(torch.sigmoid(target_scores))))
                # loss.append(torch.sum(target_scores))
                # loss.append(torch.sum(-torch.log(target_scores)))
            loss = torch.stack(loss).mean()
            if steps % args.report_loss_steps == 0:
                logger.info('Loss: {}'.format(loss))
            loss.backward()
            # 进行梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_gradient_norm)
            optimizer.step()
            if steps % args.valid_steps == 0:
                logger.info('Start validating...')
                hits_1 = 0
                hits_3 = 0
                hits_10 = 0
                cnt = 0
                for _question, _mask, _head, _answers in data_loader.batch_generator('valid'):
                    scores = model(_question, _mask, _head).cpu()
                    # todo: 这里似乎搞反了
                    predicts = torch.sort(scores, dim=1, descending=True).indices
                    # predicts = torch.max(scores, 1, keepdim=True).indices.numpy()
                    for predict, _answer in zip(predicts, _answers):
                        cnt += 1
                        for j in range(10):
                            if predict[j] in _answer:
                                if j == 0:
                                    hits_1 += 1
                                if j < 3:
                                    hits_3 += 1
                                if j < 10:
                                    hits_10 += 1
                                break
                cur_performance = {'hits_1': hits_1 / cnt, 'hits_3': hits_3 / cnt, 'hits_10': hits_10 / cnt}
                logger.info('EPOCH: {}, STEP: {}, Hits_1: {}, Hits_3: {}, Hits_10: {}'
                            .format(i, steps, cur_performance['hits_1'], cur_performance['hits_3'],
                                    cur_performance['hits_10']))
                if cur_performance['hits_1'] > best_performance['hits_1']:
                    best_performance = cur_performance
                    if args.require_save:
                        logger.info('Saving model...')
                        torch.save(model.state_dict(), args.save_path + 'model.pkl')
                        with open(args.save_path + 'performance.txt', 'w') as f:
                            f.write("Best Performance, Hits_1: {}, Hits_3: {}, Hist_10: {}".
                                    format(best_performance['hits_1'], best_performance['hits_3'],
                                           best_performance['hits_10']))
                    require_improvement = 0
                else:
                    require_improvement += 1
                    if require_improvement == args.require_improvement:
                        logger.warning('EXIT: training finished because of no improvement')
                        exit(-1)
            steps += 1
        # 进行学习率衰减
        # scheduler.step()
    logger.info('finish training')


def main():
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
    train(model, data_loader)


if __name__ == '__main__':
    main()
