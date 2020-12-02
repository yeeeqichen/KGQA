import argparse
import time
import torch
from load_data import DataLoader
from model import KnowledgeEmbeddingModel
from tqdm import tqdm
import numpy as np


def evaluate():
    pass


if __name__ == '__main__':
    def handle_sample(sample, cuda, num_entity):
        idx1 = torch.tensor(sample[:, 0])
        idx2 = torch.tensor(sample[:, 1])
        idx3 = torch.tensor(sample[:, 2])
        if args.cuda:
            idx1 = idx1.cuda()
            idx2 = idx2.cuda()
            idx3 = idx3.cuda()
        one_hot = torch.nn.functional.one_hot(idx3, num_classes=num_entity)
        return idx1, idx2, idx3, one_hot

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MetaQA", nargs="?",
                        help="Which dataset to use: FB15k, FB15k-237, WN18 or WN18RR.")
    parser.add_argument("--EPOCH", type=int, default=500, nargs="?",
                        help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=32, nargs="?",
                        help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.001, nargs="?",
                        help="Learning rate.")
    parser.add_argument("--model", type=str, default='complEx', nargs="?",
                        help="Model.")
    parser.add_argument("--embed_dim", type=int, default=128, nargs="?",
                        help="embedding dimension")
    parser.add_argument("--valid_step", type=int, default=1, nargs="?",
                        help="do validation after how many train iter")
    parser.add_argument("--cuda", type=bool, default=False, nargs="?",
                        help="use gpu or not")
    args = parser.parse_args()
    loader = DataLoader(data_directory='../data/' + args.dataset, batch_size=args.batch_size)
    loader.load_files()
    model = KnowledgeEmbeddingModel(loader.num_entity, loader.num_relation, args.embed_dim,
                                    score_func=args.model)
    if args.cuda:
        model = model.cuda()
    print(model)
    opt = torch.optim.Adam(lr=args.lr, params=model.parameters())
    start_train = time.time()
    total_loss = 0
    for epoch in range(args.EPOCH):
        model.train()
        for train_sample in tqdm(loader.batch_generator(purpose='train')):
            head_idx, relation_idx, _, target_one_hot = handle_sample(train_sample, args.cuda, loader.num_entity)
            predict = model(head_idx, relation_idx)
            loss = model.ce_loss(predict, target_one_hot)
            total_loss += loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % 100 == 0:
            print('Epoch', epoch, ' Epoch time', time.time() - start_train, ' Loss:', total_loss / 100)
            total_loss = 0
        if epoch % args.valid_step == 0:
            model.eval()
            with torch.no_grad():
                hits10 = 0
                hits3 = 0
                hits1 = 0
                ranks = []
                for valid_sample in tqdm(loader.batch_generator(purpose='valid')):
                    head_idx, relation_idx, tail_idx, _ = handle_sample(valid_sample, args.cuda, loader.num_entity)
                    predict = model(head_idx, relation_idx)
                    sort_values, sort_idx = torch.sort(predict, dim=1, descending=True)
                    # (batch, num_entity)
                    sort_idx = sort_idx.cpu().numpy()
                    for j in range(valid_sample.shape[0]):
                        rank = np.where(sort_idx[j] == tail_idx[j].item())[0][0]
                        ranks.append(rank + 1)
                        if rank < 10:
                            hits10 += 1
                        if rank < 3:
                            hits3 += 1
                        if rank < 1:
                            hits1 += 1
                hits10_score = hits10 / len(ranks)
                hits3_score = hits3 / len(ranks)
                hits1_score = hits1 / len(ranks)
                mean_rank = np.mean(ranks)
                mrr = np.mean(1.0 / np.array(ranks))
                print("valid\nhits10:{}\nhits3:{}\nhits1:{}\nmean_rank:{}\nmrr:{}".
                      format(hits10_score, hits3_score, hits1_score, mean_rank, mrr))
