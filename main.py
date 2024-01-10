import argparse
import itertools
import logging
import time
import datetime
import os
import torch
# from torch.utils.tensorboard import SummaryWriter

from utils.data_helper import DataSet
from utils.link_prediction import run_link_prediction
from model.framework import LAN
from TransE_PyTorch.model import TransE
from tqdm import tqdm
from NSCaching.BernCorrupter import BernCorrupter

logger = logging.getLogger()


def main():
    config = parse_arguments()
    run_training(config)


def parse_arguments():
    """ Parses arguments from CLI. """
    parser = argparse.ArgumentParser(description="Configuration for LAN model")
    parser.add_argument('--data_dir', '-D', type=str, default="data/FB15k-237/head-10")
    parser.add_argument('--save_dir', '-S', type=str, default="data/FB15k-237/head-10")
    # model
    parser.add_argument('--use_relation', type=int, default=1)
    parser.add_argument('--embedding_dim', '-e', type=int, default=100)
    parser.add_argument('--max_neighbor', type=int, default=64)
    parser.add_argument('--n_neg', '-n', type=int, default=1)
    parser.add_argument('--aggregate_type', type=str, default='attention')
    parser.add_argument('--score_function', type=str, default='TransE')
    parser.add_argument('--loss_function', type=str, default='margin')
    parser.add_argument('--margin', type=float, default='1.0')
    parser.add_argument('--corrupt_mode', type=str, default='both')
    # training
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--num_epoch', type=int, default=2000)
    parser.add_argument('--weight_decay', '-w', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--evaluate_size', type=int, default=250)
    parser.add_argument('--steps_per_display', type=int, default=100)
    parser.add_argument('--epoch_per_checkpoint', type=int, default=50)
    # NSCaching
    parser.add_argument('--is_use_NSCaching', type=bool, default=False)
    parser.add_argument('--N_1', type=int, default=30)
    parser.add_argument('--N_2', type=int, default=90)
    # gpu option
    parser.add_argument('--gpu_fraction', type=float, default=0.2)
    parser.add_argument('--gpu_device', type=str, default='0')
    parser.add_argument('--allow_soft_placement', type=bool, default=False)
    # for analysis
    parser.add_argument('--attention_record', type=bool, default=False)

    return parser.parse_args()


def run_training(config):
    # set up GPU
    config.device = torch.device("cuda:0")

    set_up_logger(config)

    logger.info('args: {}'.format(config))

    # writer = SummaryWriter("display")
    # initialize
    model_pretrain = TransE(10336, 1170, config.device)
    # checkpoint = torch.load("./TransE_PyTorch/checkpoint.tar")
    # model_pretrain.load_state_dict(checkpoint["model_state_dict"])

    entity_embeddings = model_pretrain.entities_emb.weight.data
    relation_embeddings = model_pretrain.relations_emb.weight.data

    # 可选：转换为NumPy数组
    entity_embeddings_np = entity_embeddings.cpu().numpy()
    relation_embeddings_np = relation_embeddings.cpu().numpy()

    # prepare data
    logger.info("Loading data...")
    dataset = DataSet(config, logger)
    logger.info("Loading finish...")

    # dataset.get_cache()

    # corrputer = BernCorrupter(dataset.triplets_train, dataset.num_entity, dataset.num_relation*2+1)
    # dataset.corrupter = corrputer

    # 网格搜索超参数
    embedding_dims = [200, 100, 500]
    learning_rates = [1e-3, 1e-2, 1e-1]
    batch_sizes = [512, 1024, 2048]
    nums_epochs = [2000, 1000]
    losses = ['margin', 'CE']
    # optimizers = ['Adam', 'SGD', 'RMSprop']

    best_performance = -float("inf")

    for embedding_dim, learning_rate, batch_size, num_epoch in itertools.product(embedding_dims, learning_rates, batch_sizes, nums_epochs):
        config.embedding_dim = embedding_dim
        config.learning_rate = learning_rate
        config.batch_size = batch_size
        config.num_epoch = num_epoch
        current_hyperparameters_message = 'current hyperparameters,' \
                                          'embedding_dim:{}, ' \
                                          'learning_rate:{}, ' \
                                          'batch_size:{}, ' \
                                          'num_epoch:{}'.format(embedding_dim, learning_rate, batch_size, num_epoch)
        print(current_hyperparameters_message)

        model = LAN(config, dataset.num_training_entity, dataset.num_relation)
        save_path = os.path.join(config.save_dir, "train_model_DKGT_new.pt")
        model.to(config.device)
        # L2正则化?
        optim = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

        dataset.model = model

        # training
        num_batch = dataset.num_sample // config.batch_size
        logger.info('Train with {} batches'.format(num_batch))

        for epoch in range(config.num_epoch):
            st_epoch = time.time()
            loss_epoch = 0.
            cnt_batch = 0
            for batch_data in tqdm(dataset.batch_iter_epoch(dataset.triplets_train, config.batch_size, config.n_neg)):
                model.train()
                st_batch = time.time()
                loss_batch = model.loss(batch_data)
                cnt_batch += 1
                loss_epoch += loss_batch.item()
                loss_batch.backward()
                optim.step()
                model.zero_grad()
                en_batch = time.time()
                # print an overview every some batches
                # if (cnt_batch + 1) % config.steps_per_display == 0 or (cnt_batch + 1) == num_batch:
                # batch_info = 'epoch {}, batch {}, loss: {:.3f}, time: {:.3f}s'.format(epoch, cnt_batch, loss_batch, en_batch - st_batch)
                # print(batch_info)
                # logger.info(batch_info)
            en_epoch = time.time()
            epoch_info = 'epoch {}, mean loss: {:.3f}, time: {:.3f}s'.format(epoch, loss_epoch / cnt_batch,
                                                                             en_epoch - st_epoch)
            # writer.add_scalar("DKGT-gatv2-singlehead-train1", loss_epoch / cnt_batch, epoch)
            print(epoch_info)
            logger.info(epoch_info)

            # evaluate the model every some steps
            if (epoch + 1) % config.epoch_per_checkpoint == 0 or (epoch + 1) == config.num_epoch:
                model.eval()
                st_test = time.time()
                with torch.no_grad():
                    performance, hit10 = run_link_prediction(config, model, dataset, epoch, logger, is_test=False)
                    # writer.add_scalar("DKGT-gatv2-singlehead-test1", hit10, epoch)
                if performance > best_performance:
                    best_performance = performance
                    torch.save(model.state_dict(), save_path)
                    time_str = datetime.datetime.now().isoformat()
                    saved_message = '{}: model at epoch {} save in file {}'.format(time_str, epoch, save_path)
                    print(saved_message)
                    hyperparameters_message = 'better performance with hyperparameters,' \
                                          'embedding_dim:{}, ' \
                                          'learning_rate:{}, ' \
                                          'batch_size:{}, ' \
                                          'num_epoch:{}'.format(embedding_dim, learning_rate, batch_size, num_epoch)
                    print(hyperparameters_message)
                    logger.info(saved_message)
                en_test = time.time()
                test_finish_message = 'testing finished with time: {:.3f}s'.format(en_test - st_test)
                print(test_finish_message)
                logger.info(test_finish_message)

    finished_message = 'Training finished'
    print(finished_message)
    logger.info(finished_message)

    epoch = config.num_epoch
    st_test = time.time()
    run_link_prediction(config, model, dataset, epoch, logger, is_test=True)
    en_test = time.time()
    logger.info('Testing finished with time: {:.3f}s'.format(en_test - st_test))


def set_up_logger(config):
    checkpoint_dir = config.save_dir
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(checkpoint_dir + 'train.log', 'w+')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


if __name__ == '__main__':
    main()
