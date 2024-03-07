import argparse
from torch import distributed as dist
import torch
from data import build_dataloader
from evaluation.config_prcc import get_img_config as get_img_config_prcc
from main import parse_option
from models import build_model
from evaluation.test_reranking import test_reranking_prcc
from tools.utils import get_logger, set_seed
import os.path as osp
import os


def parse_option():
    parser = argparse.ArgumentParser(
        description='Train clothes-changing re-id model with clothes-based adversarial loss')
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
    # Datasets
    parser.add_argument('--dataset', type=str, default='prcc', help="ltcc, prcc, vcclothes, ccvid, last, deepchange")
    parser.add_argument('--root', type=str, default='', help="dataset path")
    parser.add_argument('--gen_path', type=str, default='', help="generated queries")
    # Miscs
    parser.add_argument('--output', type=str, help="your output path to save model and logs")

    parser.add_argument('--tag', type=str, help='tag for log file')
    parser.add_argument('--gpu', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

    args, unparsed = parser.parse_known_args()
    config = get_img_config_prcc(args)

    return config


def main(config, model):
    local_rank = dist.get_rank()
    model = model.cuda(local_rank)

    torch.cuda.set_device(local_rank)

    if config.EVAL_MODE:
        logger.info("Evaluate only")
        with torch.no_grad():
            test_reranking_prcc(model, queryloader_diff, galleryloader, dataset, config)



def create_model(config, dataset):
    # Build model
    model, _, _ = build_model(config, dataset)
    # Build optimizer
    model.load_state_dict(
        torch.load("./logs/prcc/res50-cels-cal/best_model.pth.tar")[
            'model_state_dict'], strict=False)

    return model


if __name__ == "__main__":
    config = parse_option()
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU
    # Init dist
    dist.init_process_group(backend="nccl", init_method='env://')
    local_rank = dist.get_rank()
    # Set random seed
    set_seed(config.SEED + local_rank)
    # get logger
    output_file = osp.join(config.OUTPUT, 'log_test.log')
    logger = get_logger(output_file, local_rank, 'reid')
    logger.info("Config:\n-----------------------------------------")
    logger.info(config)
    logger.info("-----------------------------------------")

    _, queryloader_same, queryloader_diff, galleryloader, dataset, train_sampler = build_dataloader(config)


    model = create_model(config, dataset)

    main(config, model)
