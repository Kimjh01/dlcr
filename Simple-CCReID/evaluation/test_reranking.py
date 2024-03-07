import os
import random
import time
import datetime
import logging
import numpy as np
import torch
import os
import sys
import time

# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)

# adding the parent directory to
# the sys.path.
sys.path.append(parent)
import torch.nn.functional as F
from torch import distributed as dist
from tools.eval_metrics import evaluate, evaluate_with_clothes
import copy

VID_DATASET = ['ccvid']


@torch.no_grad()
def extract_img_feature(model, dataloader, image_dir=None):
    features, pids, camids, clothes_ids = [], torch.tensor([]), torch.tensor([]), torch.tensor([])
    img_names = []
    features_top5 = []
    dataloader.dataset.image_dir = image_dir
    for batch_idx, (imgs, batch_pids, batch_camids, batch_clothes_ids, batch_img_names) in enumerate(dataloader):
        imgs = imgs.cuda()
        batch_features = model(imgs)
        batch_features = F.normalize(batch_features, p=2, dim=1)
        features.append(batch_features.cpu())
        img_names.extend(batch_img_names)
        pids = torch.cat((pids, batch_pids.cpu()), dim=0)
        camids = torch.cat((camids, batch_camids.cpu()), dim=0)
        clothes_ids = torch.cat((clothes_ids, batch_clothes_ids.cpu()), dim=0)
    features = torch.cat(features, 0)
    return features, features_top5, pids, camids, clothes_ids, img_names

def test_reranking_prcc(model, queryloader_diff, galleryloader, dataset, config):
    logger = logging.getLogger('reid.test')
    since = time.time()
    model.eval()
    path_reid = os.path.join(config.DATA.ROOT, dataset.dataset_dir)
    image_dir_gallery = path_reid + '/rgb/test/A'
    dist_matrices = []
    print(os.path.exists(config.DATA.QUERY_CLOTHES))
    query_clothes = os.listdir(config.DATA.QUERY_CLOTHES)
    query_clothes = ['original'] + query_clothes[:4]
    gf, gf_top5, g_pids, g_camids, g_clothes_ids, _ = extract_img_feature(model, galleryloader,
                                                                          image_dir=image_dir_gallery)
    g_pids, g_camids, g_clothes_ids = g_pids.numpy(), g_camids.numpy(), g_clothes_ids.numpy()
    top1s = []
    times = []
    for i in range(len(query_clothes)):
        cloth_id = query_clothes[i]
        if cloth_id == 'original':
            image_dir_query = os.path.join(path_reid, "rgb", "test", "C")
        else:
            image_dir_query = os.path.join(config.DATA.ROOT_QUERY, cloth_id)

        qf, qf_top5, q_pids, q_camids, q_clothes_ids, q_img_names = extract_img_feature(model, queryloader_diff,
                                                                                        image_dir_query)
        torch.cuda.empty_cache()
        time_elapsed = time.time() - since

        logger.info("Extracted features for query set, obtained {} matrix".format(qf.shape))
        logger.info("Extracted features for gallery set, obtained {} matrix".format(gf.shape))
        logger.info('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        # Compute distance matrix between query and gallery
        since = time.time()
        m, n = qf.size(0), gf.size(0)
        distmat = torch.zeros((m, n))
        qf, gf = qf.cuda(), gf.cuda()
        # Cosine similarity
        for j in range(m):
            distmat[j] = (- torch.mm(qf[j:j + 1], gf.t())).cpu()
        distmat = distmat.numpy()
        if i == 0:
            original_distmat = (distmat * (-1)) + 1
        dist_matrices.append((distmat * (-1)) + 1)
        q_pids, q_camids, q_clothes_ids = q_pids.numpy(), q_camids.numpy(), q_clothes_ids.numpy()

        time_elapsed = time.time() - since
        logger.info('Distance computing in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        since = time.time()

        logger.info("Computing CMC and mAP only for clothes-changing")
        cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
        logger.info("Results ---------------------------------------------------")
        logger.info(
            'top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
        logger.info("-----------------------------------------------------------")
        top1s.append(cmc[0])


    list_dict_top5 = {}
    for i, dist_mat in enumerate(dist_matrices):

        index = np.argsort(dist_matrices[i], axis=1)
        for j in range(len(q_pids)):
            query_index = np.argwhere(g_pids == q_pids[j])
            camera_index = np.argwhere(g_camids == q_camids[j])
            good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
            if good_index.size == 0:
                continue
            # remove junk_index
            junk_index = np.intersect1d(query_index, camera_index)

            mask = np.in1d(index[j], junk_index, invert=True)
            top_5_indices = index[j][mask]

            top_5_indices = top_5_indices[-5:]

            dict_top5 = {'pid': q_pids[j], 'index': i}
            for k, gallery_pid in enumerate(g_pids[top_5_indices]):
                dict_top5[gallery_pid] = dist_mat[j][top_5_indices[k]]

            if (str(q_pids[j]) + q_img_names[j]) in list_dict_top5:
                list_dict_top5[str(q_pids[j]) + q_img_names[j]].append(dict_top5)
            else:
                list_dict_top5[str(q_pids[j]) + q_img_names[j]] = [dict_top5]

    pred = 0
    list_new_top5 = []
    for example in list_dict_top5:

        list_top5 = list_dict_top5[example]
        current_pid = list_top5[0]['pid']
        single_top5 = copy.deepcopy(list_top5[0])  # {}#
        for top5 in list_top5:
            max_s = 0
            for person in top5:
                if person == 'pid' or person == 'index':
                    continue
                if max_s < top5[person]:
                    max_s = top5[person]
            for person in top5:
                if person == 'pid' or person == 'index':
                    continue
                if person in single_top5:
                    single_top5[person] += (top5[person] / (max_s)) * top5[person]  # (top5[person]/max_s)*
        pids = []
        values = []
        list_new_top5.append(single_top5)
        for person in single_top5:
            if person == 'pid' or person == 'index':
                continue
            pids.append(person)
            values.append(single_top5[person])
        values = np.array(values)
        pos_max = np.argmax(values)

        pred_pid = pids[pos_max]
        pred += int(pred_pid == current_pid)
    arg_original_distmat = np.argsort(original_distmat, axis=1)
    indices_top5 = arg_original_distmat[:, -5:]
    for i, top5 in enumerate(list_new_top5):

        for index in indices_top5[i]:
            pid = g_pids[index]

            original_distmat[i, index] = max(top5[int(pid)], original_distmat[i, index])

    cmc, mAP = evaluate(-original_distmat, q_pids, g_pids, q_camids, g_camids)
    logger.info("Results ---------------------------------------------------")
    logger.info(
        'top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP))
    logger.info("-----------------------------------------------------------")

    logger.info(f'Top-1 acc:{pred / len(list_dict_top5)}')
    return cmc[0]