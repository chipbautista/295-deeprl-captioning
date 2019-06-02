import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from bert_serving.client import BertClient

from agent import Agent
from environment import Environment
from settings import *
from data import MSCOCO


def evaluate_cider(data_loader):
    ground_truths = {}
    predictions = {}
    for img_ids, img_features, captions in data_loader:
        ground_truths.update(
            dict(zip(img_ids, map(list, np.array(captions).T))))

        with torch.no_grad():
            greedy_captions = agent.predict_captions(
                img_features, 'greedy', constrain=True)
        predictions.update(dict(zip(img_ids, greedy_captions)))
    cider_score, _ = env.cider.compute_score(ground_truths, predictions)
    return cider_score


def get_scores(gt_bert_vectors, preds, gt_dict, preds_dict):
    _, cider_scores = env.cider.compute_score(gt_dict, preds_dict)
    if INCLUDE_CONTEXT_SCORE:
        context_scores = env.get_context_score(gt_bert_vectors, preds)
        reward = (BETA * cider_scores) + ((1 - BETA) * context_scores)
        return cider_scores, context_scores, reward
    return cider_scores, None, None


RUN_IDENTIFIER = time.strftime('RL-%m%d-%H%M-E')

if INCLUDE_CONTEXT_SCORE:
    # Run this first, else the program won't continue
    # bert-serving-start -model_dir ../data/bert_models/uncased_L-12_H-768_A-12/ -max_seq_len 30
    # add "-cpu" if running on CPU.
    bert_client = BertClient(check_length=False)
else:
    bert_client = None

env = Environment(bert_client)
agent = Agent(LEARNING_RATE_RL, env)

agent.actor.load_state_dict(torch.load(
    MODEL_WEIGHTS, map_location=None if USE_CUDA else 'cpu'
)['model_state_dict'])

train_loader = DataLoader(MSCOCO('train', evaluation=True),
                          batch_size=BATCH_SIZE_RL, shuffle=SHUFFLE,
                          pin_memory=True)
val_loader = DataLoader(MSCOCO('val', evaluation=True),
                        batch_size=500, shuffle=SHUFFLE, pin_memory=True)

# val_reward = evaluate_cider(val_loader)
val_reward = 0
print('Starting val CIDEr score: ', val_reward)
max_val_reward = val_reward

print('RUN IDENTIFIER: ', RUN_IDENTIFIER)
print('Loaded model weights: ', MODEL_WEIGHTS)
print('LEARNING RATE: ', LEARNING_RATE_RL)
print('INCLUDE_CONTEXT_SCORE: {}. BETA={}'.format(INCLUDE_CONTEXT_SCORE, BETA))
# print('DECAY PER {} EPOCHS: {}'.format(LR_DECAY_STEP_SIZE, LR_DECAY_PER_EPOCH))
print('BATCH SIZE: ', BATCH_SIZE_RL)
print('TOTAL BATCHES: ', len(train_loader), '\n')
print('\nStarting REINFORCE training.\n')


for e in range(20):
    epoch_start = time.time()

    # TRAINING
    rewards = {
        'sample_cider': [],
        'sample_context': [],
        'sample_reward': [],  # actual reward, controlled by beta
        'greedy_cider': [],
        'greedy_context': [],
        'greedy_reward': []
    }
    for b, (img_ids, img_features, captions) in enumerate(train_loader):
        agent.actor_optim.zero_grad()

        sampled_captions, sampled_log_probs = agent.predict_captions(
            img_features, mode='sample')
        sampled_dict = dict(zip(img_ids, sampled_captions))
        with torch.no_grad():
            greedy_captions = agent.predict_captions(img_features, 'greedy')
            greedy_dict = dict(zip(img_ids, greedy_captions))

        # load the ground truth BERT vectors in here
        gt_bert_vectors = [
            np.load(CAPTION_VECTORS_DIR.format('train', img_id))
            for img_id in img_ids]

        # transform ground truth and results to the format needed for eval
        gt_dict = dict(zip(img_ids, map(list, np.array(captions).T)))

        sample_cider_score, sample_context_score, sample_reward = get_scores(
            gt_bert_vectors, sampled_captions, gt_dict, sampled_dict)
        greedy_cider_score, greedy_context_score, greedy_reward = get_scores(
            gt_bert_vectors, greedy_captions, gt_dict, greedy_dict)

        # self-critical: score from sampling - score from test time
        advantages = torch.Tensor((sample_reward - greedy_reward).reshape(-1))

        # normalize advantages
        advantages = ((advantages - advantages.mean()) /
                      advantages.std() + 1e-9)
        if USE_CUDA:
            advantages = advantages.cuda()
        loss = -(advantages *
                 torch.stack(sampled_log_probs).reshape(-1)).mean()
        loss.backward()
        agent.actor_optim.step()

        rewards['sample_cider'].extend(sample_cider_score)
        rewards['sample_context'].extend(sample_context_score)
        rewards['sample_reward'].extend(sample_reward)
        rewards['greedy_cider'].extend(greedy_cider_score)
        rewards['greedy_context'].extend(greedy_context_score)
        rewards['greedy_reward'].extend(greedy_reward)

        if (b + 1) % 200 == 0:
            print('\t[Batch {} running metrics] - R train {:.2f} - R train (greedy): {:.2f}'.format(
                b + 1, np.mean(rewards['sample_reward']), np.mean(rewards['greedy_reward'])))

    # VALIDATION
    # val_reward = evaluate_cider(val_loader)
    gt_dict = {}
    greedy_dict = {}
    gt_bert_vectors = []
    greedy_captions = []
    for img_ids, img_features, captions in val_loader:
        with torch.no_grad():
            greedy_captions_ = agent.predict_captions(
                img_features, 'greedy', constrain=True)

        greedy_captions.extend(greedy_captions_)
        greedy_dict.update(dict(zip(img_ids, greedy_captions_)))
        gt_dict.update(dict(zip(img_ids, map(list, np.array(captions).T))))
        gt_bert_vectors.extend([
            np.load(CAPTION_VECTORS_DIR.format('val', img_id))
            for img_id in img_ids])

    val_cider_score, val_context_score, val_score = get_scores(
            gt_bert_vectors, greedy_captions, gt_dict, greedy_dict)
    # import pdb; pdb.set_trace()
    print('Epoch {} - R train: {:.2f} - R train (greedy): {:.2f} - R val (CIDEr): {:.2f} - R val (Context): {:.2f} ({:.2f}s)'.format(
        e, np.mean(rewards['sample_reward']), np.mean(rewards['greedy_reward']),
        val_cider_score.mean(), val_context_score.mean(), time.time() - epoch_start))
    print('Sample CIDEr: {:.4f}. Sample context: {:.4f}. Greedy CIDEr: {:.4f}. Greedy context: {:.4f}'.format(
        np.mean(rewards['sample_cider']), np.mean(rewards['sample_context']),
        np.mean(rewards['greedy_cider']), np.mean(rewards['greedy_context'])))

    val_score = val_score.mean()
    if (val_score > max_val_reward) or e == 9:
        max_val_reward = val_score
        print('Higher mean validation reward achieved. Saving model.')
        torch.save({
            'epoch': e,
            'model_state_dict': agent.actor.state_dict(),
            'optimizer_state_dict': agent.actor_optim.state_dict()
        }, MODEL_DIR.format(RUN_IDENTIFIER + str(e)))
