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


def get_scores(gt, gt_, preds, preds_):
    # predictions_ is basically the same as predictions
    # but it's a list instead of a dict.
    _, cider_scores = env.cider.compute_score(gt, preds)
    if INCLUDE_CONTEXT_SCORE:
        context_scores = env.get_context_score(gt_, preds_)
        return (BETA * cider_scores) + ((1 - BETA) * context_scores)
    return cider_scores


RUN_IDENTIFIER = time.strftime('RL-%m%d-%H%M-E')
env = Environment(BertClient())
agent = Agent(LEARNING_RATE_RL, env)

agent.actor.load_state_dict(torch.load(
    MODEL_WEIGHTS, map_location=None if USE_CUDA else 'cpu'
)['model_state_dict'])
print('Loaded model weights: ', MODEL_WEIGHTS)

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
print('LEARNING RATE: ', LEARNING_RATE_RL)
print('DECAY PER {} EPOCHS: {}'.format(LR_DECAY_STEP_SIZE, LR_DECAY_PER_EPOCH))
print('BATCH SIZE: ', BATCH_SIZE_RL)
print('TOTAL BATCHES: ', len(train_loader), '\n')
print('\nStarting REINFORCE training.\n')


for e in range(20):
    epoch_start = time.time()

    # TRAINING
    rewards = []
    g_rewards = []
    for b, (img_ids, img_features, captions) in enumerate(train_loader):
        agent.actor_optim.zero_grad()

        sampled_captions_, sampled_log_probs = agent.predict_captions(
            img_features)
        sampled_captions = dict(zip(img_ids, sampled_captions_))
        with torch.no_grad():
            greedy_captions_ = agent.predict_captions(img_features, 'greedy')
            greedy_captions = dict(zip(img_ids, greedy_captions_))

        # pytorch's dataloader does something to the lists,
        # but i dont need it and i dont know how to turn it off.
        # so have to do this:
        captions = np.array(captions).T

        # transform ground truth and results to the format needed for eval
        ground_truth = dict(zip(img_ids, map(list, captions)))
        sample_reward = get_scores(
            ground_truth, captions, sampled_captions, sampled_captions_)
        greedy_reward = get_scores(
            ground_truth, captions, greedy_captions, greedy_captions_)

        # self-critical: score from sampling - score from test time algo
        advantages = torch.Tensor(
            (sample_reward - greedy_reward).reshape(-1))

        # normalize advantages
        advantages = ((advantages - advantages.mean()) /
                      advantages.std() + 1e-9)
        if USE_CUDA:
            advantages = advantages.cuda()
        loss = -(advantages *
                 torch.stack(sampled_log_probs).reshape(-1)).mean()
        loss.backward()
        agent.actor_optim.step()

        rewards.extend(sample_cider_scores)
        g_rewards.extend(greedy_cider_scores)

        if (b + 1) % 50 == 0:
            print('\t[Batch {} running metrics] - R train {:.2f} - R train (greedy): {:.2f}'.format(
                b + 1, np.mean(rewards), np.mean(g_rewards)))

    # VALIDATION
    val_reward = evaluate_cider(val_loader)
    print('Epoch {} - R train: {:.2f} - R train (greedy): {:.2f} - R val (CIDEr): {:.2f} ({:.2f}s)'.format(
        e, np.mean(rewards), np.mean(g_rewards), val_reward,
        time.time() - epoch_start))

    if val_reward > max_val_reward:
        max_val_reward = val_reward
        print('Higher mean validation reward achieved. Saving model.')
        torch.save({
            'epoch': e,
            'model_state_dict': agent.actor.state_dict(),
            'optimizer_state_dict': agent.actor_optim.state_dict()
        }, MODEL_DIR.format(RUN_IDENTIFIER + str(e)))
