import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from agent import Agent
from environment import Environment
from settings import *
from data import MSCOCO


def captions_to_coco_format(img_ids, captions):
    return {
        img_id: [cap]
        for img_id, cap in zip(img_ids, captions)
    }


RUN_IDENTIFIER = time.strftime('RL-%m%d-%H%M-E')
env = Environment()
agent = Agent(LEARNING_RATE_RL)

agent.actor.load_state_dict(torch.load(
    MODEL_WEIGHTS, map_location=None if USE_CUDA else 'cpu'
)['model_state_dict'])

train_loader = DataLoader(MSCOCO('train', evaluation=True),
                          batch_size=BATCH_SIZE_RL, shuffle=SHUFFLE)
val_loader = DataLoader(MSCOCO('val', evaluation=True),
                        batch_size=BATCH_SIZE_RL, shuffle=SHUFFLE)

min_val_reward = 0.0
for e in range(10):
    epoch_start = time.time()

    # TRAINING
    rewards = []
    for img_ids, img_features, captions in train_loader:
        agent.actor_optim.zero_grad()

        # original shape: (5, 36, 2048), but when accessed per sample,
        # the resulting shape is (36, 2048), need it to be (1, 36, 2048)
        # use .unsqueeze() to make the original shape (5, 1, 36, 2048)
        img_features = img_features.unsqueeze(1)

        sampled_captions = []
        greedy_captions = []
        sampled_log_probs = []
        for img_feat in img_features:  # each iteration is a single image
            _, init_state, init_lstm_states = env.reset(img_feat)
            sampled_caption, probs = agent.inference(
                init_state, init_lstm_states, env)
            with torch.no_grad():
                greedy_caption, _ = agent.inference(
                    init_state, init_lstm_states, env, 'greedy')

            sampled_log_probs.append(probs)
            sampled_captions.append(sampled_caption)
            greedy_captions.append(greedy_caption)

        # pytorch's dataloader does something to the lists,
        # but i dont need it and i dont know how to turn it off.
        # so have to do this:
        captions = np.array(captions).T

        # transform ground truth and results to the format needed for eval
        ground_truth = dict(zip(img_ids, map(list, captions)))
        sampled_captions = captions_to_coco_format(
            img_ids, sampled_captions)
        greedy_captions = captions_to_coco_format(
            img_ids, greedy_captions)

        # calculate CIDEr scores
        mean_sample_scores, sample_scores = env.get_cider_score(
            ground_truth, sampled_captions)
        mean_greedy_scores, greedy_scores = env.get_cider_score(
            ground_truth, greedy_captions)

        # self-critical: score from sampling - score from test time algo
        advantages = torch.Tensor((sample_scores - greedy_scores).reshape(-1))
        if USE_CUDA:
            advantages = advantages.cuda()

        loss = -(advantages * torch.stack(sampled_log_probs)).mean()
        loss.backward()
        agent.actor_optim.step()
        rewards.append(mean_sample_scores)

    # VALIDATION
    rewards_val = []
    with torch.no_grad():
        for img_ids, img_features, captions in val_loader:
            img_features = img_features.unsqueeze(1)
            greedy_captions = []
            for img_feat in img_features:
                greedy_caption, _ = agent.inference(
                    init_state, init_lstm_states, env, 'greedy')
                greedy_captions.append(greedy_caption)

            captions = np.array(captions).T
            ground_truth = dict(zip(img_ids, map(list, captions)))
            greedy_captions = captions_to_coco_format(
                img_ids, greedy_captions)
            mean_greedy_scores, _ = env.get_cider_score(
                ground_truth, greedy_captions)
            rewards_val.append(mean_greedy_scores)

    val_reward = np.mean(rewards_val)

    # print('[B] Loss: {:.2f}. Mean reward: {:.2f}. Mean advantage: {:.2f}'.format(
    #     loss.item(), mean_sample_scores, advantages.mean()))

    print('Epoch {}. R train: {:.2f} R val: {:.2f}. {:.2f}'.format(
        e, np.mean(rewards), val_reward,
        time.time() - epoch_start))

    if val_reward > min_val_reward:
        min_val_reward = val_reward
        print('Higher mean validation reward achieved. Saving model.')
        torch.save({
            'epoch': e,
            'model_state_dict': agent.actor.state_dict(),
            'optimizer_state_dict': agent.actor_optim.state_dict()
        }, MODEL_DIR.format(RUN_IDENTIFIER + str(e)))
