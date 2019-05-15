import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from agent import Agent
from environment import Environment
from settings import *
from data import MSCOCO


def forward(img_features, mode='sample'):
    predictions = []
    log_probs = []
    _, state, lstm_states = env.reset(img_features)

    # this should store the index of the first occurrence of <EOS>
    # for each sample in the batch
    EOS_tracker = np.full(img_features.shape[0], None)
    for i in range(MAX_WORDS):
        word_logits, lstm_states = agent.actor(state, lstm_states)
        probs = F.softmax(word_logits, dim=1)  # check if dim=1 is correct
        if mode == 'sample':
            idxs = torch.multinomial(probs, 1)
        else:
            idxs = torch.argmax(probs, dim=1)
        words = env.vocabulary[idxs]
        predictions.append(words.reshape(-1))
        log_probs.append([
            lp[i] for (lp, i) in zip(torch.log(probs), idxs)
        ])

        # inefficient but this should be fast enough... ? :(
        eos_idxs = (words == '<EOS>').nonzero()[0]
        for idx in eos_idxs:
            if EOS_tracker[idx] is None:
                EOS_tracker[idx] = i + 1

        # finish loop if they're all done
        if all(EOS_tracker != None):
            break

        state['language_lstm_h'] = lstm_states['language_h']
        state['prev_word_indeces'] = idxs.reshape(-1)

    captions = [
        [' '.join(w[:i])] for (w, i) in
        zip(np.array(predictions).T, EOS_tracker)
    ]
    if mode == 'sample':
        log_probs = [
            lp[:i].sum() for (lp, i) in
            zip(np.array(log_probs).T, EOS_tracker)
        ]
        return captions, log_probs
    return captions



RUN_IDENTIFIER = time.strftime('RL-%m%d-%H%M-E')
env = Environment()
agent = Agent(LEARNING_RATE_RL)

agent.actor.load_state_dict(torch.load(
    MODEL_WEIGHTS, map_location=None if USE_CUDA else 'cpu'
)['model_state_dict'])

train_loader = DataLoader(MSCOCO('train', evaluation=True),
                          batch_size=BATCH_SIZE_RL, shuffle=SHUFFLE, pin_memory=True)
val_loader = DataLoader(MSCOCO('val', evaluation=True),
                        batch_size=1000, shuffle=SHUFFLE, pin_memory=True)


val_scores = []
for img_ids, img_features, captions in val_loader:
    with torch.no_grad():
        greedy_captions = forward(img_features, 'greedy')
    greedy_captions = dict(zip(img_ids, greedy_captions))

    captions = np.array(captions).T
    ground_truth = dict(zip(img_ids, map(list, captions)))

    _, val_greedy_scores = env.cider.compute_score(
        ground_truth, greedy_captions)
    val_scores.append(val_greedy_scores)

val_reward = np.mean(val_greedy_scores)

print('Starting val CIDEr score: ', val_reward)

print('RUN IDENTIFIER: ', RUN_IDENTIFIER)
print('LEARNING RATE: ', LEARNING_RATE_RL)
print('DECAY PER {} EPOCHS: {}'.format(LR_DECAY_STEP_SIZE, LR_DECAY_PER_EPOCH))
print('BATCH SIZE: ', BATCH_SIZE_RL)
print('TOTAL BATCHES: ', len(train_loader), '\n')
print('\nStarting REINFORCE training.\n')

max_val_reward = val_reward
for e in range(10):
    epoch_start = time.time()

    # TRAINING
    rewards = []
    g_rewards = []
    for b, (img_ids, img_features, captions) in enumerate(train_loader):
        agent.actor_optim.zero_grad()

        sampled_captions, sampled_log_probs = forward(img_features)
        sampled_captions = dict(zip(img_ids, sampled_captions))
        with torch.no_grad():
            greedy_captions = forward(img_features, 'greedy')
            greedy_captions = dict(zip(img_ids, greedy_captions))

        # pytorch's dataloader does something to the lists,
        # but i dont need it and i dont know how to turn it off.
        # so have to do this:
        captions = np.array(captions).T

        # transform ground truth and results to the format needed for eval
        ground_truth = dict(zip(img_ids, map(list, captions)))
        _, sample_scores = env.cider.compute_score(
            ground_truth, sampled_captions)
        _, greedy_scores = env.cider.compute_score(
            ground_truth, greedy_captions)

        # self-critical: score from sampling - score from test time algo
        advantages = torch.Tensor((sample_scores - greedy_scores).reshape(-1))
        # normalize advantages
        advantages = ((advantages - advantages.mean()) /
                      advantages.std() + 1e-9)
        if USE_CUDA:
            advantages = advantages.cuda()
        loss = -(advantages * torch.stack(sampled_log_probs).reshape(-1)).mean()
        loss.backward()
        agent.actor_optim.step()

        rewards.extend(sample_scores)
        g_rewards.extend(greedy_scores)

        if (b + 1) % 50 == 0:
            print('\t[Batch {} running metrics] - R train {:.2f} - R train (greedy): {:.2f}'.format(
                b + 1, np.mean(rewards), np.mean(g_rewards)))

    # VALIDATION
    val_scores = []
    for img_ids, img_features, captions in val_loader:
        with torch.no_grad():
            greedy_captions = forward(img_features, 'greedy')
        greedy_captions = dict(zip(img_ids, greedy_captions))

        captions = np.array(captions).T
        ground_truth = dict(zip(img_ids, map(list, captions)))

        _, val_greedy_scores = env.cider.compute_score(
            ground_truth, greedy_captions)
        val_scores.append(val_greedy_scores)
    val_reward = np.mean(val_greedy_scores)

    print('Epoch {} - R train: {:.2f} - R train (greedy): {:.2f} - R val: {:.2f} ({:.2f}s)'.format(
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
