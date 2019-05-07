import time
# from torch.multiprocessing import Pool

import numpy as np
import torch
from torch.utils.data import DataLoader

from agent import Agent
from environment import Environment
from settings import *
from data import MSCOCO


# Try Monte-Carlo First


env = Environment()
agent = Agent()
agent.actor.load_state_dict(
    torch.load(MODEL_WEIGHTS, map_location='cpu')['model_state_dict'])

train_loader = DataLoader(MSCOCO('val', evaluation=True),
                          batch_size=BATCH_SIZE_RL, shuffle=SHUFFLE)
# val_loader = DataLoader(MSCOCO('val', evaluation=True), shuffle=SHUFFLE)

for e in range(MAX_EPOCH):
    batch_rewards = []
    epoch_start = time.time()
    for i, (img_ids, img_features, captions) in enumerate(train_loader):
        agent.actor_optim.zero_grad()

        # pytorch's dataloader does something to the lists,
        # but i dont need it and i dont know how to turn it off.
        # so have to do this:
        captions = np.array(captions).T

        # original shape: (5, 36, 2048), but when accessed per sample,
        # the resulting shape is (36, 2048), need it to be (1, 36, 2048)
        # use .unsqueeze() to make the original shape (5, 1, 36, 2048)
        img_features = img_features.unsqueeze(1)

        # run them async (for monte-carlo case.) (how to make this work?)
        # with Pool() as p:
        #     predictions = p.map(agent.predict_captions, img_features)

        sampled_captions = []
        greedy_captions = []
        sampled_probs = []
        for img_feat in img_features:  # each iteration is a single image
            _, init_state, init_lstm_states = env.reset(img_feat)
            sampled_caption, probs = agent.inference(
                init_state, init_lstm_states, env)
            with torch.no_grad():
                greedy_caption, _ = agent.inference(
                    init_state, init_lstm_states, env, 'greedy')

            sampled_probs.append(probs)
            sampled_captions.append(sampled_caption)
            greedy_captions.append(greedy_caption)

        log_probs = torch.log(torch.stack(sampled_probs))

        # just some rearranging of variables
        # sampled_predictions = predictions[:, 0]
        # sampled_p = torch.Tensor(predictions[:, 1].astype('float32'))
        # greedy_predictions = predictions[:, 2]
        # sampled_p = predictions[:, 3]  # don't need this actually?

        # transform ground truth and results to the format needed for eval
        ground_truth = dict(zip(img_ids, map(list, captions)))
        sampled_captions = {
            img_id: [pred]
            for img_id, pred in zip(img_ids, sampled_captions)
        }
        greedy_captions = {
            img_id: [pred]
            for img_id, pred in zip(img_ids, greedy_captions)
        }

        # calculate CIDEr scores
        mean_sample_scores, sample_scores = env.get_cider_score(
            ground_truth, sampled_captions)
        mean_greedy_scores, greedy_scores = env.get_cider_score(
            ground_truth, greedy_captions)

        # self-critical: score from sampling - score from test time algo
        advantages = torch.Tensor(
            (sample_scores - greedy_scores).reshape(-1))
        # try normalizing the advantage
        # norm_advantages = (
        #     (advantages - advantages.mean()) /
        #     (advantages.std() + 1e-9)
        # )
        # print(log_probs.mean())
        # print(log_probs)

        loss = -(advantages * log_probs).mean()
        # loss.backward()
        # print('* LR: ', loss.item() * LEARNING_RATE)
        agent.actor_optim.step()

        batch_rewards.append(mean_sample_scores)
        # print('[B] Loss: {:.2f}. Mean reward: {:.2f}. Mean advantage: {:.2f}'.format(
        #     loss.item(), mean_sample_scores, advantages.mean()))

    print('Epoch {}. Mean batch reward: {:.2f}'.format(
        e, np.mean(batch_rewards)))
    print('Elapsed: {:.2f}'.format(time.time() - epoch_start))
