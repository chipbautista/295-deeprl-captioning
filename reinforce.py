
from agent import Agent
from environment import Environment
from settings import *
# REINFORCE ALGORITHM


### IS THIS MONTE CARLO THOUGH???
def play_episode(img_features, captions):
    _, state, lstm_states = env.reset(img_features, captions)
    # get the caption's mean context vector using BERT
    gt_caption_context = env.encode_captions_to_bert(captions)

    sampled_caption = ''
    greedy_caption = ''
    rewards = []
    advantages = []
    probabilities = []
    for _ in range(MAX_WORDS):
        word_logits, lstm_states = agent.forward(state, lstm_states)

        word_probs = F.softmax(word_logits.detach().cpu()).numpy()
        sampled_word = env.probs_to_word(word_probs)
        greedy_word = env.probs_to_word(word_probs, 'greedy')

        reward_sampled, done = env.get_context_reward(
            caption=' '.join([sampled_caption, sampled_word]),
            gt_caption_context)
        reward_greedy, _ = env.get_context_reward(
            caption=' '.join([greedy_caption, greedy_word]),
            gt_caption_context)

        rewards.append(reward_sampled)
        advantages.append(reward_sampled - reward_greedy)
        probabilities.append(word_probs)

        if done:
            break

    return advantages, probabilities, torch.mean(rewards)


env = Environment()
agent = Agent()
# default batch size 1
train_loader = DataLoader(MSCOCO('train'), shuffle=SHUFFLE)
train_loader = DataLoader(MSCOCO('val'), shuffle=SHUFFLE)

for e in range(MAX_EPOCH):
    epoch_start = time.time()

    # play each image separately because we don't know how long
    # each generated sentence will be
    for i, (img_features, captions) in enumerate(train_loader):
        agent.actor_optim.zero_grad()

        # this will be in batches?
        advantages, probabilities, mean_reward = play_episode(
            img_features, captions)
        agent.update_policy(advantages, torch.log(probabilities))

    for img_features, captions in val_loader:
        _, _, mean_reward = play_episode(img_features, captions)

    print('Elapsed: {:.2f}'.format(time.time() - epoch_start))
