
from agent import Agent
from environment import Environment
from settings import *
# REINFORCE ALGORITHM



"""
# For each episode (a set of trajectories)
for i, (img_features, gt_captions) in train_loader:
    # init episode memory
    trajectories = []
    log_probabilities = []
    rewards = []
    # play episode
    for T in range(MAX_WORDS):
        # init state

        # choose action
        probabilities, lstm_states = agent.select_action(
            state, lstm_states)
        action = actions[np.max(probabilities)]
        # store log probabilities
        log_probabilities.append(
            torch.log(probabilities).cpu())
        # receive reward
        reward, done = env.step(action)
        # store reward
        rewards.append(reward)
        # construct next_state
        steps.append((state, action, reward, next_state))
        if done:
            break

    # at this point, `T` holds the number of steps the agent took
    # Calculate the discounted rewards at every time step:
    # NOTE: THIS WILL BE REPLACED BY A VALUE NETWORK THOUGH
    discounted_rewards = []
    for t in range(T):
        G = [r * (DISCOUNT_FACTOR**k)
             for (k, r) in enumerate(rewards[t:])].sum()
        discounted_rewards.append(G)

    # Accdg to @thecrisyoon's Medium post, it's better to normalize
    discounted_rewards = (
        (discounted_rewards - discounted_rewards.mean()) /
        (discounted_rewards.std() + 1e-9)
    )

    policy_gradients = [
        -(log_probabilities[t] * discounted_rewards[t])
        for t in range(T)
    ]

    # Update policy network now
    agent.actor_optim.zero_grad()
    loss = torch.stack(policy_gradients).sum()
    loss.backward()
    agent.actor_optim.step()
"""


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

        word_probs = F.softmax(word_logits)
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
