import math
import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim
import learning_env
import librarian
from actor_critic import ActorCritic
from torch.autograd import Variable
from torchvision import datasets, transforms


def train(args, dynet):
    torch.manual_seed(args.seed)

    embedding_size = args.embedding_size

    lstm_size = args.lstm_size

    num_modules = len(dynet.library) + 1

    libr = librarian.SimpleLibrarian(num_modules, embedding_size)
    print type(libr)
    model = ActorCritic(num_modules, libr, lstm_size)

    env = learning_env.Environment(args, dynet, libr)

    optimizer = optim.Adam(model.parameters(), lr=args.ac_lr)

    model.train()

    values = []
    log_probs = []

    state = env.reset()
    #state = torch.from_numpy(state)
    done = True

    episode_length = 0
    while True:
        episode_length += 1
        # Sync with the shared model
        # model.load_state_dict(shared_model.state_dict())
        if done:
            cx = Variable(torch.zeros(1, lstm_size))
            hx = Variable(torch.zeros(1, lstm_size))
        else:
            cx = Variable(cx.data)
            hx = Variable(hx.data)

        values = []
        log_probs = []
        rewards = []
        entropies = []

        for step in range(args.num_steps):
            value, logit, (hx, cx) = model(
                (state.unsqueeze(0)), (hx, cx))
            prob = F.softmax(logit)
            log_prob = F.log_softmax(logit)
            entropy = -(log_prob * prob).sum(1)
            entropies.append(entropy)

            action = prob.multinomial().data
            log_prob = log_prob.gather(1, Variable(action))

            state, reward, done, _ = env.step(action.numpy())
            done = done or episode_length >= args.num_steps
            reward = max(min(reward, 1), -1)

            if done:
                episode_length = 0
                state = env.reset()

            #state = torch.from_numpy(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break

        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model((state.unsqueeze(0)), (hx, cx))
            R = value.data

        values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        R = Variable(R)
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = rewards[i] + args.gamma * \
                values[i + 1].data - values[i].data
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - \
                log_probs[i] * Variable(gae) - 0.01 * entropies[i]

        optimizer.zero_grad()
        (policy_loss + 0.5 * value_loss).backward()

        global_norm = 0
        for param in model.parameters():
            global_norm += param.grad.data.pow(2).sum()
        global_norm = math.sqrt(global_norm)
        ratio = 40 / global_norm
        if ratio < 1:
            for param in model.parameters():
                param.grad.data.mul_(ratio)
        optimizer.step()
