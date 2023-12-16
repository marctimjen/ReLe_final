import collections
import torch
import numpy as np

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

class ExperienceBuffer:  # to keep past actions
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # set buffer and size of que

    def __len__(self):
        return len(self.buffer)  # return length

    def append(self, experience):
        self.buffer.append(experience)  # append the experience to the end of the que

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)  # select random batches...
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])

        return states, actions, rewards, dones, next_states


class DQFD_experience_buffer:
    """
    Firstly we start by random sampling of the expert and game states.
    """

    def __init__(self, capacity, amount_expert, state_shape: int = 6,
                 prob_alpha: float = 0.6, n_step: int = 1, device: torch.device = torch.device('cuda:0'),
                 beta_start: float = 0.4, beta_frames: int = 10000000, gamma: float = 0.99):
        self.device = device
        self.amount_expert = amount_expert

        self.expert_epsilon = 1 ** (-4.8)
        self.game_epsilon = 1 ** (-5)
        self.beta_start = beta_start
        self.beta = beta_start
        self.beta_frames = beta_frames
        self.prob_alpha = prob_alpha
        self.gamma = gamma

        self.sample_index = 0
        self.capacity = capacity

        self.state = torch.zeros((capacity, state_shape), dtype=torch.float64, device=device)
        self.next_state = torch.zeros((capacity, state_shape), dtype=torch.float64, device=device)

        self.rewards = torch.zeros((capacity, 1), dtype=torch.float64, device=device)
        self.actions = torch.zeros((capacity, 1), dtype=torch.int64, device=device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.bool, device=device)

        self.steps = torch.zeros((capacity, 1), dtype=torch.int64, device=device)

        self.priorities = torch.zeros((capacity, 1), dtype=torch.float64, device=device)  # Priority score

        self.size = 0

        # For the n-step learning
        if n_step > 1:
            self.append = self.append_both
            self.n_step = n_step
        else:
            self.append = self.append_1step

        self.ls_iter = []

    def __len__(self):
        return self.size

    def append_1step(self, experience):
        state, action, reward, done, new_state = experience

        self.state[self.sample_index] = state
        self.next_state[self.sample_index] = new_state
        self.rewards[self.sample_index] = reward
        self.actions[self.sample_index] = action
        self.dones[self.sample_index] = done
        self.steps[self.sample_index] = 1

        max_prio = self.priorities.max() if self.priorities.sum() else 1.0  # get the max priority
        self.priorities[self.sample_index] = max_prio  # set the max priority for the new sample

        # Update the index and size, and wrap around if necessary
        self.sample_index = (self.sample_index + 1) % self.capacity

        if not(self.sample_index):  # do not overwrite the first expert samples
            self.sample_index = self.amount_expert

        self.size = min(self.size + 1, self.capacity)

    def append_both(self, experience):
        self.append_1step(experience)
        self.append_nstep(experience)

    def update_beta(self, idx):  # update the beta values so that beta -> 1 during training
        v = self.beta_start + idx * (1.0 - self.beta_start) / self.beta_frames
        self.beta = min(torch.tensor(1.0), v)
        return self.beta

    def sample(self, batch_size):
        probs = self.priorities ** self.prob_alpha
        probs /= probs.sum()

        # Select random indices for the batch
        indices = torch.multinomial(probs.flatten(), batch_size, replacement=False)

        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        batch = (self.state[indices], self.actions[indices], self.rewards[indices], self.dones[indices],
                    self.next_state[indices], self.steps[indices])

        return indices, batch, weights.flatten()  # get control over the states retuned!

    def update_priorities(self, batch_indices, batch_priorities):  # update the new priorities!
        """
        This function is to be called after the loss function has been calculated. This is to update the priorities
        of the different samples.
        """

        exp_ind = batch_indices < self.amount_expert  # get the indicies of the expert samples

        self.priorities[batch_indices[exp_ind]] = batch_priorities[exp_ind].view(-1, 1) + self.expert_epsilon
        self.priorities[batch_indices[~exp_ind]] = batch_priorities[~exp_ind].view(-1, 1) + self.game_epsilon

    def updater_iter(self):
        """
        This function is used for the n-step learning.
        """
        up_iter = 0
        rew = 0
        while True:
            experience = yield
            state, action, reward, done, new_state = experience

            # calculate the reward for the n-step learning
            rew = rew + (self.gamma ** up_iter) * reward
            up_iter += 1

            if up_iter == 1:
                first = state
                act = action

            if up_iter == self.n_step:
                last = new_state
                yield first, act, rew, done, last

    def append_nstep(self, experience):

        self.ls_iter.append(self.updater_iter())  # store the generators
        next(self.ls_iter[-1])  # go to the "= yield"

        who_to_pop = []
        iter = 0

        for gen in self.ls_iter:
            exp = gen.send(experience)

            if exp is None:
                continue

            state, action, reward, done, new_state = exp

            self.state[self.sample_index] = state
            self.next_state[self.sample_index] = new_state
            self.rewards[self.sample_index] = reward
            self.actions[self.sample_index] = action
            self.dones[self.sample_index] = done
            self.steps[self.sample_index] = self.n_step

            max_prio = self.priorities.max() if self.priorities.sum() else 1.0  # get the max priority
            self.priorities[self.sample_index] = max_prio  # set the max priority for the new sample

            # Update the index and size, and wrap around if necessary
            self.sample_index = (self.sample_index + 1) % self.capacity

            if not(self.sample_index):  # do not overwrite the first expert samples
                self.sample_index = self.amount_expert

            self.size = min(self.size + 1, self.capacity)

            who_to_pop.append(iter)
            iter += 1

        for i in who_to_pop[::-1]:
            self.ls_iter.pop(i)

    def clear_at_end_of_episode(self):
        self.ls_iter = []  # so that we do not mix old transitions with new ones
