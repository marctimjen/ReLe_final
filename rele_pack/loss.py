import torch
import torch.nn as nn

def calc_loss_double_dqn(batch, net, tgt_net, gamma, double=False):
    states_v, actions_v, rewards_v, done_mask, next_states_v = batch

    # states_v = torch.cat(states).view(-1, 6)
    # next_states_v = torch.cat(next_states).view(-1, 6)
    # actions_v = torch.cat(actions)
    # rewards_v = torch.cat(rewards)  # get the reward for the action "actions"
    # done_mask = torch.cat(dones)

    # actions_v = actions_v.unsqueeze(-1)
    state_action_vals = net(states_v).gather(1, actions_v)
    state_action_vals = state_action_vals.squeeze(-1)  # predicted Q-values from the main network

    with torch.no_grad():
        if double:
            next_state_acts = net(next_states_v).max(1)[1]  # get argmax of out network (get the action)
            # This is also the best action to take

            next_state_acts = next_state_acts.unsqueeze(-1)

            next_state_vals = tgt_net(next_states_v).gather(1, next_state_acts).squeeze(-1)
            # Use the target network to get the next_state vals - using the argmax from the network - instead of the
            # action chosen.
        else:
            next_state_vals = tgt_net(next_states_v).max(1)[0]
        next_state_vals[done_mask.view(-1)] = 0.0

        exp_sa_vals = next_state_vals * gamma + rewards_v.view(-1)  # approximated Q-values

    return nn.MSELoss()(state_action_vals, exp_sa_vals)