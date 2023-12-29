import torch
import torch.nn as nn
import torch.nn.functional as F


def calc_loss_double_dqn(batch, net, tgt_net, gamma, double=False):
    states_v, actions_v, rewards_v, done_mask, next_states_v = batch
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


def calc_loss_DQfD(buf_sample, net, tgt_net, gamma, double: bool = False, run = None, nr_exp_samples: int = 50000,
                   lambdas: dict = {"lambda_dq": 1, "lambda_n": 1, "lambda_je": 1, "lambda_l2": 0.0005},
                   device = torch.device("cpu")):
    """
    :param buf_sample: Sample from the replay buffer.
    :param net: DQN.
    :param tgt_net: Target DQN.
    :param gamma: The gamma value.
    :param double: (bool) If true use double DQN loss else use basic DQN loss.
    :param run: Neptune run instance to upload loss.
    :param nr_exp_samples: The first nr_exp_samples are from the expert policy.
    :param lambdas: The different weights of the losses.

    :return:
    """
    indices, batch, weights = buf_sample
    states_v, actions_v, rewards_v, done_mask, next_states_v, steps_v = batch
    steps_v = steps_v.flatten()

    Q_vals_first = net(states_v)
    Q_vals_last = tgt_net(next_states_v)

    state_action_vals = Q_vals_first.gather(1, actions_v)
    state_action_vals = state_action_vals.squeeze(-1)  # predicted Q-values from the main network

    def calc_loss_double_dqn_combi(Q_f, Q_l, done_mask, gamma, steps_v, double=double):
        """
        This function calculates the loss function for the double DQN algorithm.

        :param Q_f: The Q-value for the first state
        :param Q_l: The Q-values for the state after the action
        :param done_mask: A mask to check if the game is done
        """
        with torch.no_grad():
            if double:
                next_state_acts = Q_f.max(1)[1]  # get argmax of out network (get the action)
                # This is also the best action to take
                next_state_acts = next_state_acts.unsqueeze(-1)
                next_state_vals = Q_l.gather(1, next_state_acts).squeeze(-1)
                # Use the target network to get the next_state vals - using the argmax from the network - instead of the
                # action chosen.
            else:
                next_state_vals = Q_l.max(1)[0]

            next_state_vals[done_mask.view(-1)] = 0.0

            exp_sa_vals = next_state_vals * (gamma ** steps_v) + rewards_v.view(-1)  # approximated Q-values
        l = (state_action_vals - exp_sa_vals) ** 2  # New loss function
        return l

    def calc_loss_supervised_marginal(Q_f, indices, actions_v, nr_exp_samples, device):
        """
        This function calculates the loss function for the expert actions only.
        :return:
        """

        pos_param = torch.tensor(0.2, dtype=torch.float32, device=device)
        margins = torch.max((Q_f + (F.one_hot(actions_v.flatten(), num_classes=9) - 1) * (-pos_param)), axis=1)[0]
        state_action_vals = Q_f.gather(1, actions_v).flatten()
        exp_indicies = indices < nr_exp_samples  # get the indicies of the expert samples
        return exp_indicies * (margins - state_action_vals)

    def calc_loss_L2(net):
        l2_reg_loss = 0
        for param in net.parameters():
            l2_reg_loss += torch.norm(param, p=2)

        return l2_reg_loss

    lambda_dq = lambdas["lambda_dq"]
    lambda_n = lambdas["lambda_n"]
    lambda_je = lambdas["lambda_je"]
    lambda_l2 = lambdas["lambda_l2"]

    je = calc_loss_supervised_marginal(Q_f=Q_vals_first, indices=indices, actions_v=actions_v,
                                       nr_exp_samples=nr_exp_samples, device=device)

    jdq_n_jn = calc_loss_double_dqn_combi(Q_f=Q_vals_first, Q_l=Q_vals_last, done_mask=done_mask, gamma=gamma,
                                          steps_v=steps_v, double=double)
    jl2 = calc_loss_L2(net)

    dqn_loss = weights * jdq_n_jn * lambda_dq * (steps_v == 1)
    nstep_loss = weights * jdq_n_jn * lambda_n * (steps_v != 1)
    marginal_loss = weights * je * lambda_je
    L2_loss = jl2 * lambda_l2


    if run:
        run["loss/DQN_loss"].log(dqn_loss.mean())
        run["loss/N_step_loss"].log(nstep_loss.mean())
        run["loss/marginal_loss"].log(marginal_loss.mean())
        run["loss/L2_loss"].log(L2_loss.mean())

    loss = dqn_loss + nstep_loss + marginal_loss + L2_loss
                        # manually multiply the MSE loss with the weights to get the desired output
                        # we need the batch_weights to correct for bias in the sampling

    return loss.mean(), loss.data.abs()
