#pragma once

#include <torch/torch.h>

#include "Models.h"

// Generalized advantage estimate.
using GAE = std::vector<std::tuple<torch::Tensor /*state*/,
                                   torch::Tensor /*action*/,
                                   torch::Tensor /*policy*/,
                                   torch::Tensor /*return*/,
                                   torch::Tensor /*advantage*/>>;



// Vector of tensors.
using VT = std::vector<torch::Tensor>;

// Optimizer.
using OPT = torch::optim::Optimizer;

// Random engine for shuffling memory.
std::random_device rd;
std::mt19937 re(rd());

// Proximal policy optimization, https://arxiv.org/abs/1707.06347
class PPO
{
public:
    static auto returns(VT& rewards, VT& dones, VT& vals, double gamma, double lambda) -> VT; // Generalized advantage estimate, https://arxiv.org/abs/1506.02438
    static auto update(ActorCritic& ac, GAE& gaes, OPT& opt, uint epochs, uint mini_batch_size, double clip_param=.2) -> void; // Update the policy after T time steps for K epochs

    static auto update(ActorCritic& ac,
                       torch::Tensor& states,
                       torch::Tensor& actions,
                       torch::Tensor& log_probs,
                       torch::Tensor& returns,
                       torch::Tensor& advantages, 
                       OPT& opt, 
                       uint steps, uint epochs, uint mini_batch_size, double clip_param=.2) -> void;
};

auto PPO::returns(VT& rewards, VT& dones, VT& vals, double gamma, double lambda) -> VT
{
    // Compute the returns.
    torch::Tensor gae = torch::zeros({1}, torch::kFloat64);
    VT returns(rewards.size(), torch::zeros({1}, torch::kFloat64));

    for (uint i=rewards.size();i-- >0;) // inverse for loops over unsigned: https://stackoverflow.com/questions/665745/whats-the-best-way-to-do-a-reverse-for-loop-with-an-unsigned-index/665773
    {
        // Advantage.
        auto delta = rewards[i] + gamma*vals[i+1]*(1-dones[i]) - vals[i];
        gae = delta + gamma*lambda*(1-dones[i])*gae;

        returns[i] = gae + vals[i];
    }

    return returns;
}

auto PPO::update(ActorCritic& ac, GAE& gaes, OPT& opt, uint epochs, uint mini_batch_size, double clip_param) -> void
{
    for (uint i=0;i<epochs;i++)
    {
        // Generate random indices.
        std::vector<uint> idx;
        idx.reserve(mini_batch_size);

        for (uint i=0;i<mini_batch_size;i++) {

            idx.push_back(std::uniform_int_distribution<uint>(0, gaes.size()-1)(re));
        }

        for (auto& i: idx)
        {
            auto state = std::get<0>(gaes[i]);
            auto av = ac.forward(state); // action value pairs
            auto action = std::get<0>(av);
            auto entropy = ac.entropy();
            auto new_log_prob = ac.log_prob(action);

            auto old_log_prob = std::get<2>(gaes[i]);
            auto ratio = (new_log_prob - old_log_prob).exp();
            auto advantage = std::get<4>(gaes[i]);
            auto surr1 = ratio*advantage;
            auto surr2 = torch::clamp(ratio, 1. - clip_param, 1. + clip_param)*advantage;

            auto ret = std::get<4>(gaes[i]);
            auto val = std::get<1>(av);
            auto actor_loss = -torch::min(surr1, surr2);
            auto critic_loss = (ret-val).pow(2);

            auto loss = 0.5*critic_loss+actor_loss-0.001*entropy;

            opt.zero_grad();
            loss.backward();
            opt.step();
        }
    }
}

auto PPO::update(ActorCritic& ac,
                 torch::Tensor& states,
                 torch::Tensor& actions,
                 torch::Tensor& log_probs,
                 torch::Tensor& returns,
                 torch::Tensor& advantages, 
                 OPT& opt, 
                 uint steps, uint epochs, uint mini_batch_size, double clip_param) -> void
{
    for (uint i=0;i<epochs;i++)
    {
        // Generate random indices.
        std::vector<uint> idx;
        idx.reserve(mini_batch_size);

        for (uint i=0;i<mini_batch_size;i++) {

            idx.push_back(std::uniform_int_distribution<uint>(0, steps-1)(re));
        }

        for (auto& i: idx)
        {
            auto av = ac.forward(states[i]); // action value pairs
            auto action = std::get<0>(av);
            auto entropy = ac.entropy();
            auto new_log_prob = ac.log_prob(actions[i]);

            auto old_log_prob = log_probs[i];
            auto ratio = (new_log_prob - old_log_prob).exp();
            auto surr1 = ratio*advantages[i];
            auto surr2 = torch::clamp(ratio, 1. - clip_param, 1. + clip_param)*advantages[i];

            auto val = std::get<1>(av);
            auto actor_loss = -torch::min(surr1, surr2);
            auto critic_loss = (returns[i]-val).pow(2);

            auto loss = 0.5*critic_loss+actor_loss-0.001*entropy;

            opt.zero_grad();
            loss.backward();
            opt.step();
        }
    }
}