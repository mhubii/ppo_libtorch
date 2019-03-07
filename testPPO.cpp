#include <fstream>
#include <Eigen/Core>
#include <torch/torch.h>
#include "ProximalPolicyOptimization.h"
#include "Models.h"

using vec = Eigen::Vector2d;

enum AGENT {
    PLAYING,
    WON,
    LOST
};

struct TestEnv
{
    TestEnv(double x, double y) : goal_(2), pos_(2), state_(4)
    {
        goal_ << x, y;
        pos_.setZero();
        state_ << pos_, goal_;  

        old_dist_ = GoalDist(pos_);
    };

    auto Act(double act_x, double act_y) -> std::tuple<Eigen::VectorXd, int>
    { 
        old_dist_ = GoalDist(pos_);

        pos_(0) += act_x;
        pos_(1) += act_y;

        state_ << pos_, goal_;

        AGENT agent;

        if (GoalDist(pos_) < 1e-1) {
            agent = WON;
        }
        else if (GoalDist(pos_) > 1e1) {
            agent = LOST;
        }
        else {
            agent = PLAYING;
        }

        return std::make_tuple(state_, agent);
    }
    double Reward()
    {
        return old_dist_ - GoalDist(pos_);
    }
    double GoalDist(vec& x) 
    { 
        return (goal_ - x).norm();
    }
    void Reset()
    {
        pos_.setZero();
        state_ << pos_, goal_;
    }

    Eigen::Vector2d pos_;
    Eigen::Vector2d goal_;
    Eigen::VectorXd state_;

    double old_dist_;
};


int main(int argc, char** argv) {

    // Environment.
    double x = std::stod(argv[1]); // goal x pos
    double y = std::stod(argv[2]); // goal y pos
    TestEnv env(x, y);

    // Model.
    uint n_in = 4;
    uint n_out = 2;
    double std = 1e-2;

    ActorCritic ac(n_in, n_out, std);
    ac.to(torch::kF64);
    ac.normal(0., std);
    torch::optim::Adam opt(ac.parameters(), 1e-3);

    // Training loop.
    uint n_iter = 10000;
    uint n_steps = 64;
    uint n_epochs = 8;
    uint mini_batch_size = 16;
    uint ppo_epochs = uint(n_steps/mini_batch_size);

    torch::Tensor state = torch::zeros({1, n_in}, torch::kF64);
    torch::Tensor action = torch::zeros({1, n_out}, torch::kF64);
    torch::Tensor reward = torch::zeros({1, 1}, torch::kF64);
    torch::Tensor next_state = torch::zeros({1, n_in}, torch::kF64);
    torch::Tensor done = torch::zeros({1, 1}, torch::kF64);

    torch::Tensor log_prob = torch::zeros({1, 1}, torch::kF64);
    torch::Tensor value = torch::zeros({1, 1}, torch::kF64);

    VT states(n_steps, torch::zeros({1, n_in}, torch::kF64));
    VT actions(n_steps, torch::zeros({1, n_out}, torch::kF64));
    VT rewards(n_steps, torch::zeros({1, 1}, torch::kF64));
    VT next_states(n_steps, torch::zeros({1, n_in}, torch::kF64));
    VT dones(n_steps, torch::zeros({1, 1}, torch::kF64));

    VT log_probs(n_steps, torch::zeros({1, n_out}, torch::kF64));
    VT returns(n_steps, torch::zeros({1, 1}, torch::kF64));
    VT values(n_steps+1, torch::zeros({1, 1}, torch::kF64));

    // Output.
    std::ofstream out;
    out.open("data.csv");

    // Initial state of env.
    for (uint i=0;i<n_in;i++)
    {
        state[0][i] = env.state_(i);
    }

    // Counter.
    uint c = 0;

    for (uint e=0;e<n_epochs;e++)
    {
        printf("epoch %u/%u\n", e+1, n_epochs);

        for (uint i=0;i<n_iter;i++)
        {
            // Play.
            auto av = ac.forward(state);
            action = std::get<0>(av);
            value = std::get<1>(av);
            log_prob = ac.log_prob(action);

            double x_act = *(action.data<double>());
            double y_act = *(action.data<double>()+1);
            auto sd = env.Act(x_act, y_act);

            // New state.
            reward[0][0] = env.Reward();
            for (uint i=0;i<n_in;i++)
            {
                next_state[0][i] = std::get<0>(sd)(i);
            }
            switch (std::get<1>(sd))
            {
                case PLAYING:
                    done[0][0] = 0.;
                    break;
                case WON:
                    printf("won, reward: %f\n", env.Reward());
                    done[0][0] = 1.;
                    break;
                case LOST:
                    printf("lost, reward: %f\n", env.Reward());
                    done[0][0] = 1.;
                    break;
            }

            out << e << ", " << env.pos_(0) << ", " << env.pos_(1) << ", " << env.goal_(0) << ", " << env.goal_(1) << "\n";

            // Store everything.
            states[c].copy_(state);
            rewards[c].copy_(reward);
            actions[c].copy_(action);
            next_states[c].copy_(next_state);
            dones[c].copy_(done);

            log_probs[c].copy_(log_prob);
            values[c].copy_(value);
            
            c++;

            // Update.
            if (c%n_steps == 0)
            {
                values[c] = std::get<1>(ac.forward(next_state));

                returns = PPO::returns(rewards, dones, values, .99, .95);

                torch::Tensor t_log_probs = torch::cat(log_probs).detach();
                torch::Tensor t_returns = torch::cat(returns).detach();
                torch::Tensor t_values = torch::cat(values).detach();
                torch::Tensor t_states = torch::cat(states);
                torch::Tensor t_actions = torch::cat(actions);
                torch::Tensor t_advantages = t_returns - t_values.slice(0, 0, n_steps);

                PPO::update(ac, t_states, t_actions, t_log_probs, t_returns, t_advantages, opt, n_steps, ppo_epochs, mini_batch_size);
            
                c = 0;
            }

            state.copy_(next_state);

            if (*(done.data<double>()) == 1.) 
            {
                env.Reset();

                // Initial state of env.
                for (uint i=0;i<n_in;i++)
                {
                    state[0][i] = env.state_(i);
                }
            }
        }
    }

    out.close();

    return 0;
}
