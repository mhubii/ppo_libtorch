#include <fstream>
#include <Eigen/Core>
#include <torch/torch.h>
#include "ProximalPolicyOptimization.h"
#include "Models.h"

using vec = Eigen::Vector2d;

enum STATUS {
    PLAYING,
    WON,
    LOST,
    RESETTING
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

        STATUS status;

        if (GoalDist(pos_) < 4e-1) {
            status = WON;
        }
        else if (GoalDist(pos_) > 1e1) {
            status = LOST;
        }
        else {
            status = PLAYING;
        }

        return std::make_tuple(state_, status);
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
    void SetGoal(double x, double y)
    {
        goal_(0) = x;
        goal_(1) = y;

        old_dist_ = GoalDist(pos_);
        state_ << pos_, goal_;
    }

    Eigen::Vector2d pos_;
    Eigen::Vector2d goal_;
    Eigen::VectorXd state_;

    double old_dist_;
};


int main() {

    // Random engine.
    std::random_device rd;
    std::mt19937 re(rd());
    std::uniform_int_distribution<> dist(-5, 5);

    // Environment.
    double x = double(dist(re)); // goal x pos
    double y = double(dist(re)); // goal y pos
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
    uint n_epochs = 20;
    uint mini_batch_size = 16;
    uint ppo_epochs = uint(n_steps/mini_batch_size);

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
    out.open("../data/data.csv");

    // episode, agent_x, agent_y, goal_x, goal_y, STATUS=(PLAYING, WON, LOST)
    out << 1 << ", " << env.pos_(0) << ", " << env.pos_(1) << ", " << env.goal_(0) << ", " << env.goal_(1) << ", " << RESETTING << "\n";

    // Counter.
    uint c = 0;

    for (uint e=0;e<n_epochs;e++)
    {
        printf("epoch %u/%u\n", e+1, n_epochs);

        for (uint i=0;i<n_iter;i++)
        {
            torch::Tensor state = torch::zeros({1, n_in}, torch::kF64);
            torch::Tensor action = torch::zeros({1, n_out}, torch::kF64);
            torch::Tensor reward = torch::zeros({1, 1}, torch::kF64);
            torch::Tensor next_state = torch::zeros({1, n_in}, torch::kF64);
            torch::Tensor done = torch::zeros({1, 1}, torch::kF64);

            torch::Tensor log_prob = torch::zeros({1, 1}, torch::kF64);
            torch::Tensor value = torch::zeros({1, 1}, torch::kF64);

            // State of env.
            for (uint i=0;i<n_in;i++)
            {
                state[0][i] = env.state_(i);
            }

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
                    reward[0][0] += 10.;
                    done[0][0] = 1.;
                    printf("won, reward: %f\n", *(reward.data<double>()));
                    break;
                case LOST:
                    reward[0][0] -= 10.;
                    done[0][0] = 1.;
                    printf("lost, reward: %f\n", *(reward.data<double>()));
                    break;
            }

            // episode, agent_x, agent_y, goal_x, goal_y, AGENT=(PLAYING, WON, LOST)
            out << e+1 << ", " << env.pos_(0) << ", " << env.pos_(1) << ", " << env.goal_(0) << ", " << env.goal_(1) << ", " << std::get<1>(sd) << "\n";

            // Store everything.
            states[c] = state;
            rewards[c] = reward;
            actions[c] = action;
            next_states[c] = next_state;
            dones[c] = done;

            log_probs[c] = log_prob;
            values[c] = value;
            
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

            if (*(done.data<double>()) == 1.) 
            {
                // Set new goal.
                double x_new = double(dist(re)); 
                double y_new = double(dist(re));
                env.SetGoal(x_new, y_new);

                // Reset the position of the agent.
                env.Reset();

                // episode, agent_x, agent_y, goal_x, goal_y, STATUS=(PLAYING, WON, LOST)
                out << e+1 << ", " << env.pos_(0) << ", " << env.pos_(1) << ", " << env.goal_(0) << ", " << env.goal_(1) << ", " << RESETTING << "\n";
            }
        }

        // Reset at the end of an epoch.
        double x_new = double(dist(re)); 
        double y_new = double(dist(re));
        env.SetGoal(x_new, y_new);

        // Reset the position of the agent.
        env.Reset();

        // episode, agent_x, agent_y, goal_x, goal_y, STATUS=(PLAYING, WON, LOST)
        out << e+1 << ", " << env.pos_(0) << ", " << env.pos_(1) << ", " << env.goal_(0) << ", " << env.goal_(1) << ", " << RESETTING << "\n";
    }

    out.close();

    return 0;
}
