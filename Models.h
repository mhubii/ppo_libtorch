#pragma once

#include <torch/torch.h>
#include <math.h>

// Network model for Proximal Policy Optimization on Incy Wincy.
struct ActorCritic : public torch::nn::Module 
{
    // Actor.
    torch::nn::Linear a_lin1_{nullptr}, a_lin2_{nullptr}, a_lin3_{nullptr};
    torch::Tensor mu_;
    torch::Tensor std_;

    // Critic.
    torch::nn::Linear c_lin1_{nullptr}, c_lin2_{nullptr}, c_lin3_{nullptr}, c_val_{nullptr};

    ActorCritic(int64_t n_in, int64_t n_out, double std)
        : // Actor.
          a_lin1_(torch::nn::Linear(n_in, 16)),
          a_lin2_(torch::nn::Linear(16, 32)),
          a_lin3_(torch::nn::Linear(32, n_out)),
          mu_(torch::full(n_out, 0.)),
          std_(torch::full(n_out, std, torch::kFloat64)),
          
          // Critic
          c_lin1_(torch::nn::Linear(n_in, 16)),
          c_lin2_(torch::nn::Linear(16, 32)),
          c_lin3_(torch::nn::Linear(32, n_out)),
          c_val_(torch::nn::Linear(n_out, 1)) 
    {
        // Register the modules.
        register_module("a_lin1", a_lin1_);
        register_module("a_lin2", a_lin2_);
        register_module("a_lin3", a_lin3_);
        register_parameter("std", std_);

        register_module("c_lin1", c_lin1_);
        register_module("c_lin2", c_lin2_);
        register_module("c_lin3", c_lin3_);
        register_module("c_val", c_val_);
    }

    ActorCritic() = default;

    // Forward pass.
    auto forward(torch::Tensor x) -> std::tuple<torch::Tensor, torch::Tensor> 
    {

        // Actor.
        mu_ = torch::relu(a_lin1_->forward(x));
        mu_ = torch::relu(a_lin2_->forward(mu_));
        mu_ = torch::tanh(a_lin3_->forward(mu_));

        // Critic.
        torch::Tensor val = torch::relu(c_lin1_->forward(x));
        val = torch::relu(c_lin2_->forward(val));
        val = torch::tanh(c_lin3_->forward(val));
        val = c_val_->forward(val);

        // Reparametrization trick.
        if (this->is_training()) 
        {
            torch::NoGradGuard no_grad;

            torch::Tensor action = torch::normal(mu_, std_.abs().expand_as(mu_));
            return std::make_tuple(action, val);  
        }
        else 
        {
            return std::make_tuple(mu_, val);  
        }
    }

    // Initialize network.
    void normal(double mu, double std) 
    {
        torch::NoGradGuard no_grad;

        for (auto& p: this->parameters()) 
        {
            p.normal_(mu,std);
        }         
    }

    void zero() 
    {
        torch::NoGradGuard no_grad;

        for (auto& p: this->parameters()) 
        {
            p.zero_();
        }         
    }

    auto entropy() -> torch::Tensor
    {
        // Differential entropy of normal distribution. For reference https://pytorch.org/docs/stable/_modules/torch/distributions/normal.html#Normal
        return 0.5 + 0.5*log(2*M_PI) + std_.abs().log();
    }

    auto log_prob(torch::Tensor action) -> torch::Tensor
    {
        // Logarithmic probability of taken action, given the current distribution.
        torch::Tensor var = std_*std_;
        torch::Tensor log_scale = std_.abs().log();

        return -((action - mu_)*(action - mu_))/(2*var) - log_scale - log(sqrt(2*M_PI));
    }

    // Forward pass using reparametrization.
    auto rforward(torch::Tensor x) -> std::tuple<torch::Tensor, torch::Tensor> 
    {
 
        // Actor.
        mu_ = torch::relu(a_lin1_->forward(x));
        mu_ = torch::relu(a_lin2_->forward(mu_));
        mu_ = torch::tanh(a_lin3_->forward(mu_));

        // Critic.
        torch::Tensor val = torch::relu(c_lin1_->forward(x));
        val = torch::relu(c_lin2_->forward(val));
        val = torch::tanh(c_lin3_->forward(val));
        val = c_val_->forward(val);

        // Reparametrization trick.
        if (this->is_training()) 
        {
            torch::Tensor action = torch::normal(torch::zeros(mu_.sizes()), torch::ones(mu_.sizes())).detach();
            action = action.mul(std_.abs()).add(mu_).expand_as(mu_);
            return std::make_tuple(action, val);  
        }
        else 
        {
            return std::make_tuple(mu_, val);  
        }
    }
};
