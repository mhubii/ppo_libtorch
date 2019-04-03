#pragma once

#include <torch/torch.h>
#include <math.h>

// Network model for Proximal Policy Optimization on Incy Wincy.
struct ActorCriticImpl : public torch::nn::Module 
{
    // Actor.

    // *** Fill in code ***
    //
    //     - add layers to the actor, for example torch::nn::Linear

    torch::Tensor mu_;
    torch::Tensor log_std_;

    // Critic.

    // *** Fill in code ***
    //
    //     - add layers to the critic, for example torch::nn::Linear

    torch::nn::Linear c_val_; // value layer

    ActorCriticImpl(int64_t n_in, int64_t n_out, double std)
        : // Actor.

          // *** Fill in code ***
          //
          //     - initialize the layers, for example actor_layer1_(torch::nn::Linear(n_in, 16))

          mu_(torch::full(n_out, 0.)),
          log_std_(torch::full(n_out, std)),
          
          // Critic

          // *** Fill in code ***
          //
          //     - initialize the layers

          c_val_(torch::nn::Linear(n_out, 1)) 
    {
        // Register the modules.

        // *** Fill in code ***
        //
        //     - register the actor layers, for example register_module("actor_layer1", actor_layer1_)

        register_parameter("log_std", log_std_);

        // *** Fill in code ***
        //
        //     - register the critic layers
    }

    // Forward pass.
    auto forward(torch::Tensor x) -> std::tuple<torch::Tensor, torch::Tensor> 
    {

        // Actor.
        // *** Fill in code ***
        //
        //     - feed forward the actor, for example x = actor_layer1_->forward(x)
        //     - add activation functions, for example x = torch::relu(actor_layer1_->forward(x))

        // Critic.
        torch::Tensor val = 
        // *** Fill in code ***
        //
        //     - feed forward the critic, add activation functions
        val = c_val_->forward(val);

        if (this->is_training()) 
        {
            torch::NoGradGuard no_grad;

            torch::Tensor action = torch::normal(mu_, log_std_.exp().expand_as(mu_));
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

    auto entropy() -> torch::Tensor
    {
        // Differential entropy of normal distribution. For reference https://pytorch.org/docs/stable/_modules/torch/distributions/normal.html#Normal
        return 0.5 + 0.5*log(2*M_PI) + log_std_;
    }

    auto log_prob(torch::Tensor action) -> torch::Tensor
    {
        // Logarithmic probability of taken action, given the current distribution.
        torch::Tensor var = (log_std_+log_std_).exp();

        return -((action - mu_)*(action - mu_))/(2*var) - log_std_ - log(sqrt(2*M_PI));
    }
};

TORCH_MODULE(ActorCritic);
