#include <torch/torch.h>
#include <iostream>

struct Net : torch::nn::Module
{
  Net(int64_t N, int64_t M)
      : linear(register_module("linear", torch::nn::Linear(N, M)))
  {
    another_bias = register_parameter("b", torch::randn(M));
  }
  torch::Tensor forward(torch::Tensor input)
  {
    return linear(input) + another_bias;
  }
  torch::nn::Linear linear;
  torch::Tensor another_bias;
};

int main()
{
  // initialize neural net
  Net net(4, 5);
  // loop over the parameters using iterator
  std::cout << "Initialize NN and loop over parameters using iterator" << std::endl;
  for (const auto &pair : net.named_parameters())
  {
    std::cout << pair.key() << ": " << pair.value() << std::endl;
  }

  // do a forward pass
  std::cout << "Do a forward pass" << std::endl;
  std::cout << net.forward(torch::ones({2, 4})) << std::endl;
}