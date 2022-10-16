#include <cmath>
#include <cstring>
#include <numeric>
#include <cassert>
#include <iostream>
#include <sstream>
#include <string>
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <iostream>
#include <memory>

int main(int argc, const char *argv[])
{

  double cutoff;
  torch::jit::script::Module model;
  torch::Device device = torch::kCPU;

  std::unordered_map<std::string, std::string> metadata = {
      {"config", ""},
      {"nequip_version", ""},
      {"r_max", ""},
      {"n_species", ""},
      {"type_names", ""},
      {"_jit_bailout_depth", ""},
      {"_jit_fusion_strategy", ""},
      {"allow_tf32", ""}};

  if (argc != 2)
  {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }

  if (torch::cuda::is_available())
  {
    device = torch::kCUDA;
  }
  else
  {
    device = torch::kCPU;
  }
  std::cout << "NEQUIP is using device " << device << "\n";

  try
  {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    model = torch::jit::load(argv[1], device, metadata);
  }
  catch (const c10::Error &e)
  {
    std::cerr << "error loading the model\n";
    return -1;
  }

  // Create a vector of inputs.
  // std::vector<torch::jit::IValue> inputs;
  // inputs.push_back(torch::ones({1, 3, 224, 224}));

  // Execute the model and turn its output into a tensor.
  // at::Tensor output = module.forward(inputs).toTensor();
  // std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
  std::cout << "Loading model from " << argv[1] << "\n";

  // Check if model is a NequIP model
  if (metadata["nequip_version"].empty())
  {
    std::cout << "The indicated TorchScript file does not appear to be a deployed NequIP model; did you forget to run `nequip-deploy`?\n";
  }

  if (model.hasattr("training"))
  {
    std::cout << "Freezing TorchScript model...\n";
    model = torch::jit::freeze(model);
  }

  // std::shared_ptr<torch::jit::script::Module> model = torch::jit::load(argv[1]);

  //  model = torch::jit::load(std::string(arg[1]), device, metadata);
  model.eval();

  // Check if model is a NequIP model
  // if (metadata["nequip_version"].empty())
  // {
  //   error->all(FLERR, "The indicated TorchScript file does not appear to be a deployed NequIP model; did you forget to run `nequip-deploy`?");
  // }
}
