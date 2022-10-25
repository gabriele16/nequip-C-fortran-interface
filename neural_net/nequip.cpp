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
#include <memory>
#include "nequip.h"

using namespace nequip;

NequipPot::
    NequipPot(const std::string &model)
{
  init(model);
}

// DeepPot::~DeepPot()
// {
//   delete graph_def;
// }

void NequipPot::
    init(const std::string &model)
{
  torch::Device device = torch::kCPU;

  if (torch::cuda::is_available())
  {
    device = torch::kCUDA;
  }
  else
  {
    device = torch::kCPU;
  }

  std::unordered_map<std::string, std::string> metadata = {
      {"config", ""},
      {"nequip_version", ""},
      {"r_max", ""},
      {"n_species", ""},
      {"type_names", ""},
      {"_jit_bailout_depth", ""},
      {"_jit_fusion_strategy", ""},
      {"allow_tf32", ""}};

  std::cout << "Information from model: " << metadata.size() << " key-value pairs\n";

  // Deserialize the ScriptModule from a file using torch::jit::load().
  nequipmodel = torch::jit::load(model, device, metadata);
  nequipmodel.eval();

  std::cout << "Loading model from " << model << "\n";

  std::cout << "Information from model: " << metadata.size() << " key-value pairs\n";
  for (const auto &n : metadata)
  {
    std::cout << "Key:[" << n.first << "] \n";
    //        std::cout  << "Key:[" << n.first << "] Value:[" << n.second << "]\n";
  }

  cutoff = std::stod(metadata["r_max"]);

  // Freeze the model. Improve efficiency. Do only later.
  // if (nequipmodel.hasattr("training"))
  // {
  //   std::cout << "Freezing TorchScript model...\n";
  //   nequipmodel = torch::jit::freeze(nequipmodel);
  // }
}

void NequipPot::
    compute(double &ener,
            std::vector<double> &force,
            std::vector<double> &atom_energy,
            const std::vector<double> &coord,
            const std::vector<int> &atype,
            const std::vector<double> &box)
{

  if (debug_mode)
  {
    for (const auto &p : nequipmodel.parameters())
    {
      std::cout << p << std::endl;
    }
  }

  if (nequipmodel.hasattr("training"))
  {
    std::cout << "Connection made!\n";
  }
  else
  {
    std::cout << "Connection made but has no training!\n";
  }
  std::cout << "cutoff: " << cutoff << std::endl;
}