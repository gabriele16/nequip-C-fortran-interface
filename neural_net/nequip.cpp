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
using namespace torch::indexing;

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

  if (debug_mode)
  {
    std::cout << "Information from model: " << metadata.size() << " key-value pairs\n";
    for (const auto &n : metadata)
    {
      std::cout << "Key:[" << n.first << "] \n";
      //      std::cout << "Key:[" << n.first << "] Value:[" << n.second << "]\n";
    }
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
    distance(torch::Tensor x1, torch::Tensor x2,
             torch::Tensor h, torch::Tensor hinv, double &rsq)
{

  auto s1 = torch::matmul(hinv, x1);
  auto s2 = torch::matmul(hinv, x2);
  auto s21 = s2 - s1;
  s21 = s21 - torch::round(s21);
  auto r21 = torch::matmul(h, s21);
  rsq = torch::dot(r21, r21).item<double>();
}

void NequipPot::
    compute(const int natoms,
            const std::vector<double> &box,
            double &ener,
            std::vector<double> &f,
            std::vector<double> &atom_energy,
            const std::vector<double> &x,
            const std::vector<int> &atype)
{

  torch::Tensor pos_tensor = torch::zeros({natoms, 3});
  torch::Tensor tag2type_tensor = torch::zeros({natoms}, torch::TensorOptions().dtype(torch::kInt64));
  torch::Tensor periodic_shift_tensor = torch::zeros({3});
  torch::Tensor cell_tensor = torch::zeros({3, 3});
  torch::Tensor x1_tensor = torch::zeros({3});
  torch::Tensor x2_tensor = torch::zeros({3});
  double rsq = 0.0;

  auto pos = pos_tensor.accessor<float, 2>();
  //  long edges[2 * nedges];
  //  float edge_cell_shifts[3 * nedges];
  //  auto tag2type = tag2type_tensor.accessor<long, 1>();
  auto periodic_shift = periodic_shift_tensor.accessor<float, 1>();
  auto cell = cell_tensor.accessor<float, 2>();
  auto x1 = x1_tensor.accessor<float, 1>();
  auto x2 = x2_tensor.accessor<float, 1>();

  // Get cell
  int ii = 0;
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      cell[i][j] = box[ii];
      ii++;
    }
  }

  std::cout << "cell: " << cell_tensor << std::endl;

  for (int i = 0; i < natoms; i++)
  {
    std::cout << "atom type " << atype[i] << std::endl;
  }

  auto cell_inv = cell_tensor.inverse().transpose(0, 1);

  for (int i = 0; i < natoms; i++)
  {
    pos[i][0] = x[i * 3];
    pos[i][1] = x[i * 3 + 1];
    pos[i][2] = x[i * 3 + 2];
  }

  for (int ii = 0; ii < natoms; ii++)
  {
    x1[0] = pos[ii][0];
    x1[1] = pos[ii][1];
    x1[2] = pos[ii][2];
    for (int jj = 0; jj < natoms; jj++)
    {
      x2[0] = pos[jj][0];
      x2[1] = pos[jj][1];
      x2[2] = pos[jj][2];
      distance(x1_tensor, x2_tensor, cell_tensor, cell_inv, rsq);
    }
  }

  // if (debug_mode)
  // {
  //   for (const auto &p : nequipmodel.parameters())
  //   {
  //     std::cout << p << std::endl;
  //   }
  //   std::cout << "cutoff: " << cutoff << std::endl;
  // }
}