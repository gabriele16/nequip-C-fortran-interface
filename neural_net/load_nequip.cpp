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
  int debug_mode = 0;

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

  if (const char *env_p = std::getenv("NEQUIP_DEBUG"))
  {
    std::cout << "PairNEQUIP is in DEBUG mode, since NEQUIP_DEBUG is in env\n";
    debug_mode = 1;
  }

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
  // In PyTorch >=1.11, this is now set_fusion_strategy
  torch::jit::FusionStrategy strategy;
  if (metadata["_jit_fusion_strategy"].empty())
  {
    // This is the default used in the Python code
    std::cout << "Jit Fusion Strategy\n";
    strategy = {{torch::jit::FusionBehavior::DYNAMIC, 3}};
  }
  else
  {
    std::stringstream strat_stream(metadata["_jit_fusion_strategy"]);
    std::string fusion_type, fusion_depth;
    while (std::getline(strat_stream, fusion_type, ','))
    {
      std::getline(strat_stream, fusion_depth, ';');
      strategy.push_back({fusion_type == "STATIC" ? torch::jit::FusionBehavior::STATIC : torch::jit::FusionBehavior::DYNAMIC, std::stoi(fusion_depth)});
    }
  }
  torch::jit::setFusionStrategy(strategy);

  // Set whether to allow TF32:
  bool allow_tf32;
  if (metadata["allow_tf32"].empty())
  {
    // Better safe than sorry
    allow_tf32 = false;
  }
  else
  {
    // It gets saved as an int 0/1
    allow_tf32 = std::stoi(metadata["allow_tf32"]);
  }
  // See https://pytorch.org/docs/stable/notes/cuda.html
  at::globalContext().setAllowTF32CuBLAS(allow_tf32);
  at::globalContext().setAllowTF32CuDNN(allow_tf32);

  std::cout << "Information from model: " << metadata.size() << " key-value pairs\n";
  for (const auto &n : metadata)
  {
    std::cout << "Key:[" << n.first << "] Value:[" << n.second << "]\n";
  }

  cutoff = std::stod(metadata["r_max"]);

  // TODO: Now we have to get the info from CP2K, e.g. elements/kinds present, number of atoms
  // For now we can also hard code it for our example though.
  // HOW: Write a Class or a struct for this.

  // Number of atoms (nelocal) is 96, i.e. 32 water molecules;
  // nedges is hardcoded for the water example (5026), as printed from
  // the nequip-lammps tutorial using pair_nequip.cpp
  // inum is the real atoms, for now equal to nlocal
  int nlocal = 96;

  // Initialize tensors for positions and cell
  torch::Tensor pos_tensor = torch::zeros({nlocal, 3});
  torch::Tensor tag2type_tensor = torch::zeros({nlocal}, torch::TensorOptions().dtype(torch::kInt64));
  torch::Tensor periodic_shift_tensor = torch::zeros({3});
  torch::Tensor cell_tensor = torch::zeros({3, 3});

  auto pos = pos_tensor.accessor<float, 2>();
  //  long edges[2 * nedges];
  //  float edge_cell_shifts[3 * nedges];
  auto tag2type = tag2type_tensor.accessor<long, 1>();
  auto periodic_shift = periodic_shift_tensor.accessor<float, 1>();
  auto cell = cell_tensor.accessor<float, 2>();

  // Inverse mapping from tag to "real" atom index
  // std::vector<int> tag2i(inum);

  // Loop over real atoms to store tags, types and positions
  // for (int ii = 0; ii < inum; ii++)
  // {
  //   int i = ilist[ii];
  //   int itag = tag[i];
  //   int itype = type[i];

  //   // Inverse mapping from tag to x/f atom index
  //   tag2i[itag - 1] = i; // tag is probably 1-based
  //   tag2type[itag - 1] = type_mapper[itype];
  //   pos[itag - 1][0] = x[i][0];
  //   pos[itag - 1][1] = x[i][1];
  //   pos[itag - 1][2] = x[i][2];
  // }

  // Get cell
  cell[0][0] = 9.85;

  cell[1][0] = 0.0;
  cell[1][1] = 9.85;

  cell[2][0] = 0.0;
  cell[2][1] = 0.0;
  cell[2][2] = 9.85;

  std::cout << "cell: " << cell_tensor << "\n";
  std::cout << "tag2i: "
            << "\n";

  model.eval();
}
