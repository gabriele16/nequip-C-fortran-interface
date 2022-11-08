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

  std::cout << "r_max: " << cutoff << std::endl;

  // Freeze the model. Improve efficiency. Do only later.
  // if (nequipmodel.hasattr("training"))
  // {
  //   std::cout << "Freezing TorchScript model...\n";
  //   nequipmodel = torch::jit::freeze(nequipmodel);
  // }
}

void NequipPot::
    distance_vec_and_shifts(torch::Tensor x1, torch::Tensor x2,
                            torch::Tensor h, torch::Tensor hinv,
                            torch::Tensor &dx_vec,
                            torch::Tensor &cell_shift)
{
  auto s1 = torch::matmul(hinv, x1);
  auto s2 = torch::matmul(hinv, x2);
  auto s12 = s1 - s2;
  cell_shift = torch::round(s12);
  s12 = s12 - cell_shift;
  dx_vec = torch::matmul(h, s12);
}

void NequipPot::wrap_positions(torch::Tensor pos, torch::Tensor cell,
                               torch::Tensor &wrapped_positions)
{
  auto fractional = torch::linalg_solve(cell.transpose(0, 1),
                                        pos.transpose(0, 1))
                        .transpose(0, 1);

  auto frac_shifted_pos = torch::remainder(fractional, 1.0);
  wrapped_positions = torch::matmul(frac_shifted_pos, cell);
}

void NequipPot::
    compute(const int natoms,
            const std::vector<int> &atype,
            const std::vector<double> &box,
            const std::vector<double> &x,
            std::vector<double> &f,
            std::vector<double> &atom_ener,
            double &ener)
{

  torch::Tensor pos_tensor = torch::zeros({natoms, 3});
  torch::Tensor wrap_pos_tensor = torch::zeros({natoms, 3});
  torch::Tensor tag2type_tensor = torch::zeros({natoms}, torch::TensorOptions().dtype(torch::kInt64));
  torch::Tensor periodic_shift_tensor = torch::zeros({3});
  torch::Tensor cell_tensor = torch::zeros({3, 3});
  torch::Tensor x1_tensor = torch::zeros({3});
  torch::Tensor x2_tensor = torch::zeros({3});
  torch::Tensor dx_vec_tensor = torch::zeros({3});
  torch::Tensor cell_shift_tensor = torch::zeros({3});

  // vector of edges that needs to be populated
  std::vector<long> edges;
  std::vector<float> edge_cell_shifts;
  std::vector<float> e_vec(3);
  int nedges = 0;
  int edge_counter = 0;
  double rsq = 0.0;

  auto pos = pos_tensor.accessor<float, 2>();
  //  long edges[2 * nedges];
  //  float edge_cell_shifts[3 * nedges];
  auto tag2type = tag2type_tensor.accessor<long, 1>();
  auto periodic_shift = periodic_shift_tensor.accessor<float, 1>();
  auto cell = cell_tensor.accessor<float, 2>();
  auto x1 = x1_tensor.accessor<float, 1>();
  auto x2 = x2_tensor.accessor<float, 1>();
  auto dx_vec = dx_vec_tensor.accessor<float, 1>();

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

  if (debug_mode)
  {
    std::cout << "cell: " << cell_tensor << std::endl;
  }

  auto cell_inv_tensor = cell_tensor.inverse().transpose(0, 1);

  for (int i = 0; i < natoms; i++)
  {
    pos[i][0] = x[i * 3];
    pos[i][1] = x[i * 3 + 1];
    pos[i][2] = x[i * 3 + 2];
    // atom type is 0-based in nequip but 1-based in LAMMPS
    // make sure how it is in CP2K
    tag2type[i] = atype[i] - 1;
  }

  // std::cout << "atom type " << atype << std::endl;
  wrap_positions(pos_tensor, cell_tensor, wrap_pos_tensor);

  // std::cout << "wrapped positions: " << wrap_pos_tensor << std::endl;

  // need to define the accessor after wrapping!
  auto wrap_pos = wrap_pos_tensor.accessor<float, 2>();

  int jnum = 0;
  std::cout << "About to enter double loop" << std::endl;
  for (int ii = 0; ii < natoms; ii++)
  {
    x1[0] = wrap_pos[ii][0];
    x1[1] = wrap_pos[ii][1];
    x1[2] = wrap_pos[ii][2];

    jnum = 0;
    for (int jj = 0; jj < natoms; jj++)
    {
      if (ii != jj)
      {
        x2[0] = wrap_pos[jj][0];
        x2[1] = wrap_pos[jj][1];
        x2[2] = wrap_pos[jj][2];

        // The calc. below should really be
        // x[j][0] - pos[jtag-1][0]
        // as it calculates the periodic shift
        // of coordinates due to neighbor lists and domain decomp. in LAMMPS
        // see https://github.com/mir-group/pair_nequip/blob/main/pair_nequip.cpp

        // periodic_shift[0] = wrap_pos[jj][0] - wrap_pos[jj][0];
        // periodic_shift[1] = wrap_pos[jj][1] - wrap_pos[jj][1];
        // periodic_shift[2] = wrap_pos[jj][2] - wrap_pos[jj][2];

        distance_vec_and_shifts(x1_tensor, x2_tensor, cell_tensor, cell_inv_tensor,
                                dx_vec_tensor, cell_shift_tensor);

        rsq = torch::dot(dx_vec_tensor, dx_vec_tensor).item<double>();

        // std::cout << "x1: " << x1[0] << std::endl;
        // std::cout << "x2: " << x2[0] << std::endl;
        // std::cout << "dx_vec" << dx_vec_tensor << std::endl;
        // std::cout << "cell_shift_tensor" << cell_shift_tensor << std::endl;
        // std::cout << "rsq: " << sqrt(rsq) << std::endl;

        if (rsq < cutoff * cutoff)
        {
          auto cell_shift = cell_shift_tensor.accessor<float, 1>();

          e_vec[0] = std::round(cell_shift[0]);
          e_vec[1] = std::round(cell_shift[1]);
          e_vec[2] = std::round(cell_shift[2]);

          edge_cell_shifts.push_back(e_vec[0]);
          edge_cell_shifts.push_back(e_vec[1]);
          edge_cell_shifts.push_back(e_vec[2]);

          edges.push_back(ii);
          edges.push_back(jj);
          edge_counter++;
          jnum++;
          // std::cout << "edge_i: " << ii << " edge_j: " << jj << std::endl;
        }
      }
    }
    // std::cout << "atom_id: " << ii << " num_neigh: " << jnum << " x: " << x1[0] << std::endl;
  }

  // shorten the list before sending to nequip
  // the following stuff is probably redundant
  // if we don't use neighborlist, we keep it for now.
  torch::Tensor edges_tensor = torch::zeros({2, edge_counter}, torch::TensorOptions().dtype(torch::kInt64));
  torch::Tensor edge_cell_shifts_tensor = torch::zeros({edge_counter, 3});
  auto new_edges = edges_tensor.accessor<long, 2>();
  auto new_edge_cell_shifts = edge_cell_shifts_tensor.accessor<float, 2>();

  //  std::cout << "edge_counter: " << edge_counter << std::endl;

  for (int i = 0; i < edge_counter; i++)
  {
    long *e = &edges[i * 2];
    new_edges[0][i] = e[0];
    new_edges[1][i] = e[1];

    float *ev = &edge_cell_shifts[i * 3];
    new_edge_cell_shifts[i][0] = ev[0];
    new_edge_cell_shifts[i][1] = ev[1];
    new_edge_cell_shifts[i][2] = ev[2];
  }

  std::cout << "Sending information to Pytorch!" << std::endl;

  c10::Dict<std::string, torch::Tensor> input;
  input.insert("pos", wrap_pos_tensor.to(device));
  input.insert("edge_index", edges_tensor.to(device));
  input.insert("edge_cell_shift", edge_cell_shifts_tensor.to(device));
  input.insert("cell", cell_tensor.to(device));
  input.insert("atom_types", tag2type_tensor.to(device));
  std::vector<torch::IValue> input_vector(1, input);

  if (debug_mode)
  {
    std::cout << "NequIP model input:\n";
    std::cout << "pos:\n"
              << wrap_pos_tensor << "\n";
    std::cout << "edge_index:\n"
              << edges_tensor << "\n";
    std::cout << "edge_cell_shifts:\n"
              << edge_cell_shifts_tensor << "\n";
    std::cout << "cell:\n"
              << cell_tensor << "\n";
    std::cout << "atom_types:\n"
              << tag2type_tensor << "\n";
  }

  auto output = nequipmodel.forward(input_vector).toGenericDict();

  torch::Tensor forces_tensor = output.at("forces").toTensor().cpu();
  auto forces = forces_tensor.accessor<float, 2>();

  torch::Tensor total_energy_tensor = output.at("total_energy").toTensor().cpu();
  ener = total_energy_tensor.item<double>();

  torch::Tensor atomic_energy_tensor = output.at("atomic_energy").toTensor().cpu();
  auto atomic_energy = atomic_energy_tensor.accessor<float, 2>();
  float atomic_energy_sum = atomic_energy_tensor.sum().data_ptr<float>()[0];

  for (int i = 0; i < natoms; i++)
  {
    f[i * 3] = forces[i][0];
    f[i * 3 + 1] = forces[i][1];
    f[i * 3 + 2] = forces[i][2];
    atom_ener[i] = atomic_energy[i][0];
  }

  if (debug_mode)
  {
    std::cout << "NequIP model output:\n";
    std::cout << "forces: " << forces_tensor << "\n";
    std::cout << "total_energy: " << total_energy_tensor << "\n";
    std::cout << "atomic_energy: " << atomic_energy_tensor << "\n";
  }
}