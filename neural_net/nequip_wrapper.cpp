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
#include "nequip_wrapper.h"

// define a new data type, this struct allows us to use it a safe way in C.
struct NEQUIP
{
    // obj is the pointer where we store the nequip object.
    void *obj;
};

// create wrapper function for constructor in NEQUIP
nequip *create_nequip(char *model)
{
    nequip *neq;
    torch::jit::script::Module obj;
    neq = (typeof(neq))malloc(sizeof(*neq));
    torch::Device device = torch::kCPU;
    double cutoff;

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
    obj = torch::jit::load(model, device, metadata);
    obj.eval();
    neq->obj = &obj;

    std::cout << "Loading model from " << model << "\n";

    std::cout << "Information from model: " << metadata.size() << " key-value pairs\n";
    for (const auto &n : metadata)
    {
        std::cout << "Key:[" << n.first << "] Value:[" << n.second << "]\n";
    }

    cutoff = std::stod(metadata["r_max"]);

    return neq;
}
void delete_nequip(nequip *neq)
{
    free(neq);
}
void compute_nequip(nequip *neq,
                    int *vecsize,
                    double *dener,
                    double *dforce,
                    double *datom_ener,
                    double *dcoord_,
                    int *datype_,
                    double *dbox)
{

    //     //    NNPInter *obj;
    torch::jit::script::Module *obj;

    if (neq == NULL)
        return;
    obj = static_cast<torch::jit::script::Module *>(neq->obj);
    // convert array to vector
    double ener = 0.0;
    int vsize = *vecsize;
    std::vector<double> force_(vsize * 3, 0.0);
    std::vector<double> atom_energy_(vsize, 0.0);
    std::vector<double> coord_(dcoord_, dcoord_ + vsize * 3);
    std::vector<double> box(dbox, dbox + 9);
    std::vector<int> atype_(datype_, datype_ + vsize);
    std::cout << "define ok" << std::endl;

    // obj->compute(ener,
    //              force_,
    //              atom_energy_,
    //              coord_,
    //              atype_,
    //              box);

    std::cout << "input ok" << std::endl;

    torch::Tensor pos_tensor = torch::zeros({vsize, 3});
    torch::Tensor tag2type_tensor = torch::zeros({vsize}, torch::TensorOptions().dtype(torch::kInt64));
    torch::Tensor periodic_shift_tensor = torch::zeros({3});
    torch::Tensor cell_tensor = torch::zeros({3, 3});

    auto pos = pos_tensor.accessor<float, 2>();
    //  long edges[2 * nedges];
    //  float edge_cell_shifts[3 * nedges];
    //    auto tag2type = tag2type_tensor.accessor<long, 1>();
    //    auto periodic_shift = periodic_shift_tensor.accessor<float, 1>();
    auto cell = cell_tensor.accessor<float, 2>();

    for (int ii = 0; ii < vsize; ii++)
    {
        pos[ii][0] = coord_[ii * 3];
        pos[ii][1] = coord_[ii * 3 + 1];
        pos[ii][2] = coord_[ii * 3 + 2];
    }

    // Get cell
    std::cout << "Get cell" << std::endl;
    int i = 0;
    for (int ii = 0; ii < 3; ii++)
    {
        for (int jj = 0; jj < 3; jj++)
        {
            cell[ii][jj] = box[i];
            i++;
            std::cout << cell[ii][jj] << std::endl;
        }
    }
    auto cell_inv = cell_tensor.inverse().transpose(0, 1);

    // int edge_counter = 0;
    // int debug_mode = 1;
    // if (debug_mode)
    //     printf("NEQUIP edges: i j xi[:] xj[:] cell_shift[:] rij\n");
    // for (int ii = 0; ii < nlocal; ii++)
    // {
    //     int i = ilist[ii];
    //     int itag = tag[i];
    //     int itype = type[i];

    //     int jnum = numneigh[i];
    //     int *jlist = firstneigh[i];
    //     for (int jj = 0; jj < jnum; jj++)
    //     {
    //         int j = jlist[jj];
    //         j &= NEIGHMASK;
    //         int jtag = tag[j];
    //         int jtype = type[j];

    //         // TODO: check sign
    //         periodic_shift[0] = x[j][0] - pos[jtag - 1][0];
    //         periodic_shift[1] = x[j][1] - pos[jtag - 1][1];
    //         periodic_shift[2] = x[j][2] - pos[jtag - 1][2];

    //         double dx = x[i][0] - x[j][0];
    //         double dy = x[i][1] - x[j][1];
    //         double dz = x[i][2] - x[j][2];

    //         double rsq = dx * dx + dy * dy + dz * dz;
    //         if (rsq < cutoff * cutoff)
    //         {
    //             torch::Tensor cell_shift_tensor = cell_inv.matmul(periodic_shift_tensor);
    //             auto cell_shift = cell_shift_tensor.accessor<float, 1>();
    //             float *e_vec = &edge_cell_shifts[edge_counter * 3];
    //             e_vec[0] = std::round(cell_shift[0]);
    //             e_vec[1] = std::round(cell_shift[1]);
    //             e_vec[2] = std::round(cell_shift[2]);
    //             // std::cout << "cell shift: " << cell_shift_tensor << "\n";

    //             // TODO: double check order
    //             edges[edge_counter * 2] = itag - 1;     // tag is probably 1-based
    //             edges[edge_counter * 2 + 1] = jtag - 1; // tag is probably 1-based
    //             edge_counter++;

    //             if (debug_mode)
    //             {
    //                 printf("%d %d %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g\n", itag - 1, jtag - 1,
    //                        pos[itag - 1][0], pos[itag - 1][1], pos[itag - 1][2], pos[jtag - 1][0], pos[jtag - 1][1], pos[jtag - 1][2],
    //                        e_vec[0], e_vec[1], e_vec[2], sqrt(rsq));
    //             }
    //         }
    //     }
    // }
    // if (debug_mode)
    //     printf("end NEQUIP edges\n");

    // *dener = ener;
    // std::cout << "energy is " << *dener << std::endl;
    // for (int i = 0; i < vsize * 3; i++)
    // {
    //     dforce[i] = force_[i];
    // }
    // for (int i = 0; i < vsize; i++)
    // {
    //     datom_ener[i] = atom_energy_[i];
    // }
    // std::cout << "this means vector function wrap successfully" << std::endl;
}
