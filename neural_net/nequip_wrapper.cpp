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

// create wrapper function for constructor in NNPInter
nequip *create_nequip(char *model)
{
    nequip *neq;
    torch::jit::script::Module obj;
    neq = (typeof(neq))malloc(sizeof(*neq));
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

    // Deserialize the ScriptModule from a file using torch::jit::load().
    obj = torch::jit::load(model, device, metadata);
    neq->obj = &obj;

    std::cout << "Loading model from " << model << "\n";

    std::cout << "Information from model: " << metadata.size() << " key-value pairs\n";
    for (const auto &n : metadata)
    {
        std::cout << "Key:[" << n.first << "] Value:[" << n.second << "]\n";
    }

    return neq;
}
void delete_nequip(nequip *neq)
{
    free(neq);
}
// void compute_nequip(nnp *neq,
//                  int *vecsize,
//                  double *dener,
//                  double *dforce,
//                  double *datom_ener,
//                  double *dcoord_,
//                  int *datype_,
//                  double *dbox)
// {

//     NNPInter *obj;
//     if (neq == NULL)
//         return;
//     obj = static_cast<NNPInter *>(n->obj);
//     // covert array to vector
//     ENERGYTYPE ener = 0.0;
//     int vsize = *vecsize;
//     std::vector<double> force_(vsize * 3, 0.0);
//     std::vector<double> atom_energy_(vsize, 0.0);
//     std::vector<double> coord_(dcoord_, dcoord_ + vsize * 3);
//     std::vector<double> box(dbox, dbox + 9);
//     std::vector<int> atype_(datype_, datype_ + vsize);
//     //	cout << "define ok" << endl;

//     obj->compute(ener,
//                  force_,
//                  atom_energy_,
//                  coord_,
//                  atype_,
//                  box);
//     //	cout << "input ok" << endl;
//     *dener = ener;
//     //	cout << "energy is " << *dener << endl;
//     for (int i = 0; i < vsize * 3; i++)
//     {
//         dforce[i] = force_[i];
//     }
//     for (int i = 0; i < vsize; i++)
//     {
//         datom_ener[i] = atom_energy_[i];
//     }
//     //	cout << "this means vector function wrap successfully" << endl;
// }