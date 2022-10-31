#include <vector>
#include <string>
#include <iostream>
#include <torch/torch.h>

namespace nequip
{
    /**
     * @brief Deep Potential.
     **/
    class NequipPot
    {
    public:
        NequipPot(const std::string &model);
        void init(const std::string &model);
        void print_summary(const std::string &pre) const;

        void compute(const int natoms,
                     const std::vector<double> &box,
                     double &ener,
                     std::vector<double> &force,
                     std::vector<double> &atom_energy,
                     const std::vector<double> &coord,
                     const std::vector<int> &atype);

        void distance(torch::Tensor x1, torch::Tensor x2,
                      torch::Tensor h, torch::Tensor hinv, double &rsq);

        double cutoff;
        torch::jit::Module nequipmodel;
        torch::Device device = torch::kCPU;

    protected:
        int *type_mapper;
        int debug_mode = 1;
    };
}
