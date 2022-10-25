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

        void compute(double &ener,
                     std::vector<double> &force,
                     std::vector<double> &atom_energy,
                     const std::vector<double> &coord,
                     const std::vector<int> &atype,
                     const std::vector<double> &box);

        double cutoff;
        torch::jit::Module nequipmodel;
        torch::Device device = torch::kCPU;

    protected:
        int *type_mapper;
        int debug_mode = 1;
    };
}
