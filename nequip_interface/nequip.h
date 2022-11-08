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
                     const std::vector<int> &atype,
                     const std::vector<double> &box,
                     const std::vector<double> &x,
                     std::vector<double> &f,
                     std::vector<double> &atom_energy,
                     double &ener);

        void distance_vec_and_shifts(torch::Tensor x1, torch::Tensor x2,
                                     torch::Tensor h, torch::Tensor hinv,
                                     torch::Tensor &dx_vec,
                                     torch::Tensor &cell_shift);

        void wrap_positions(torch::Tensor pos, torch::Tensor cell,
                            torch::Tensor &wrapped_positions);

        double cutoff;
        torch::jit::Module nequipmodel;
        torch::Device device = torch::kCPU;

    protected:
        int *type_mapper;
        int debug_mode = 0;
    };
}
