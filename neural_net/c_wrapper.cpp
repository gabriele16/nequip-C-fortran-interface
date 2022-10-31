
#include <iostream>
#include "nequip.h"
#include "c_wrapper.h"

// create wrapper function for constructor in NEQUIP
nequipwrap *create_nequip(char *model)
{
    nequipwrap *neq;
    nequip::NequipPot *obj;
    neq = (typeof(neq))malloc(sizeof(*neq));
    obj = new nequip::NequipPot(model);
    neq->obj = obj;
    return neq;
}
void delete_nequip(nequipwrap *neq)
{
    free(neq);
}
void compute_nequip(nequipwrap *neq,
                    int *vecsize,
                    double *dener,
                    double *dforce,
                    double *datom_ener,
                    double *dcoord_,
                    int *datype_,
                    double *dbox)
{

    nequip::NequipPot *obj;
    if (neq == NULL)
        return;
    obj = static_cast<nequip::NequipPot *>(neq->obj);
    // covert array to vector
    double ener = 0.0;
    int natoms = *vecsize;
    std::vector<double> force_(natoms * 3, 0.0);
    std::vector<double> atom_energy_(natoms, 0.0);
    std::vector<double> coord_(dcoord_, dcoord_ + natoms * 3);
    std::vector<double> box(dbox, dbox + 9);
    std::vector<int> atype_(datype_, datype_ + natoms);

    std::cout << "define ok" << std::endl;

    obj->compute(natoms,
                 box,
                 ener,
                 force_,
                 atom_energy_,
                 coord_,
                 atype_);

    std::cout << "input ok" << std::endl;
    *dener = ener;
    //	cout << "energy is " << *dener << endl;
    for (int i = 0; i < natoms * 3; i++)
    {
        dforce[i] = force_[i];
    }
    for (int i = 0; i < natoms; i++)
    {
        datom_ener[i] = atom_energy_[i];
    }

    //	cout << "this means vector function wrap successfully" << endl;
}