/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2022 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: GPL-2.0-or-later                                 */
/*----------------------------------------------------------------------------*/

/*******************************************************************************
 * \brief A minimal wrapper for DeepMD-kit C++ interface.
 * \author Yongbin Zhuang and Yunpei Liu, modified by Gabriele Tocci
 ******************************************************************************/

#ifdef __cplusplus
extern "C"
{
#endif
    // define a new data type, this struct allows us to use it a safe way in C.
    struct NEQUIP
    {
        // obj is the pointer where we store the nequip object.
        void *obj;
        //    torch::jit::script::Module obj;
        char *model;
        double cutoff;
    };

    typedef struct NEQUIP nequip;

    nequip *create_nequip(char *model);

    void delete_nequip(nequip *neq);

    void compute_nequip(nequip *neq, int *vecsize, double *dener, double *dforce,
                        double *datom_ener, double *dcoord_, int *datype_, double *dbox);

#ifdef __cplusplus
}
#endif