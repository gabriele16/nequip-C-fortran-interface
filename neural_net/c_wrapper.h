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
    struct NEQUIPWRAP
    {
        // obj is the pointer where we store the nequip object.
        void *obj;
        //    torch::jit::script::Module obj;
    };

    typedef struct NEQUIPWRAP nequipwrap;

    nequipwrap *create_nequip(char *model);

    void delete_nequip(nequipwrap *neq);

    void compute_nequip(nequipwrap *neq, int *vecsize, double *dener, double *dforce,
                        double *datom_ener, double *dcoord_, int *datype_, double *dbox);

#ifdef __cplusplus
}
#endif