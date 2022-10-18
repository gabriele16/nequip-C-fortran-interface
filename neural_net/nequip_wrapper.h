

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
    struct NEQUIP;
    typedef struct NEQUIP nequip;

    nequip *create_nequip(char *model);

    void delete_nequip(nequip *neq);

    // void compute_nequip(nequip *neq, int *vecsize, double *dener, double *dforce,
    //                     double *datom_ener, double *dcoord_, int *datype_, double *dbox);

#ifdef __cplusplus
}
#endif