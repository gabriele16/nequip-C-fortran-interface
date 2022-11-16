! **************************************************************************************************
!> \brief Interface to the DeePMD-kit or a c++ wrapper.
!> \par History
!>      07.2019 created [Yongbin Zhuang]
!> \author Yongbin Zhuang
! **************************************************************************************************

! **************************************************************************************************
!> \brief Interface to Nequip with c++ wrapper.
!> \par History
!>      10.2022 created Gabriele Tocci, modified from DeePMD-kit interface by Yongbin Zhuang
!> \author Gabriele Tocci
! **************************************************************************************************


MODULE wrap_nequip

   USE ISO_C_BINDING,                   ONLY: C_PTR,&
                                              C_CHAR,&
                                              C_DOUBLE,&
                                              C_INT,&
                                              C_NULL_CHAR,&
                                              C_LOC

   IMPLICIT NONE   
   PRIVATE
   PUBLIC :: nequip_nnp, create_nequip, delete_nequip_c, compute_nequip

   INTERFACE
      FUNCTION create_nequip_c(model) BIND(C, name="create_nequip")
         USE ISO_C_BINDING, ONLY: C_CHAR, C_PTR
         IMPLICIT NONE
         TYPE(C_PTR)                    :: create_nequip_c
         CHARACTER(KIND=C_CHAR)         :: model(*)
      END FUNCTION
      SUBROUTINE delete_nequip_c(nequip) BIND(C, name="delete_nequip")
         USE ISO_C_BINDING, ONLY: C_PTR
         IMPLICIT NONE
         TYPE(C_PTR), INTENT(IN), VALUE   :: nequip 
      END SUBROUTINE
      SUBROUTINE compute_nequip_c(nequip, vecsize, &
                    dener, dforce,  datom_ener, &
                     dcoord, datype, dbox) BIND(C, name="compute_nequip")
          USE ISO_C_BINDING
          IMPLICIT NONE
          TYPE(C_PTR), INTENT(IN), VALUE :: nequip
          TYPE(C_PTR), INTENT(IN), VALUE :: vecsize
          TYPE(C_PTR), INTENT(IN), VALUE :: dener
          TYPE(C_PTR), INTENT(IN), VALUE :: dforce
          TYPE(C_PTR), INTENT(IN), VALUE :: datom_ener
          TYPE(C_PTR), INTENT(IN), VALUE :: dcoord
          TYPE(C_PTR), INTENT(IN), VALUE :: datype
          TYPE(C_PTR), INTENT(IN), VALUE :: dbox
       END SUBROUTINE
   END INTERFACE
     TYPE nequip_nnp
             TYPE(C_PTR) :: ptr
     END TYPE
     
CONTAINS

   FUNCTION create_nequip(model)
      IMPLICIT NONE
      TYPE(nequip_nnp) :: create_nequip
      CHARACTER(len=*), INTENT(IN), TARGET :: model
      CHARACTER(len=1, kind=C_CHAR) :: c_model(LEN_TRIM(model) + 1)
      INTEGER   :: N,i 
             N = LEN_TRIM(model)
             DO i = 1, N
                c_model(i) = model(i:i)
             END DO
             c_model(N+1) = C_NULL_CHAR
             create_nequip%ptr = create_nequip_c(c_model)
   END FUNCTION

   SUBROUTINE compute_nequip(pot, vecsize, dener, dforce,  datom_ener,  dcoord, datype, dbox)
       IMPLICIT NONE
       TYPE(C_PTR) :: pot
       INTEGER(C_INT), TARGET  :: vecsize
       REAL(C_DOUBLE), POINTER :: dener
       REAL(C_DOUBLE), POINTER :: dforce(:)
       REAL(C_DOUBLE), POINTER :: datom_ener(:)
       REAL(C_DOUBLE), POINTER :: dcoord(:)
       INTEGER(C_INT), POINTER :: datype(:)
       REAL(C_DOUBLE), POINTER :: dbox(:)
       CALL compute_nequip_c(pot, C_LOC(vecsize), C_LOC(dener), C_LOC(dforce(1)), &
                                 C_LOC(datom_ener(1)),C_LOC(dcoord(1)), C_LOC(datype(1)), C_LOC(dbox(1)))
                               
    END SUBROUTINE
END MODULE wrap_nequip
