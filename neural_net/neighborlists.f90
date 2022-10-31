
! **************************************************************************************************
!> \brief Interface to Nequip with c++ wrapper.
!> \par History
!>      10.2022 created Gabriele Tocci, modified from DeePMD-kit interface by Yongbin Zhuang
!> \author Gabriele Tocci
! **************************************************************************************************


MODULE neighborslist

IMPLICIT NONE
     
CONTAINS

   SUBROUTINE pbc_dx(dx,dmatbox)
      IMPLICIT NONE 
      DOUBLE PRECISION, DIMENSION(3), INTENT (INOUT) :: dx
      DOUBLE PRECISION, DIMENSION(3,3), INTENT(IN) :: dmatbox

      dx(1) = dx(1) - dmatbox(1,1) * ANINT ( dx(1) / dmatbox(1,1) )
      dx(2) = dx(2) - dmatbox(2,2) * ANINT ( dx(2) / dmatbox(2,2) )
      dx(3) = dx(3) - dmatbox(3,3) * ANINT ( dx(3) / dmatbox(3,3) )            

   END SUBROUTINE

   SUBROUTINE compute_neighborslist( dcoord,  dbox, natoms, cutoff)

      IMPLICIT NONE
      INTEGER, INTENT(IN) :: natoms
      INTEGER :: i,j
      DOUBLE PRECISION, DIMENSION(natoms*3), INTENT(IN) :: dcoord
      DOUBLE PRECISION, DIMENSION(9), INTENT(IN) :: dbox
      INTEGER, DIMENSION(natoms), INTENT(OUT) :: nlist
      INTEGER, DIMENSION(natoms), INTENT(OUT) :: list
      
      DOUBLE PRECISION, DIMENSION(3,3) :: dmatbox
      DOUBLE PRECISION, DIMENSION(natoms,3) :: dmatcoord
      DOUBLE PRECISION, DIMENSION(3) :: dx
      DOUBLE PRECISION :: cutoff
 
      dmatbox = reshape( dbox, (/ 3, 3 /) )
      dmatcoord = transpose(reshape(dcoord,(/3,natoms/) ))
  
      DO i = 1, natoms -1
         DO j = i+1, natoms
            dx(:) = dmatcoord(i,:) - dmatcoord(j,:)
            CALL pbc_dx(dx,dmatbox)
            IF (SUM(dx*dx) <= cutoff*cutoff) THEN
               nlist(i) = nlist(i) + 1
               nlist(j) = nlist(j) + 1
               list(i,nlist(i)) = j
               list(j,nlist(j)) = i
            ENDIF
         ENDDO
      ENDDO
  
      END SUBROUTINE compute_neighborslist

END MODULE neighborslist
