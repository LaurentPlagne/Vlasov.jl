      subroutine f02agf(a,ia,n,rr,ri,vr,ivr,vi,ivi,intger,ifail)
c     Shim NAG f02agf -> LAPACK dgeev
      implicit none
      integer ia,n,ivr,ivi,ifail,intger(*)
      double precision a(ia,*),rr(*),ri(*),vr(ivr,*),vi(ivi,*)
      integer lwmax
      parameter (lwmax=20000)
      double precision work(lwmax),vtmp(2000,2000),vldum(1,1)
      integer info,lwork,i,j
      lwork=lwmax
      call dgeev('N','V',n,a,ia,rr,ri,vldum,1,vtmp,2000,
     +           work,lwork,info)
      do j=1,n
         intger(j)=0
      end do
      j=1
 10   if (j.le.n) then
         if (ri(j).eq.0.0d0) then
            do i=1,n
               vr(i,j)=vtmp(i,j)
               vi(i,j)=0.0d0
            end do
            j=j+1
         else
            do i=1,n
               vr(i,j)  = vtmp(i,j)
               vi(i,j)  = vtmp(i,j+1)
               vr(i,j+1)= vtmp(i,j)
               vi(i,j+1)=-vtmp(i,j+1)
            end do
            j=j+2
         end if
         goto 10
      end if
      ifail=info
      end
