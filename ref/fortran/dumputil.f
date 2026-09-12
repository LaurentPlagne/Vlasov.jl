      subroutine dumpmat(nom,a,lda,n)
      implicit none
      character*(*) nom
      integer lda,n,i,j
      double precision a(lda,*)
      open(77,file=trim(nom),form='unformatted',access='stream')
      write(77) n
      do j=1,n
         do i=1,n
            write(77) a(i,j)
         end do
      end do
      close(77)
      end
      subroutine dumpvec(nom,v,n)
      implicit none
      character*(*) nom
      integer n,i
      double precision v(*)
      open(77,file=trim(nom),form='unformatted',access='stream')
      write(77) n
      do i=1,n
         write(77) v(i)
      end do
      close(77)
      end
      subroutine dumpflat(nom,a,n)
c     Ecrit n reels consecutifs d'un tableau vu a plat (ordre colonne).
      implicit none
      character*(*) nom
      integer n,i
      double precision a(*)
      open(78,file=trim(nom),form='unformatted',access='stream')
      write(78) n
      do i=1,n
         write(78) a(i)
      end do
      close(78)
      end
      subroutine dumpint(nom,n)
      implicit none
      character*(*) nom
      integer n
      double precision v(1)
      v(1)=dble(n)
      open(79,file=trim(nom),form='unformatted',access='stream')
      write(79) 1
      write(79) v(1)
      close(79)
      end
      subroutine dumpran2(n)
c     Dump des n premieres valeurs de ran2, amorce a idum=-1.
      implicit none
      integer n,i,idum
      real*4 ran2
      external ran2
      double precision v(20000)
      idum=-1
      do i=1,n
         v(i)=dble(ran2(idum))
      end do
      call dumpflat('rand2.bin',v,n)
      end
