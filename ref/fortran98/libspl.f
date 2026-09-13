c-------------------- subroutine spline cubique -------------------
c--- G est une grille de dimension max=n --------------------------
c--- I est l'indice du spline -------------------------------------
c-------------------------------------------------------------------
       subroutine scub2(g,n,i,isig,x,spline)
       include 'ceq3d.f'
       real*8 g,x,spline,hsup,hinf,formu1
       integer n,i,isig,ii,ido
       dimension g(0:NHF)
       external formu1
       spline=0.d0
       ii=i
       ido=2*ii+isig
       if ((ido.gt.(2*n+1)).or.(ii.lt.0)) then
              print*,'1 mauvaise valeur de l''indice'
              print*,'i,isig,ido,n',i,isig,ido,n
              return
       end if
       if (ii.lt.n) hsup=g(ii+1)-g(ii)
       if (ii.gt.0) hinf=g(ii)-g(ii-1)
       if (ii.eq.0) then
              if ((x.lt.g(0)).or.(x.gt.g(1))) then
                     spline=0.d0
              else
                     spline=formu1(ido,g(1),x,hsup,1)
              end if
       else
              if (ii.eq.n) then
                     if ((x.gt.g(n)).or.(x.lt.g(n-1))) then
                            spline=0.d0
                     else
                            spline=formu1(ido,x,g(n-1),hinf,-1)
                     end if
              else
                     if ((x.gt.g(ii+1)).or.(x.lt.g(ii-1))) then
                            spline=0.d0
c                     print*,'yo x g(ii+1),g(ii-1)',x,g(ii+1),g(ii-1)
                     else
                            if (x.gt.g(ii)) then
                                   spline=formu1(ido,g(ii+1),x,hsup,1)
                            else
                                   spline=formu1(ido,x,g(ii-1),hinf,-1)
                            end if
                     end if
              end if
       end if
       end
c-------------------- subroutine spline cubique -------------------
c--- G est une grille de dimension max=n --------------------------
c--- I est l'indice du spline -------------------------------------
c-------------------------------------------------------------------
       subroutine scub3(g,n,i,isig,x,spline)
       include 'ceq3d.f'
       real*8 g,x,spline,hsup,hinf,formu2
       integer n,i,isig,ii,ido
       dimension g(0:NHF)
       external formu2
       spline=0.d0
       ii=i
       ido=2*ii+isig
       if ((ido.gt.(2*n+1)).or.(ii.lt.0)) then
              print*,'1 mauvaise valeur de l''indice'
              print*,'i,isig,ido,n',i,isig,ido,n
              return
       end if
       if (ii.lt.n) hsup=g(ii+1)-g(ii)
       if (ii.gt.0) hinf=g(ii)-g(ii-1)
       if (ii.eq.0) then
              if ((x.lt.g(0)).or.(x.gt.g(1))) then
                     spline=0.d0
              else
                     spline=formu2(ido,g(1),x,hsup,1)
              end if
       else
              if (ii.eq.n) then
                     if ((x.gt.g(n)).or.(x.lt.g(n-1))) then
                            spline=0.d0
                     else
                            spline=formu2(ido,x,g(n-1),hinf,-1)
                     end if
              else
                     if ((x.gt.g(ii+1)).or.(x.lt.g(ii-1))) then
                            spline=0.d0
c                     print*,'yo x g(ii+1),g(ii-1)',x,g(ii+1),g(ii-1)
                     else
                            if (x.gt.g(ii)) then
                                   spline=formu2(ido,g(ii+1),x,hsup,1)
                            else
                                   spline=formu2(ido,x,g(ii-1),hinf,-1)
                            end if
                     end if
              end if
       end if
       end
c------------------------------------------------------------------
c--------------------subroutine spline cubique -------------------
c---  G est une grille de dimension max=n --------------------------
c---  I est l'indice du spline -------------------------------------
c-------------------------------------------------------------------
      subroutine scub(g,n,i,isig,x,spline,s1,s2,s3)
      include 'ceq3d.f'
      real*8 g,x,spline,s1,s2,s3,hsup,hinf,formu,formu1,formu2
      integer n,i,isig,ii,ido
      dimension g(0:NHF)
      external formu,formu1,formu2
      spline=0.d0
      s3=0.d0
      ii=i
      ido=2*ii+isig
      if ((ido.gt.(2*n+1)).or.(ii.lt.0)) then
         print*,'mauvaise valeur de l''indice'
         print*,'i,isig,ido,n',i,isig,ido,n
         return
      end if
      if (ii.lt.n) hsup=g(ii+1)-g(ii)
      if (ii.gt.0) hinf=g(ii)-g(ii-1)
      if (ii.eq.0) then
         if ((x.lt.g(0)).or.(x.gt.g(1))) then
            spline=0.d0
            s1=0.d0
            s2=0.d0
         else
            spline=formu(ido,g(1),x,hsup,1)
            s1=formu1(ido,g(1),x,hsup,1)
            s2=formu2(ido,g(1),x,hsup,1)
         end if
      else
         if (ii.eq.n) then
            if ((x.gt.g(n)).or.(x.lt.g(n-1))) then
               spline=0.d0
               s1=0.d0
               s2=0.d0
            else
               spline=formu(ido,x,g(n-1),hinf,-1)
               s1=formu1(ido,x,g(n-1),hinf,-1)
               s2=formu2(ido,x,g(n-1),hinf,-1)
            end if
         else
            if ((x.gt.g(ii+1)).or.(x.lt.g(ii-1))) then
               spline=0.d0
               s1=0.d0
               s2=0.d0
c     print*,'yo x g(ii+1),g(ii-1)',x,g(ii+1),g(ii-1)
            else
               if (x.gt.g(ii)) then
                  spline=formu(ido,g(ii+1),x,hsup,1)
                  s1=formu1(ido,g(ii+1),x,hsup,1)
                  s2=formu2(ido,g(ii+1),x,hsup,1)
               else
                  spline=formu(ido,x,g(ii-1),hinf,-1)
                  s1=formu1(ido,x,g(ii-1),hinf,-1)
                  s2=formu2(ido,x,g(ii-1),hinf,-1)
               end if
            end if
         end if
      end if
      end                     
c--------------------subroutine spline cubique -------------------
c---  G est une grille de dimension max=n --------------------------
c---  I est l'indice du spline -------------------------------------
c-------------------------------------------------------------------
      subroutine scub13(g,n,i,isig,x,spline,s2)
      include 'ceq3d.f'
      real*8 g,x,spline,s2,hsup,hinf,formu,formu2
      integer n,i,isig,ii,ido
      dimension g(0:NHF)
      external formu,formu2
      spline=0.d0
      ii=i
      ido=2*ii+isig
      if ((ido.gt.(2*n+1)).or.(ii.lt.0)) then
         print*,'13 mauvaise valeur de l''indice'
         print*,'i,isig,ido,n',i,isig,ido,n
         return
      end if
      if (ii.lt.n) hsup=g(ii+1)-g(ii)
      if (ii.gt.0) hinf=g(ii)-g(ii-1)
      if (ii.eq.0) then
         if ((x.lt.g(0)).or.(x.gt.g(1))) then
            spline=0.d0
            s2=0.d0
         else
            spline=formu(ido,g(1),x,hsup,1)
            s2=formu2(ido,g(1),x,hsup,1)
         end if
      else
         if (ii.eq.n) then
            if ((x.gt.g(n)).or.(x.lt.g(n-1))) then
               spline=0.d0
               s2=0.d0
            else
               spline=formu(ido,x,g(n-1),hinf,-1)
               s2=formu2(ido,x,g(n-1),hinf,-1)
            end if
         else
            if ((x.gt.g(ii+1)).or.(x.lt.g(ii-1))) then
               spline=0.d0
               s2=0.d0
c     print*,'yo x g(ii+1),g(ii-1)',x,g(ii+1),g(ii-1)
            else
               if (x.gt.g(ii)) then
                  spline=formu(ido,g(ii+1),x,hsup,1)
                  s2=formu2(ido,g(ii+1),x,hsup,1)
               else
                  spline=formu(ido,x,g(ii-1),hinf,-1)
                  s2=formu2(ido,x,g(ii-1),hinf,-1)
               end if
            end if
         end if
      end if
      end                     
c--------------------subroutine spline cubique -------------------
c---  G est une grille de dimension max=n --------------------------
c---  I est l'indice du spline -------------------------------------
c-------------------------------------------------------------------
      subroutine scub1(g,n,i,isig,x,spline)
      include 'ceq3d.f'
      real*8 g,x,spline,hsup,hinf,formu
      integer n,i,isig,ii,ido
      dimension g(0:NHF)
      external formu
      spline=0.d0
      ii=i
      ido=2*ii+isig
      if ((ido.gt.(2*n+1)).or.(ii.lt.0)) then
         print*,'1 mauvaise valeur de l''indice'
         print*,'i,isig,ido,n',i,isig,ido,n
         return
      end if
      if (ii.lt.n) hsup=g(ii+1)-g(ii)
      if (ii.gt.0) hinf=g(ii)-g(ii-1)
      if (ii.eq.0) then
         if ((x.lt.g(0)).or.(x.gt.g(1))) then
            spline=0.d0
         else
            spline=formu(ido,g(1),x,hsup,1)
         end if
      else
         if (ii.eq.n) then
            if ((x.gt.g(n)).or.(x.lt.g(n-1))) then
               spline=0.d0
            else
               spline=formu(ido,x,g(n-1),hinf,-1)
            end if
         else
            if ((x.gt.g(ii+1)).or.(x.lt.g(ii-1))) then
               spline=0.d0
c     print*,'yo x g(ii+1),g(ii-1)',x,g(ii+1),g(ii-1)
            else
               if (x.gt.g(ii)) then
                  spline=formu(ido,g(ii+1),x,hsup,1)
               else
                  spline=formu(ido,x,g(ii-1),hinf,-1)
               end if
            end if
         end if
      end if
      end
c--------------------subroutine spline cubique -------------------
c---  G est une grille de dimension max=n --------------------------
c---  I est l'indice du spline -------------------------------------
c-------------------------------------------------------------------
      subroutine scub12(g,n,i,isig,x,spline,s1)
      include 'ceq3d.f'
      real*8 g,x,spline,s1,hsup,hinf,formu,formu1
      integer n,i,isig,ii,ido
      dimension g(0:NHF)
      external formu,formu1
      spline=0.d0
      ii=i
      ido=2*ii+isig
      if ((ido.gt.(2*n+1)).or.(ii.lt.0)) then
         print*,'12 mauvaise valeur de l''indice'
         print*,'i,isig,ido,n',i,isig,ido,n
         return
      end if
      if (ii.lt.n) hsup=g(ii+1)-g(ii)
      if (ii.gt.0) hinf=g(ii)-g(ii-1)
      if (ii.eq.0) then
         if ((x.lt.g(0)).or.(x.gt.g(1))) then
            spline=0.d0
            s1=0.d0
         else
            spline=formu(ido,g(1),x,hsup,1)
            s1=formu1(ido,g(1),x,hsup,1)
         end if
      else
         if (ii.eq.n) then
            if ((x.gt.g(n)).or.(x.lt.g(n-1))) then
               spline=0.d0
               s1=0.d0
            else
               spline=formu(ido,x,g(n-1),hinf,-1)
               s1=formu1(ido,x,g(n-1),hinf,-1)
            end if
         else
            if ((x.gt.g(ii+1)).or.(x.lt.g(ii-1))) then
               spline=0.d0
               s1=0.d0
c     print*,'yo x g(ii+1),g(ii-1)',x,g(ii+1),g(ii-1)
            else
               if (x.gt.g(ii)) then
                  spline=formu(ido,g(ii+1),x,hsup,1)
                  s1=formu1(ido,g(ii+1),x,hsup,1)
               else
                  spline=formu(ido,x,g(ii-1),hinf,-1)
                  s1=formu1(ido,x,g(ii-1),hinf,-1)
               end if
            end if
         end if
      end if
      end
c---------------------formuleintemediaire ------------------------
      function formu(ido,g,p,h,isi)
      implicit none
      real*8 a,g,p,h,formu
      integer ido,isi,imp
      a=(g-p)/h
      imp=mod(ido,2)
      if (imp.eq.0) then
         formu=3.d0*a*a-2.d0*a*a*a
      else
         formu=h*(a*a-a*a*a)*isi
      end if
      end
c--------------------formulederivee premiere----------------------
      function formu1(ido,g,p,h,isi)
      implicit none
      real*8 a,g,p,h,formu1
      integer ido,isi,imp
      a=(g-p)/h
      imp=mod(ido,2)
      if (imp.eq.0) then
         formu1=-6.d0*(a-1.d0*a*a)*isi/h
      else
         formu1=-2.d0*a+3.d0*a*a
      end if
      end      
c--------------------formulederivee seconde----------------------
      function formu2(ido,g,p,h,isi)
      implicit none
      real*8 a,g,p,h,formu2
      integer ido,isi,imp
      a=(g-p)/h
      imp=mod(ido,2)
      if (imp.eq.0) then
         formu2=6.d0*(1.d0-2.d0*a)/(h*h)
      else
         formu2=(2.d0-6.d0*a)*isi/h
      end if
      end      
c--------------------subroutine primitive --------------------------
c---  G est une grille de dimension max=n --------------------------
c---  I est l'indice du spline -------------------------------------
c-------------------------------------------------------------------
      subroutine prim(g,n,i,isig,x,primit)
      include 'ceq3d.f'
      real*8 x,spline,hsup,hinf,formu,primit,formp
      integer n,i,isig,ii,ido
      real*8 g(0:NHF),c
      ii=i
      ido=2*ii+isig
      if ((ido.gt.(2*n+1)).or.(ii.lt.0)) then
         print*,'1 mauvaise valeur de l''indice'
         print*,'i,isig,ido,n',i,isig,ido,n
         return
      end if
      hsup=0.d0
      hinf=0.d0
      if (ii.lt.n) hsup=g(ii+1)-g(ii)
      if (ii.gt.0) hinf=g(ii)-g(ii-1)
      c=(hsup+hinf)*0.5
      if (isig.eq.1) c=(hsup*hsup-hinf*hinf)/12.d0
      if (ii.eq.0) then
         if (x.le.g(0)) then
            primit=0.d0
         else
            if (x.ge.g(1)) then
               primit=c
            else
               primit=formp(isig,g(1),x,hsup,1,c)
            end if
         end if
      else
         if (ii.eq.n) then
            if (x.gt.g(n)) then
               primit=c
            else
               if (x.le.g(n-1)) then
                  primit=0.d0
               else
                  primit=formp(isig,x,g(n-1),hinf,-1,c)
               end if
            end if
         else
            if (x.le.g(ii-1)) then
               primit=0.d0
            else
               if (x.ge.g(ii+1)) then
                  primit=c
               else
                  if (x.gt.g(ii)) then
                     primit=formp(isig,g(ii+1),x,hsup,1,c)
                  else
                     primit=formp(isig,x,g(ii-1),hinf,-1,c)
                  end if
               end if
            end if
         end if
      end if
      end
c---------------------------------------------------
      subroutine primx2(g,n,i,isig,x,primit)
      include 'ceq3d.f'
      real*8 x,spline,hsup,hinf,formu,primit,formpx2
      integer n,i,isig,ii,ido
      real*8 g(0:NHF),c
      ii=i
      ido=2*ii+isig
      if ((ido.gt.(2*n+1)).or.(ii.lt.0)) then
         print*,'1 mauvaise valeur de l''indice'
         print*,'i,isig,ido,n',i,isig,ido,n
         return
      end if
      hsup=0.d0
      hinf=0.d0
      if (ii.lt.n) hsup=g(ii+1)-g(ii)
      if (ii.gt.0) hinf=g(ii)-g(ii-1)
      if ((ii.gt.0).and.(ii.lt.n)) then
         c=hinf*((-1.d0/3.d0)*(hinf**2) 
     +        +0.2d0*(3.d0*hinf**2-4*hinf*g(ii-1))
     +        +0.5d0*(3.d0*hinf*g(ii-1)-g(ii-1)**2)+(g(ii-1)**2))
     +        -(hsup*((1.d0/3.d0)*(hsup**2) 
     +        -0.2d0*(3.d0*hsup**2+4*hsup*g(ii+1))
     +        +0.5d0*(3.d0*hsup*g(ii+1)+g(ii+1)**2)-(g(ii+1)**2))) 
         if (isig.eq.1) then
            c=-1.d0*(hinf**2)*((hinf**2)/(-6.d0)
     +           +0.2d0*(hinf**2-2.d0*g(ii-1)*hinf)
     +           +0.25d0*(2.d0*hinf*g(ii-1)-g(ii-1)**2)
     +           +(g(ii-1)**2)/(3.d0))-(
     +           -1.d0*(hsup**2)*((hsup**2)/(-6.d0)
     +           +0.2d0*(hsup**2+2.d0*g(ii+1)*hsup)
     +           -0.25d0*(2.d0*hsup*g(ii+1)+g(ii+1)**2)
     +           +(g(ii+1)**2)/(3.d0)))
         end if
      else
        if (ii.eq.0) then
           c= -(hsup*((1.d0/3.d0)*(hsup**2) 
     +          -0.2d0*(3.d0*hsup**2+4*hsup*g(ii+1))
     +          +0.5d0*(3.d0*hsup*g(ii+1)+g(ii+1)**2)-(g(ii+1)**2))) 
           if (isig.eq.1) then
              c=(hsup**2)*((hsup**2)/(-6.d0)
     +           +0.2d0*(hsup**2+2.d0*g(ii+1)*hsup)
     +           -0.25d0*(2.d0*hsup*g(ii+1)+g(ii+1)**2)
     +           +(g(ii+1)**2)/(3.d0))
           end if
         else
            c=0.d0
         end if
      end if
      if (ii.eq.0) then
         if (x.le.g(0) ) then
            primit=0.d0
         else
            if (x.ge.g(1)) then
               primit=c
            else
               primit=formpx2(isig,g(1),x,hsup,1,c)
            end if
         end if
      else
         if (ii.eq.n) then
            if (x.gt.g(n)) then
               primit=c
            else
               if (x.le.g(n-1)) then
                  primit=0.d0
               else
                  primit=formpx2(isig,x,g(n-1),hinf,-1,c)
               end if
            end if
         else  
            if (x.le.g(ii-1)) then
               primit=0.d0
            else
               if (x.ge.g(ii+1)) then
                  primit=c
               else
                  if (x.gt.g(ii)) then
                     primit=formpx2(isig,g(ii+1),x,hsup,1,c)
                  else
                     primit=formpx2(isig,x,g(ii-1),hinf,-1,c)
                  end if
               end if
            end if
         end if
      end if
      end
c---------------------------------------------------------------------
      subroutine primx(g,n,i,isig,x,primit)
      include 'ceq3d.f'
      real*8 x,spline,hsup,hinf,formu,primit,formpx
      integer n,i,isig,ii,ido
      real*8 g(0:NHF),c
      ii=i
      ido=2*ii+isig
      if ((ido.gt.(2*n+1)).or.(ii.lt.0)) then
         print*,'1 mauvaise valeur de l''indice'
         print*,'i,isig,ido,n',i,isig,ido,n
         return
      end if
      hsup=0.d0
      hinf=0.d0
      if (ii.lt.n) hsup=g(ii+1)-g(ii)
      if (ii.gt.0) hinf=g(ii)-g(ii-1)
      if ((ii.gt.0).and.(ii.lt.n)) then
         c=hinf*((-2.d0/5.d0)*hinf+.25d0*
     +        (3.d0*hinf-2.d0*g(ii-1))+g(ii-1))
     +        -hsup*((-2.d0/5.d0)*hsup+0.25d0*
     +        (3.d0*hsup+2.d0*g(ii+1))-g(ii+1))
         if (isig.eq.1) then
            c=(hinf**2)*(hinf/5.d0+0.25d0
     +           *(g(ii-1)-hinf)-g(ii-1)/3.d0)-
     +           (hsup**2)*(-hsup/5.d0+0.25d0
     +           *(hsup+g(ii+1))-g(ii+1)/3.d0)
         end if
      else
        if (ii.eq.0) then
           c=-hsup*((-2.d0/5.d0)*hsup+0.25d0*
     +          (3.d0*hsup+2.d0*g(ii+1))-g(ii+1))
           if (isig.eq.1) then
              c=-(hsup**2)*(-hsup/5.d0+0.25d0
     +             *(hsup+g(ii+1))-g(ii+1)/3.d0)
           end if
         else
            c=0.d0
         end if
      end if
      if (ii.eq.0) then
         if (x.le.g(0) ) then
            primit=0.d0
         else
            if (x.ge.g(1)) then
               primit=c
            else
               primit=formpx(isig,g(1),x,hsup,1,c)
            end if
         end if
      else
         if (ii.eq.n) then
            if (x.gt.g(n)) then
               primit=c
            else
               if (x.le.g(n-1)) then
                  primit=0.d0
               else
                  primit=formpx(isig,x,g(n-1),hinf,-1,c)
               end if
            end if
         else
            if (x.le.g(ii-1)) then
               primit=0.d0
            else
               if (x.ge.g(ii+1)) then
                  primit=c
               else
                  if (x.gt.g(ii)) then
                     primit=formpx(isig,g(ii+1),x,hsup,1,c)
                  else
                     primit=formpx(isig,x,g(ii-1),hinf,-1,c)
                  end if
               end if
            end if
         end if
      end if
      end
c---------------------------------------------------------------------
c---------------------------------------------------
      function formp(isig,g,p,h,isi,c)
      implicit none
      real*8 a,g,p,h,formu,c,formp
      integer isig,isi
      a=(g-p)/h
      if (isig.eq.0) then
         formp=isi*h*(a**4*0.5-a**3)+(isi+1.d0)*c*0.5
      else
         formp=h*h*(a**4*0.25-(1.d0/3.d0)*a**3)+(isi+1.d0)*c*0.5
      end if
      end
c---------------------------------------------------
      function formpx(isig,g,p,h,isi,c)
      implicit none
      real*8 a,g,p,h,formu,c,formpx
      integer isig,isi
      a=(g-p)/h
      if (isig.eq.0) then
         if (isi.eq.-1) then
            formpx=h*((-2.d0/5.d0)*h*a**5+0.25d0*
     +           (3.d0*h-2.d0*p)*a**4+p*a**3)
         else
            formpx=h*((-2.d0/5.d0)*h*a**5+0.25d0*
     +           (3.d0*h+2.d0*g)*a**4-g*a**3)+c
         end if
      else
         if (isi.eq.-1) then
            formpx=(0.2d0*h*a**5+0.25d0*(p-h)*
     +           a**4-(1.d0/3.d0)*p*a**3)*h**2
         else
            formpx=(-0.2d0*h*a**5+0.25d0*(h+g)*
     +           a**4-(1.d0/3.d0)*g*a**3)*h**2+c
         end if
      end if
      end
c---------------------------------------------------
c---------------------------------------------------
      function formpx2(isig,g,p,h,isi,c)
      implicit none
      real*8 a,g,p,h,formu,c,formpx2
      integer isig,isi
      a=(g-p)/h
      if (isig.eq.0) then
         if (isi.eq.-1) then
            formpx2=h*((-1.d0/3.d0)*(h**2)*(a**6) 
     +           +0.2d0*(3.d0*h**2-4*h*p)*a**5
     +           +0.5d0*(3.d0*h*p-p**2)*a**4+(p**2)*(a**3))
         else
            formpx2=h*((1.d0/3.d0)*(h**2)*(a**6) 
     +           -0.2d0*(3.d0*h**2+4*h*g)*a**5
     +           +0.5d0*(3.d0*h*g+g**2)*a**4-(g**2)*(a**3))+c
         end if
      else
         if (isi.eq.-1) then
            formpx2=-1.d0*(h**2)*((h**2)*(a**6)/(-6.d0)
     +           +0.2d0*(h**2-2.d0*p*h)*a**5
     +           +0.25d0*(2.d0*h*p-p**2)*a**4
     +           +(p**2)*(a**3)/(3.d0))
         else
            formpx2=-1.d0*(h**2)*((h**2)*(a**6)/(-6.d0)
     +           +0.2d0*(h**2+2.d0*g*h)*a**5
     +           -0.25d0*(2.d0*h*g+g**2)*a**4
     +           +(g**2)*(a**3)/(3.d0))+c
         end if
      end if
      end
c-------------------------------------------------------------
c-----------------------------------------------------------------    
      subroutine rajoute(i,j,k,gx,gy,gz,lxs2,uslx,rnbdt,gausstab,
     +           x,y,z,rho)
      include 'ceq3d.f'
      real*8 gausstab(1:8,0:NBTDMAX)
      real*8 gx(0:NHFX)
      real*8 gy(0:NHFY)
      real*8 gz(0:NHFZ)
      real*8 rho(0:NTXT,0:NTYT,0:NTZT)
      real*8 rx(8),ry(8),rz(8)
      real*8 lxs2,uslx,rnbdt
      real*8 x,y,z,coef
      real*8 aa1,aa2,aa3,aa4
      real*8 aa5,aa6,aa7,aa8
      integer compt
      integer i,j,k
      logical inx,iny,inz
      integer indix,indiy,indiz,id,jd,kd,kk,jj,ii
      indix=int((x-gx(i)+lxs2)*uslx*rnbdt+0.5d0)
      indiy=int((y-gy(j)+lxs2)*uslx*rnbdt+0.5d0)
      indiz=int((z-gz(k)+lxs2)*uslx*rnbdt+0.5d0)
      do compt=1,8
         rx(compt)=gausstab(compt,indix)
         ry(compt)=gausstab(compt,indiy)
         rz(compt)=gausstab(compt,indiz)
      end do
      id=2*i-4
      jd=2*j-4
      kd=2*k-4
      do kk=1,8
         do jj=1,8
            coef=ry(jj)*rz(kk)
            j=jd+jj
            k=kd+kk
            aa1=rho(id+1,j,k)+rx(1)*coef
            aa2=rho(id+2,j,k)+rx(2)*coef
            aa3=rho(id+3,j,k)+rx(3)*coef
            aa4=rho(id+4,j,k)+rx(4)*coef
            aa5=rho(id+5,j,k)+rx(5)*coef
            aa6=rho(id+6,j,k)+rx(6)*coef
            aa7=rho(id+7,j,k)+rx(7)*coef
            aa8=rho(id+8,j,k)+rx(8)*coef
            rho(id+1,j,k)=aa1
            rho(id+2,j,k)=aa2
            rho(id+3,j,k)=aa3
            rho(id+4,j,k)=aa4
            rho(id+5,j,k)=aa5
            rho(id+6,j,k)=aa6
            rho(id+7,j,k)=aa7
            rho(id+8,j,k)=aa8
         end do
      end do
      end
c---------------------------------------------------------------
      subroutine champsg(x,nx,gx,y,ny,gy,z,nz,gz,
     +     csol,champE,inttab1,inttab2,nbdt,pasgrid)
      include 'ceq3d.f'
      real*8 x,y,z
      real*8 gx(0:NHFX)
      real*8 gy(0:NHFY)
      real*8 gz(0:NHFZ)
      real*8 csol(0:NTXT,0:NTYT,0:NTZT)
      real*8 champE(3)
      real*8 inttab1(0:9,0:NBTDMAX)
      real*8 inttab2(0:9,0:NBTDMAX)
      real*8 intx1(0:9),intx2(0:9)
      real*8 inty1(0:9),inty2(0:9)
      real*8 intz1(0:9),intz2(0:9)
      real*8 fx,fy,fz,lxs2,uslx,rnbdt,pasgrid
      real*8 ctab(0:9),coef1,coef2,coef3,fxp,fyp
      integer ii,jj,kk,gid,gjd,gkd
      integer gi,gj,gk,nx,ny,nz,nbdt
      integer i,j,k,indix,indiy,indiz
      lxs2=pasgrid*0.5d0
      uslx=1.d0/pasgrid
      rnbdt=dfloat(nbdt)
      call xig(gx,nx,x,gi)
      call xig(gy,ny,y,gj)
      call xig(gz,nz,z,gk)
      gid=2*gi-4
      gjd=2*gj-4
      gkd=2*gk-4
      indix=int((x-gx(gi)+lxs2)*uslx*rnbdt+0.5d0)
      indiy=int((y-gy(gj)+lxs2)*uslx*rnbdt+0.5d0)
      indiz=int((z-gz(gk)+lxs2)*uslx*rnbdt+0.5d0)
      intx1(0)=inttab1(0,indix)
      intx1(1)=inttab1(1,indix)
      intx1(2)=inttab1(2,indix)
      intx1(3)=inttab1(3,indix)
      intx1(4)=inttab1(4,indix)
      intx1(5)=inttab1(5,indix)
      intx1(6)=inttab1(6,indix)
      intx1(7)=inttab1(7,indix)
      intx1(8)=inttab1(8,indix)
      intx1(9)=inttab1(9,indix)
c
      intx2(0)=inttab2(0,indix)
      intx2(1)=inttab2(1,indix)
      intx2(2)=inttab2(2,indix)
      intx2(3)=inttab2(3,indix)
      intx2(4)=inttab2(4,indix)
      intx2(5)=inttab2(5,indix)
      intx2(6)=inttab2(6,indix)
      intx2(7)=inttab2(7,indix)
      intx2(8)=inttab2(8,indix)
      intx2(9)=inttab2(9,indix)
c
      inty1(0)=inttab1(0,indiy)
      inty1(1)=inttab1(1,indiy)
      inty1(2)=inttab1(2,indiy)
      inty1(3)=inttab1(3,indiy)
      inty1(4)=inttab1(4,indiy)
      inty1(5)=inttab1(5,indiy)
      inty1(6)=inttab1(6,indiy)
      inty1(7)=inttab1(7,indiy)
      inty1(8)=inttab1(8,indiy)
      inty1(9)=inttab1(9,indiy)
c
      inty2(0)=inttab2(0,indiy)
      inty2(1)=inttab2(1,indiy)
      inty2(2)=inttab2(2,indiy)
      inty2(3)=inttab2(3,indiy)
      inty2(4)=inttab2(4,indiy)
      inty2(5)=inttab2(5,indiy)
      inty2(6)=inttab2(6,indiy)
      inty2(7)=inttab2(7,indiy)
      inty2(8)=inttab2(8,indiy)
      inty2(9)=inttab2(9,indiy)
c
      intz1(0)=inttab1(0,indiz)
      intz1(1)=inttab1(1,indiz)
      intz1(2)=inttab1(2,indiz)
      intz1(3)=inttab1(3,indiz)
      intz1(4)=inttab1(4,indiz)
      intz1(5)=inttab1(5,indiz)
      intz1(6)=inttab1(6,indiz)
      intz1(7)=inttab1(7,indiz)
      intz1(8)=inttab1(8,indiz)
      intz1(9)=inttab1(9,indiz)
c
      intz2(0)=inttab2(0,indiz)
      intz2(1)=inttab2(1,indiz)
      intz2(2)=inttab2(2,indiz)
      intz2(3)=inttab2(3,indiz)
      intz2(4)=inttab2(4,indiz)
      intz2(5)=inttab2(5,indiz)
      intz2(6)=inttab2(6,indiz)
      intz2(7)=inttab2(7,indiz)
      intz2(8)=inttab2(8,indiz)
      intz2(9)=inttab2(9,indiz)
c
      fx=0.d0
      fy=0.d0
      fz=0.d0
      do kk=0,9
         k=gkd+kk
         do jj=0,9
            j=gjd+jj
            coef1=inty1(jj)*intz1(kk)
            coef2=inty2(jj)*intz1(kk)
            coef3=inty1(jj)*intz2(kk)
            fxp=csol(gid+0,j,k)*intx2(0)
            fyp=csol(gid+0,j,k)*intx1(0)
            fxp=fxp+csol(gid+1,j,k)*intx2(1)
            fyp=fyp+csol(gid+1,j,k)*intx1(1)
            fxp=fxp+csol(gid+2,j,k)*intx2(2)
            fyp=fyp+csol(gid+2,j,k)*intx1(2)
            fxp=fxp+csol(gid+3,j,k)*intx2(3)
            fyp=fyp+csol(gid+3,j,k)*intx1(3)
            fxp=fxp+csol(gid+4,j,k)*intx2(4)
            fyp=fyp+csol(gid+4,j,k)*intx1(4)
            fxp=fxp+csol(gid+5,j,k)*intx2(5)
            fyp=fyp+csol(gid+5,j,k)*intx1(5)
            fxp=fxp+csol(gid+6,j,k)*intx2(6)
            fyp=fyp+csol(gid+6,j,k)*intx1(6)
            fxp=fxp+csol(gid+7,j,k)*intx2(7)
            fyp=fyp+csol(gid+7,j,k)*intx1(7)
            fxp=fxp+csol(gid+8,j,k)*intx2(8)
            fyp=fyp+csol(gid+8,j,k)*intx1(8)
            fxp=fxp+csol(gid+9,j,k)*intx2(9)
            fyp=fyp+csol(gid+9,j,k)*intx1(9)
	    fx=fx+coef1*fxp
	    fy=fy+coef2*fyp
            fz=fz+coef3*fyp
c
         end do
      end do
      champE(1)=-1.d0*fx
      champE(2)=-1.d0*fy
      champE(3)=-1.d0*fz
      end
c---------------------------------------------------------------
      subroutine champsg2(x,nx,gx,y,ny,gy,z,nz,gz,
     +     csol,champE,inttab1,inttab2,nbdt,pasgrid)
      include 'ceq3d.f'
      real*8 x,y,z
      real*8 gx(0:NHFX)
      real*8 gy(0:NHFY)
      real*8 gz(0:NHFZ)
      real*8 csol(0:NTXT,0:NTYT,0:NTZT)
      real*8 champE(3)
      real*8 inttab1(0:9,0:NBTDMAX)
      real*8 inttab2(0:9,0:NBTDMAX)
      real*8 intx1(0:9),intx2(0:9)
      real*8 inty1(0:9),inty2(0:9)
      real*8 intz1(0:9),intz2(0:9)
      real*8 fx,fy,fz,lxs2,uslx,rnbdt,pasgrid
      real*8 ctab(0:9),coef1,coef2,coef3,fxp,fyp
      integer ii,jj,kk,gid,gjd,gkd
      integer gi,gj,gk,nx,ny,nz,nbdt
      integer i,j,k,indix,indiy,indiz
      lxs2=pasgrid*0.5d0
      uslx=1.d0/pasgrid
      rnbdt=dfloat(nbdt)
      call xig(gx,nx,x,gi)
      call xig(gy,ny,y,gj)
      call xig(gz,nz,z,gk)
      gid=2*gi-4
      gjd=2*gj-4
      gkd=2*gk-4
      indix=int((x-gx(gi)+lxs2)*uslx*rnbdt+0.5d0)
      indiy=int((y-gy(gj)+lxs2)*uslx*rnbdt+0.5d0)
      indiz=int((z-gz(gk)+lxs2)*uslx*rnbdt+0.5d0)
      do i=0,9
         intx1(i)=inttab1(i,indix)
         intx2(i)=inttab2(i,indix)
         inty1(i)=inttab1(i,indiy)
         inty2(i)=inttab2(i,indiy)
         intz1(i)=inttab1(i,indiz)
         intz2(i)=inttab2(i,indiz)
      end do
c
      fx=0.d0
      fy=0.d0
      fz=0.d0
      do kk=0,9
         k=gkd+kk
         do jj=0,9
            j=gjd+jj
            coef1=inty1(jj)*intz1(kk)
            coef2=inty2(jj)*intz1(kk)
            coef3=inty1(jj)*intz2(kk)
            fxp=0.d0
            fyp=0.d0
            do ii=0,9
               i=gid+ii
               fxp=fxp+csol(i,j,k)*intx2(ii)
               fyp=fyp+csol(i,j,k)*intx1(ii)
            end do
	    fx=fx+coef1*fxp
	    fy=fy+coef2*fyp
            fz=fz+coef3*fyp
         end do
      end do
      champE(1)=-1.d0*fx
      champE(2)=-1.d0*fy
      champE(3)=-1.d0*fz
      end
