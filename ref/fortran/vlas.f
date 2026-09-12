      program vlas 
c---------------------declaration---------------------------------
      include 'ceq3d.f'
      real*8 posprold(3)
      real*8 gx(0:NHFX)
      real*8 gy(0:NHFY)
      real*8 gz(0:NHFZ)
      real*8 gtx(0:NTXT)
      real*8 gty(0:NTYT)
      real*8 gtz(0:NTZT)
      real*8 rho(0:NTXT,0:NTYT,0:NTZT)
      real*8 cofrho(0:NTXT,0:NTYT,0:NTZT)
      real*8 gtech(0:100)
      real*8 qpold(3,npartmax)
      real*8 qp(3,npartmax)
      real*8 fp(3,npartmax)
      real*8 dltt
      real*8 nbion,nbelec
      real*8 dx(0:NTXT,0:NTXT),dy(0:NTYT,0:NTYT),dz(0:NTZT,0:NTZT)
      real*8 sx(0:NTXT,0:NTXT),sy(0:NTYT,0:NTYT),sz(0:NTZT,0:NTZT)
      real*8 sxm1(0:NTXT,0:NTXT),sym1(0:NTYT,0:NTYT),szm1(0:NTZT,0:NTZT)
      real*8 s1x(0:NTXT,0:NTXT),s1y(0:NTYT,0:NTYT),s1z(0:NTZT,0:NTZT)
      real*8 rev(NXS,NYS,NZS)
      real*8 mxm1(NXS,NXS),mym1(NYS,NYS),mzm1(NZS,NZS)
      real*8 mx(NXS,NXS),my(NYS,NYS),mz(NZS,NZS)
      real*8 phi(0:NTXT,0:NTYT,0:NTZT)
      real*8 csol(0:NTXT,0:NTYT,0:NTZT)
      real*8 rhsl(NXS,NYS,NZS)
      real*8 gxbig(0:NHFX)
      real*8 gybig(0:NHFY)
      real*8 gzbig(0:NHFZ)
      real*8 gtxbig(0:NTXT)
      real*8 gtybig(0:NTYT)
      real*8 gtzbig(0:NTZT)
      real*8 rhobig(0:NTXT,0:NTYT,0:NTZT)
      real*8 volm1big(0:NTXT,0:NTYT,0:NTZT)
      real*8 dxbig(0:NTXT,0:NTXT)
      real*8 dybig(0:NTYT,0:NTYT)
      real*8 dzbig(0:NTZT,0:NTZT)
      real*8 sxbig(0:NTXT,0:NTXT)
      real*8 sybig(0:NTYT,0:NTYT)
      real*8 szbig(0:NTZT,0:NTZT)
      real*8 sxm1big(0:NTXT,0:NTXT)
      real*8 sym1big(0:NTYT,0:NTYT)
      real*8 szm1big(0:NTZT,0:NTZT)
      real*8 s1xbig(0:NTXT,0:NTXT)
      real*8 s1ybig(0:NTYT,0:NTYT)
      real*8 s1zbig(0:NTZT,0:NTZT)
      real*8 revbig(NXS,NYS,NZS)
      real*8 mxm1big(NXS,NXS)
      real*8 mym1big(NYS,NYS)
      real*8 mzm1big(NZS,NZS)
      real*8 mxbig(NXS,NXS)
      real*8 mybig(NYS,NYS)
      real*8 mzbig(NZS,NZS)
      real*8 phibig(0:NTXT,0:NTYT,0:NTZT)
      real*8 csolbig(0:NTXT,0:NTYT,0:NTZT)
      real*8 rhslbig(NXS,NYS,NZS)
      real*8 psxbig(0:NTXT)
      real*8 psybig(0:NTYT)
      real*8 pszbig(0:NTZT)
      real*8 psxxbig(0:NTXT)
      real*8 psyybig(0:NTYT)
      real*8 pszzbig(0:NTZT)
      real*8 psx2big(0:NTXT)
      real*8 psy2big(0:NTYT)
      real*8 psz2big(0:NTZT)
      real*8 xclu,xboite
      real*8 psx(0:NTXT),psy(0:NTYT),psz(0:NTZT)
      real*8 psxx(0:NTXT),psyy(0:NTYT),pszz(0:NTZT)
      real*8 psx2(0:NTXT),psy2(0:NTYT),psz2(0:NTZT)
      real*8 pasgrid,sigr
      real*8 inttab1(0:9,0:NBTDMAX)
      real*8 inttab2(0:9,0:NBTDMAX)
      real*8 gausstab(1:8,0:NBTDMAX)
      real*8 enele,enejel,ekin,ekinout,enetot,einout,lcine(3)
      real*8 mpro,epro,impara,cutoff,xinit
      real*8 vpro(3),pospro(3),chapro
      real*8 rcutee,elin,einterne
      real*8 delvm
      character*3 chnomb
      character*4 titi,tito
      integer nbdt
      integer npart
      integer nx,ny,nz
      integer nbt,i
      integer tabout(npartmax),nbout
      integer tabouti(npartmax),nbouti
      integer liste2(npartmax)
      integer ntx,nty,ntz,nsx,nsy,nsz
      integer n1xyz
      integer nbatt,nbphot
      integer n1big,n2big
      integer nbcap
      logical first,last,capturee,first2
      character*23 titreq
      nbdt=1000
      nbcap=0
      einterne=0.d0
c     ----------------------main-------------------------------------
      print*,'programme TFD ***'
c     ---   -- initialisation -- -- -- 
      call litinput(n1xyz,xclu,xboite,
     +     nbion,nbelec,npart,dltt,nbt,
     +     mpro,epro,impara,cutoff,xinit,chapro,rcutee,
     +     n1big,n2big,delvm)
      call initpro(mpro,epro,impara,xinit,
     +     pospro,vpro,posprold,dltt)
      call makeinit(nbelec,qp,qpold,dltt,npart)
      print*,'qp(1,1)=',qp(1,1)
      print*,'pp(1,1)=',qpold(1,1)
      call grille(nx,gx,gtx,ny,gy,gty,nz,gz,gtz,
     +            xclu,n1xyz,ntx,nty,ntz,nsx,nsy,nsz)
      print*,'nx,ny,nz',nx,ny,nz,ntx,nty,ntz
      call biggrille(nx,gxbig,gtxbig,
     +               ny,gybig,gtybig,
     +               nz,gzbig,gtzbig,
     +               xclu,xboite,n1big,n2big,
     +               ntx,nty,ntz,nsx,nsy,nsz)
      print*,'nx,ny,nz',nx,ny,nz,ntx,nty,ntz
      call griech(100,gtech,50.d0)
      call maketable(gx,gtx,pasgrid,sigr,nbdt,gausstab)
      print*,'gausstab ok'
      call maketaint(nx,gx,sigr,nbdt,inttab1,inttab2)
      print*,'inttab ok'
      call static(ntx,nty,ntz,nx,ny,nz,
     +     gtx,gty,gtz,gx,gy,gz,
     +     dx,dy,dz,mx,my,mz,psx,psy,psz,psxx,psyy,pszz,
     +     mxm1,mym1,mzm1,rev,nsx,nsy,nsz,psx2,psy2,psz2,
     +     sx,sy,sz,s1x,s1y,s1z,sxm1,sym1,szm1)
      call static(ntx,nty,ntz,nx,ny,nz,
     +     gtxbig,gtybig,gtzbig,gxbig,gybig,gzbig,
     +     dxbig,dybig,dzbig,mxbig,mybig,mzbig,
     +     psxbig,psybig,pszbig,psxxbig,psyybig,pszzbig,
     +     mxm1big,mym1big,mzm1big,revbig,nsx,nsy,nsz,
     +     psx2big,psy2big,psz2big,
     +     sxbig,sybig,szbig,s1xbig,s1ybig,s1zbig,
     +     sxm1big,sym1big,szm1big)
      print*,'static ok'
      call volumes(volm1big,gtxbig,gxbig,nx,
     +                      gtybig,gybig,ny,
     +                      gtzbig,gzbig,nz)     
c           ------- calcul de qpold -------------------
      call moveback1(qp,qpold,nbelec,npart,dltt)
      print*,'qpold(1,1) t=-0.5dt=',qpold(1,1)
      call makerhog(rho,npart,qpold,nx,ny,nz,gx,gy,gz,
     +     nbelec,tabout,nbout,nbdt,pasgrid,gausstab,nbcap,
     +     cofrho,sxm1,sym1,szm1,ntx,nty,ntz,psx,psy,psz)
      print*,'nbou grille fine',nbout
      call makerho(rhobig,npart,qpold,ntx,nty,ntz,
     +              gtxbig,gtybig,gtzbig,
     +              nbelec,volm1big,tabout,nbout,nbcap)
      print*,'nbou grille coarse',nbout
      print*,'solve ok'
      print*,'entree ds pspech'
c      call sortierho(nx,ny,nz,ntx,nty,ntz,'rhd',
c     +     sxm1,sym1,szm1,gx,gy,gz,rho,
c     +     sxm1big,sym1big,szm1big,gxbig,gybig,gzbig,rhobig)
c      call sortierho(gx,gy,gz,nx,ny,nz,rho,'rhd',
c     +     sxm1,sym1,szm1,ntx,nty,ntz)
      call makerh2(rhobig,dxbig,dybig,dzbig,nsx,nsy,nsz,
     +     gtxbig,gtybig,gtzbig,phibig,rhslbig,
     +     sxm1big,sym1big,szm1big,
     +     psxbig,psybig,pszbig,psxxbig,
     +     psyybig,pszzbig,psx2big,psy2big,psz2big)
      call solve(mxbig,mybig,mzbig,mxm1big,mym1big,mzm1big
     +     ,nsx,nsy,nsz,revbig,rhslbig,phibig,csolbig
     +     ,sxm1big,sym1big,szm1big)
      call makerhsf(rho,dx,dy,dz,nsx,nsy,nsz,
     +     gtx,gty,gtz,phi,rhsl,sxm1,sym1,szm1,
     +     psx,psy,psz,psxx,psyy,pszz,psx2,psy2,psz2,
     +     gxbig,gybig,gzbig,nx,ny,nz,csolbig)
      call solve(mx,my,mz,mxm1,mym1,mzm1,nsx,nsy,nsz,
     +     rev,rhsl,phi,csol,sxm1,sym1,szm1)
c      call sortie(gx,gy,gz,nx,ny,nz,csol,'ele')
c      call sortie(gxbig,gybig,gzbig,nx,ny,nz,csolbig,'elb')
c      call sortietest(gx,gy,gz,gxbig,gybig,gzbig,
c     +     nx,ny,nz,csol,csolbig)
      call pspech(ntx,nty,ntz,gtx,gty,gtz,
     +     rho,csol,sxm1,sym1,szm1,nbion)
      call pspech(ntx,nty,ntz,gtxbig,gtybig,gtzbig,
     +     rhobig,csolbig,sxm1big,sym1big,szm1big,nbion)
c      call statistique(csol,nx,ny,nz,
c     +     gx,gy,gz,npart,qp,qpold,nbion,dltt,
c     +     inttab1,nbdt,pasgrid)
c      call sortie(gx,gy,gz,nx,ny,nz,csol,'tot')
c      call sortie(gxbig,gybig,gzbig,nx,ny,nz,csolbig,'tob')
c      call sortietest(gx,gy,gz,gxbig,gybig,gzbig,
c     +     nx,ny,nz,csol,csolbig)
      call force2g(npart,nbelec,fp,qpold,nx,ny,nz,
     +      csolbig,gxbig,gybig,gzbig,nbout,
     +      csol,gx,gy,gz,
     +      inttab1,inttab2,nbdt,pasgrid,nbcap,liste2)
      call moveback2(qp,qpold,fp,nbelec,npart,dltt)
      print*,'qpold(1,1) t=-dt=',qpold(1,1)
c               -----------------------------
      call makerhog(rho,npart,qp,nx,ny,nz,gx,gy,gz,
     +     nbelec,tabout,nbout,nbdt,pasgrid,gausstab,nbcap,
     +     cofrho,sxm1,sym1,szm1,ntx,nty,ntz,psx,psy,psz)
      first=.true.
      last=.false.
      call trace(qp,npart,first,1,nbion,nbcap)
c      call incproj2(delvm,mpro,pospro,vpro,
c     +     dltt,qp,qpold,fp,cutoff,
c     +     nbion,nbelec,npart,chapro,first,1,
c     +     rcutee,epro,impara,last,nbcap,einterne,titreq,
c     +     csolbig,gxbig,gybig,gzbig,nbout,
c     +     csol,gx,gy,gz,nx,ny,nz,
c     +     inttab1,inttab2,nbdt,pasgrid,liste2)
      call incproj(mpro,pospro,posprold,vpro,dltt,qp,fp,cutoff,
     +     nbion,nbelec,npart,chapro,first,1,rcutee,
     +     epro,impara,last,nbcap,einterne,titreq)
c      call echanti2(npart,qp,100,gtech,0,nbcap,nbelec)
      call makerhog(rho,npart,qp,nx,ny,nz,gx,gy,gz,
     +     nbelec,tabout,nbout,nbdt,pasgrid,gausstab,nbcap,
     +     cofrho,sxm1,sym1,szm1,ntx,nty,ntz,psx,psy,psz)
      call makerh2(rho,dx,dy,dz,nsx,nsy,nsz,
     +     gtx,gty,gtz,phi,rhsl,
     +     sxm1,sym1,szm1,psx,psy,psz,psxx,
     +     psyy,pszz,psx2,psy2,psz2)
      call solve(mx,my,mz,mxm1,mym1,mzm1,nsx,nsy,nsz,
     +     rev,rhsl,phi,csol,sxm1,sym1,szm1)
      call makerho(rhobig,npart,qp,ntx,nty,ntz,
     +              gtxbig,gtybig,gtzbig,
     +              nbelec,volm1big,tabout,nbout,nbcap)
      call makerh2(rhobig,dxbig,dybig,dzbig,nsx,nsy,nsz,
     +     gtxbig,gtybig,gtzbig,phibig,rhslbig,
     +     sxm1big,sym1big,szm1big,
     +     psxbig,psybig,pszbig,psxxbig,
     +     psyybig,pszzbig,psx2big,psy2big,psz2big)
      call solve(mxbig,mybig,mzbig,mxm1big,mym1big,mzm1big
     +     ,nsx,nsy,nsz,revbig,rhslbig,phibig,csolbig
     +     ,sxm1big,sym1big,szm1big)
      call pspech(ntx,nty,ntz,gtx,gty,gtz,
     +     rho,csol,sxm1,sym1,szm1,nbion)
      call pspech(ntx,nty,ntz,gtxbig,gtybig,gtzbig,
     +     rhobig,csolbig,sxm1big,sym1big,szm1big,nbion)
      call force2g(npart,nbelec,fp,qp,nx,ny,nz,
     +      csolbig,gxbig,gybig,gzbig,nbout,
     +      csol,gx,gy,gz,
     +      inttab1,inttab2,nbdt,pasgrid,nbcap,liste2)
c      call sortie(gx,gy,gz,nx,ny,nz,csol,'tot')
      print*,'sortie de pspech'
      print*,'fin initialisation'
      first=.false.
      first2=.true.
      capturee=.false.
c     ------------------------------------------
c     ---    --   boucle sur le temps  --   ---
c     ------------------------------------------  
      nbatt=0
      nbphot=2000
      do i=1,nbt
         if (i.gt.1) first2=.false.
         print*,'tour no ',i
         call makerhog(rho,npart,qp,nx,ny,nz,gx,gy,gz,
     +        nbelec,tabouti,nbouti,nbdt,pasgrid,gausstab,nbcap,
     +     cofrho,sxm1,sym1,szm1,ntx,nty,ntz,psx,psy,psz)
         call makerho(rhobig,npart,qp,ntx,nty,ntz,
     +              gtxbig,gtybig,gtzbig,
     +              nbelec,volm1big,tabout,nbout,nbcap)
         call makerh2(rhobig,dxbig,dybig,dzbig,nsx,nsy,nsz,
     +     gtxbig,gtybig,gtzbig,phibig,rhslbig,
     +     sxm1big,sym1big,szm1big,
     +     psxbig,psybig,pszbig,psxxbig,
     +     psyybig,pszzbig,psx2big,psy2big,psz2big)
         call solve(mxbig,mybig,mzbig,mxm1big,mym1big,mzm1big
     +     ,nsx,nsy,nsz,revbig,rhslbig,phibig,csolbig
     +     ,sxm1big,sym1big,szm1big)
         call makerhsf(rho,dx,dy,dz,nsx,nsy,nsz,
     +     gtx,gty,gtz,phi,rhsl,sxm1,sym1,szm1,
     +     psx,psy,psz,psxx,psyy,pszz,psx2,psy2,psz2,
     +     gxbig,gybig,gzbig,nx,ny,nz,csolbig)
         call solve(mx,my,mz,mxm1,mym1,mzm1,nsx,nsy,nsz,
     +        rev,rhsl,phi,csol,sxm1,sym1,szm1)
         if (mod(i-1,10).eq.0) then
         call enerele2g(npart,nbelec,qp,nx,ny,nz,
     +     csolbig,gxbig,gybig,gzbig,nbout,
     +     csol,gx,gy,gz,inttab1,nbdt,pasgrid,
     +     enele,einout,elin,nbcap,liste2)
         end if
         call pspech(ntx,nty,ntz,gtxbig,gtybig,gtzbig,
     +     rhobig,csolbig,sxm1big,sym1big,szm1big,nbion)
         call pspech(ntx,nty,ntz,gtx,gty,gtz,
     +        rho,csol,sxm1,sym1,szm1,nbion)
         print*,'entree ds forceg,nbout',nbout
         call force2g(npart,nbelec,fp,qp,nx,ny,nz,
     +      csolbig,gxbig,gybig,gzbig,nbout,
     +      csol,gx,gy,gz,
     +      inttab1,inttab2,nbdt,pasgrid,nbcap,liste2)
c         call incproj2(delvm,mpro,pospro,vpro,
c     +     dltt,qp,qpold,fp,cutoff,
c     +     nbion,nbelec,npart,chapro,first,i+1,
c     +     rcutee,epro,impara,last,nbcap,einterne,titreq,
c     +     csolbig,gxbig,gybig,gzbig,nbout,
c     +     csol,gx,gy,gz,nx,ny,nz,
c     +     inttab1,inttab2,nbdt,pasgrid,liste2)
c
         call incproj(mpro,pospro,posprold,vpro,dltt,qp,fp,cutoff,
     +     nbion,nbelec,npart,chapro,first,i+1,rcutee,epro,
     +        impara,last,nbcap,einterne,titreq)
         print*,'sortie de forceg'
         call move(qp,qpold,fp,nbelec,npart,dltt,ekin,ekinout,
     +        nbout,tabout,lcine,nbcap)
c         epoti=epotip1
c         if ((mod(i,20).eq.0).or.(i.eq.nbt)) then
         if (mod(i-1,10).eq.0) then    
         call pspech2(ntx,nty,ntz,gtxbig,gtybig,gtzbig,
     +     rhobig,csolbig,sxm1big,sym1big,szm1big,nbion)
         call pspech2(ntx,nty,ntz,gtx,gty,gtz,
     +        rho,csol,sxm1,sym1,szm1,nbion)
         call enertot2g(npart,nbion,nbelec,qp,nx,ny,nz,
     +     csolbig,gxbig,gybig,gzbig,nbout,
     +     csol,gx,gy,gz,inttab1,nbdt,pasgrid,
     +     enele,einout,elin,enejel,ekin,
     +     ekinout,enetot,i/10+1,lcine,first2,nbcap,liste2,titreq)
         call angular(npart,qp,gx,nx,i/10+1)
c         call echanti2(npart,qp,100,gtech,i/10+1,nbcap,nbelec)
         end if 
         call trace(qp,npart,first,i+1,nbion,nbcap)
         if ((i.gt.nbatt).and.(i.le.(nbatt+nbphot))) then
            call makecha(chnomb,i-nbatt)
            titi='t'//chnomb
            tito='r'//chnomb
            print*,'titi=',titi
            if (mod(i-1,5).eq.0) then
               call sortie(gxbig,gybig,gzbig,nx,ny,nz,csolbig,titi)            
            end if
            call sortierho(nx,ny,nz,ntx,nty,ntz,tito,
     +           sxm1,sym1,szm1,gx,gy,gz,rho,
     +           sxm1big,sym1big,szm1big,gxbig,gybig,gzbig,rhobig)
c            call sortietest(gx,gy,gz,gxbig,gybig,gzbig,
c     +           nx,ny,nz,csol,csolbig)
         end if
         call capture(npart,qp,pospro,nbelec,nbcap)
         if (capturee) then
         else
            if (pospro(1).gt.(1000.d0)) then
               call docapture(npart,qp,qpold,
     +           fp,pospro,nbcap,chapro,nbelec,cutoff,einterne)
               capturee=.true.
               print*,'capture effectuee !!!'
            end if
         end if
      end do 
c     --------- fin de la boucle sur le temps -----------------
      last=.true.
      call incproj(mpro,pospro,posprold,vpro,dltt,qp,fp,cutoff,
     +     nbion,nbelec,npart,chapro,first,i+1,
     +     rcutee,epro,impara,last,nbcap,einterne,titreq)
c      call sortie(gx,gy,gz,nx,ny,nz,csol,'tfi')
      end
c----------------------subroutinemakeinit --------
c-    Sous routine qui distribue les pseudo particules en utilisant le  genera --
c-    teur aleatoire sobol et la densite issue ue modele de Thomas-fermi.      --
c-    Une photo des positions et impulsions des pseudo-part est ensuite faite. --
c-----------------------------------------------------------------------------
      subroutine makeinit(nbelec,qp,qpold,dltt,npart)
      include 'ceq3d.f'
      real*8 qp(3,npartmax)
      real*8 qpold(3,npartmax)
      real*8 nbelec,dltt
      real*8 rmax,pmax,rhorp(nrmax,npmax)
      integer npart,ntab,nr,np
      print*,'initialisation a partir d''un fichier exterieur'
c      call readfrp(rhorp,rmax,pmax,nr,np)
c      call initialize(qp,qpold,rhorp,rmax,pmax,npart,nr,np,nbelec)
      call initialise(qp,qpold,npart,nbelec)
      end
c---------------subroutinegrille(x0,xn,nx,gx,colx + (y) + (z)--------------
c-    Sous-Routine qui calcule la grille a trois dimension sur laquelle  vont -
c-    etre fait les calculs. Elle calcule ensuite les positions des points    -
c-    de collocation sur cette grile (colxyz) a partir des absices de gauss  --
c---------------------------------------------------------------------------
      subroutine grille(nx,gx,gtx,ny,gy,gty,nz,gz,gtz,
     +             xclu,n1xyz,ntx,nty,ntz,nsx,nsy,nsz)
      include 'ceq3d.f'
      real*8 x0,xn
      real*8 y0,yn
      real*8 z0,zn
      real*8 gx(0:NHFX)
      real*8 gy(0:NHFY)
      real*8 gz(0:NHFZ)
      real*8 colx(DE*NHFX)
      real*8 coly(DE*NHFY)
      real*8 colz(DE*NHFZ)
      real*8 gtx(0:NTXT)
      real*8 gty(0:NTYT)
      real*8 gtz(0:NTZT)
      real*8 xclu
      integer nx,ny,nz,ntx,nty,ntz,nsx,nsy,nsz
      integer n1xyz
      integer i
      print*,'debut construction grille fine'
      x0=-1.d0*xclu
      y0=-1.d0*xclu
      z0=-1.d0*xclu
      xn=xclu
      yn=xclu
      zn=xclu
      nx=n1xyz
      ny=n1xyz
      nz=n1xyz      
      call mkgri(x0,xn,gx,1.d0,nx)
      call mkgri(y0,yn,gy,1.d0,ny)
      call mkgri(z0,zn,gz,1.d0,nz)
      ntx=2*nx+1
      nty=2*ny+1
      ntz=2*nz+1
      nsx=ntx-1
      nsy=nty-1
      nsz=ntz-1
      call colloc(gx,colx,nx)
      call colloc(gy,coly,ny)
      call colloc(gz,colz,nz)
      call makegt(nx,ny,nz,gx,gy,gz,colx,coly,colz,
     +     ntx,nty,ntz,gtx,gty,gtz)
      do i=0,ntx
         print*,'grille fine:',i,gtx(i),gty(i),gtz(i)
      end do
      print*,'fin construction grille fine'
      end
c-----------------------subroutinelitinput ---------------------------------
c     Sous routine qui lit le fichier tfd.inp qui contient les parametres de la -
c     simulation. Ces parametres sont detailles dans ce fichier.                -
c----------------------------------------------------------------------------
      subroutine litinput(n1xyz,xclu,xboite,
     +     nbion,nbelec,npart,dltt,nbt,
     +     mpro,epro,impara,cutoff,xinit,chapro,rcutee,
     +     n1big,n2big,delvm)
      include 'ceq3d.f'
      integer n1xyz,npart,nbt
      integer n1big,n2big
      real*8 xclu,xboite,nbion,dltt,chapro
      real*8 mpro,epro,impara,cutoff,xinit
      real*8 rcutee,nbelec,delvm
      open (1,file='vlas.inp',status='old')
      read (1,*)
      read (1,*),n1xyz
      read (1,*)
      read (1,*),n1big
      read (1,*)
      read (1,*),n2big
      read (1,*)
      read (1,*),xclu
      read (1,*)
      read (1,*),xboite
      read (1,*)
      read (1,*),nbion
      read (1,*)
      read (1,*),nbelec
      read (1,*)
      read (1,*),npart
      read (1,*)
      read (1,*),nbt
      read (1,*)
      read (1,*),dltt
      read (1,*)
      read (1,*),mpro
      read (1,*)
      read (1,*),epro
      read (1,*)
      read (1,*),impara
      read (1,*)
      read (1,*),cutoff
      read (1,*)
      read (1,*),xinit
      read (1,*)
      read (1,*),chapro
      read (1,*)
      read (1,*),rcutee
      read (1,*)
      read (1,*),delvm
      close (1)
      end
c-----------------------subroutinemkgri -----------------------------------
c-    Sous routine utilise lors de l'appel de grille(). Elle calcule les coor -
c-    donnees d'une grille unidimentionelle. la grille a 3d etant le produit  -
c-    de trois de ces grilles :                                               -
c-    x0 : premier point de la grille (=g(0))                           -
c-    xn : dernier point de la grille (=g(n))                           -
c-    g : tableau de real*8 de dimension NHF (defini ds ceq3d.f)       -
c-    n : indice du dernier point de grille (n<NHF)                    -
c---------------------------------------------------------------------------
      subroutine mkgri(x0,xn,g,a,n)
      include 'ceq3d.f'
      real*8 g(0:NHF)
      real*8 a,dx,x0,xn
      integer i,n
      dx=(xn-x0)/dfloat(n)
      if (a.ne.1.d0) dx=(xn-x0)*(a-1.d0)/(a**(n-1)-1.d0)
      g(0)=x0
      do i=1,n
         g(i)=g(i-1)+dx
         dx=dx*a
      enddo
      end
c----------------------------------------------------------------
      subroutine mkgri2(xclu,xboite,n1,n2,g)
      include 'ceq3d.f'
      real*8 g(0:NHF)
      real*8 xclu,xboite
      real*8 a,dg(1:NHF),h1,h2b
      integer i,n1,n2
      h1=xclu/dfloat(n1-1)
      do i=1,n1
         dg(i)=dfloat(i-1)*h1
      end do
      call findacc(h1,xboite-xclu,a,n2)
      print*,'a',a
      if (a.ne.1.d0) then
         h2b=(xboite-xclu)*(1.d0-a)/(1.d0-a**n2)
         print*,'h2b',h2b
         do i=1,n2
            dg(n1+i)=dg(n1+i-1)+h2b*(a**(i-1))
         end do
      else
         h2b=(xboite-xclu)/dfloat(n2)
         do i=1,n2
            dg(n1+i)=dg(n1)+dfloat(i)*h2b
         end do
      end if
      do i=0,n1+n2-1
         g(i)=-1.d0*dg(n1+n2-i)
      end do
      do i=n1+n2,2*(n1+n2)-2
         g(i)=-1.d0*g(2*(n1+n2)-i-2)
      end do
      end
c----------------------------------------------------
      subroutine findacc(h1,l,a,n)
      implicit none
      real*8 h1,l,a,amin,amax,h2,eps
      integer nbt,n
      eps=1d-10
      amin=1.d0+eps
      amax=10.d0
      if ((h2(amin,l,n)-h1)*(h2(amax,l,n)-h1).ge.0.d0) then
         print*,'probleme sous routine findacc'
         stop
      else
         nbt=1
         do while (((amax-amin).gt.eps).and.(nbt.lt.100))
            a=(amin+amax)*0.5d0
            if (((h2(a,l,n)-h1)*(h2(amin,l,n)-h1)).gt.0.d0) then
               amin=a
            else
               amax=a
            end if
            nbt=nbt+1
         end do
      end if
      end
c------------------------------------------------------------
      function h2(a,l,n)
      implicit none 
      real*8 a,l,h2
      integer n
      h2=l*(1.d0-a)/(1.d0-a**n)
      end
c-----------------------colloc (g,col,n) ----------------------------------
c     S.R qui calcule les coord des points de colloc sur une grille donnee      -
c     Symboliquement on a:                                                     -
c     -
c     g0  x1   x2   g1  x3 //  gi x2i+1  x2i+2  gi+1 // gn-1 x2n-1  x2n   gn-
c     .   x    x    .     x     .    x      x     .       .     x     x    -
c     -
c     ou on a x21+1-gi=0.21(gi+1-gi) et x2i+2-gi=0.78(gi+1-gi)              -
c     les xi st stocke dans col(1,2n).
c----------------------------------------------------------------------------
      subroutine colloc(g,col,n)
      include 'ceq3d.f'
      real*8 g(0:NHF)
      real*8 col(DE*NHF)
      integer i,j,n
      real*8 u
      dimension u(DE)
      u(1)=0.21132486540519d0
      u(2)=0.78867513459481d0 
      do i=0,n-1
         do j=1,DE
            col(DE*i+j)=g(i)+(g(i+1)-g(i))*u(j)
         end do
      end do
      end
c-----------------subroutinemakegt ------------------------------------
c-    S.R qui cree un tableau gt qui reuni les tableau g et col           -
c-    gt(0)=g(0), gt(i)=col(i), gt(n)=g(n). Ce tableau est utilise lors du-
c-    calcul de la force press                                            -
c-----------------------------------------------------------------------
      subroutine makegt(nx,ny,nz,gx,gy,gz,colx,coly,colz,
     +     ntx,nty,ntz,gtx,gty,gtz)
      include 'ceq3d.f'
      real*8 gx(0:NHFX)
      real*8 gy(0:NHFY)
      real*8 gz(0:NHFZ)
      real*8 colx(DE*NHFX)
      real*8 coly(DE*NHFY)
      real*8 colz(DE*NHFZ)
      real*8 gtx(0:NTXT)
      real*8 gty(0:NTYT)
      real*8 gtz(0:NTZT)
      integer ntx,nty,ntz
      integer nx,ny,nz
      integer i
      do i=0,ntx
         call gricol(i,colx,gx,nx,gtx(i))
      end do
      do i=0,nty
         call gricol(i,coly,gy,ny,gty(i))
      end do
      do i=0,ntz
         call gricol(i,colz,gz,nz,gtz(i))
      end do
      end
c-----------------procedure xi--------g(i)----x-----g(i+1)---------------
      subroutine xi(g,imax,x,i,in)
      include 'ceq3d.f'
      real*8 x
      real*8 g(0:NHF)
      integer imax,i
      logical in
      in=.true.
      if ((x.lt.g(0)).or.(x.gt.g(imax))) then
         in=.false.
         return
      end if
      if (x.eq.g(0)) then
         i=0
      else  
         i=0
         do while (x.gt.g(i+1))
            i=i+1
         end do
      end if
      end   
c------subroutinegriech(ntech,gtech,rmax)-----------------
      subroutine griech(ntech,gtech,rmax)
      include 'ceq3d.f'
      real*8 gtech(0:100)
      real*8 rmax
      integer ntech,nbprem
      integer l
      nbprem=10
      gtech(0)=0.d0
      gtech(1)=rmax/ntech*nbprem
      do l=2,ntech
         gtech(l)=gtech(1)+l*(rmax-gtech(1))/ntech
      end do
      end
c------------echanti2-------------------------------------------
      subroutine echanti2(npart,qp,ntech,gtech,j,nbcap,nbelec)
      include 'ceq3d.f'
      real*8 gtech(0:100)
      real*8 rhoech(0:100),nbelec
      real*8 ri,rmin,dumx,dumy
      real*8 qp(3,npartmax),rl,vol,normal
      integer i,l,nbout,j
      integer ntech
      integer npart,nbcap
      nbout=0
      do i=0,ntech
         rhoech(i)=0.d0
      end do
      rmin=1.d10
      do l=1,npart-nbcap
         rl=dsqrt(qp(1,l)**2+qp(2,l)**2+qp(3,l)**2)
         if (rmin.gt.rl) rmin=rl
         if ((rl.ge.gtech(0)).and.(rl.le.gtech(ntech))) then
            if (rl.lt.(gtech(ntech)+gtech(ntech-1))/2) then
               i=0
               ri=0.5d0*(gtech(i+1)+gtech(i))
               do while (rl.gt.ri)
                  i=i+1
                  ri=0.5d0*(gtech(i+1)+gtech(i))
               end do
            else 
               i=ntech
            end if
            rhoech(i)=rhoech(i)+1
         else 
            nbout=nbout+1    
         end if
      end do
      do i=0,ntech
         if (i.eq.0) then
            vol=4.d0*pi*((gtech(1)/2.d0)**3)/3.d0
         else
            if (i.eq.ntech) then
               vol=4.d0*pi*(gtech(i)**3-
     +              ((gtech(i)+gtech(i-1))/2.d0)**3)/3.d0
            else
               vol=4.d0*pi*(((gtech(i+1)+gtech(i))/2.d0)**3
     +              -((gtech(i)+gtech(i-1))/2.d0)**3)/3.d0
            end if
         end if 
         rhoech(i)=rhoech(i)/vol
      end do
      normal=nbelec/dfloat(npart)
      if (j.eq.0) then
         open (1,file='echanti2.dat',status='unknown')
         do i=0,ntech
            write (1,'(2e14.6)') gtech(i),rhoech(i)*normal
         end do
         write (1,'(2e14.6)') 0.d0,0.d0	
         close (1)
      else
         open (1,file='echanti2.dat',status='old')
         do i=1,j*(ntech+2)
            read (1,'(2e14.6)') dumx,dumy
         end do
         do i=0,ntech
            write (1,'(2e14.6)') gtech(i),rhoech(i)*normal
         end do
         write (1,'(2e14.6)') 0.d0,0.d0	
         close (1)
      end if
      end
c-----------------sousroutine volumes -------------------
      subroutine volumes(volm1,gtx,gx,nx,gty,gy,ny,gtz,gz,nz)     
      include 'ceq3d.f'
      real*8 volm1(0:DE*NHFX+1,0:DE*NHFY+1,0:DE*NHFZ+1)
      real*8 gx(0:NHFX)
      real*8 gy(0:NHFY)
      real*8 gz(0:NHFZ)
      real*8 gtx(0:NTXT)
      real*8 gty(0:NTYT)
      real*8 gtz(0:NTZT)
      real*8 lx,ly,lz
      integer nx,ny,nz
      integer i,j,k
      do i=0,de*nx+1
         do j=0,de*ny+1
            do k=0,de*nz+1
               call longueur(gx,gtx,nx,i,lx)
               call longueur(gy,gty,ny,j,ly)
               call longueur(gz,gtz,nz,k,lz)
               if (lx*ly*lz.le.0) then
                  print*,'probleme procedure volm1',lx,ly,lz,i,j,k
               else
                  volm1(i,j,k)=1.d0/(lx*ly*lz)
               end if
            end do
         end do
      end do
      end
c---------------sousroutine longueur ---------------------
      subroutine longueur(g,gt,n,i,l)
      include 'ceq3d.f'
      real*8 g(0:NHFX)
      real*8 gt(0:NTXT)
      real*8 l
      integer i,n
      if ((i.ge.0).and.(i.le.de*n+1)) then
         if (i.eq.0) then
            l=(gt(1)-g(0))*0.5d0
         else
            if (i.eq.1) then
               l=(gt(2)-g(0))*0.5d0
            else
               if (i.eq.de*n+1) then
                  l=(g(n)-gt(de*n))*0.5d0
               else
                  if (i.eq.de*n) then
                     l=(g(n)-gt(de*n-1))*0.5d0
                  else
                     l=(gt(i+1)-gt(i-1))*0.5d0
                  end if
               end if
            end if
         end if
      else
         print*,'probleme procedure longueur',i,de*n+1
      end if
      end
c----------subroutine gricol(i,col,g,n,x) ------------------------
      subroutine gricol(i,col,g,n,x)
      include 'ceq3d.f'
      real*8 g(0:NHF)
      real*8 col(DE*NHF)
      real*8 x
      integer i,n
      if (i.eq.0) then
         x=g(0)
      else
         if (i.eq.de*n+1) then
            x=g(n)
         else
            x=col(i)
         end if
      end if
      end
c------------subroutinefindig -------------------------------------
      subroutine findig(g,n,x,i,in)
      include 'ceq3d.f'
      real*8 g(0:NHF)
      real*8 x,lxs2
      integer i,n
      logical in
      in=.true.
      lxs2=(g(1)-g(0))*0.5d0
      if ((x.ge.(g(1)+lxs2)).and.(x.le.(g(n-1)-lxs2))) then
         i=0
         do while(g(i+1).lt.x)
            i=i+1
         end do
         if (((g(i+1)-x)).lt.(x-g(i))) i=i+1
      else
         in=.false.
      end if
      end
c------------subroutinefindi -------------------------------------
      subroutine findi(g,n,x,i,a,in)
      include 'ceq3d.f'
      real*8 g(0:NTXT)
      real*8 x,a
      integer i,n
      logical in
      in=.true.
      if ((x.ge.g(0)).and.(x.le.g(n))) then
         i=0
         do while(g(i+1).lt.x)
            i=i+1
         end do
         a=(g(i+1)-x)/(g(i+1)-g(i))
      else
         in=.false.
      end if
      end
c----------------sousroutine trace(qp,npart,first,nb,nbion)---------  
      subroutine trace(qp,npart,first,nb,nbion,nbcap) 
      include 'ceq3d.f'
      real*8 qp(3,npartmax)
      real*8 r2,rp2,rc2
      real*8 spill,rayjel2,nbion
      logical first
      integer npart,i,nb,j,k,out
      integer nbspill,nbcap
      character*1 lchif(10)
      character*11 nameq
      rayjel2=((nbion**(1.d0/3.d0))*rs)**2.d0
      print*,'rayjel2,16*rayjel2',rayjel2,16.d0*rayjel2
      lchif(1)='1'
      lchif(2)='2'
      lchif(3)='3'
      lchif(4)='4'
      lchif(5)='5'
      lchif(6)='6'
      lchif(7)='7'
      lchif(8)='8'
      lchif(9)='9'
      lchif(10)='0'
      if (first) then
         do k=1,10
            nameq='traceq'//lchif(k)//'.dat'
            open (1,file=nameq,status='unknown')
            write(1,'(3e15.7)') qp(1,k),qp(2,k),qp(3,k)
            close(1)
         end do
         r2=0.d0
	 rc2=0.d0
         out=0
         nbspill=0
         do i=1,npart-nbcap
            rp2=qp(1,i)**2+qp(2,i)**2+qp(3,i)**2      
            if (rp2.gt.3600.d0) then
               out=out+1
               rc2=rc2+rp2
            else
               rc2=rc2+rp2
               r2=r2+rp2
            end if
            if ((rp2.gt.rayjel2).and.
     +          (rp2.lt.(16.d0*rayjel2))) then
               nbspill=nbspill+1 
            end if
         end do
         spill=100.d0*dfloat(nbspill)/dfloat(npart-nbcap)
         r2=dsqrt(r2/(npart-nbcap-out))
         rc2=dsqrt(rc2/(npart-nbcap))
         open (3,file='moye.dat',status='unknown')
         write (3,'(I10,3e15.7)') nb,r2,rc2,spill
         close(3)
      else
         do k=1,10
            nameq='traceq'//lchif(k)//'.dat'
            open (1,file=nameq,status='old')
            do j=1,nb-1
               read(1,'(3e15.7)')
            end do
            write(1,'(3e15.7)') qp(1,k),qp(2,k),qp(3,k)
            close(1)
         end do
         r2=0.d0
	 rc2=0.d0
         out=0
         nbspill=0
         do i=1,npart-nbcap
            rp2=qp(1,i)**2+qp(2,i)**2+qp(3,i)**2      
            if (rp2.gt.3600.d0) then
               out=out+1
               rc2=rc2+rp2
            else
               r2=r2+rp2
               rc2=rc2+rp2
            end if
            if ((rp2.gt.rayjel2).and.
     +          (rp2.lt.(16.d0*rayjel2))) then
               nbspill=nbspill+1 
            end if
         end do
         spill=100.d0*dfloat(nbspill)/dfloat(npart-nbcap)
         r2=dsqrt(r2/(npart-out-nbcap))
         rc2=dsqrt(rc2/(npart-nbcap))
         open (3,file='moye.dat',status='old')
         do j=1,nb-1
            read(3,'(I10,3e15.7)')
         end do
         write (3,'(I10,3e15.7)') nb,r2,rc2,spill
         close(3)
      end if
      end 
c---------------subroutineprod1(n,a,b,c)----------------------------------
      subroutine prod1(n,a,b,c)
      include'ceq3d.f'
      integer i,j,n,m
      real*8 a(NXS,NXS),b(NXS,NXS),c(NXS,NXS)
      do i=1,n
         do j=1,n
            c(i,j)=0.d0
            do m=1,n
               c(i,j)=c(i,j)+a(i,m)*b(m,j)
            end do
         end do
      end do
      end
c---------------subroutineprod(n,a,b,c)----------------------------------
      subroutine prod(n,a,b,c)
      include'ceq3d.f'
      integer i,j,n,m
      real*8 a(0:NTXT,0:NTXT),b(0:NTXT,0:NTXT),c(0:NTXT,0:NTXT)
      do i=0,n-1
         do j=0,n-1
            c(i,j)=0.d0
            do m=0,n-1
               c(i,j)=c(i,j)+a(i,m)*b(m,j)
            end do
         end do
      end do
      end
c---------------subroutineprod3(n,a,b,c,d)----------------------------------
      subroutine prod3(n,a,b,c,d)
      include'ceq3d.f'
      integer n
      real*8 a(NXS,NXS),b(NXS,NXS),c(NXS,NXS)
      real*8 d(NXS,NXS),inter(NXS,NXS)
      call prod1(n,a,b,inter)
      call prod1(n,inter,c,d)
      end
c---------------subroutinedup1(n,a,b)----------------------------------
      subroutine dup1(n,a,b)
      include'ceq3d.f'
      integer i,j,n
      real*8 a(NXS,NXS),b(NXS,NXS)
      do i=1,n
         do j=1,n
            b(i,j)=a(i,j)
         end do
      end do
      end
c---------------subroutinedup(n,a,b)----------------------------------
      subroutine dup(n,a,b)
      include'ceq3d.f'
      integer i,j,n
      real*8 a(0:NTXT,0:NTXT),b(0:NTXT,0:NTXT)
      do i=0,n-1
         do j=0,n-1
            b(i,j)=a(i,j)
         end do
      end do
      end
c-----------------subroutinediagonal(n,a,vr,vi,rr,ri)------------
      subroutine diagonal(n,a,vr,vi,rr,ri)	      
      include 'ceq3d.f'
      integer ifail
      real*8 a(NXS,NXS),ri(NXS),rr(NXS)
      real*8 vi(NXS,NXS),vr(NXS,NXS)
      real*8 inter(NXS,NXS)
      integer intger(NXS),n
      external f02agf
      call dup1(n,a,inter)
      ifail = 1
      call f02agf(inter,NXS,n,rr,ri,vr,
     +     NXS,vi,NXS,intger,ifail)
      if (ifail.ne.0) then
         print*, 'error in f02agf. ifail =', ifail
      end if
      end
c------------------subroutineinverse(n,a,am1)----------------
      subroutine inverse(n,a,am1)
      include 'ceq3d.f'	
      integer lwork
      parameter (lwork=64*(NTXT+1))
      integer ifail,info,n
      real*8 a(0:NTXT,0:NTXT),work(lwork),am1(0:NTXT,0:NTXT)
      integer ipiv(0:NTXT)
      external dgetrf, dgetri
      call dup(n,a,am1)
      if (n.le.NTXT+1) then
         call dgetrf(n,n,am1,NTXT+1,ipiv,info)
         if (info.eq.0) then
            call dgetri(n,am1,NTXT+1,ipiv,work,lwork,info)
            ifail = 0
         else
            print*, 'matrice non inversible'
         end if
      end if
      end
c------------------subroutineinverse1(n,a,am1)----------------
      subroutine inverse1(n,a,am1)
      include 'ceq3d.f'	
      integer lwork
      parameter (lwork=64*(NXS+2))
      integer ifail,info,n
      real*8 a(NXS,NXS),work(lwork),am1(NXS,NXS)
      integer ipiv(0:NTXT)
      external dgetrf, dgetri
      call dup1(n,a,am1)
      if (n.le.NXS) then
         call dgetrf(n,n,am1,NXS,ipiv,info)
         if (info.eq.0) then
            call dgetri(n,am1,NXS,ipiv,work,lwork,info)
            ifail = 0
         else
            print*, 'matrice non inversible'
         end if
      end if
      end
c----------------------------------------------------------------
      subroutine makes(ntx,gtx,nx,gx,sx)
      include 'ceq3d.f'          
      real*8 gtx(0:NTXT)
      real*8 gx(0:NHFX)
      real*8 sx(0:NTXT,0:NTXT),spl	
      integer ntx,nx
      integer k,i,j
      do k=0,ntx
         do i=0,ntx
            sx(k,i)=0.d0
         end do
      end do
      do j=0,nx-1
         do i=2*j,2*j+3
            do k=2*j+1,2*j+2
               call scub1(gx,nx,i/2,mod(i,2),gtx(k),spl)
               sx(k,i)=spl
            end do
         end do
      end do
      call scub1(gx,nx,0,0,gtx(0),spl)
      sx(0,0)=spl
      call scub1(gx,nx,nx,0,gtx(ntx),spl)
      sx(ntx,ntx-1)=spl
      end              
c----------------------------------------------------------------
      subroutine makes2(ntx,gtx,nx,gx,sx)
      include 'ceq3d.f'          
      real*8 gtx(0:NTXT)
      real*8 gx(0:NHFX)
      real*8 sx(0:NTXT,0:NTXT),spl	
      integer ntx,nx
      integer k,i,j
      do k=0,ntx
         do i=0,ntx
            sx(k,i)=0.d0
         end do
      end do
      do j=0,nx-1
         do i=2*j,2*j+3
            do k=2*j+1,2*j+2
               call scub3(gx,nx,i/2,mod(i,2),gtx(k),spl)
               sx(k,i)=spl
            end do
         end do
      end do
      call scub3(gx,nx,0,0,gtx(0),spl)
      sx(0,0)=spl
      call scub3(gx,nx,nx,0,gtx(ntx),spl)
      sx(ntx,ntx-1)=spl
      end              
c-----------------------------------------------------------------
      subroutine makes1(ntx,gtx,nx,gx,sx)
      include 'ceq3d.f'          
      real*8 gtx(0:NTXT)
      real*8 gx(0:NHFX)
      real*8 sx(0:NTXT,0:NTXT),spl	
      integer ntx,nx
      integer k,i,j
      do k=0,ntx
         do i=0,ntx
            sx(k,i)=0.d0
         end do
      end do
      do j=0,nx-1
         do i=2*j,2*j+3
            do k=2*j+1,2*j+2
               call scub2(gx,nx,i/2,mod(i,2),gtx(k),spl)
               sx(k,i)=spl
            end do
         end do
      end do
      call scub2(gx,nx,0,0,gtx(0),spl)
      sx(0,0)=spl
      call scub2(gx,nx,nx,0,gtx(ntx),spl)
      sx(ntx,ntx-1)=spl
      end              
c------------------------------------------------------------
      subroutine extract(ntx,dd,d)
      include 'ceq3d.f'
      real*8 dd(0:NTXT,0:NTXT)
      real*8 d(NXS,NXS)
      integer i,j,ntx
      do i=1,ntx
         do j=1,ntx
            d(i,j)=dd(i,j)
         end do
      end do
      end
c-----------------------------------------------------------
      subroutine static(ntx,nty,ntz,nx,ny,nz,
     +     gtx,gty,gtz,gx,gy,gz,
     +     dx,dy,dz,mx,my,mz,psx,psy,psz,psxx,psyy,pszz,
     +     mxm1,mym1,mzm1,rev,nsx,nsy,nsz,psx2,psy2,psz2,
     +     sx,sy,sz,s1x,s1y,s1z,sxm1,sym1,szm1)
      include 'ceq3d.f'
      real*8 psx(0:NTXT),psy(0:NTYT),psz(0:NTZT)
      real*8 psxx(0:NTXT),psyy(0:NTYT),pszz(0:NTZT)
      real*8 psx2(0:NTXT),psy2(0:NTYT),psz2(0:NTZT)
      real*8 gx(0:NHFX),gy(0:NHFY),gz(0:NHFZ)
      real*8 gtx(0:NTXT),gty(0:NTYT),gtz(0:NTZT)
      real*8 sx(0:NTXT,0:NTXT),sy(0:NTYT,0:NTYT),sz(0:NTZT,0:NTZT)
      real*8 s1x(0:NTXT,0:NTXT),s1y(0:NTYT,0:NTYT),s1z(0:NTZT,0:NTZT)
      real*8 dx(0:NTXT,0:NTXT),dy(0:NTYT,0:NTYT),dz(0:NTZT,0:NTZT)
      real*8 s2x(0:NTXT,0:NTXT),s2y(0:NTYT,0:NTYT),s2z(0:NTZT,0:NTZT)
      real*8 sxm1(0:NTXT,0:NTXT)	
      real*8 sym1(0:NTYT,0:NTYT)	
      real*8 szm1(0:NTZT,0:NTZT)	
      real*8 dex(NXS,NXS),dey(NYS,NYS),dez(NZS,NZS)
      real*8 lxr(NXS),lxi(NXS)
      real*8 lyr(NYS),lyi(NYS)
      real*8 lzr(NZS),lzi(NZS)
      real*8 rev(NXS,NYS,NZS)
      real*8 mxm1(NXS,NXS),mym1(NYS,NYS),mzm1(NZS,NZS)
      real*8 mx(NXS,NXS),my(NYS,NYS),mz(NZS,NZS)
      real*8 mxi(NXS,NXS),myi(NYS,NYS),mzi(NZS,NZS)
      real*8 interx(NXS,NXS)
      real*8 intery(NYS,NYS)
      real*8 interz(NZS,NZS)
      real*8 deno,prima,primb
      integer ntx,nty,ntz,nx,ny,nz,nsx,nsy,nsz
      integer i,j,k
      call makes(ntx,gtx,nx,gx,sx)
      call makes(nty,gty,ny,gy,sy)
      call makes(ntz,gtz,nz,gz,sz)
      call makes1(ntx,gtx,nx,gx,s1x)
      call makes1(nty,gty,ny,gy,s1y)
      call makes1(ntz,gtz,nz,gz,s1z)
      call makes2(ntx,gtx,nx,gx,s2x)
      call makes2(nty,gty,ny,gy,s2y)
      call makes2(ntz,gtz,nz,gz,s2z)
      call inverse(ntx+1,sx,sxm1)
      call inverse(nty+1,sy,sym1)
      call inverse(ntz+1,sz,szm1)
      call prod(ntx+1,s2x,sxm1,dx)     
      call prod(nty+1,s2y,sym1,dy)     
      call prod(ntz+1,s2z,szm1,dz)     
      nsx=ntx-1
      nsy=nty-1
      nsz=ntz-1
      call extract(nsx,dx,dex)
      call extract(nsy,dy,dey)
      call extract(nsz,dz,dez)
      call diagonal(nsx,dex,mx,mxi,lxr,lxi)
      call diagonal(nsy,dey,my,myi,lyr,lyi)
      call diagonal(nsz,dez,mz,mzi,lzr,lzi)
      call inverse1(nsx,mx,mxm1)
      call inverse1(nsy,my,mym1)
      call inverse1(nsz,mz,mzm1)
      call prod3(nsx,mxm1,dex,mx,interx)
      call prod3(nsy,mym1,dey,my,intery)
      call prod3(nsz,mzm1,dez,mz,interz)
      do i=0,ntx
         call prim(gx,nx,i/2,mod(i,2),gtx(0),prima)
         call prim(gx,nx,i/2,mod(i,2),gtx(ntx),primb)
         psx(i)=primb-prima
      end do
      do j=0,nty
         call prim(gy,ny,j/2,mod(j,2),gty(0),prima)
         call prim(gy,ny,j/2,mod(j,2),gty(nty),primb)
         psy(j)=primb-prima
      end do
      do k=0,ntz
         call prim(gz,nz,k/2,mod(k,2),gtz(0),prima)
         call prim(gz,nz,k/2,mod(k,2),gtz(ntz),primb)
         psz(k)=primb-prima
      end do
      do i=0,ntx
         call primx(gx,nx,i/2,mod(i,2),gtx(0),prima)
         call primx(gx,nx,i/2,mod(i,2),gtx(ntx),primb)
         psxx(i)=primb-prima
      end do
      do j=0,nty
         call primx(gy,ny,j/2,mod(j,2),gty(0),prima)
         call primx(gy,ny,j/2,mod(j,2),gty(nty),primb)
         psyy(j)=primb-prima
      end do
      do k=0,ntz
         call primx(gz,nz,k/2,mod(k,2),gtz(0),prima)
         call primx(gz,nz,k/2,mod(k,2),gtz(ntz),primb)
         pszz(k)=primb-prima
      end do
      do i=0,ntx
         call primx2(gx,nx,i/2,mod(i,2),gtx(0),prima)
         call primx2(gx,nx,i/2,mod(i,2),gtx(ntx),primb)
         psx2(i)=primb-prima
      end do
      do j=0,nty
         call primx2(gy,ny,j/2,mod(j,2),gty(0),prima)
         call primx2(gy,ny,j/2,mod(j,2),gty(nty),primb)
         psy2(j)=primb-prima
      end do
      do k=0,ntz
         call primx2(gz,nz,k/2,mod(k,2),gtz(0),prima)
         call primx2(gz,nz,k/2,mod(k,2),gtz(ntz),primb)
         psz2(k)=primb-prima
      end do
      do i=1,nsx
         do j=1,nsy
            do k=1,nsz
               deno=lxr(i)+lyr(j)+lzr(k)
               if (deno.eq.0) then
                  print*,'pb deno'
                  stop
               else
                  rev(i,j,k)=1.d0/deno
               end if
            end do
         end do
      end do
      end
c-----------------------------------------------------------------
      subroutine solve(mx,my,mz,mxm1,mym1,mzm1,nsx,nsy,nsz,
     +     rev,rhsl,phi,csol,sxm1,sym1,szm1)
      include 'ceq3d.f'              
      real*8 rev(NXS,NYS,NZS)
      real*8 mxm1(NXS,NXS),mym1(NYS,NYS),mzm1(NZS,NZS)
      real*8 sxm1(0:NTXT,0:NTXT),sym1(0:NTYT,0:NTYT),szm1(0:NTZT,0:NTZT)
      real*8 mx(NXS,NXS),my(NYS,NYS),mz(NZS,NZS)
      real*8 ww(NXS,NYS,NZS)
      real*8 vect(NXS,NYS,NZS)
      real*8 ptilde(NXS,NYS,NZS)
      real*8 rhsl(NXS,NYS,NZS)
      real*8 phi(0:NTXT,0:NTYT,0:NTZT)
      real*8 csol(0:NTXT,0:NTYT,0:NTZT)
      integer nsx,nsy,nsz
      integer i,j,k
      call tensrus(nsx,nsy,nsz,mxm1,mym1,mzm1,rhsl,ww)
      do i=1,nsx
         do j=1,nsy
            do k=1,nsz
               vect(i,j,k)=rev(i,j,k)*ww(i,j,k)
            end do
         end do
      end do
      call tensrus(nsx,nsy,nsz,mx,my,mz,vect,ptilde)
      do i=1,nsx
         do j=1,nsy
            do k=1,nsz
               phi(i,j,k)=ptilde(i,j,k)
            end do
         end do
      end do
      call tensrus2(nsx+1,nsy+1,nsz+1,sxm1,sym1,szm1,phi,csol)
      print*,'csol(1,1,1)',csol(1,1,1)
      end 
c-------------------------------------------------
      subroutine tensrus(nsx,nsy,nsz,matx,maty,matz,vect,te)
      include 'ceq3d.f'
      real*8 te(NXS,NYS,NZS)
      real*8 fp(NXS,NYS,NZS)
      real*8 fpp(NXS,NYS,NZS)
      real*8 vect(NXS,NYS,NZS)
      real*8 matx(NXS,NXS)
      real*8 maty(NYS,NYS)
      real*8 matz(NZS,NZS)
      integer a,b,c,i,j,k
      integer nsx,nsy,nsz
      do a=1,nsx
         do b=1,nsy
            do k=1,nsz
               fp(a,b,k)=0.d0
               do c=1,nsz
                  fp(a,b,k)=fp(a,b,k)+matz(k,c)*vect(a,b,c)
               end do
            end do
         end do
      end do
      do a=1,nsx
         do k=1,nsz
            do j=1,nsy
               fpp(a,j,k)=0.d0
               do b=1,nsy
                  fpp(a,j,k)=fpp(a,j,k)+maty(j,b)*fp(a,b,k)
               end do
            end do
         end do
      end do   
      do i=1,nsx
         do j=1,nsy
            do k=1,nsz
               te(i,j,k)=0.d0
               do a=1,nsx
                  te(i,j,k)=te(i,j,k)+matx(i,a)*fpp(a,j,k)
               end do
            end do
         end do
      end do
      end
c-------------------------------------------------------------------
      subroutine tensrus2(ntx,nty,ntz,matx,maty,matz,vect,te)
      include 'ceq3d.f'
      real*8 te(0:NTXT,0:NTYT,0:NTZT)
      real*8 fp(0:NTXT,0:NTYT,0:NTZT)
      real*8 fpp(0:NTXT,0:NTYT,0:NTZT)
      real*8 vect(0:NTXT,0:NTYT,0:NTZT)
      real*8 matx(0:NTXT,0:NTXT)
      real*8 maty(0:NTYT,0:NTYT)
      real*8 matz(0:NTZT,0:NTZT)
      integer a,b,c,i,j,k
      integer ntx,nty,ntz
      do a=0,ntx
         do b=0,nty
            do k=0,ntz
               fp(a,b,k)=0.d0
               do c=0,ntz
                  fp(a,b,k)=fp(a,b,k)+matz(k,c)*vect(a,b,c)
               end do
            end do
         end do
      end do
      do a=0,ntx
         do k=0,ntz
            do j=0,nty
               fpp(a,j,k)=0.d0
               do b=0,nty
                  fpp(a,j,k)=fpp(a,j,k)+maty(j,b)*fp(a,b,k)
               end do
            end do
         end do
      end do   
      do i=0,ntx
         do j=0,nty
            do k=0,ntz
               te(i,j,k)=0.d0
               do a=0,ntx
                  te(i,j,k)=te(i,j,k)+matx(i,a)*fpp(a,j,k)
               end do
            end do
         end do
      end do
      end
c--------------------------------------------------------------
      subroutine potentiel(x,nx,gx,y,ny,gy,z,nz,gz,csol,
     +     pote,nbout,npart,nbion)
      include 'ceq3d.f'
      real*8 pote
      real*8 csol(0:NTXT,0:NTYT,0:NTZT)
      real*8 gx(0:NHFX)
      real*8 gy(0:NHFY)
      real*8 gz(0:NHFZ)
      real*8 x,y,z,r
      real*8 sp(3,0:1,0:1),sp1(3,0:1,0:1),nbion
      integer si,sj,sk
      integer nx,ny,nz
      integer gi,gj,gk,nbout,npart
      integer ai,aj,ak,bi,bj,bk,degi,degj,degk
      logical inx,iny,inz,in
      call xi(gx,nx,x,gi,inx)
      call xi(gy,ny,y,gj,iny)
      call xi(gz,nz,z,gk,inz)
      in=((inx.and.iny).and.inz)
      if (in) then
         pote=0.d0
         degi=DE*gi
         degj=DE*gj
         degk=DE*gk
         do ai=0,1
            do bi=0,1       
               call scub12(gx,nx,gi+ai,bi,x,
     +              sp(1,ai,bi),sp1(1,ai,bi))
               call scub12(gy,ny,gj+ai,bi,y,
     +              sp(2,ai,bi),sp1(2,ai,bi))
               call scub12(gz,nz,gk+ai,bi,z,
     +              sp(3,ai,bi),sp1(3,ai,bi))
            end do
         end do
         do ai=0,1
            do bi=0,1
               si=degi+de*ai+bi
               do aj=0,1
                  do bj=0,1
                     sj=degj+de*aj+bj
                     do ak=0,1
                        do bk=0,1
                           sk=degk+de*ak+bk
                           pote=pote+csol(si,sj,sk)*
     +                          sp(1,ai,bi)*sp(2,aj,bj)*sp(3,ak,bk)
                        end do
                     end do
                  end do                  
               end do                  
            end do                  
         end do         
      else
         r=dsqrt(x**2+y**2+z**2)
         pote=(nbion*dfloat(nbout)*qel/dfloat(npart))/r
      end if
      end
c--------------------------------------------------------------
      subroutine champ(x,nx,gx,y,ny,gy,z,nz,gz,csol,champE,in)
      include 'ceq3d.f'
      real*8 champE(3)
      real*8 csol(0:NTXT,0:NTYT,0:NTZT)
      real*8 gx(0:NHFX)
      real*8 gy(0:NHFY)
      real*8 gz(0:NHFZ)
      real*8 x,y,z
      real*8 sp(3,0:1,0:1),sp1(3,0:1,0:1)
      integer si,sj,sk
      integer nx,ny,nz
      integer gi,gj,gk
      integer ai,aj,ak,bi,bj,bk,degi,degj,degk
      logical inx,iny,inz,in
      call xi(gx,nx,x,gi,inx)
      call xi(gy,ny,y,gj,iny)
      call xi(gz,nz,z,gk,inz)
      in=((inx.and.iny).and.inz)
      if (in) then
         champE(1)=0.d0
         champE(2)=0.d0
         champE(3)=0.d0
         degi=DE*gi
         degj=DE*gj
         degk=DE*gk
         do ai=0,1
            do bi=0,1       
               call scub12(gx,nx,gi+ai,bi,x,
     +              sp(1,ai,bi),sp1(1,ai,bi))
               call scub12(gy,ny,gj+ai,bi,y,
     +              sp(2,ai,bi),sp1(2,ai,bi))
               call scub12(gz,nz,gk+ai,bi,z,
     +              sp(3,ai,bi),sp1(3,ai,bi))
            end do
         end do
         do ai=0,1
            do bi=0,1
               si=degi+de*ai+bi
               do aj=0,1
                  do bj=0,1
                     sj=degj+de*aj+bj
                     do ak=0,1
                        do bk=0,1
                           sk=degk+de*ak+bk
                           champE(1)=champE(1)-csol(si,sj,sk)*
     +                          sp1(1,ai,bi)*sp(2,aj,bj)*sp(3,ak,bk)
                           champE(2)=champE(2)-csol(si,sj,sk)*
     +                          sp(1,ai,bi)*sp1(2,aj,bj)*sp(3,ak,bk)
                           champE(3)=champE(3)-csol(si,sj,sk)*
     +                          sp(1,ai,bi)*sp(2,aj,bj)*sp1(3,ak,bk)
                        end do
                     end do
                  end do                  
               end do                  
            end do                  
         end do         
      else
         return
      end if
      end
C-----------------------------------------------------
      subroutine sortie(gx,gy,gz,nx,ny,nz,csol,titre)
      include 'ceq3d.f'
      real*8 gx(0:NHFX)
      real*8 gy(0:NHFY)
      real*8 gz(0:NHFZ)
      real*8 csol(0:NTXT,0:NTYT,0:NTZT)
      integer nx,ny,nz
      real*8 x,h,pot,h2,z,dumr
      integer i,j,l,rep,dumi,dumj
      character*4 titre
      character*25 titrec
      dumi=0
      dumj=0
      dumr=0.d0
      do i=0,nx
      end do
      titrec='out/                '//titre 
      open (2,file=titrec,status='unknown')
      print*,'trec=',titrec
c      print*,'plot x=0 => entrez 1'
c      print*,'plot y=0 => entrez 2'
c      print*,'plot z=0 => entrez 3'
c      read*, rep
      rep=1
      do j=0,100
         h=(gx(nx)-gx(0))/100.d0-1d-10
         x=gx(0)+j*h
         do l=0,100
            h2=(gz(nz)-gz(0))/100.d0-1d-10
            z=gz(0)+l*h2
            if (rep.eq.1) then
               call potentiel(x,nx,gx,z,ny,gy,0.d0,nz,gz,csol,pot,
     +              dumi,dumj,dumr)
            else
               if (rep.eq.2) then
                  call potentiel(x,nx,gx,0.d0,ny,gy,z,nz,gz,csol,pot,
     +              dumi,dumj,dumr)
               else
                  call potentiel(x,nx,gx,z,ny,gy,0.d0,nz,gz,csol,pot,
     +              dumi,dumj,dumr)
               end if
            end if
            if (dabs(pot).lt.1e-10) pot=0.d0
            write(2,'(3e14.6)') x,z,pot
         end do
      end do
      end
c-----------------procedure xis--------g(i)----x-----g(i+1)---------------
      subroutine xis(g,imax,x,i)
      include 'ceq3d.f'
      real*8 x
      real*8 g(0:NHF)
      integer imax,i
      if (x.eq.g(0)) then
         i=0
      else  
         i=0
         do while (x.gt.g(i+1))
            i=i+1
         end do
      end if 
      end   
c-----------------procedure xig--------g(i)----x-----g(i+1)---------------
      subroutine xig(g,imax,x,i)
      include 'ceq3d.f'
      real*8 x
      real*8 g(0:NHF)
      integer imax,i
      i=0
      do while (x.gt.g(i+1))
         i=i+1
      end do
      if (((g(i+1)-x)).lt.(x-g(i))) i=i+1
      end   
c------------------------------------------------------------------
      subroutine pspech(ntx,nty,ntz,gtx,gty,gtz,
     +     rho,csol,sxm1,sym1,szm1,nbion)
      include 'ceq3d.f'              
      real*8 gtx(0:NTXT)
      real*8 gty(0:NTYT)
      real*8 gtz(0:NTZT)
      real*8 sxm1(0:NTXT,0:NTXT),sym1(0:NTYT,0:NTYT),szm1(0:NTZT,0:NTZT)
      real*8 rho(0:NTXT,0:NTYT,0:NTZT)
      real*8 csol(0:NTXT,0:NTYT,0:NTZT)
      real*8 echsol(0:NTXT,0:NTYT,0:NTZT)
      real*8 ech(0:NTXT,0:NTYT,0:NTZT)
      real*8 nbion,potjel,rr
      integer ntx,nty,ntz
      integer i,j,k
      real*8 us3,coefech,coffcor,cfc2,rsrm1
      us3=1.d0/3.d0
      coefech=-((3.d0/PI)**us3)
      coffcor=-0.0333d0
      cfc2=11.4d0*((4.d0*PI/3.d0)**us3)
      do k=0,ntz
         do j=0,nty
            do i=0,ntx
               rsrm1=rho(i,j,k)**us3
               rr=dsqrt(gtx(i)**2+gty(j)**2+gtz(k)**2)
               ech(i,j,k)=coefech*rsrm1+coffcor*dlog(1.d0+cfc2*rsrm1)
     +              +potjel(rr,nbion)
c               ech(i,j,k)=potjel(rr,nbion)
            end do
         end do
      end do
      call tensrus2(ntx,nty,ntz,sxm1,sym1,szm1,ech,echsol)
      do k=0,ntz
         do j=0,nty
            do i=0,ntx
               csol(i,j,k)=csol(i,j,k)+echsol(i,j,k)
            end do
         end do
      end do
      end
c---------------- subroutine readfrp ------------------------------
      subroutine readfrp(rhorp,rmax,pmax,nr,np)
      include 'ceq3d.f'
      real*8 rmax,pmax,rhorp(nrmax,npmax),dumx,dumy
      integer nr,np,dumi,i,j
      open (1,file='frp.d',status='unknown')
      read(1,'(3I10)') nr,np,dumi
      print*,'dumi',dumi
      do i=1,nr
         do j=1,np
            read (1,'(3e15.6)') dumx,dumy,rhorp(i,j)
         end do
      end do
      rmax=dumx
      pmax=dumy
      print*,'pmax,rmax',rmax,pmax
      end
c----------------------------------------------------------------------
C initialize test particle distribution according to density stored
C in file density_file
C
      subroutine initialize(rt,pt,rhorp,rmax,pmax,npart,nr,np,nbelec)
      include 'ceq3d.f'
      real*8 rt(3,npartmax)
      real*8 pt(3,npartmax)
      real*8 rmax,pmax,rhorp(nrmax,npmax)
      real*4 ran2
      real*8 x(6),nbelec
      integer i,nr,np,npart
      real*8 r,p
      real*8 pht
      integer tries,lr,lp,nbi,idum
      real*8 stheta,ee
C Set up initial distribution by using rejection method
C First, pick up quasi random set f coordinates
C then, check random number between 0 and 1 and reject if greater than f at
C that position
      tries = 0
      ee = nbelec/dfloat(npart)
      idum=-1
      do i=1,npart 
30       do nbi=1,6
             x(nbi)=ran2(idum)
         end do
         r = rmax*x(1)**(1.d0/3.d0)
         stheta = sqrt(1.d0-(2.d0*x(3)-1.d0)**2)
         rt(1,i) = r*cos(2.d0*PI*x(2))*stheta
         rt(2,i) = r*sin(2.d0*PI*x(2))*stheta
         rt(3,i) = r*(2.d0*x(3)-1.d0)
         p = pmax*x(4)**(1.d0/3.d0)
         stheta = sqrt(1.d0-(2.d0*x(6)-1.d0)**2)
         pt(1,i) = p*ee*cos(2.d0*PI*x(5))*stheta
         pt(2,i) = p*ee*sin(2.d0*PI*x(5))*stheta
         pt(3,i) = p*ee*(2.d0*x(6)-1.d0)
         lr = int(r*dfloat(nr-1)/rmax)
         lp = int(p*dfloat(np-1)/pmax)
         r  = r*dfloat(nr-1)/rmax - lr
         p  = p*dfloat(np-1)/pmax - lp
C     density function at that point
         pht = rhorp(lr+2,lp+2)  *r     *p
     &        + rhorp(lr+1,lp+2)  *(1.d0-r)*p
     &        + rhorp(lr+2,lp+1)  *r     *(1.d0-p)
     &        + rhorp(lr+1,lp+1)  *(1.d0-r)*(1.d0-p)
C     reject with probability 1-pht
         tries = tries + 1
         if (pht.lt.0.5d0) then
            goto 30
         end if 
      end do
c      do i=1,npart
c         if (mod(i,39).eq.0) then
c            rt(1,i)=10000.d0
c            rt(2,i)=10000.d0
c            rt(3,i)=10000.d0
c            pt(1,i)=0.d0
c            pt(2,i)=0.d0
c            pt(3,i)=0.d0
c         end if
c      end do
      write (*,'(''Number of tries: '',i7," ,number of accepted:",i6)')
     &     tries,i-1
      write (*,'(''Initialization of test particles completed'')')
c      do i=1,npart
c         rt(1,i)=rt(1,i)+1.d0
c      end do
      end
c-----------------------------------------------------------------
      function potjel(r,nbion)
      include 'ceq3d.f'
      real*8 r,nbion
      real*8 r0,potjel
      r0=rs*(nbion**(1.d0/3.d0))
      if (r.lt.r0) then
         potjel=-1.d0*nbion*(3.d0-(r/r0)**2.d0)/(2.d0*r0)
      else
         potjel=-1.d0*nbion/r
      end if
      end 
c---------------------------------------------------------------
      FUNCTION ran2(idum)
      INTEGER idum,IM1,IM2,IMM1,IA1,IA2,IQ1,IQ2,IR1,IR2,NTAB,NDIV
      REAL ran2,AM,EPS,RNMX
      PARAMETER (IM1=2147483563,IM2=2147483399,AM=1./IM1,IMM1=IM1-1,
     *IA1=40014,IA2=40692,IQ1=3668,IQ2=52774,IR1=12211,IR2=3791,
     *NTAB=32,NDIV=1+IMM1/NTAB,EPS=1.2e-7,RNMX=1.-EPS)
      INTEGER idum2,j,k,iv(NTAB),iy
      SAVE iv,iy,idum2
      DATA idum2/123456789/, iv/NTAB*0/, iy/0/
      if (idum.le.0) then
        idum=max(-idum,1)
        idum2=idum
        do 11 j=NTAB+8,1,-1
          k=idum/IQ1
          idum=IA1*(idum-k*IQ1)-k*IR1
          if (idum.lt.0) idum=idum+IM1
          if (j.le.NTAB) iv(j)=idum
11      continue
        iy=iv(1)
      endif
      k=idum/IQ1
      idum=IA1*(idum-k*IQ1)-k*IR1
      if (idum.lt.0) idum=idum+IM1
      k=idum2/IQ2
      idum2=IA2*(idum2-k*IQ2)-k*IR2
      if (idum2.lt.0) idum2=idum2+IM2
      j=1+iy/NDIV
      iy=iv(j)-idum2
      iv(j)=idum
      if(iy.lt.1)iy=iy+IMM1
      ran2=min(AM*iy,RNMX)
      return
      END
c------------------------------------------------------------------
      subroutine maketable(gx,gtx,pasgrid,sigr,nbdt,gausstab)
      include 'ceq3d.f'
      real*8 gx(0:NHFX)
      real*8 gtx(0:NTXT)
      real*8 pasgrid,sigr
      integer nbdt,i
      real*8 gausstab(1:8,0:NBTDMAX),k
      real*8 rmin,rmax,kus3,r
      pasgrid=gx(1)-gx(0)
      sigr=pasgrid/3.d0  
      rmin=(gx(2)+gx(1))*0.5d0
      rmax=(gx(3)+gx(2))*0.5d0
      k=(dsqrt(2.d0*PI)*sigr)**3.d0
      kus3=1.d0/(k**(1.d0/3.d0))
      do i=0,nbdt
         r=rmin+dfloat(i)*(rmax-rmin)/dfloat(nbdt)
         gausstab(1,i)=kus3*dexp(-((r-gtx(1))**2.d0)
     +        /(2.d0*(sigr**2.d0)))
         gausstab(2,i)=kus3*dexp(-((r-gtx(2))**2.d0)
     +        /(2.d0*(sigr**2.d0)))
         gausstab(3,i)=kus3*dexp(-((r-gtx(3))**2.d0)
     +        /(2.d0*(sigr**2.d0)))
         gausstab(4,i)=kus3*dexp(-((r-gtx(4))**2.d0)
     +        /(2.d0*(sigr**2.d0)))
         gausstab(5,i)=kus3*dexp(-((r-gtx(5))**2.d0)
     +        /(2.d0*(sigr**2.d0)))
         gausstab(6,i)=kus3*dexp(-((r-gtx(6))**2.d0)
     +        /(2.d0*(sigr**2.d0)))
         gausstab(7,i)=kus3*dexp(-((r-gtx(7))**2.d0)
     +        /(2.d0*(sigr**2.d0)))
         gausstab(8,i)=kus3*dexp(-((r-gtx(8))**2.d0)
     +        /(2.d0*(sigr**2.d0)))
      end do
      end
c---------------------------------------------------------------
      subroutine maketaint(nx,gx,sigr,nbdt,inttab1,inttab2)
      include 'ceq3d.f'
      real*8 gx(0:NHFX)
      real*8 sigr,k
      integer nbdt,i,nx
      real*8 inttab1(0:9,0:NBTDMAX)
      real*8 inttab2(0:9,0:NBTDMAX)
      real*8 rmin,rmax,kus3,r
      rmin=(gx(2)+gx(1))*0.5d0
      rmax=(gx(3)+gx(2))*0.5d0
      k=(dsqrt(2.d0*PI)*sigr)**3.d0
      kus3=1.d0/(k**(1.d0/3.d0))
      do i=0,nbdt
         r=rmin+dfloat(i)*(rmax-rmin)/dfloat(nbdt)
         call intvg1(gx,nx,gx(0),gx(1),0,sigr,r,kus3,inttab1(0,i))
         call intvg1(gx,nx,gx(0),gx(1),1,sigr,r,kus3,inttab1(1,i))
         call intvg2(gx,nx,gx(0),gx(1),0,sigr,r,kus3,inttab2(0,i))
         call intvg2(gx,nx,gx(0),gx(1),1,sigr,r,kus3,inttab2(1,i))
         call intvg1(gx,nx,gx(0),gx(2),2,sigr,r,kus3,inttab1(2,i))
         call intvg1(gx,nx,gx(0),gx(2),3,sigr,r,kus3,inttab1(3,i))
         call intvg2(gx,nx,gx(0),gx(2),2,sigr,r,kus3,inttab2(2,i))
         call intvg2(gx,nx,gx(0),gx(2),3,sigr,r,kus3,inttab2(3,i))
         call intvg1(gx,nx,gx(1),gx(3),4,sigr,r,kus3,inttab1(4,i))
         call intvg1(gx,nx,gx(1),gx(3),5,sigr,r,kus3,inttab1(5,i))
         call intvg2(gx,nx,gx(1),gx(3),4,sigr,r,kus3,inttab2(4,i))
         call intvg2(gx,nx,gx(1),gx(3),5,sigr,r,kus3,inttab2(5,i))
         call intvg1(gx,nx,gx(2),gx(4),6,sigr,r,kus3,inttab1(6,i))
         call intvg1(gx,nx,gx(2),gx(4),7,sigr,r,kus3,inttab1(7,i))
         call intvg2(gx,nx,gx(2),gx(4),6,sigr,r,kus3,inttab2(6,i))
         call intvg2(gx,nx,gx(2),gx(4),7,sigr,r,kus3,inttab2(7,i))
         call intvg1(gx,nx,gx(3),gx(4),8,sigr,r,kus3,inttab1(8,i))
         call intvg1(gx,nx,gx(3),gx(4),9,sigr,r,kus3,inttab1(9,i))
         call intvg2(gx,nx,gx(3),gx(4),8,sigr,r,kus3,inttab2(8,i))
         call intvg2(gx,nx,gx(3),gx(4),9,sigr,r,kus3,inttab2(9,i))
      end do
      end
c-----------------------------------------------------------
      subroutine intvg1(gx,nx,xdeb,xfin,indice,sigr,r,kus3,inte)
      include 'ceq3d.f'
      real*8 gx(0:NHFX)
      real*8 kus3,xdeb,xfin,sigr,r,inte,spl
      real*8 funci,funcip1,xi,xip1,pas
      integer i,indice,ind,isig,nbinterv,nx
      ind=indice/2
      isig=mod(indice,2)
      nbinterv=1000
      inte=0.d0
      pas=(xfin-xdeb)/dfloat(nbinterv)
      xi=xdeb
      call scub1(gx,nx,ind,isig,xi,spl)         
      funci=kus3*spl*dexp(-((r-xdeb)**(2.d0))/(2.d0*(sigr**2.d0)))
      do i=1,nbinterv
         xip1=xi+pas
         call scub1(gx,nx,ind,isig,xip1,spl)         
         funcip1=kus3*spl*dexp(-((r-xip1)**(2.d0))/(2.d0*(sigr**2.d0)))
         inte=inte+0.5d0*pas*(funci+funcip1)
         xi=xip1
         funci=funcip1
      end do
      end
c-----------------------------------------------------------
      subroutine intvg2(gx,nx,xdeb,xfin,indice,sigr,r,kus3,inte)
      include 'ceq3d.f'
      real*8 gx(0:NHFX)
      real*8 kus3,xdeb,xfin,sigr,r,inte,spl
      real*8 funci,funcip1,xi,xip1,pas
      integer i,indice,ind,isig,nbinterv,nx
      ind=indice/2
      isig=mod(indice,2)
      nbinterv=1000
      inte=0.d0
      pas=(xfin-xdeb)/dfloat(nbinterv)
      xi=xdeb
      call scub1(gx,nx,ind,isig,xi,spl)         
      funci=-2.d0*kus3*(r-xdeb)*spl*
     +     dexp(-((r-xdeb)**(2.d0))/(2.d0*(sigr**2.d0)))
     +        /(2.d0*(sigr**2.d0))
      do i=1,nbinterv
         xip1=xi+pas
         call scub1(gx,nx,ind,isig,xip1,spl)         
         funcip1=-2.d0*kus3*(r-xip1)*spl*
     +        dexp(-((r-xip1)**(2.d0))/(2.d0*(sigr**2.d0)))
     +        /(2.d0*(sigr**2.d0))
         inte=inte+0.5d0*pas*(funci+funcip1)
         xi=xip1
         funci=funcip1
      end do
      end
c---------------------- subroutine makerhog ----------------------
      subroutine makerhog(rho,npart,qp,nx,ny,nz,gx,gy,gz,
     +     nbelec,tabout,nbout,nbdt,pasgrid,gausstab,nbcap,
     +     cofrho,sxm1,sym1,szm1,ntx,nty,ntz,psx,psy,psz)
      include 'ceq3d.f'
      real*8 psx(0:NTXT),psy(0:NTYT),psz(0:NTZT)
      real*8 gausstab(1:8,0:NBTDMAX)
      real*8 gx(0:NHFX)
      real*8 gy(0:NHFY)
      real*8 gz(0:NHFZ)
      real*8 nbelec,pasgrid
      real*8 sxm1(0:NTXT,0:NTXT),sym1(0:NTYT,0:NTYT),szm1(0:NTZT,0:NTZT)
      real*8 rho(0:NTXT,0:NTYT,0:NTZT)
      real*8 cofrho(0:NTXT,0:NTYT,0:NTZT)
      real*8 charge,qtot,coef
      real*8 qp(3,npartmax),lxs2,uslx,rnbdt
      integer tabout(npartmax),nbdt
      integer i,j,k,l,nbout,ntx,nty,ntz
      logical inx,iny,inz
      integer nx,ny,nz
      integer npart
      integer nbcap
      nbout=0
      lxs2=pasgrid*0.5d0
      uslx=1.d0/pasgrid
      rnbdt=dfloat(nbdt)
      charge=nbelec/dfloat(npart)
      do i=0,ntx
         do j=0,nty
            do k=0,ntz
               rho(i,j,k)=0.d0
            end do
         end do
      end do
      do l=1,npart-nbcap
         call findig(gx,nx,qp(1,l),i,inx)
         call findig(gy,ny,qp(2,l),j,iny)
         call findig(gz,nz,qp(3,l),k,inz)
         if (inx.and.iny.and.inz) then
            call rajoute(i,j,k,gx,gy,gz,lxs2,uslx,rnbdt,gausstab,
     +           qp(1,l),qp(2,l),qp(3,l),rho)
         else 
            nbout=nbout+1
            tabout(nbout)=l
         end if
      end do
      do i=0,ntx
         do j=0,nty
            do k=0,ntz
               rho(i,j,k)=rho(i,j,k)*charge
            end do
         end do
      end do
      qtot=0.d0
      call tensrus2(ntx,nty,ntz,sxm1,sym1,szm1,rho,cofrho)
      do k=0,ntz
         do j=0,nty
            do i=0,ntx
               qtot=qtot+cofrho(i,j,k)*psx(i)*psy(j)*psz(k)
            end do
         end do
      end do
      coef=dfloat(npart-nbout)*charge/qtot
      do k=0,ntz
         do j=0,nty
            do i=0,ntx
               rho(i,j,k)=rho(i,j,k)*coef
            end do
         end do
      end do

      print*,'nb de pseudo part hors grille :',nbout
      end  
c-----------------------------------------------------
      subroutine potensg(x,nx,gx,y,ny,gy,z,nz,gz,
     +     csol,pote,inttab1,nbdt,pasgrid)
      include 'ceq3d.f'
      real*8 x,y,z
      real*8 gx(0:NHFX)
      real*8 gy(0:NHFY)
      real*8 gz(0:NHFZ)
      real*8 csol(0:NTXT,0:NTYT,0:NTZT)
      real*8 pote
      real*8 inttab1(0:9,0:NBTDMAX)
      real*8 intx1(0:9)
      real*8 inty1(0:9)
      real*8 intz1(0:9)
      real*8 lxs2,uslx,rnbdt,pasgrid,coef,potep
      integer jj,kk,gid,gjd,gkd
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
c
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
      pote=0.d0
      do kk=0,9
         k=gkd+kk
         do jj=0,9
            j=gjd+jj
            coef=inty1(jj)*intz1(kk)
            potep=0.d0
            potep=potep+csol(gid+0,j,k)*intx1(0)
            potep=potep+csol(gid+1,j,k)*intx1(1)
            potep=potep+csol(gid+2,j,k)*intx1(2)
            potep=potep+csol(gid+3,j,k)*intx1(3)
            potep=potep+csol(gid+4,j,k)*intx1(4)
            potep=potep+csol(gid+5,j,k)*intx1(5)
            potep=potep+csol(gid+6,j,k)*intx1(6)
            potep=potep+csol(gid+7,j,k)*intx1(7)
            potep=potep+csol(gid+8,j,k)*intx1(8)
            potep=potep+csol(gid+9,j,k)*intx1(9)
            pote=pote+potep*coef
         end do
      end do
      end
c------------------------------------------------------------
      function gaussir(r,sigr)
      include 'ceq3d.f'
      real*8 r,gaussir
      real*8 sigr,r2,k,kus3
      k=(dsqrt(2.d0*PI)*sigr)**3.d0
c      kus3=1.d0/(k**(1.d0/3.d0))
      kus3=1.d0/k
      r2=r**2.d0
      gaussir=dexp(-1.d0*r2/(2.d0*(sigr**2.d0)))*kus3
      end
c------------------------------------------------------------
c------------------------------------------------------------
      function erfsr(r,sigr)
      include 'ceq3d.f'
      real*8 erfsr,erf
      real*8 sigr,rr,x,r
      rr=r
      x=rr/(dsqrt(2.d0)*sigr)
      if (rr.lt.1.d-7) then
         erfsr=1.d0/(dsqrt(2*PI)*sigr)
      else
         erfsr=erf(x)/rr
      end if
      end
c------------------------------------------------------------
      function gammp(a,x)
      real*8 a,gammp,x
      real*8 gammcf,gamser,gln
      if(x.lt.0..or.a.le.0.)pause 'bad arguments in gammp'
      if(x.lt.a+1.d0)then
        call gser(gamser,a,x,gln)
        gammp=gamser
      else
        call gcf(gammcf,a,x,gln)
        gammp=1.d0-gammcf
      endif
      return
      end
c------------------------------------------------------------
      function erf(x)
      real*8 erf,x
      real*8 gammp
      if(x.lt.0.)then
        erf=-gammp(.5d0,x**2)
      else
        erf=gammp(.5d0,x**2)
      endif
      return
      end
c------------------------------------------------------------
      subroutine gser(gamser,a,x,gln)
      integer itmax
      real*8 a,gamser,gln,x,eps
      parameter (itmax=100,eps=3.d-7)
      integer n
      real*8 ap,del,sum,gammln
      gln=gammln(a)
      if(x.le.0.)then
        if(x.lt.0.)pause 'x < 0 in gser'
        gamser=0.d0
        return
      endif
      ap=a
      sum=1.d0/a
      del=sum
      do 11 n=1,itmax
        ap=ap+1.d0
        del=del*x/ap
        sum=sum+del
        if(abs(del).lt.abs(sum)*eps)goto 1
11    continue
      pause 'a too large, itmax too small in gser'
1     gamser=sum*exp(-x+a*log(x)-gln)
      return
      end
c------------------------------------------------------------
      subroutine gcf(gammcf,a,x,gln)
      integer itmax
      real*8 a,gammcf,gln,x,eps,fpmin
      parameter (itmax=100,eps=3.d-7,fpmin=1.d-30)
      integer i
      real*8 an,b,c,d,del,h,gammln
      gln=gammln(a)
      b=x+1.d0-a
      c=1.d0/fpmin
      d=1.d0/b
      h=d
      do 11 i=1,itmax
        an=-i*(i-a)
        b=b+2.d0
        d=an*d+b
        if(abs(d).lt.fpmin)d=fpmin
        c=b+an/c
        if(abs(c).lt.fpmin)c=fpmin
        d=1.d0/d
        del=d*c
        h=h*del
        if(abs(del-1.d0).lt.eps)goto 1
11    continue
      pause 'a too large, itmax too small in gcf'
1     gammcf=exp(-x+a*log(x)-gln)*h
      return
      end
c------------------------------------------------------------
      function gammln(xx)
      real*8 gammln,xx
      integer j
      real*8 ser,stp,tmp,x,y,cof(6)
      cof(1)=76.18009172947146d0
      cof(2)=-86.50532032941677d0
      cof(3)=24.01409824083091d0
      cof(4)=-1.231739572450155d0
      cof(5)=.1208650973866179d-2
      cof(6)=-.5395239384953d-5
      stp=2.5066282746310005d0
      x=xx
      y=x
      tmp=x+5.5d0
      tmp=(x+0.5d0)*log(tmp)-tmp
      ser=1.000000000190015d0
      do 11 j=1,6
        y=y+1.d0
        ser=ser+cof(j)/y
11    continue
      gammln=tmp+log(stp*ser/x)
      return
      end
c--------------------------------------------------------------
      subroutine sortierho(nx,ny,nz,ntx,nty,ntz,titre,
     +     sxm1,sym1,szm1,gx,gy,gz,rho,
     +     sxm1big,sym1big,szm1big,gxbig,gybig,gzbig,rhobig)
      include 'ceq3d.f'
      real*8 sxm1(0:NTXT,0:NTXT),sym1(0:NTYT,0:NTYT),szm1(0:NTZT,0:NTZT)
      real*8 gx(0:NHFX)
      real*8 gy(0:NHFY)
      real*8 gz(0:NHFZ)
      real*8 gxbig(0:NHFX)
      real*8 gybig(0:NHFY)
      real*8 gzbig(0:NHFZ)
      real*8 sxm1big(0:NTXT,0:NTXT)
      real*8 sym1big(0:NTYT,0:NTYT)
      real*8 szm1big(0:NTZT,0:NTZT)
      integer ntx,nty,ntz
      real*8 csol(0:NTXT,0:NTYT,0:NTZT)
      real*8 csolbig(0:NTXT,0:NTYT,0:NTZT)
      real*8 rho(0:NTXT,0:NTYT,0:NTZT)
      real*8 rhobig(0:NTXT,0:NTYT,0:NTZT)
      real*8 longbase
      integer nx,ny,nz,big
      real*8 x,h,pot,h2,y,dumr,xmin,xmax,ymin,ymax,eps
      integer i,j,l,rep,dumi,dumj
      character*4 titre
      character*25 titrec
      call tensrus2(ntx,nty,ntz,sxm1,sym1,szm1,rho,csol)
      call tensrus2(ntx,nty,ntz,sxm1big,sym1big,szm1big,rhobig,csolbig)
      dumi=0
      dumj=0
      dumr=0.d0
      titrec='out/                '//titre 
      open (2,file=titrec,status='unknown')
      print*,'trec=',titrec
      eps=1.d-5
      longbase=50.d0
      xmin=-1.d0*longbase+eps
      xmax=longbase-eps
      ymin=-1.d0*longbase+eps
      ymax=longbase-eps
      do j=0,120
         h=(xmax-xmin)/120.d0-1.d-10
         x=xmin+j*h
         do l=0,120
            h2=(ymax-ymin)/120.d0-1.d-10
            y=ymin+l*h2
            big=0
            if (x.gt.gx(nx)) big=big+1
            if (y.gt.gy(ny)) big=big+1
            if (x.lt.gx(0)) big=big+1
            if (y.lt.gy(0)) big=big+1
            if (big.eq.0) then
               call potentiel(x,nx,gx,y,ny,gy,0.d0,nz,gz,
     +              csol,pot,dumi,dumj,dumr)
            else
               call potentiel(x,nx,gxbig,y,ny,gybig,0.d0,nz,gzbig,
     +              csolbig,pot,dumi,dumj,dumr)
            end if
            if (dabs(pot).lt.1e-10) pot=0.d0
            write(2,'(3e14.6)') x,y,pot
         end do
      end do
      end
c----------------------------------------------------------------
      subroutine initpro(mpro,epro,impara,xinit
     +     ,pospro,vpro,posprold,dltt)
      include 'ceq3d.f'
      real*8 mpro,epro,impara,xinit
      real*8 pospro(3)
      real*8 vpro(3)
      real*8 posprold(3)
      real*8 dltt
      pospro(1)=xinit
      pospro(2)=impara
      pospro(3)=0.d0
      vpro(1)=dsqrt(2.d0*Epro/mpro)
      vpro(2)=0.d0
      vpro(3)=0.d0
      posprold(1)=pospro(1)-vpro(1)*dltt
      posprold(2)=pospro(2)-vpro(2)*dltt
      posprold(3)=pospro(3)-vpro(3)*dltt
      end
c---------------------------------------------------------------------
      subroutine makecha(chnomb,k)
      implicit none
      integer k
      character*3 chnomb
      character*2 chnomb2
      character*1 chnomb1
      write(chnomb,'(I3)') k
      if (k.ge.100) then
         write(chnomb,'(I3)') k
      else
         if (k.ge.10) then
            write(chnomb2,'(I2)') k               
            chnomb='0'//chnomb2
         else
            if (k.ge.1) then
               write(chnomb1,'(I1)') k               
               chnomb='00'//chnomb1
            else
               chnomb='000'
            end if
         end if
      end if
      end           
c-----------------------------------------------------------------
      subroutine incproj(mpro,pospro,posprold,vpro,dltt,qp,fp,cutoff,
     +     nbion,nbelec,npart,chapro,first,nb,
     +     rcutee,epro,impara,last,nbcap,einterne,titreq)
      include 'ceq3d.f'
      real*8 qp(3,npartmax)
      real*8 fp(3,npartmax)
      real*8 mpro,pospro(3),vpro(3),dltt,cutoff
      real*8 r2,x,y,z,cut2,rayon2,coef,coef2,rhoin
      real*8 chapro,fx,fy,fz,fxt,fyt,fzt
      real*8 modf,nbion,nbelec,ekinp,chaprot,einterne
      real*8 epro,impara,Eelpro,Ejelpro,rayon,potjel
      real*8 posprold(3),xpro,ypro,zpro
      integer tabcb(nbcbmax),nbcap
      real*8 rcutee,qpart,vcent,coefcent
      integer i,nb,npart,j,k
      logical first
      logical last
      character*1 chch,chma
      character*3 chen,chim
      character*23 titrep
      character*23 titreq
      write(chma,'(I1)') int(mpro/1836.154d0+0.1d0)
      chaprot=chapro+nbelec*dfloat(nbcap)/dfloat(npart)
      write(chch,'(I1)') int(chaprot+0.1d0)
      call makecha(chen,int(epro*27.211d0/1000.d0+0.1d0))
      call makecha(chim,int(impara+0.1d0))
      titrep='Eloss/'//'Em'//chma//
     +     'q'//chch//'e'//chen//'i'//chim//'.dat'
      titreq='Eloss/'//'Fm'//chma//
     +     'q'//chch//'e'//chen//'i'//chim//'.dat'
      k=0
      x=pospro(1)
      y=pospro(2)
      z=pospro(3)
      rayon=dsqrt(x**2.d0+y**2.d0+z**2.d0)
      Ejelpro=chapro*potjel(rayon,nbion)
      cut2=cutoff**2
      rayon2=((nbion**(1.d0/3.d0))*rs)**2.d0
      r2=x**2.d0+y**2.d0+z**2.d0
      coef2=nbion*chapro
      rhoin=3.d0/(4.d0*pi*(rs**3.d0))
      qpart=-1.d0*(nbelec/dfloat(npart))
      vcent=1.5d0*chapro/cutoff
      coefcent=-0.5d0*chapro/(cutoff**3)
      print*,'chapro,cutoff,cut2,vcent,coefcent'
      print*,chapro,cutoff,cut2,vcent,coefcent
c      --- projectile <-> jellium --
      if (r2.gt.rayon2) then
         modf=coef2/(r2**1.5d0)
         fxt=modf*x      
         fyt=modf*y
         fzt=modf*z
      else
         modf=4.d0*pi*chapro*rhoin/3.d0
         fxt=modf*x
         fyt=modf*y
         fzt=modf*z
      end if
c      -- projectile <-> pseudo elecs noncb --
      coef=-1.d0*(nbelec/dfloat(npart))*chapro
      Eelpro=0.d0
      do i=1,npart-nbcap
         x=pospro(1)-qp(1,i)
         y=pospro(2)-qp(2,i)
         z=pospro(3)-qp(3,i)
         r2=x**2.d0+y**2.d0+z**2.d0
         if (r2.gt.cut2) then
            modf=coef/(r2**1.5d0)
            fx=modf*x 
            fy=modf*y 
            fz=modf*z 
            fxt=fxt+fx
            fyt=fyt+fy
            fzt=fzt+fz
            fp(1,i)=fp(1,i)-fx
            fp(2,i)=fp(2,i)-fy
            fp(3,i)=fp(3,i)-fz
            Eelpro=Eelpro+coef/dsqrt(r2)
         else
c            Eelpro=Eelpro+coef/dsqrt(cut2)
            modf=coef/(cut2**1.5d0)
            fx=modf*x 
            fy=modf*y 
            fz=modf*z 
            fxt=fxt+fx
            fyt=fyt+fy
            fzt=fzt+fz
            fp(1,i)=fp(1,i)-fx
            fp(2,i)=fp(2,i)-fy
            fp(3,i)=fp(3,i)-fz
            Eelpro=Eelpro+qpart*(vcent+r2*coefcent)
            k=k+1
            if (k.lt.nbcbmax) then
               tabcb(k)=i
            else
               print*,'trop de pp cb'
               stop
            end if
         end if
      end do   
      print*,'closelybound:',k
c      if (k.ge.2) then
c         print*,'av rcutee,ncb,nbion,npart,tabcb(1)'
c         print*,rcutee,k,nbion,npart,tabcb(1)
c         call closebo(tabcb,qp,fp,npart,nbelec,k,rcutee)
c      end if
c      print*,'closebo ok'
      xpro=2.d0*pospro(1)-posprold(1)+(dltt**2)*fxt/mpro
      ypro=2.d0*pospro(2)-posprold(2)+(dltt**2)*fyt/mpro
      zpro=2.d0*pospro(3)-posprold(3)+(dltt**2)*fzt/mpro
      vpro(1)=(xpro-posprold(1))/(2.d0*dltt)
      vpro(2)=(ypro-posprold(2))/(2.d0*dltt)
      vpro(3)=(zpro-posprold(3))/(2.d0*dltt)
      posprold(1)=pospro(1)
      posprold(2)=pospro(2)
      posprold(3)=pospro(3)
      pospro(1)=xpro
      pospro(2)=ypro
      pospro(3)=zpro
      ekinp=0.5d0*mpro*(vpro(1)**2+vpro(2)**2+vpro(3)**2)
      if (first) then
         open (1,file=titrep,status='unknown')
         write(1,*) 'Ekin,charge,impara du projectile'
         write(1,*) epro,mpro,impara
         write(1,*) 'x(t),y(t),z(t),Ekin(t)'
         write(1,'(7e15.7)') pospro(1),pospro(2),pospro(3),epro-ekinp
     +        ,Eelpro,Ejelpro,einterne
         close(1)
      else
         if (last) then
            open (1,file=titrep,status='old')
            read(1,*)
            read(1,*)
            read(1,*)
            do j=1,nb-1
               read(1,'(7e15.7)')
            end do
            write(1,*) 'Energie perdue par le projectile > 0 (eV)'
            write(1,*) (epro-ekinp)*27.2116d0
            close(1)
         else
            open (1,file=titrep,status='old')
            read(1,*)
            read(1,*)
            read(1,*)
            do j=1,nb-1
               read(1,'(7e15.7)')
            end do
         write(1,'(7e15.7)') pospro(1),pospro(2),pospro(3),epro-ekinp,
     +           Eelpro,Ejelpro,einterne
            close(1)
         end if
         end if
      end
c-----------------------------------------------------------------
      subroutine incproj3(mpro,pospro,posprold,vpro,dltt,qp,fp,cutoff,
     +     nbion,nbelec,npart,chapro,first,nb,
     +     rcutee,epro,impara,last,nbcap,einterne,titreq)
      include 'ceq3d.f'
      real*8 posprold(3),xpro,ypro,zpro
      real*8 qp(3,npartmax)
      real*8 fp(3,npartmax)
      real*8 mpro,pospro(3),vpro(3),dltt,cutoff
      real*8 r2,x,y,z,cut2,rayon2,coef,coef2,rhoin
      real*8 chapro,fx,fy,fz,fxt,fyt,fzt
      real*8 modf,nbion,nbelec,ekinp,chaprot,einterne
      real*8 epro,impara,Eelpro,Ejelpro,rayon,potjel
      integer tabcb(nbcbmax),nbcap
      real*8 rcutee,qpart,vcent,coefcent
      integer i,nb,npart,j,k
      logical first
      logical last
      character*1 chch,chma
      character*3 chen,chim
      character*23 titrep
      character*23 titreq
      write(chma,'(I1)') int(mpro/1836.154d0+0.1d0)
      chaprot=chapro+nbelec*dfloat(nbcap)/dfloat(npart)
      write(chch,'(I1)') int(chaprot+0.1d0)
      call makecha(chen,int(epro*27.211d0/1000.d0+0.1d0))
      call makecha(chim,int(impara+0.1d0))
      titrep='Eloss/'//'Em'//chma//
     +     'q'//chch//'e'//chen//'i'//chim//'.dat'
      titreq='Eloss/'//'Fm'//chma//
     +     'q'//chch//'e'//chen//'i'//chim//'.dat'
      k=0
      x=pospro(1)
      y=pospro(2)
      z=pospro(3)
      rayon=dsqrt(x**2.d0+y**2.d0+z**2.d0)
      Ejelpro=chapro*potjel(rayon,nbion)
      cut2=cutoff**2
      rayon2=((nbion**(1.d0/3.d0))*rs)**2.d0
      r2=x**2.d0+y**2.d0+z**2.d0
      coef2=nbion*chapro
      rhoin=3.d0/(4.d0*pi*(rs**3.d0))
      qpart=-1.d0*(nbelec/dfloat(npart))
      vcent=2.d0*chapro/cutoff
      coefcent=-1.d0*chapro/(cutoff**3)
c      --- projectile <-> jellium --
      if (r2.gt.rayon2) then
         modf=coef2/(r2**1.5d0)
         fxt=modf*x      
         fyt=modf*y
         fzt=modf*z
      else
         modf=4.d0*pi*chapro*rhoin/3.d0
         fxt=modf*x
         fyt=modf*y
         fzt=modf*z
      end if
c      -- projectile <-> pseudo elecs noncb --
      coef=-1.d0*(nbelec/dfloat(npart))*chapro
      Eelpro=0.d0
      do i=1,npart-nbcap
         x=pospro(1)-qp(1,i)
         y=pospro(2)-qp(2,i)
         z=pospro(3)-qp(3,i)
         r2=x**2.d0+y**2.d0+z**2.d0
         if (r2.gt.cut2) then
            modf=coef/(r2**1.5d0)
            fx=modf*x 
            fy=modf*y 
            fz=modf*z 
            fxt=fxt+fx
            fyt=fyt+fy
            fzt=fzt+fz
            fp(1,i)=fp(1,i)-fx
            fp(2,i)=fp(2,i)-fy
            fp(3,i)=fp(3,i)-fz
            Eelpro=Eelpro+coef/dsqrt(r2)
         else
c            Eelpro=Eelpro+coef/dsqrt(cut2)
            modf=coef/(cut2**1.5d0)
            fx=modf*x 
            fy=modf*y 
            fz=modf*z 
            fxt=fxt+fx
            fyt=fyt+fy
            fzt=fzt+fz
            fp(1,i)=fp(1,i)-fx
            fp(2,i)=fp(2,i)-fy
            fp(3,i)=fp(3,i)-fz
            Eelpro=Eelpro+qpart*(vcent+r2*coefcent)
c            Eelpro=Eelpro+coef/dsqrt(cut2)
            k=k+1
            if (k.lt.nbcbmax) then
               tabcb(k)=i
            else
               print*,'trop de pp cb'
            end if
         end if
      end do   
      print*,'closelybound:',k
c      if (k.ge.2) then
c         print*,'av rcutee,ncb,nbion,npart,tabcb(1)'
c         print*,rcutee,k,nbion,npart,tabcb(1)
c         call closebo(tabcb,qp,fp,npart,nbelec,k,rcutee)
c      end if
c      print*,'closebo ok'
      xpro=2.d0*pospro(1)-posprold(1)+(dltt*2)*fxt/mpro
      ypro=2.d0*pospro(2)-posprold(2)+(dltt*2)*fyt/mpro
      zpro=2.d0*pospro(3)-posprold(3)+(dltt*2)*fzt/mpro
      vpro(1)=(xpro-posprold(1))/(2.d0*dltt)
      vpro(2)=(ypro-posprold(2))/(2.d0*dltt)
      vpro(3)=(zpro-posprold(3))/(2.d0*dltt)
      posprold(1)=pospro(1)
      posprold(2)=pospro(2)
      posprold(3)=pospro(3)
      pospro(1)=xpro
      pospro(2)=ypro
      pospro(3)=zpro
      ekinp=0.5d0*mpro*(vpro(1)**2+vpro(2)**2+vpro(3)**2)
      if (first) then
         open (1,file=titrep,status='unknown')
         write(1,*) 'Ekin,charge,impara du projectile'
         write(1,*) epro,mpro,impara
         write(1,*) 'x(t),y(t),z(t),Ekin(t)'
         write(1,'(7e15.7)') pospro(1),pospro(2),pospro(3),epro-ekinp
     +        ,Eelpro,Ejelpro,einterne
         close(1)
      else
         if (last) then
            open (1,file=titrep,status='old')
            read(1,*)
            read(1,*)
            read(1,*)
            do j=1,nb-1
               read(1,'(7e15.7)')
            end do
            write(1,*) 'Energie perdue par le projectile > 0 (eV)'
            write(1,*) (epro-ekinp)*27.2116d0
            close(1)
         else
            open (1,file=titrep,status='old')
            read(1,*)
            read(1,*)
            read(1,*)
            do j=1,nb-1
               read(1,'(7e15.7)')
            end do
         write(1,'(7e15.7)') pospro(1),pospro(2),pospro(3),epro-ekinp,
     +           Eelpro,Ejelpro,einterne
            close(1)
         end if
         end if
      end
c--------------------------------------------------------------
      subroutine closebo(tabcb,qp,fp,npart,nbelec,ncb,rcutee)
      include 'ceq3d.f'
      real*8 qp(3,npartmax)
      real*8 fp(3,npartmax)
      real*8 x,y,z
      real*8 fx,fy,fz
      real*8 r2,r2min,valf
      real*8 rcutee,coef,nbelec
      integer tabcb(nbcbmax)
      integer i,j,n1,n2,npart,ncb
c      print*,'rcutee,ncb,nbion,npart,tabcb(1)'
c      print*,rcutee,ncb,nbion,npart,tabcb(1)
      r2min=rcutee**2.d0
      coef=-1.d0*(nbelec/dfloat(npart))**2.d0
      do i=1,ncb-1
         n1=tabcb(i)
         do j=i+1,ncb
c            print*,'i,j',i,j
            n2=tabcb(j)
            x=qp(1,n2)-qp(1,n1)
            y=qp(2,n2)-qp(2,n1)
            z=qp(3,n2)-qp(3,n1)
            r2=x**2.d0+y**2.d0+z**2.d0
            if (r2.lt.r2min) r2=r2min
            valf=coef/(r2**1.5d0)
            fx=valf*x
            fy=valf*y
            fz=valf*z
            fp(1,n1)=fp(1,n1)+fx
            fp(2,n1)=fp(2,n1)+fy
            fp(3,n1)=fp(3,n1)+fz
            fp(1,n2)=fp(1,n2)-fx
            fp(2,n2)=fp(2,n2)-fy
            fp(3,n2)=fp(3,n2)-fz
         end do
      end do
      end

c----------------------------------------------------------------------
      subroutine initializ2(rt,pt,rhorp,rmax,pmax,npart,nr,np,nbion)
      include 'ceq3d.f'
      real*8 rt(3,npartmax)
      real*8 pt(3,npartmax)
      real*8 rmax,pmax,rhorp(nrmax,npmax)
      real*4 ran2
      real*8 x(6),nbion
      integer i,nr,np,npart
      real*8 r,p
      real*8 pht
      integer tries,lr,lp,nbi,idum
      real*8 stheta,ee
      real*8 ptheta,pphi,palpha
      tries = 0
      ee = nbion/dfloat(npart)
      idum=-1
      do i=1,npart 
 30      do nbi=1,6
            x(nbi)=ran2(idum)
         end do
         r = rmax*x(1)**(1.d0/3.d0)
         stheta = sqrt(1.d0-(2.d0*x(3)-1.d0)**2)
         rt(1,i) = r*cos(2.d0*PI*x(2))*stheta
         rt(2,i) = r*sin(2.d0*PI*x(2))*stheta
         rt(3,i) = r*(2.d0*x(3)-1.d0)
         p = pmax*x(4)**(1.d0/3.d0)
         palpha=x(5)*2.d0*PI
         if (r.ne.0.d0) then
            ptheta=acos(rt(3,i)/r)
         else
            ptheta=0.d0
         end if
         if (rt(1,i).ne.0.d0) then
            pphi=atan(rt(2,i)/rt(1,i))
         else
            pphi=0.5d0*PI
         end if
         pt(1,i)=p*ee*(cos(pphi)*cos(palpha)*cos(ptheta)
     +                -sin(pphi)*sin(palpha))
         pt(2,i)=p*ee*(sin(pphi)*cos(ptheta)*cos(palpha)
     +                +cos(pphi)*sin(palpha))
         pt(3,i)=p*ee*(-1.d0*sin(ptheta)*cos(palpha))
         lr = int(r*dfloat(nr-1)/rmax)
         lp = int(p*dfloat(np-1)/pmax)
         r  = r*dfloat(nr-1)/rmax - lr
         p  = p*dfloat(np-1)/pmax - lp
C     density function at that point
         pht = rhorp(lr+2,lp+2)  *r     *p
     &        + rhorp(lr+1,lp+2)  *(1.d0-r)*p
     &        + rhorp(lr+2,lp+1)  *r     *(1.d0-p)
     &        + rhorp(lr+1,lp+1)  *(1.d0-r)*(1.d0-p)
C     reject with probability 1-pht
         tries = tries + 1
         if (pht.lt.0.5d0) then
            goto 30
         end if 
      end do
      write (*,'(''Number of tries: '',i7," ,number of accepted:",i6)')
     &     tries,i-1
      write (*,'(''Initialization of test particles completed'')')
      end
c-------------------------------------------------------------------
      subroutine moveback1(qp,qpold,nbelec,npart,dltt)
      include 'ceq3d.f'
      real*8 qp(3,npartmax)
      real*8 qpold(3,npartmax)
      real*8 nbelec
      real*8 dltt,coef
      integer npart
      integer i
      coef=-0.5d0*dltt*dfloat(npart)/(mel*nbelec)
      do i=1,npart
         qpold(1,i)=qp(1,i)+coef*qpold(1,i)
         qpold(2,i)=qp(2,i)+coef*qpold(2,i)
         qpold(3,i)=qp(3,i)+coef*qpold(3,i)
      end do
      end
c-------------------------------------------------------------------
      subroutine moveback2(qp,qpold,fp,nbelec,npart,dltt)
      include 'ceq3d.f'
      real*8 qp(3,npartmax)
      real*8 fp(3,npartmax)
      real*8 qpold(3,npartmax)
      real*8 nbelec
      real*8 dltt,coef2
      integer npart
      integer i
      coef2=0.5d0*dltt*2*dfloat(npart)/(mel*nbelec)
      do i=1,npart
         qpold(1,i)=-qp(1,i)+2.d0*qpold(1,i)+coef2*fp(1,i)
         qpold(2,i)=-qp(2,i)+2.d0*qpold(2,i)+coef2*fp(2,i)              
         qpold(3,i)=-qp(3,i)+2.d0*qpold(3,i)+coef2*fp(3,i)
      end do
      end
c---------------------------------------------------------------------
      subroutine biggrille(nx,gx,gtx,ny,gy,gty,nz,gz,gtz,
     +               xclu,xboite,n1big,n2big,
     +               ntx,nty,ntz,nsx,nsy,nsz)
      include 'ceq3d.f'
      real*8 gx(0:NHFX)
      real*8 gy(0:NHFY)
      real*8 gz(0:NHFZ)
      real*8 colx(DE*NHFX)
      real*8 coly(DE*NHFY)
      real*8 colz(DE*NHFZ)
      real*8 gtx(0:NTXT)
      real*8 gty(0:NTYT)
      real*8 gtz(0:NTZT)
      real*8 xclu,xboite
      integer nx,ny,nz,ntx,nty,ntz,nsx,nsy,nsz
      integer n1big,n2big
      integer n1xyz,n2xyz
      integer i
      print*,'debut construction grille coarse'
      n1xyz=n1big/2
      n2xyz=(n2big+2)/2
      call mkgri2(xclu,xboite,n1xyz,n2xyz,gx)
      call mkgri2(xclu,xboite,n1xyz,n2xyz,gy)
      call mkgri2(xclu,xboite,n1xyz,n2xyz,gz)
      nx=n1big+n2big
      ny=n1big+n2big
      nz=n1big+n2big
      ntx=2*nx+1
      nty=2*ny+1
      ntz=2*nz+1
      nsx=ntx-1
      nsy=nty-1
      nsz=ntz-1
      call colloc(gx,colx,nx)
      call colloc(gy,coly,ny)
      call colloc(gz,colz,nz)
      call makegt(nx,ny,nz,gx,gy,gz,colx,coly,colz,
     +     ntx,nty,ntz,gtx,gty,gtz)
      print*,'fin construction grille coarse'
      do i=0,ntx
         print*,'grille coarse',i,gtx(i)
      end do
      end
c------subroutinemakerho(nbion,ntx,nty,ntz,gtx,gty,gtz,rhojel,volm1)--
      subroutine makerho(rho,npart,qp,ntx,nty,ntz,gtx,gty,gtz,
     +     nbelec,volm1,tabout,nbout,nbcap)
      include 'ceq3d.f'
      real*8 gtx(0:NTXT)
      real*8 gty(0:NTYT)
      real*8 gtz(0:NTZT)
      real*8 nbelec
      real*8 volm1(0:NTXT,0:NTYT,0:NTZT)
      real*8 rho(0:NTXT,0:NTYT,0:NTZT)
      real*8 charge,ax,ay,az
      real*8 somme,qp(3,npartmax)
      integer tabout(npartmax)
      integer i,j,k,l,nbout
      logical inx,iny,inz
      integer ntx,nty,ntz
      integer npart,nbcap
      nbout=0
      charge=nbelec/dfloat(npart)
      do i=0,ntx
         do j=0,nty
            do k=0,ntz
               rho(i,j,k)=0.d0
            end do
         end do
      end do
      do l=1,npart-nbcap
         call findi(gtx,ntx,qp(1,l),i,ax,inx)
         call findi(gty,nty,qp(2,l),j,ay,iny)
         call findi(gtz,ntz,qp(3,l),k,az,inz)
         if (inx.and.iny.and.inz) then
            rho(i,j,k)=rho(i,j,k)+(ax*ay*az)
            rho(i,j,k+1)=rho(i,j,k+1)+(ax*ay*(1-az))
            rho(i,j+1,k)=rho(i,j+1,k)+(ax*(1-ay)*az)
            rho(i,j+1,k+1)=rho(i,j+1,k+1)+(ax*(1-ay)*(1-az))
            rho(i+1,j,k)=rho(i+1,j,k)+((1-ax)*ay*az)
            rho(i+1,j,k+1)=rho(i+1,j,k+1)+((1-ax)*ay*(1-az))
            rho(i+1,j+1,k)=rho(i+1,j+1,k)+((1-ax)*(1-ay)*az)
            rho(i+1,j+1,k+1)=rho(i+1,j+1,k+1)
     +           +((1-ax)*(1-ay)*(1-az))
            somme=(ax*ay*az)+(ax*ay*(1-az))+(ax*(1-ay)*az)+
     +           ((1-ax)*ay*az)+((1-ax)*ay*(1-az))+((1-ax)*(1-ay)*az)+
     +           ((1-ax)*(1-ay)*(1-az))+(ax*(1-ay)*(1-az))
            if (abs(somme-1.d0).gt.1e-4) then
               print*,'somme=',somme,i,j,k
            end if
         else 
            nbout=nbout+1
            tabout(nbout)=l
         end if
      end do
      do i=0,NTXT
         do j=0,NTYT
            do k=0,NTZT
               rho(i,j,k)=rho(i,j,k)*charge*volm1(i,j,k)
            end do
         end do
      end do
      print*,'nb de pseudo part hors grille :',nbout
      end
C-----------------------------------------------------
      subroutine sortietest(gx,gy,gz,gx2,gy2,gz2,nx,ny,nz,csol,csol2)
      include 'ceq3d.f'
      real*8 gx(0:NHFX)
      real*8 gy(0:NHFY)
      real*8 gz(0:NHFZ)
      real*8 csol(0:NTXT,0:NTYT,0:NTZT)
      real*8 gx2(0:NHFX)
      real*8 gy2(0:NHFY)
      real*8 gz2(0:NHFY)
      real*8 csol2(0:NTXT,0:NTYT,0:NTZT)
      integer nx,ny,nz
      real*8 x,h,pot,h2,z,dumr,pot2
      integer i,j,l,dumi,dumj
      dumi=0
      dumj=0
      dumr=0.d0
      do i=0,nx
      end do
      open (2,file='test.dat',status='unknown')
c      print*,'plot x=0 => entrez 1'
c      print*,'plot y=0 => entrez 2'
c      print*,'plot z=0 => entrez 3'
c      read*, rep
      do j=0,100
         h=(gx(nx)-gx(0))/100.d0-1.d-10
         x=gx(0)+j*h
         do l=0,100
            h2=(gz(nz)-gz(0))/100.d0-1.d-10
            z=gz(0)+l*h2
            call potentiel(x,nx,gx,z,ny,gy,0.d0,nz,gz,csol,pot,
     +           dumi,dumj,dumr)
            call potentiel(x,nx,gx2,z,ny,gy2,0.d0,nz,gz,csol2,pot2,
     +           dumi,dumj,dumr)
            write(2,'(5e14.6)') x,z,pot,pot2,pot-pot2
         end do
      end do
      end
c--------------------------------------------------------------------
      subroutine makerhsf(rho,dx,dy,dz,nsx,nsy,nsz,
     +     gtx,gty,gtz,phi,rhsl,sxm1,sym1,szm1,
     +     psx,psy,psz,psxx,psyy,pszz,psx2,psy2,psz2,
     +     gx,gy,gz,nx,ny,nz,csol)
      include 'ceq3d.f'
      real*8 gx(0:NHFX)
      real*8 gy(0:NHFY)
      real*8 gz(0:NHFZ)
      integer nx,ny,nz
      real*8 gtx(0:NTXT),gty(0:NTYT),gtz(0:NTZT)
      real*8 rho(0:NTXT,0:NTYT,0:NTZT)
      real*8 csol(0:NTXT,0:NTYT,0:NTZT)
      real*8 cofrho(0:NTXT,0:NTYT,0:NTZT)
      real*8 rhsl(NXS,NYS,NZS)
      real*8 comp(NXS,NYS,NZS)
      real*8 dx(0:NTXT,0:NTXT),dy(0:NTYT,0:NTYT),dz(0:NTZT,0:NTZT)
      real*8 phi(0:NTXT,0:NTYT,0:NTZT)
      real*8 coef,somme,qtot,bari(3),quad(3,3)
      real*8 psx(0:NTXT),psy(0:NTYT),psz(0:NTZT)
      real*8 psxx(0:NTXT),psyy(0:NTYT),pszz(0:NTZT)
      real*8 psx2(0:NTXT),psy2(0:NTYT),psz2(0:NTZT)
      real*8 sxm1(0:NTXT,0:NTXT),sym1(0:NTYT,0:NTYT),szm1(0:NTZT,0:NTZT)
      integer nsx,nsy,nsz
      integer ntx,nty,ntz
      integer a,b,c,i,j,k
      integer bord
      real*8 dumr,pot,x,y,z
      integer dumi,dumj
      dumi=0
      dumj=0
      dumr=0.d0
      ntx=nsx+1
      nty=nsy+1
      ntz=nsz+1
      call tensrus2(ntx,nty,ntz,sxm1,sym1,szm1,rho,cofrho)
      do i=1,3
         bari(i)=0.d0
         do j=1,3
            quad(i,j)=0.d0
         end do
      end do      
      qtot=0.d0
      do i=0,ntx
         do j=0,nty
            do k=0,ntz
               qtot=qtot+cofrho(i,j,k)*psx(i)*psy(j)*psz(k)
               bari(1)=bari(1)+cofrho(i,j,k)*psxx(i)*psy(j)*psz(k)
               bari(2)=bari(2)+cofrho(i,j,k)*psx(i)*psyy(j)*psz(k)
               bari(3)=bari(3)+cofrho(i,j,k)*psx(i)*psy(j)*pszz(k)
               quad(1,1)=quad(1,1)+cofrho(i,j,k)*(
     +              2.d0*psx2(i)*psy(j)*psz(k)
     +                  -psx(i)*psy2(j)*psz(k)
     +                  -psx(i)*psy(j)*psz2(k))
               quad(2,2)=quad(2,2)+cofrho(i,j,k)*(
     +                  -psx2(i)*psy(j)*psz(k)
     +             +2.d0*psx(i)*psy2(j)*psz(k)
     +                  -psx(i)*psy(j)*psz2(k))
               quad(3,3)=quad(3,3)+cofrho(i,j,k)*(
     +                  -psx2(i)*psy(j)*psz(k)
     +                  -psx(i)*psy2(j)*psz(k)
     +             +2.d0*psx(i)*psy(j)*psz2(k))
               quad(1,2)=quad(1,2)+cofrho(i,j,k)*(
     +              3.d0*psxx(i)*psyy(j)*psz(k))
               quad(1,3)=quad(1,3)+cofrho(i,j,k)*(
     +              3.d0*psxx(i)*psy(j)*pszz(k))
               quad(2,3)=quad(2,3)+cofrho(i,j,k)*(
     +              3.d0*psx(i)*psyy(j)*pszz(k))
            end do
         end do
      end do
      quad(2,1)=quad(1,2)
      quad(3,2)=quad(2,3)
      quad(3,1)=quad(1,3)
      print*,'somme des charges =',qtot
      coef=-4.d0*pi
      if (qtot.ne.0.d0) then
         do i=1,3
            bari(i)=bari(i)/qtot
         end do
      end if
      print*,'cxyz',bari(1),bari(2),bari(3)
      do i=1,3
         do j=1,3
            print*,'i,j,qij',i,j,quad(i,j)
         end do
      end do
      do i=0,ntx
         do j=0,nty
            do k=0,ntz
               bord=0
               if (i.eq.ntx) bord=bord+1
               if (j.eq.nty) bord=bord+1
               if (k.eq.ntz) bord=bord+1
               if (i.eq.0) bord=bord+1
               if (j.eq.0) bord=bord+1
               if (k.eq.0) bord=bord+1
               if (bord.ge.1) then
               x=gtx(i)
               y=gty(j)
               z=gtz(k)
               call potentiel(x,nx,gx,y,ny,gy,z,nz,gz,csol,pot,
     +              dumi,dumj,dumr)
                  phi(i,j,k)=pot
               end if
            end do
         end do
      end do
      do i=1,nsx
         do j=1,nsy
            do k=1,nsz
               rhsl(i,j,k)=coef*rho(i,j,k)
            end do
         end do
      end do
      do i=1,nsx
         do j=1,nsy
            do k=1,nsz
               somme=0.d0
               do a=0,ntx,ntx
                  somme=somme-dx(i,a)*phi(a,j,k)
               end do
               do b=0,nty,nty
                  somme=somme-dy(j,b)*phi(i,b,k)
               end do
               do c=0,ntz,ntz
                  somme=somme-dz(k,c)*phi(i,j,c)
               end do
               comp(i,j,k)=somme
            end do
         end do
      end do
      do i=1,nsx
         do j=1,nsy
            do k=1,nsz
               rhsl(i,j,k)=rhsl(i,j,k)+comp(i,j,k)
            end do
         end do
      end do
      end
c--------------------------------------------------------------------
      subroutine makerh2(rho,dx,dy,dz,nsx,nsy,nsz,
     +     gtx,gty,gtz,phi,rhsl,sxm1,sym1,szm1,
     +     psx,psy,psz,psxx,psyy,pszz,psx2,psy2,psz2)
      include 'ceq3d.f'
      real*8 gtx(0:NTXT),gty(0:NTYT),gtz(0:NTZT)
      real*8 rho(0:NTXT,0:NTYT,0:NTZT)
      real*8 cofrho(0:NTXT,0:NTYT,0:NTZT)
      real*8 rhsl(NXS,NYS,NZS)
      real*8 comp(NXS,NYS,NZS)
      real*8 dx(0:NTXT,0:NTXT),dy(0:NTYT,0:NTYT),dz(0:NTZT,0:NTZT)
      real*8 phi(0:NTXT,0:NTYT,0:NTZT)
      real*8 unsr,unsr5,xp(3),bari2
      real*8 coef,somme,qtot,bari(3),quad(3,3),quadri,monop
      real*8 psx(0:NTXT),psy(0:NTYT),psz(0:NTZT)
      real*8 psxx(0:NTXT),psyy(0:NTYT),pszz(0:NTZT)
      real*8 psx2(0:NTXT),psy2(0:NTYT),psz2(0:NTZT)
      real*8 sxm1(0:NTXT,0:NTXT),sym1(0:NTYT,0:NTYT),szm1(0:NTZT,0:NTZT)
c      real*8 norma
      integer nsx,nsy,nsz
      integer ntx,nty,ntz
      integer a,b,c,i,j,k,l,m
      integer bord
      ntx=nsx+1
      nty=nsy+1
      ntz=nsz+1
      call tensrus2(ntx,nty,ntz,sxm1,sym1,szm1,rho,cofrho)
      do i=1,3
         bari(i)=0.d0
         do j=1,3
            quad(i,j)=0.d0
         end do
      end do
      qtot=0.d0
      do i=0,ntx
         do j=0,nty
            do k=0,ntz
               qtot=qtot+cofrho(i,j,k)*psx(i)*psy(j)*psz(k)
               bari(1)=bari(1)+cofrho(i,j,k)*psxx(i)*psy(j)*psz(k)
               bari(2)=bari(2)+cofrho(i,j,k)*psx(i)*psyy(j)*psz(k)
               bari(3)=bari(3)+cofrho(i,j,k)*psx(i)*psy(j)*pszz(k)
               quad(1,1)=quad(1,1)+cofrho(i,j,k)*(
     +              2.d0*psx2(i)*psy(j)*psz(k)
     +                  -psx(i)*psy2(j)*psz(k)
     +                  -psx(i)*psy(j)*psz2(k))
               quad(2,2)=quad(2,2)+cofrho(i,j,k)*(
     +                  -psx2(i)*psy(j)*psz(k)
     +             +2.d0*psx(i)*psy2(j)*psz(k)
     +                  -psx(i)*psy(j)*psz2(k))
               quad(3,3)=quad(3,3)+cofrho(i,j,k)*(
     +                  -psx2(i)*psy(j)*psz(k)
     +                  -psx(i)*psy2(j)*psz(k)
     +             +2.d0*psx(i)*psy(j)*psz2(k))
               quad(1,2)=quad(1,2)+cofrho(i,j,k)*(
     +              3.d0*psxx(i)*psyy(j)*psz(k))
               quad(1,3)=quad(1,3)+cofrho(i,j,k)*(
     +              3.d0*psxx(i)*psy(j)*pszz(k))
               quad(2,3)=quad(2,3)+cofrho(i,j,k)*(
     +              3.d0*psx(i)*psyy(j)*pszz(k))
            end do
         end do
      end do
      quad(2,1)=quad(1,2)
      quad(3,2)=quad(2,3)
      quad(3,1)=quad(1,3)
      print*,'somme des charges =',qtot
      coef=-4.d0*pi
      if (qtot.ne.0.d0) then
         do i=1,3
            bari(i)=bari(i)/qtot
         end do
      end if
      print*,'cxyz',bari(1),bari(2),bari(3)
c      do i=1,3
c         do j=1,3
c            print*,'i,j,qij',i,j,quad(i,j)
c         end do
c      end do
      bari2=bari(1)**2+bari(2)**2+bari(3)**2
      do l=1,3
         do m=1,3
            if (l.eq.m) then
               quad(l,l)=quad(l,l)-qtot*(3.d0*bari(l)**2-bari2)
            else
               quad(l,m)=quad(l,m)-qtot*(3.d0*bari(l)*bari(m))
            end if
         end do
      end do
      do i=0,ntx
         do j=0,nty
            do k=0,ntz
               bord=0
               if (i.eq.ntx) bord=bord+1
               if (j.eq.nty) bord=bord+1
               if (k.eq.ntz) bord=bord+1
               if (i.eq.0) bord=bord+1
               if (j.eq.0) bord=bord+1
               if (k.eq.0) bord=bord+1
               if (bord.ge.1) then
                  xp(1)=gtx(i)-bari(1)
                  xp(2)=gty(j)-bari(2)
                  xp(3)=gtz(k)-bari(3)
c                  unsrj=1.d0/dsqrt(gtx(i)**2+gty(j)**2+gtx(k)**2)
                  unsr=1.d0/dsqrt(xp(1)**2+xp(2)**2+xp(3)**2)
                  monop=qtot*unsr
                  quadri=0.d0
                  do l=1,3
                     do m=1,3
                        quadri=quadri+quad(l,m)*xp(l)*xp(m)
                     end do
                  end do
                  unsr5=unsr**5
                  quadri=quadri*unsr5*0.5d0
                  phi(i,j,k)=monop+quadri
c                  phi(i,j,k)=monop
               end if
            end do
         end do
      end do
      do i=1,nsx
         do j=1,nsy
            do k=1,nsz
               rhsl(i,j,k)=coef*rho(i,j,k)
            end do
         end do
      end do
      do i=1,nsx
         do j=1,nsy
            do k=1,nsz
               somme=0.d0
               do a=0,ntx,ntx
                  somme=somme-dx(i,a)*phi(a,j,k)
               end do
               do b=0,nty,nty
                  somme=somme-dy(j,b)*phi(i,b,k)
               end do
               do c=0,ntz,ntz
                  somme=somme-dz(k,c)*phi(i,j,c)
               end do
               comp(i,j,k)=somme
            end do
         end do
      end do
      do i=1,nsx
         do j=1,nsy
            do k=1,nsz
               rhsl(i,j,k)=rhsl(i,j,k)+comp(i,j,k)
            end do
         end do
      end do
      end
c-----------------------------------------------------------
      subroutine force2g(npart,nbelec,fp,qp,nx,ny,nz,
     +     csolbig,gxbig,gybig,gzbig,nboutbig,
     +     csol,gx,gy,gz,
     +     inttab1,inttab2,nbdt,pasgrid,nbcap,liste2)
      include 'ceq3d.f'
      real*8 gx(0:NHFX)
      real*8 gy(0:NHFY)
      real*8 gz(0:NHFZ)
      real*8 csol(0:NTXT,0:NTYT,0:NTZT)
      real*8 gxbig(0:NHFX)
      real*8 gybig(0:NHFY)
      real*8 gzbig(0:NHFZ)
      real*8 csolbig(0:NTXT,0:NTYT,0:NTZT)
      real*8 inttab1(0:9,0:NBTDMAX)
      real*8 inttab2(0:9,0:NBTDMAX)
      real*8 qp(3,npartmax)
      real*8 fp(3,npartmax)
      real*8 nbelec,pasgrid
      integer nx,ny,nz
      integer nboutbig
      integer liste2(npartmax),compt
      integer nbdt,npart
      integer nbcap
      real*8 coef
      real*8 coef2,csr3s2,champE(3)
      real*8 xmin,xmax,ymin,ymax,zmin,zmax
      real*8 x,y,z
      integer i,j
      logical in
      coef=nbelec/dfloat(npart)
      coef2=coef*coef       
      xmin=gx(2)
      xmax=gx(nx-2)
      ymin=gy(2)
      ymax=gy(ny-2)
      zmin=gz(2)
      zmax=gz(nz-2)
      compt=0
      do i=1,npart-nbcap
         x=qp(1,i)
         y=qp(2,i)
         z=qp(3,i)
         if ((x.gt.xmin).and.(x.lt.xmax).and.
     +       (y.gt.ymin).and.(y.lt.ymax).and.
     +       (z.gt.zmin).and.(z.lt.zmax)) then
            call champsg(x,nx,gx,
     +                   y,ny,gy,
     +                   z,nz,gz,csol,champE,
     +           inttab1,inttab2,nbdt,pasgrid)
            fp(1,i)=coef*champE(1)
            fp(2,i)=coef*champE(2)
            fp(3,i)=coef*champE(3)
         else
            compt=compt+1
            liste2(compt)=i
         end if
      end do
      print*,'nb de part hors du calcul 1',compt
      do i=1,compt
         j=liste2(i)
         x=qp(1,j)
         y=qp(2,j)
         z=qp(3,j)
         call champ(x,nx,gxbig,y,ny,gybig,z,nz,gzbig
     +        ,csolbig,champE,in)
         if (in) then
            fp(1,j)=coef*champE(1)
            fp(2,j)=coef*champE(2)
            fp(3,j)=coef*champE(3)
         else
            csr3s2=(x**2+y**2+z**2)**(1.5d0)
            csr3s2=-1.d0*coef2*dfloat(nboutbig+nbcap)/csr3s2
            fp(1,j)=csr3s2*x
            fp(2,j)=csr3s2*y
            fp(3,j)=csr3s2*z
         end if
      end do
      end
c-----------------------------------------------------------
      subroutine enerele2g(npart,nbelec,qp,nx,ny,nz,
     +     csolbig,gxbig,gybig,gzbig,nboutbig,
     +     csol,gx,gy,gz,inttab1,nbdt,pasgrid,
     +     enele,einout,elin,nbcap,liste2)
      include 'ceq3d.f'
      real*8 gx(0:NHFX)
      real*8 gy(0:NHFY)
      real*8 gz(0:NHFZ)
      real*8 csol(0:NTXT,0:NTYT,0:NTZT)
      real*8 gxbig(0:NHFX)
      real*8 gybig(0:NHFY)
      real*8 gzbig(0:NHFZ)
      real*8 csolbig(0:NTXT,0:NTYT,0:NTZT)
      real*8 inttab1(0:9,0:NBTDMAX)
      real*8 qp(3,npartmax)
      real*8 nbelec,pasgrid
      integer nx,ny,nz
      integer nboutbig
      integer liste2(npartmax),compt
      integer nbdt,npart,nbcap
      real*8 coef
      real*8 csr3s2,coefint
      real*8 xmin,xmax,ymin,ymax,zmin,zmax
      real*8 xmin2,xmax2,ymin2,ymax2,zmin2,zmax2
      real*8 x,y,z
      real*8 potel,potin,einout,enele,elin
      real*8 pot,dumr,pote,ract
      integer dumi,dumj
      integer i,j
      coef=nbelec/dfloat(npart)
      coefint=nbelec*dfloat(npart-nboutbig-nbcap)/dfloat(npart)
      dumi=0
      dumj=0
      dumr=0.d0
      potel=0.d0
      potin=0.d0
      einout=0.d0
      xmin=gx(2)
      xmax=gx(nx-2)
      ymin=gy(2)
      ymax=gy(ny-2)
      zmin=gz(2)
      zmax=gz(nz-2)
      xmin2=gxbig(0)
      xmax2=gxbig(nx)
      ymin2=gybig(0)
      ymax2=gybig(ny)
      zmin2=gzbig(0)
      zmax2=gzbig(nz)
      compt=0
      do i=1,npart-nbcap
         x=qp(1,i)
         y=qp(2,i)
         z=qp(3,i)
         ract=dsqrt(qp(1,i)**2+qp(2,i)**2+qp(3,i)**2)
         if ((x.gt.xmin).and.(x.lt.xmax).and.
     +       (y.gt.ymin).and.(y.lt.ymax).and.
     +       (z.gt.zmin).and.(z.lt.zmax)) then
            call potensg(x,nx,gx,y,ny,gy,z,nz,gz,csol,pote,
     +           inttab1,nbdt,pasgrid)
            potel=potel+coef*pote
            if (ract.lt.100.d0) then
               potin=potin+coef*pote
            end if
         else
            compt=compt+1
            liste2(compt)=i
         end if
      end do
      print*,'nb de part hors du calcul 1',compt
      do i=1,compt
         j=liste2(i)
         x=qp(1,j)
         y=qp(2,j)
         z=qp(3,j)
         ract=dsqrt(qp(1,j)**2+qp(2,j)**2+qp(3,j)**2)
         if ((x.gt.xmin2).and.(x.lt.xmax2).and.
     +       (y.gt.ymin2).and.(y.lt.ymax2).and.
     +       (z.gt.zmin2).and.(z.lt.zmax2)) then
            call potentiel(x,nx,gxbig,y,ny,gybig,z,nz,gzbig
     +            ,csolbig,pot,dumi,dumj,dumr)
            potel=potel+coef*pot
            if (ract.lt.100.d0) then
               potin=potin+coef*pot
            end if
         else
            csr3s2=dsqrt(x**2+y**2+z**2)
            csr3s2=coefint/csr3s2
            potel=potel+coef*csr3s2
            einout=einout+coef*csr3s2
         end if
      end do
      enele=potel*0.5d0
      elin=potin*0.5d0
      end
c-----------------------------------------------------------
      subroutine enertot2g(npart,nbion,nbelec,qp,nx,ny,nz,
     +     csolbig,gxbig,gybig,gzbig,nboutbig,
     +     csol,gx,gy,gz,inttab1,nbdt,pasgrid,
     +     enele,einout,elin,enejel,ekin,ekinout,
     +     enetot,nbtour,lcine,first,nbcap,liste2,titreq)
      include 'ceq3d.f'
      real*8 gx(0:NHFX)
      real*8 gy(0:NHFY)
      real*8 gz(0:NHFZ)
      real*8 csol(0:NTXT,0:NTYT,0:NTZT)
      real*8 gxbig(0:NHFX)
      real*8 gybig(0:NHFY)
      real*8 gzbig(0:NHFZ)
      real*8 csolbig(0:NTXT,0:NTYT,0:NTZT)
      real*8 inttab1(0:9,0:NBTDMAX)
      real*8 qp(3,npartmax)
      real*8 nbion,nbelec,pasgrid
      integer nx,ny,nz
      integer nboutbig
      integer liste2(npartmax),compt
      integer nbdt,npart,nbcap
      real*8 coef
      real*8 csr3s2,coefint
      real*8 xmin,xmax,ymin,ymax,zmin,zmax
      real*8 xmin2,xmax2,ymin2,ymax2,zmin2,zmax2
      real*8 x,y,z
      real*8 potel,potin,einout,enele,elin
      real*8 enejel,ekin,enetot,ekinout
      real*8 ejelin,rayon,eion,etotin
      real*8 lcine(3)
      real*8 pot,pote,dumr,ract
      character*23 titreq
      integer dumi,dumj,nbtour
      integer i,j
      logical first
      dumi=0
      dumj=0
      dumr=0.d0
      coef=nbelec/dfloat(npart)
      coefint=nbelec*dfloat(npart-nbcap-nboutbig)/dfloat(npart)
      coefint=nbion-coefint
      potel=0.d0
      potin=0.d0
      xmin=gx(2)
      xmax=gx(nx-2)
      ymin=gy(2)
      ymax=gy(ny-2)
      zmin=gz(2)
      zmax=gz(nz-2)
      xmin2=gxbig(0)
      xmax2=gxbig(nx)
      ymin2=gybig(0)
      ymax2=gybig(ny)
      zmin2=gzbig(0)
      zmax2=gzbig(nz)
      compt=0
      do i=1,npart-nbcap
         x=qp(1,i)
         y=qp(2,i)
         z=qp(3,i)
         ract=dsqrt(qp(1,i)**2+qp(2,i)**2+qp(3,i)**2)
         if ((x.gt.xmin).and.(x.lt.xmax).and.
     +       (y.gt.ymin).and.(y.lt.ymax).and.
     +       (z.gt.zmin).and.(z.lt.zmax)) then
            call potensg(x,nx,gx,y,ny,gy,z,nz,gz,csol,pote,
     +           inttab1,nbdt,pasgrid)
            potel=potel+coef*pote
            if (ract.lt.100.d0) then
               potin=potin+coef*pote
            end if
         else
            compt=compt+1
            liste2(compt)=i
         end if
      end do
      print*,'nb de part hors du calcul 1',compt
      do i=1,compt
         j=liste2(i)
         x=qp(1,j)
         y=qp(2,j)
         z=qp(3,j)
         ract=dsqrt(qp(1,j)**2+qp(2,j)**2+qp(3,j)**2)
         if ((x.gt.xmin2).and.(x.lt.xmax2).and.
     +       (y.gt.ymin2).and.(y.lt.ymax2).and.
     +       (z.gt.zmin2).and.(z.lt.zmax2)) then
            call potentiel(x,nx,gxbig,y,ny,gybig,z,nz,gzbig
     +            ,csolbig,pot,dumi,dumj,dumr)
            potel=potel+coef*pot
            if (ract.lt.100.d0) then
               potin=potin+coef*pot
            end if
         else
            csr3s2=dsqrt(x**2+y**2+z**2)
            csr3s2=coefint/csr3s2
            potel=potel-coef*csr3s2
         end if
      end do
      ejelin=potin-2.d0*elin
      enejel=potel-2.d0*enele
      enele=enele+0.5d0*einout
      pote=0.d0
      rayon=(nbion**(1.d0/3.d0))*rs
      eion=3.d0*(nbion**2.d0)/(rayon*5.d0)
      enetot=eion+ekin+enele+enejel
      etotin=ejelin+elin+eion+ekin-ekinout
      if (first) then
         open (1,file=titreq,status='unknown')
         open (2,file='moment.dat',status='unknown')
         write(1,*) 'enetot,ekin,enele,enejel,einout,ekinout,etin'
         write(2,*) 'moment cinetique (lx,ly,lz)'
         write(1,'(7e15.7)') 
     +        enetot,ekin,enele,enejel,einout,ekinout,etotin
         write(2,'(3e15.7)') lcine(1),lcine(2),lcine(3)
         close(1)
         close(2)
      else
         open (1,file=titreq,status='old')
         open (2,file='moment.dat',status='old')
         read(1,*)
         read(2,*)
         do j=1,nbtour-1
            read(1,'(7e15.7)')
            read(2,'(3e15.7)')
         end do
         write(1,'(7e15.7)') 
     +        enetot,ekin,enele,enejel,einout,ekinout,etotin
         write(2,'(3e15.7)') lcine(1),lcine(2),lcine(3)
         close(1)
         close(2)
      end if
      print*,'enele,enejel,ekin,enetot,ekinout'
      print*, enele,enejel,ekin,enetot,ekinout
      end
c----------------------------------------------------------
      subroutine capture(npart,qp,pospro,nbelec,nbcap)
      include 'ceq3d.f'
      real*8 qp(3,npartmax),x,y,z
      real*8 pospro(3),nbelec
      real*8 raycap1,raycap2,raycap3
      real*8 charge1,charge2,charge3,ray
      integer i,npart,nbcap
      integer compt1,compt2,compt3
      raycap1=7.d0
      raycap2=10.d0
      raycap3=15.d0
      compt1=0
      compt2=0
      compt3=0
      do i=1,npart-nbcap
         x=qp(1,i)-pospro(1)
         y=qp(2,i)-pospro(2)
         z=qp(3,i)-pospro(3)
         ray=dsqrt(x**2+y**2+z**2)
         if (ray.lt.raycap1) then
            compt1=compt1+1
         end if
         if (ray.lt.raycap2) then
            compt2=compt2+1
         end if
         if (ray.lt.raycap3) then
            compt3=compt3+1
         end if
      end do
      charge1=nbelec*dfloat(compt1)/dfloat(npart)
      charge2=nbelec*dfloat(compt2)/dfloat(npart)
      charge3=nbelec*dfloat(compt3)/dfloat(npart)
      print*,'charge autour du proj:',charge1,charge2,charge3
      end
C----------------------------------------------------------
      subroutine docapture(npart,qp,qpold,fp,
     +           pospro,nbcap,chapro,nbelec,cutoff,einterne)
      include 'ceq3d.f'
      real*8 qp(3,npartmax)
      real*8 qpold(3,npartmax)
      real*8 fp(3,npartmax)
      real*8 pospro(3),chapro
      real*8 x,y,z,raycap,ray,coef,nbelec
      real*8 cutoff,qpart,vcent,coefcent,einterne
      integer i,npart,compt,nbcap,k
      qpart=-1.d0*(nbelec/dfloat(npart))
      vcent=2.d0*chapro/cutoff
      coefcent=-1.d0*chapro/(cutoff**3)
      coef=-1.d0*(nbelec/dfloat(npart))*chapro
      raycap=10.d0
      compt=0
      k=0
      einterne=0.d0
      do i=1,npart
         x=qp(1,i)-pospro(1)
         y=qp(2,i)-pospro(2)
         z=qp(3,i)-pospro(3)
         ray=dsqrt(x**2+y**2+z**2)
         if (ray.lt.raycap) then
            compt=compt+1
            if (ray.gt.cutoff) then
               einterne=einterne+coef/ray
            else
               einterne=einterne+qpart*(vcent+(ray**2)*coefcent)
            end if
         else
            k=k+1
            qp(1,k)=qp(1,i)
            qp(2,k)=qp(2,i)
            qp(3,k)=qp(3,i)
            qpold(1,k)=qpold(1,i)
            qpold(2,k)=qpold(2,i)
            qpold(3,k)=qpold(3,i)
            fp(1,k)=fp(1,i)
            fp(2,k)=fp(2,i)
            fp(3,k)=fp(3,i)
         end if
      end do
      nbcap=compt
      print*,'docapture ok : npart,nbcap,k,nbcap+k'
     +     ,npart,nbcap,k,nbcap+k
      chapro=chapro-nbelec*dfloat(nbcap)/dfloat(npart)      
      print*,' docapturee, nouvelle charge :',chapro
      print*,' energie interne',einterne
      end     
c------------------------------------------------------------------
      subroutine pspech2(ntx,nty,ntz,gtx,gty,gtz,
     +     rho,csol,sxm1,sym1,szm1,nbion)
      include 'ceq3d.f'              
      real*8 gtx(0:NTXT)
      real*8 gty(0:NTYT)
      real*8 gtz(0:NTZT)
      real*8 sxm1(0:NTXT,0:NTXT),sym1(0:NTYT,0:NTYT),szm1(0:NTZT,0:NTZT)
      real*8 rho(0:NTXT,0:NTYT,0:NTZT)
      real*8 csol(0:NTXT,0:NTYT,0:NTZT)
      real*8 echsol(0:NTXT,0:NTYT,0:NTZT)
      real*8 ech(0:NTXT,0:NTYT,0:NTZT)
      real*8 nbion,potjel,rr
      integer ntx,nty,ntz
      integer i,j,k
      real*8 us3,coefech,coffcor,cfc2,rsrm1
      real*8 xco,rsr
      us3=1.d0/3.d0
      coefech=-((3.d0/PI)**us3)
      coffcor=-0.0333d0
      cfc2=11.4d0*((4.d0*PI/3.d0)**us3)
      do k=0,ntz
         do j=0,nty
            do i=0,ntx
               rsrm1=rho(i,j,k)**us3
               rr=dsqrt(gtx(i)**2+gty(j)**2+gtz(k)**2)
               ech(i,j,k)=coefech*rsrm1+coffcor*dlog(1.d0+cfc2*rsrm1)
     +              +potjel(rr,nbion)
               ech(i,j,k)=-1.d0*ech(i,j,k)
c               ech(i,j,k)=potjel(rr,nbion)
            end do
         end do
      end do
      call tensrus2(ntx,nty,ntz,sxm1,sym1,szm1,ech,echsol)
      do k=0,ntz
         do j=0,nty
            do i=0,ntx
               csol(i,j,k)=csol(i,j,k)+echsol(i,j,k)
            end do
         end do
      end do
      do k=0,ntz
         do j=0,nty
            do i=0,ntx
               if (rho(i,j,k).gt.1.d-7) then
                  rsrm1=rho(i,j,k)**us3
                  rr=dsqrt(gtx(i)**2+gty(j)**2+gtz(k)**2)
                  rsr=(3.d0/(4*pi*rho(i,j,k)))**(1.d0/3.d0)
                  xco=rsr/11.4d0
                  ech(i,j,k)=coefech*rsrm1*3.d0/4.d0+
     +           coffcor*((1.d0+xco**3)*dlog(1.d0+(1.d0/xco))+
     +           0.5d0*xco-xco**2-(1.d0/3.d0))
     +                 +potjel(rr,nbion)
c     ech(i,j,k)=potjel(rr,nbion)
               else
                  ech(i,j,k)=potjel(rr,nbion)
               end if
            end do
         end do
      end do
      call tensrus2(ntx,nty,ntz,sxm1,sym1,szm1,ech,echsol)
      do k=0,ntz
         do j=0,nty
            do i=0,ntx
               csol(i,j,k)=csol(i,j,k)+echsol(i,j,k)
            end do
         end do
      end do
      end            
c----------------------------------------------------------------------
      subroutine initialise(rt,pt,npart,nbelec)
      include 'ceq3d.f'
      real*8 rt(3,npartmax)
      real*8 pt(3,npartmax)
      real*8 x(6),nbelec
      real*4 ran2
      real*8 hm1(nbtdmax),rel(nbtdmax)
      integer i,npart,nbgrid,nbgrid2
      real*8 rmax
      real*8 r,p,rhor,rk1,rk2,xj1,xj2,coef,pf
      integer idum,nbi,j,k
      real*8 stheta,ee
      open (1,file='hm1.dat',status='old')
      read (1,*) nbgrid
      do i=1,nbgrid
           read (1,'(2e14.6)') hm1(i)
      end do	
      close(1)
      open (2,file='rhoinit.dat',status='old')
      read (2,*) nbgrid2
      read (2,*) rmax
      do i=1,nbgrid2
           read (2,'(2e14.6)') rel(i)
      end do	
      close(2)
      ee = nbelec/dfloat(npart)
      idum=-1
      coef=(3.d0*(PI**2))**(1.d0/3.d0)
      do i=1,npart 
         do nbi=1,6
             x(nbi)=ran2(idum)
         end do
	 j=int(x(1)*dfloat(nbgrid-1))+1
         xj1=dfloat(j-1)/dfloat(nbgrid-1)
         xj2=dfloat(j)/dfloat(nbgrid-1)
         r=hm1(j)+(hm1(j+1)-hm1(j))*
     +	          (x(1)-xj1)/(xj2-xj1)
         stheta= sqrt(1.d0-(2.d0*x(3)-1)**2)
         rt(1,i)= r*cos(2.d0*PI*x(2))*stheta
         rt(2,i)= r*sin(2.d0*PI*x(2))*stheta
         rt(3,i)= r*(2.d0*x(3)-1.d0)
         k=int(r*dfloat(nbgrid2-1)/rmax)+1
         rk1=rmax*dfloat(k-1)/dfloat(nbgrid2-1)
         rk2=rmax*dfloat(k)/dfloat(nbgrid2-1)
         rhor=rel(k)+(rel(k+1)-rel(k))*
     +	          (r-rk1)/(rk2-rk1) 
         pf=coef*(rhor**(1.d0/3.d0))
         p=x(4)**(1.d0/3.d0)*pf
         stheta = sqrt(1.d0-(2.d0*x(6)-1.d0)**2)
         pt(1,i) = p*ee*cos(2.d0*PI*x(5))*stheta
         pt(2,i) = p*ee*sin(2.d0*PI*x(5))*stheta
         pt(3,i) = p*ee*(2.d0*x(6)-1.d0)
      end do
      write (*,'(''Initialization of test particles completed'')')
c      do i=1,npart,10
c         rt(1,i)=rt(1,i)+40.d0
c         rt(2,i+1)=rt(2,i+1)+40.d0
c      end do
      end           
c-----------------------------------------------------------------
      subroutine move(qp,qpold,fp,nbelec,npart,dltt,ekin,ekinout,
     +                nbout,tabout,lcine,nbcap)
      include 'ceq3d.f'
      real*8 qp(3,npartmax)
      real*8 qpold(3,npartmax)
      real*8 fp(3,npartmax)
      real*8 nbelec
      real*8 coef
      real*8 dltt,ract
      real*8 x,y,z,px,py,pz
      real*8 dtsq,dt2,ekin,ekinout,lcine(3),lcix,lciy,lciz
      integer npart,i,nbcap
      integer tabout(npartmax),nbout,j
      coef=1.d0/(2.d0*(nbelec/dfloat(npart))*mel)
      dtsq=(dltt**2.d0)*(dfloat(npart)/(mel*nbelec))
      dt2=(1.d0/(2.d0*dltt))*mel*nbelec/dfloat(npart)
      ekin=0.d0
      ekinout=0.d0
      j=1
      lcix=0.d0
      lciy=0.d0
      lciz=0.d0
      do i=1,npart-nbcap
         x=2.d0*qp(1,i)-qpold(1,i)+dtsq*fp(1,i)
         y=2.d0*qp(2,i)-qpold(2,i)+dtsq*fp(2,i)
         z=2.d0*qp(3,i)-qpold(3,i)+dtsq*fp(3,i)
         px=(x-qpold(1,i))*dt2
         py=(y-qpold(2,i))*dt2
         pz=(z-qpold(3,i))*dt2
         lcix=lcix+pz*qp(2,i)-py*qp(3,i)
         lciy=lciy+px*qp(3,i)-pz*qp(1,i)
         lciz=lciz+py*qp(1,i)-px*qp(2,i)
         ekin=ekin+coef*(px**2.d0+py**2.d0+pz**2.d0)
c         if (j.le.nbout) then
c            if (i.eq.tabout(j)) then
c               ekinout=ekinout+coef*(px**2.d0+py**2.d0+pz**2.d0)
c               lcix=lcix-pz*qp(2,i)+py*qp(3,i)
c               lciy=lciy-px*qp(3,i)+pz*qp(1,i)
c               lciz=lciz-py*qp(1,i)+px*qp(2,i)
c               j=j+1
c            end if
c         end if
         ract=dsqrt(qp(1,i)**2+qp(2,i)**2+qp(3,i)**2)
         if (ract.gt.100.d0) then
            ekinout=ekinout+coef*(px**2.d0+py**2.d0+pz**2.d0)
         end if
         qpold(1,i)=qp(1,i)
         qpold(2,i)=qp(2,i)
         qpold(3,i)=qp(3,i)
         qp(1,i)=x
         qp(2,i)=y
         qp(3,i)=z 
      end do
      lcine(1)=lcix
      lcine(2)=lciy
      lcine(3)=lciz
      end
c-----------------------------------------------------------
      subroutine force2gi(npart,nbelec,fpi,qpi,nx,ny,nz,
     +     csolbig,gxbig,gybig,gzbig,nboutbig,
     +     csol,gx,gy,gz,
     +     inttab1,inttab2,nbdt,pasgrid,nbcap)
      include 'ceq3d.f'
      real*8 gx(0:NHFX)
      real*8 gy(0:NHFY)
      real*8 gz(0:NHFZ)
      real*8 csol(0:NTXT,0:NTYT,0:NTZT)
      real*8 gxbig(0:NHFX)
      real*8 gybig(0:NHFY)
      real*8 gzbig(0:NHFZ)
      real*8 csolbig(0:NTXT,0:NTYT,0:NTZT)
      real*8 inttab1(0:9,0:NBTDMAX)
      real*8 inttab2(0:9,0:NBTDMAX)
      real*8 qpi(3)
      real*8 fpi(3)
      real*8 nbelec,pasgrid
      integer nx,ny,nz
      integer nboutbig
      integer nbdt,npart
      integer nbcap
      real*8 coef
      real*8 coef2,csr3s2,champE(3)
      real*8 xmin,xmax,ymin,ymax,zmin,zmax
      real*8 x,y,z
      integer i,j
      logical in
      coef=nbelec/dfloat(npart)
      coef2=coef*coef       
      xmin=gx(1)
      xmax=gx(nx-1)
      ymin=gy(1)
      ymax=gy(ny-1)
      zmin=gz(1)
      zmax=gz(nz-1)
      x=qpi(1)
      y=qpi(2)
      z=qpi(3)
      if ((x.gt.xmin).and.(x.lt.xmax).and.
     +     (y.gt.ymin).and.(y.lt.ymax).and.
     +     (z.gt.zmin).and.(z.lt.zmax)) then
         call champsg(x,nx,gx,
     +        y,ny,gy,
     +        z,nz,gz,csol,champE,
     +        inttab1,inttab2,nbdt,pasgrid)
         fpi(1)=coef*champE(1)
         fpi(2)=coef*champE(2)
         fpi(3)=coef*champE(3)
      else
         call champ(x,nx,gxbig,y,ny,gybig,z,nz,gzbig
     +        ,csolbig,champE,in)
         if (in) then
            fpi(1)=coef*champE(1)
            fpi(2)=coef*champE(2)
            fpi(3)=coef*champE(3)
         else
            csr3s2=(x**2+y**2+z**2)**(1.5d0)
            csr3s2=-1.d0*coef2*dfloat(nboutbig+nbcap)/csr3s2
            fpi(1)=csr3s2*x
            fpi(2)=csr3s2*y
            fpi(3)=csr3s2*z
         end if
      end if
      end         
c-----------------------------------------------------------------
      subroutine incproj2(delvm,mpro,pospro,vpro,
     +     dltt,qp,qpold,fp,cutoff,
     +     nbion,nbelec,npart,chapro,first,nb,
     +     rcutee,epro,impara,last,nbcap,einterne,titreq,
     +     csolbig,gxbig,gybig,gzbig,nboutbig,
     +     csol,gx,gy,gz,nx,ny,nz,
     +     inttab1,inttab2,nbdt,pasgrid,liste2)
      include 'ceq3d.f'
      real*8 gx(0:NHFX)
      real*8 gy(0:NHFY)
      real*8 gz(0:NHFZ)
      real*8 csol(0:NTXT,0:NTYT,0:NTZT)
      real*8 gxbig(0:NHFX)
      real*8 gybig(0:NHFY)
      real*8 gzbig(0:NHFZ)
      real*8 csolbig(0:NTXT,0:NTYT,0:NTZT)
      real*8 inttab1(0:9,0:NBTDMAX)
      real*8 inttab2(0:9,0:NBTDMAX)
      real*8 pasgrid
      real*8 qp(3,npartmax)
      real*8 fp(3,npartmax)
      real*8 mpro,pospro(3),vpro(3),dltt,cutoff
      real*8 r2,x,y,z,cut2,rayon2,coef,coef2,rhoin
      real*8 chapro,fx,fy,fz,fxt,fyt,fzt
      real*8 modf,nbion,nbelec,ekinp,chaprot,einterne
      real*8 epro,impara,Eelpro,Ejelpro,rayon,potjel
      integer tabcb(nbcbmax),nbcap
      real*8 rcutee,qpart,vcent,coefcent
      integer i,nb,npart,j,k
      integer nbdt,nx,ny,nz,nboutbig
      logical first
      logical last
      character*1 chch,chma
      character*3 chen,chim
      character*23 titrep
      character*23 titreq
c
      real*8 unsm,delv,delvm,delvi
      real*8 nofp,deltc
      real*8 qpav(3),qpap(3),qpco(3),vco(3)
      real*8 fppi(3),fpi(3),qpold(3,npartmax)
      real*8 elproi,fmoy(3),fmoyp(3),emoypro
      integer nbdtc
      integer liste2(npartmax)
      integer compt
      integer nbtourmoy
c
      write(chma,'(I1)') int(mpro/1836.154d0+0.1d0)
      chaprot=chapro+nbelec*dfloat(nbcap)/dfloat(npart)
      write(chch,'(I1)') int(chaprot+0.1d0)
      call makecha(chen,int(epro*27.211d0/1000.d0+0.1d0))
      call makecha(chim,int(impara+0.1d0))
      titrep='Eloss/'//'Em'//chma//
     +     'q'//chch//'e'//chen//'i'//chim//'.dat'
      titreq='Eloss/'//'Fm'//chma//
     +     'q'//chch//'e'//chen//'i'//chim//'.dat'
      k=0
      x=pospro(1)
      y=pospro(2)
      z=pospro(3)
      rayon=dsqrt(x**2.d0+y**2.d0+z**2.d0)
      Ejelpro=chapro*potjel(rayon,nbion)
      cut2=cutoff**2
      rayon2=((nbion**(1.d0/3.d0))*rs)**2.d0
      r2=x**2.d0+y**2.d0+z**2.d0
      coef2=nbion*chapro
      rhoin=3.d0/(4.d0*pi*(rs**3.d0))
      qpart=-1.d0*(nbelec/dfloat(npart))
      vcent=2.d0*chapro/cutoff
      coefcent=-1.d0*chapro/(cutoff**3)
      unsm=dfloat(npart)/nbelec
c      --- projectile <-> jellium --
      if (r2.gt.rayon2) then
         modf=coef2/(r2**1.5d0)
         fxt=modf*x      
         fyt=modf*y
         fzt=modf*z
      else
         modf=4.d0*pi*chapro*rhoin/3.d0
         fxt=modf*x
         fyt=modf*y
         fzt=modf*z
      end if
c      -- projectile <-> pseudo elecs noncb --
      coef=-1.d0*(nbelec/dfloat(npart))*chapro
      Eelpro=0.d0
      do i=1,npart-nbcap
         x=pospro(1)-qp(1,i)
         y=pospro(2)-qp(2,i)
         z=pospro(3)-qp(3,i)
         r2=x**2.d0+y**2.d0+z**2.d0
         if (r2.gt.cut2) then
            modf=coef/(r2**1.5d0)
            fx=modf*x 
            fy=modf*y 
            fz=modf*z 
            fxt=fxt+fx
            fyt=fyt+fy
            fzt=fzt+fz
            fp(1,i)=fp(1,i)-fx
            fp(2,i)=fp(2,i)-fy
            fp(3,i)=fp(3,i)-fz
            Eelpro=Eelpro+coef/dsqrt(r2)
         else
c            Eelpro=Eelpro+coef/dsqrt(cut2)
            modf=coef/(cut2**1.5d0)
            fx=modf*x 
            fy=modf*y 
            fz=modf*z 
            fxt=fxt+fx
            fyt=fyt+fy
            fzt=fzt+fz
            fp(1,i)=fp(1,i)-fx
            fp(2,i)=fp(2,i)-fy
            fp(3,i)=fp(3,i)-fz
            Eelpro=Eelpro+qpart*(vcent+r2*coefcent)
c            Eelpro=Eelpro+coef/dsqrt(cut2)
            k=k+1
            if (k.lt.nbcbmax) then
               tabcb(k)=i
            else
               print*,'trop de pp cb'
            end if
         end if
      end do   
      compt=0
      do i=1,npart-nbcap
         nofp=dsqrt(fp(1,i)**2+fp(2,i)**2+fp(3,i)**2)
         delv=dltt*nofp*unsm
         if (delv.gt.delvm) then
            compt=compt+1
            liste2(compt)=i
         end if
      end do
      nbtourmoy=0.d0
      do j=1,compt
         i=liste2(j)
         deltc=dltt
         qpav(1)=qpold(1,i)
         qpav(2)=qpold(2,i)
         qpav(3)=qpold(3,i)
         qpco(1)=qp(1,i)
         qpco(2)=qp(2,i)
         qpco(3)=qp(3,i)
         vco(1)=(qpco(1)-qpav(1))/dltt
         vco(2)=(qpco(2)-qpav(2))/dltt
         vco(3)=(qpco(3)-qpav(3))/dltt
         nofp=dsqrt(fp(1,i)**2+fp(2,i)**2+fp(3,i)**2)
         delv=deltc*nofp*unsm
         nbdtc=2
         call forceproji(pospro,qpco,fppi,cutoff,
     +        nbion,nbelec,npart,chapro,elproi)
         fxt=fxt+fppi(1)
         fyt=fyt+fppi(2)
         fzt=fzt+fppi(3)
         Eelpro=Eelpro-elproi
         do while (delv.gt.delvm)
            nbtourmoy=nbtourmoy+1
            deltc=0.5d0*deltc
            nbdtc=nbdtc*2
            qpav(1)=qpold(1,i)
            qpav(2)=qpold(2,i)
            qpav(3)=qpold(3,i)
            qpco(1)=qpav(1)+vco(1)*deltc
            qpco(2)=qpav(2)+vco(2)*deltc
            qpco(3)=qpav(3)+vco(3)*deltc
            fmoy(1)=0.d0
            fmoy(2)=0.d0
            fmoy(3)=0.d0
            fmoyp(1)=0.d0
            fmoyp(2)=0.d0
            fmoyp(3)=0.d0
            emoypro=0.d0
            delv=0.d0
            do k=1,nbdtc-1
               call force2gi(npart,nbelec,fpi,qpco,nx,ny,nz,
     +              csolbig,gxbig,gybig,gzbig,nboutbig,
     +              csol,gx,gy,gz,
     +              inttab1,inttab2,nbdt,pasgrid,nbcap)               
               call forceproji(pospro,qpco,fppi,cutoff,
     +              nbion,nbelec,npart,chapro,elproi)
               fpi(1)=fpi(1)+fppi(1)
               fpi(2)=fpi(2)+fppi(2)
               fpi(3)=fpi(3)+fppi(3)
               fmoy(1)=fmoy(1)+fpi(1)
               fmoy(2)=fmoy(2)+fpi(2)
               fmoy(3)=fmoy(3)+fpi(3)
               fmoyp(1)=fmoyp(1)+fppi(1)
               fmoyp(2)=fmoyp(2)+fppi(2)
               fmoyp(3)=fmoyp(3)+fppi(3)
               emoypro=emoypro+elproi
               nofp=dsqrt(fpi(1)**2+fpi(2)**2+fpi(3)**2)
               delvi=deltc*nofp*unsm
               if (delvi.gt.delv) delv=delvi
               qpap(1)=2.d0*qpco(1)-qpav(1)+(deltc**2)*fpi(1)
               qpap(2)=2.d0*qpco(2)-qpav(2)+(deltc**2)*fpi(2)
               qpap(3)=2.d0*qpco(3)-qpav(3)+(deltc**2)*fpi(3)
               qpav(1)=qpco(1)
               qpav(2)=qpco(2)
               qpav(3)=qpco(3)
               qpco(1)=qpap(1)
               qpco(2)=qpap(2)
               qpco(3)=qpap(3)
            end do
         end do
         fxt=fxt-fmoyp(1)/dfloat(nbdtc-1)
         fyt=fyt-fmoyp(2)/dfloat(nbdtc-1)
         fzt=fzt-fmoyp(3)/dfloat(nbdtc-1)
         Eelpro=Eelpro+emoypro/dfloat(nbdtc-1)
         fp(1,i)=fmoy(1)/dfloat(nbdtc-1)
         fp(2,i)=fmoy(2)/dfloat(nbdtc-1)
         fp(3,i)=fmoy(3)/dfloat(nbdtc-1)
         qp(1,i)=0.5d0*(qpco(1)+qpold(1,i)-(dltt**2)*fp(1,i))
         qp(2,i)=0.5d0*(qpco(2)+qpold(2,i)-(dltt**2)*fp(2,i))
         qp(3,i)=0.5d0*(qpco(3)+qpold(3,i)-(dltt**2)*fp(3,i))
      end do
      if (compt.ne.0) then
         print*,'nb de trajects corrigees :',compt
         print*,'nb moyen de division :',dfloat(nbtourmoy)/dfloat(compt)
      end if
      vpro(1)=vpro(1)+fxt*dltt/mpro
      vpro(2)=vpro(2)+fyt*dltt/mpro
      vpro(3)=vpro(3)+fzt*dltt/mpro
      pospro(1)=pospro(1)+vpro(1)*dltt
      pospro(2)=pospro(2)+vpro(2)*dltt
      pospro(3)=pospro(3)+vpro(3)*dltt
      ekinp=0.5d0*mpro*(vpro(1)**2+vpro(2)**2+vpro(3)**2)
      if (first) then
         open (1,file=titrep,status='unknown')
         write(1,*) 'Ekin,charge,impara du projectile'
         write(1,*) epro,mpro,impara
         write(1,*) 'x(t),y(t),z(t),Ekin(t)'
         write(1,'(7e15.7)') pospro(1),pospro(2),pospro(3),epro-ekinp
     +        ,Eelpro,Ejelpro,einterne
         close(1)
      else
         if (last) then
            open (1,file=titrep,status='old')
            read(1,*)
            read(1,*)
            read(1,*)
            do j=1,nb-1
               read(1,'(7e15.7)')
            end do
            write(1,*) 'Energie perdue par le projectile > 0 (eV)'
            write(1,*) (epro-ekinp)*27.2116d0
            close(1)
         else
            open (1,file=titrep,status='old')
            read(1,*)
            read(1,*)
            read(1,*)
            do j=1,nb-1
               read(1,'(7e15.7)')
            end do
         write(1,'(7e15.7)') pospro(1),pospro(2),pospro(3),epro-ekinp,
     +           Eelpro,Ejelpro,einterne
            close(1)
         end if
         end if
      end         
c-----------------------------------------------------------------
      subroutine forceproji(pospro,qpi,fppi,cutoff,
     +     nbion,nbelec,npart,chapro,elproi)
      include 'ceq3d.f'
      real*8 qpi(3)
      real*8 fppi(3)
      real*8 pospro(3),cutoff
      real*8 r2,x,y,z,cut2,coef,coef2
      real*8 chapro,fx,fy,fz
      real*8 modf,nbion,nbelec
      real*8 rcutee,qpart,vcent,coefcent
      real*8 elproi
      integer i,nb,npart,j,k
      k=0
      cut2=cutoff**2
      qpart=-1.d0*(nbelec/dfloat(npart))
      vcent=2.d0*chapro/cutoff
      coefcent=-1.d0*chapro/(cutoff**3)
      coef=-1.d0*(nbelec/dfloat(npart))*chapro
c      -- projectile <-> pseudo elecs noncb --
      x=pospro(1)-qpi(1)
      y=pospro(2)-qpi(2)
      z=pospro(3)-qpi(3)
      r2=x**2.d0+y**2.d0+z**2.d0
      if (r2.gt.cut2) then
         modf=coef/(r2**1.5d0)
         fx=modf*x 
         fy=modf*y 
         fz=modf*z 
         fppi(1)=-fx
         fppi(2)=-fy
         fppi(3)=-fz
         elproi=coef/dsqrt(r2)
      else
         modf=coef/(cut2**1.5d0)
         fx=modf*x 
         fy=modf*y 
         fz=modf*z 
         fppi(1)=-fx
         fppi(2)=-fy
         fppi(3)=-fz
         elproi=qpart*(vcent+r2*coefcent)
      end if
      end
c------------------------------------------------------------------------
      subroutine angular(npart,qp,gx,nx,nbtour)
      include 'ceq3d.f'      
      real*8 qp(3,npartmax),tabang(0:100)
      real*8 gx(0:NHFX),rout,ract,dumx,ang
      integer tabdist(0:99)
      integer i,nx,j,nbtour,nbout,npart
      integer nang,dumi,dumj
      nbout=0
      nang=100
      rout=gx(nx)
      do i=0,nang
         tabang(i)=dacos(1.d0-2.d0*dfloat(i)/dfloat(nang))
      end do
      do i=0,nang-1
         tabdist(i)=0
      end do
      do i=1,npart
         ract=dsqrt(qp(1,i)**2+qp(2,i)**2+qp(3,i)**2)
         if (ract.gt.rout) then
            ang=dacos(qp(1,i)/ract)
            j=0
            do while (ang.ge.tabang(j+1))
               j=j+1
            end do
            tabdist(j)=tabdist(j)+1
            nbout=nbout+1
         end if
      end do
      if (nbtour.eq.1) then
         open (1,file='angular.dat',status='unknown')         
         do i=0,nang-1
            write (1,'(1e14.6,2I6)') tabang(i),tabdist(i),nbout
         end do
         close(1)
      else
         open (1,file='angular.dat',status='old')
         do j=2,nbtour
            do i=0,nang-1
               read (1,'(1e14.6,2I6)') dumx,dumi,dumj
            end do
         end do
         do i=0,nang-1
            write (1,'(1e14.6,2I6)') tabang(i),tabdist(i),nbout
         end do
         close(1)
      end if
      end



        
      
      
