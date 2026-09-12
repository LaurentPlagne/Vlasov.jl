       implicit none
       real*8 pi,rho0,mel,qel,rs
       integer NHFX,NHFY,NHFZ,DE,NHF,nmax
       integer licn,lirn,nzmax,nbints
       integer NTXT,NTYT,NTZT,NXS,NYS,NZS
       integer NBTDMAX
       integer npartmax,pcolmax,lanag
       integer nrmax,npmax
       integer nbcbmax
       parameter (nrmax=100,npmax=100)
       parameter (NBTDMAX=10000)
       parameter (NHFX=28,NHFY=28,NHFZ=28) 
c       parameter (NHFX=4,NHFY=4,NHFZ=4)
       parameter (pi=3.141592653589d0)
       parameter (mel=1.d0,qel=-1.d0)
c       parameter (rs=1.23d0)
       parameter (rs=4.d0)
       parameter (npartmax=800000) 
       parameter (DE=2)
       parameter (pcolmax=64)
       parameter (rho0=3.73019397872d-03)
       parameter (NTXT=DE*NHFX+1,NTYT=DE*NHFY+1,NTZT=DE*NHFZ+1)
       parameter (NXS=DE*NHFX,NYS=DE*NHFY,NZS=DE*NHFZ)
       parameter (NHF=NHFX)
       parameter (nbints=2000)
       parameter (nbcbmax=10000)







