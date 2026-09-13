#!/usr/bin/env python3
"""Reconstruit `pot.dat`, absent de l'archive, à partir de `rhoinit.dat`.

`initialise4` (version 1998-01-05) échantillonne l'état initial par rejet dans
l'espace des phases : elle tire `r = rmax·x₁^{1/3}` et `p = pmax·x₄^{1/3}`, puis
accepte si `p²/2 + V(r) < E_F`.

Poser `V(r) = E_F − p_F(r)²/2` rend ce test équivalent à `p < p_F(r)`, c'est-à-dire
exactement la distribution de Thomas-Fermi — et `E_F` s'élimine, donc on le pose à
zéro. `p_F = (3π²ρ)^{1/3}` se lit dans `rhoinit.dat`, le profil d'équilibre dont
l'ancienne `initialise` se servait déjà.

Format attendu par `initialise4` et `litpotexa` :

    nbgrid
    rmax pmax Ef
    (r_i, ·, V(r_i))  pour i = 0..nbgrid, format (3e14.6)
"""
import math

tok = open("rhoinit.dat").read().split()
n, rmax = int(tok[0]), float(tok[1])
rho = [float(v) for v in tok[2:2 + n]]
assert len(rho) == n, f"{len(rho)} valeurs lues pour {n} annoncées"

coef = (3.0 * math.pi ** 2) ** (1 / 3)
pF = [coef * r ** (1 / 3) for r in rho]
ng, pmax, Ef = n - 1, max(pF), 0.0

with open("pot.dat", "w") as f:
    print(f"{ng:12d}", file=f)
    print(f"  {rmax:.14E}  {pmax:.14E}  {Ef:.14E}", file=f)
    for i in range(ng + 1):
        print(f"{i * rmax / ng:14.6E}{0.0:14.6E}{Ef - 0.5 * pF[i] ** 2:14.6E}", file=f)

print(f"pot.dat : {ng + 1} points, rmax={rmax}, pmax={pmax:.6f}, Ef={Ef}")
