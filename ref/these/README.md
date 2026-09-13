# Données publiées de la thèse — chapitre sur le pouvoir d'arrêt

Recopiées telles quelles de `~/these_postdoc/these/arkonnen/trimresult/`, le
répertoire de travail xmgr d'où sortent les figures `Fperte0`, `Fperte1` et
`Fperte2`. Ce sont les **points effectivement tracés**, pas une relecture de
figure : `Fperte2.xmgr` porte la date du 6 novembre 1998 et l'axe
`yaxis label "dE/dx (eV/a\s0\N)"`, ce qui fixe l'unité sans ambiguïté.

## `Ekproj.dat.N` — trajectoires (figure `Fperte0`)

Énergie cinétique du proton le long de sa traversée d'un **Na₁₀₀₀**, pour
`N` = 1, 4, 9, 16, 25 **keV**. Colonnes : `x`, `E_k` (hartree), perte cumulée,
puis deux énergies de l'agrégat. La vitesse initiale se lit `v = 0,2·√N`.

## `desdx.dat.{40,250,1000}` — pouvoir d'arrêt (figures `Fperte1`, `Fperte2`)

`dE/dx` en **eV/a₀** pour Na₄₀, Na₂₅₀ et Na₁₀₀₀, à `σ_ion = 1` u.a.
Colonne 1 : énergie en keV. Colonnes 2 et 3 : deux mesures, qui diffèrent
entre elles de 2 à 4 % — l'ordre de grandeur du bruit de la méthode.

## `lind.dat`, `ziegler.dat` — références extérieures

Modèle de Lindhard et modèle semi-empirique de Ziegler (matière macroscopique).
`lind.dat` : `v`, `dE/dx` (eV/a₀), et une troisième colonne. Sa deuxième
colonne vaut exactement `2,2530 · v`.

## La recette de la thèse pour `dE/dx`

Ce n'est **pas** la perte totale divisée par le diamètre. Le texte pose :

    dE/dx ≃ [E_k(Δx/2) − E_k(−Δx/2)] / Δx,   Δx = 4 u.a.

c'est-à-dire une pente locale **au centre** de l'agrégat, sur 4 a₀.

Vérifié : appliquer cette recette aux `Ekproj.dat.N` redonne `desdx.dat.1000`
à 1–3 % près, soit l'écart entre les deux colonnes du fichier lui-même.

| keV | v | recalculé | `desdx.dat.1000` |
|---|---|---|---|
| 1 | 0,200 | 0,523 | 0,526 / 0,518 |
| 4 | 0,400 | 0,952 | 0,961 / 0,995 |
| 9 | 0,600 | 1,438 | 1,422 / 1,481 |
| 16 | 0,800 | 1,618 | 1,587 / 1,586 |
| 25 | 1,000 | 1,540 | 1,529 / 1,590 |

## Résultat du portage

`resultat-portage.txt` est la sortie de `scripts/figure53.jl` : les cinq points
recalculés, à comparer colonne à colonne avec `desdx.dat.1000`. Quatre sur cinq
tombent à ±5 %. Voir `docs/validation-chapitre6.md`.

`rhorad.Na1000.dat` est la densité radiale d'équilibre de Na₁₀₀₀ (octobre 1998,
998,7 électrons intégrés), recopiée de `…/majrel2/initial/1000/` : c'est l'état
initial des runs, le `pot.dat` d'origine n'ayant pas survécu.

## À quoi cela sert

Ces courbes ont été produites avec la **force gaussienne** de la thèse
(`GaussianSoftening`), pas avec l'adoucissement en boule que le Fortran
implémente réellement. Voir `docs/coquilles-fortran.md` et
`docs/validation-chapitre6.md`.
