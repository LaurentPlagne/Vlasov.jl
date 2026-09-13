# Portage GPU — ce que la mesure dit

## Le plan initial était le mauvais

L'idée de départ était de porter le **solveur tensoriel** sur GPU : ce sont des
produits matrice-matrice, ce qu'un GPU fait le mieux. Le profil à l'échelle de
production dit autre chose.

Na₁₀₀₀, 800 000 pseudo-particules, grille `nfine = 44`, un pas de temps :

| étage | ms | % | échelle |
|---|---|---|---|
| forces sur le nuage | 73,0 | 31 % | N |
| énergie d'interaction | 48,6 | 20 % | N |
| dépôt lissé (fine) | 36,0 | 15 % | N |
| **Poisson (GEMM + bords)** | **30,8** | **13 %** | grille |
| champ moyen (XC + jellium) | 17,0 | 7 % | grille |
| forces du projectile | 14,2 | 6 % | N |
| dépôt (grossière) | 12,7 | 5 % | N |
| coefficients spline | 2,8 | 1 % | grille |
| Verlet | 2,8 | 1 % | N |

**Les GEMM font 14 %.** Les rendre gratuits plafonnerait à **×1,16** — Amdahl.
Le travail particulaire fait 79 %, et c'est là qu'il faut aller.

Ce renversement tient à l'échelle : sur Na₁₉₆ à 20 000 particules, les GEMM
pesaient 18 % d'un pas bien plus court. La grille a grandi comme `n⁴`, le
nuage comme `N`, et c'est `N` qui a gagné.

## Un gain gratuit, avant tout GPU

`interaction_energy` est appelée **deux fois par pas** — une fois pour Hartree
dans `update_forces!`, une fois pour le total dans `step!` — soit 18 % du pas.
Les deux sont des **diagnostics**, et le Fortran ne calculait `enertot2g` qu'un
pas sur dix. La rendre périodique donne **×1,23** sans une ligne de GPU.

## La contrainte : pas de double précision

Les GPU Apple n'ont pas de `Float64` — Metal Shading Language n'a pas de type
`double`, et `MtlArray(rand(Float64, 4))` refuse explicitement. Le portage est
donc en `Float32`, et la conséquence est structurelle :

* le chemin **CPU `Float64` reste la référence**, celle qui se compare à
  l'oracle Fortran à `1e-13` ;
* le chemin **GPU se valide contre lui**, au niveau où la physique le demande.

La mise à jour de Verlet `2q − q_old + dt²F/M` est une soustraction
catastrophique : les positions doivent rester en `Float64`. C'est 1 % du pas,
autant le laisser au CPU.

## Ce qui est fait : le champ lissé

`ext/VlasovMetalExt.jl`. Le noyau est une contraction 10×10×10 par particule —
environ 3000 opérations pour 4 Ko lus dans `csol` : **borné par la mémoire**,
pas par le calcul.

La grille fine étant uniforme, l'indice du nœud le plus proche se **calcule**
au lieu de se chercher : pas de dichotomie, pas de branche. Les particules dont
le pochoir déborde écrivent un `NaN` et sont reprises par le CPU — elles sont
rares, et les traiter sur GPU demanderait des branches là où l'intérêt est de
n'en avoir aucune.

Mesuré (`scripts/bench_gpu.jl`) :

| particules | CPU (ms) | GPU (ms) | gain | écart en norme | écart médian |
|---|---|---|---|---|---|
| 200 000 | 17,2 | 6,1 | ×2,80 | 3,8e-5 | 1,0e-6 |
| 400 000 | 31,3 | 10,0 | ×3,14 | 3,6e-5 | 1,4e-6 |
| 800 000 | 71,6 | 17,1 | **×4,19** | 3,7e-5 | 1,9e-6 |

Le gain croît avec la taille : les frais fixes — transfert de `csol`, lancement
du noyau — s'amortissent. L'écart reste à `3,7e-5`, c'est-à-dire la précision
de `Float32` et rien d'autre ; il ne dérive pas avec le nombre de particules.

Décomposition à 800 000 : noyau 12,4 ms, conversion et copie de `csol` 2,4 ms,
copies des positions et des forces 2,4 ms.

**Sur le pas complet, cela fait ×1,15.** Amdahl encore : un poste à 31 %
divisé par 4 ne donne pas grand-chose seul.

## Ce qu'il reste, par ordre de rendement

1. **`interaction_energy`** (18 %) — même forme que les forces, un *gather*
   pur. Attention à l'accumulation : sommer 800 000 termes en `Float32` perd
   des chiffres, il faut réduire en `Float32` par blocs puis accumuler en
   `Float64` côté hôte.
2. **Le dépôt** (20 %) — c'est un *scatter*, et c'est le morceau difficile. Nos
   tampons par fil (5,6 Mo chacun) ne passent pas à l'échelle GPU. Deux voies :
   des atomiques, ou **trier les particules par cellule** pour en faire une
   réduction segmentée. Le tri de la thèse revient ici, pour exactement la même
   raison qu'en 1997 : la localité des données.
3. **Les forces du projectile** (6 %) — une réduction sur toutes les
   particules, triviale à porter.
4. **Les GEMM** (14 %) — en dernier, et sans illusion.

Une remarque de conception au passage : `ParticleCloud` range les positions en
`Vector{NTuple{3,T}}`, qu'il faut réempaqueter en matrice `3×N` à chaque appel.
Un rangement par composantes supprimerait cet empaquetage **et** aiderait la
vectorisation du chemin CPU.

## Comment l'exécuter

`Metal` est une dépendance **faible** : le paquet doit rester utilisable sans
GPU Apple. L'environnement `gpu/` la porte.

    julia --project=gpu -t auto scripts/bench_gpu.jl
