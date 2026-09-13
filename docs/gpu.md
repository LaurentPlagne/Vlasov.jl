# Portage GPU — ce que la mesure dit

## Mesurer en séquence, pas en boucle

⚠️ **Un profil dont les postes ne somment pas au total est un profil faux.** Une
première version de ce document donnait des étages mesurés chacun dans sa propre
boucle : leur somme faisait 238 ms pour un pas de 412. Les 42 % manquants
n'étaient ni du ramasse-miettes (0 %) ni des allocations (5 Mo, toutes dans
`poisson!`) — c'était de l'**éviction de cache**.

Répéter une seule fonction lui laisse ses données chaudes. La séquence réelle,
elle, fait défiler les deux densités, les deux potentiels, les coefficients et
les tampons de dépôt — 45 Mo pour ces derniers à huit fils — ce qui déborde les
48 Mo de cache du M1 Max à chaque tour. Les chiffres ci-dessous sont donc
relevés **dans l'ordre réel**, un chronomètre par étage, et somment à 99,8 %.

## Où passe le temps

Na₁₀₀₀, 800 000 pseudo-particules, grille `nfine = 44`, un pas de temps.

| étage | CPU seul | avec forces sur GPU | échelle |
|---|---|---|---|
| **énergie d'interaction (Hartree)** | 77,6 | 76,9 | N |
| **énergie d'interaction (totale)** | 71,8 | 69,5 | N |
| forces sur le nuage | 127,8 | **29,9** | N |
| dépôt lissé (fine) | 68,6 | 67,0 | N |
| **Poisson (GEMM + bords)** | **42,6** | 30,9 | grille |
| dépôt (grossière) | 15,5 | 28,0 | N |
| forces du projectile | 14,8 | 14,9 | N |
| champ moyen (XC + jellium) | 12,6 | 13,2 | grille |
| coefficients spline | 3,5 | 2,8 | grille |
| Verlet | 2,7 | 2,5 | N |
| **total** | **438,6** | **336,6** | |

Deux enseignements.

**Le solveur tensoriel n'était pas la bonne cible.** Les GEMM font **10 %** d'un
pas. Les rendre gratuits plafonnerait à ×1,11 — Amdahl. C'était pourtant le plan
de départ, parce que sur Na₁₉₆ à 20 000 particules la grille pesait bien plus
lourd : elle croît comme `n⁴`, le nuage comme `N`, et à l'échelle de production
c'est `N` qui l'emporte.

**Le bilan d'énergie coûte plus que les forces.** Ses deux appels font ensemble
149 ms, soit **34 %** du pas CPU et **43 %** une fois les forces sur GPU. C'est
désormais le premier poste, et de loin.

## Un gain gratuit, avant tout GPU

## Un gain gratuit, avant tout GPU

`interaction_energy` est appelée **deux fois par pas** — une fois pour Hartree
dans `update_forces!`, une fois pour le total dans `step!`. Les deux sont des
**diagnostics**, et le Fortran ne calculait `enertot2g` qu'un pas sur dix.

La rendre périodique retire 9/10 de 43 % du pas accéléré : **336 → 206 ms**,
soit **×1,63** sans une ligne de GPU, et **×2,1** depuis le point de départ.
C'est le meilleur rapport du lot, et de loin.

Cela demande de changer le contrat de `step!`, qui rend aujourd'hui un
`EnergyBudget` à chaque pas. C'est la seule difficulté.

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

**Sur le pas complet, cela fait ×1,30** — 438,6 → 336,6 ms, mesuré en séquence.
Amdahl : un poste à 29 % divisé par 4 ne rend pas plus.

## Ce qu'il reste, par ordre de rendement

1. **Rendre le bilan d'énergie périodique** (43 % du pas accéléré) — aucun GPU,
   ×1,63. À faire avant de songer à porter `interaction_energy` : une fois
   appelée un pas sur dix, elle ne vaut plus la peine d'être portée.
2. **Le dépôt** (28 %) — c'est un *scatter*, et c'est le morceau difficile. Nos
   tampons par fil (5,6 Mo chacun) ne passent pas à l'échelle GPU. Deux voies :
   des atomiques, ou **trier les particules par cellule** pour en faire une
   réduction segmentée. Le tri de la thèse revient ici, pour exactement la même
   raison qu'en 1997 : la localité des données.
3. **Les forces du projectile** (4 %) — une réduction sur toutes les
   particules, triviale à porter.
4. **Les GEMM** (10 %) — en dernier, et sans illusion.

Une remarque de conception au passage : `ParticleCloud` range les positions en
`Vector{NTuple{3,T}}`, qu'il faut réempaqueter en matrice `3×N` à chaque appel.
Un rangement par composantes supprimerait cet empaquetage **et** aiderait la
vectorisation du chemin CPU.

## Comment l'exécuter

`Metal` est une dépendance **faible** : le paquet doit rester utilisable sans
GPU Apple. L'environnement `gpu/` la porte.

    julia --project=gpu -t auto scripts/bench_gpu.jl
