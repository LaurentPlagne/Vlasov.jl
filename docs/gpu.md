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

## Le bilan d'énergie, rendu périodique — fait

`interaction_energy` est appelée **deux fois par pas** — une fois pour Hartree
dans `update_forces!`, une fois pour le total dans `step!`. Les deux sont des
**diagnostics**, et le Fortran ne calculait `enertot2g` qu'un pas sur dix.

    run!(sim; nsteps, energy_every = 10)          # le choix du Fortran
    step!(sim; energy = false)                    # un pas sans bilan

Ce qui l'autorise : le bilan **observe**, il ne rétroagit sur rien. Vérifié, et
c'est un test : douze pas avec `energy_every = 1` et avec `energy_every = 4`
laissent des positions **rigoureusement égales**, pas approximativement.

Le défaut reste `energy_every = 1`, pour que rien ne change sans qu'on l'ait
demandé.

## Apple Accelerate — le meilleur rapport du lot

Le solveur tensoriel n'a pas besoin d'un GPU : l'**AMX** d'Apple Silicon fait du
`Float64`, ce que le GPU ne sait pas faire. `AppleAccelerate.jl` y donne accès en
une ligne, et le gain dépasse de loin les GEMM.

| étage | OpenBLAS (2 fils) | Accelerate |
|---|---|---|
| forces | 74,8 | 63,2 |
| énergie (Hartree) | 44,0 | 34,9 |
| énergie (totale) | 48,9 | 34,7 |
| dépôt lissé | 43,9 | 36,3 |
| poisson! | 38,8 | **22,2** |
| champ moyen | 20,4 | 12,4 |
| coefficients spline | 10,1 | **2,5** |
| **total** | **315,4** | **238,5** |

**Les boucles particulaires gagnent aussi**, alors qu'elles n'appellent aucun
BLAS : forces −15 %, énergie −25 %, dépôt −17 %. Ce n'est donc pas la vitesse
des GEMM qui compte le plus, c'est que le **pool de fils d'OpenBLAS cesse de
disputer le processeur** aux boucles `Threads.@threads`. Le réglage du nombre de
fils OpenBLAS le montre : plus de fils accélèrent `poisson!` mais ralentissent
le pas.

| OpenBLAS | poisson! | pas complet |
|---|---|---|
| 1 fil | 55,5 | 233,6 |
| 2 fils | 39,2 | **226,0** |
| 4 fils | 33,1 | 246,9 |
| 8 fils | 30,9 | 257,0 |
| *Accelerate* | *22,1* | *168,3* |

### ⚠️ Comment l'activer, et comment ne pas le faire

```julia
using AppleAccelerate                       # suffit ; __init__ fait le nécessaire
AppleAccelerate.load_accelerate()           # pour rebasculer dans une session
```

**Ne jamais appeler `BLAS.lbt_forward(libacc)` sans `suffix_hint`.** Accelerate
expose deux LAPACK : l'ancien, en entiers 32 bits, et le nouveau
(`\x1a$NEWLAPACK$ILP64`) qu'attend Julia. Le détournement nu lie l'ancien, et
l'ABI ne correspond pas.

Ce que cela donne, constaté ici : `inv` rend du charbon — `InexactError:
Int64(1.0e-323)` dans `getri!` — donc les matrices de collocation sont fausses,
donc **le nuage explose**, rayon médian 1475 a₀ au lieu de 32. L'erreur n'est
pas silencieuse cette fois, mais elle aurait pu l'être : ce chemin corrompt
`inv`, pas `mul!`.

Vérifié après correction : 40 pas sous Accelerate et sous OpenBLAS donnent des
positions à `7,9e-10` près et un projectile à `1,2e-13` — l'arrondi attendu
entre deux BLAS, rien de plus.

## Ce que les trois changements donnent ensemble

Trois tours alternés **dans le même processus** — le basculement BLAS est
réversible, ce qui permet un A/B propre plutôt que deux processus dont on
compare les humeurs. 800 000 particules, minimum sur trois mesures.

| configuration | ms/pas | gain |
|---|---|---|
| OpenBLAS, bilan chaque pas, CPU | 307,9 | — |
| OpenBLAS + GPU | 262,4 | ×1,17 |
| Accelerate seul | 234,8 | ×1,31 |
| OpenBLAS + bilan 1/10 | 234,1 | ×1,32 |
| Accelerate + GPU | 200,9 | ×1,53 |
| OpenBLAS + GPU + bilan 1/10 | 188,3 | ×1,64 |
| Accelerate + bilan 1/10 | 181,3 | ×1,70 |
| **Accelerate + GPU + bilan 1/10** | **140,2** | **×2,20** |

Les trois leviers se composent presque multiplicativement. Le meilleur à lui
seul est **Accelerate**, qui ne coûte qu'une ligne.

### ⚠️ Deux pièges de mesure, payés tous les deux

**Chauffer avant de chronométrer.** Le premier appel GPU paie la compilation du
noyau Metal : sans chauffe, la même mesure donnait 614 ms/pas — un GPU *plus
lent* que le CPU.

**Vérifier l'état avant de mesurer.** Un banc qui réutilise la même `Simulation`
la fait vieillir. Après la corruption LAPACK ci-dessus, 98 % des particules
étaient hors de la grille fine : elles prenaient le chemin grossier, bien moins
cher, et le GPU rendait des `NaN` que le CPU refaisait. Le classement s'en
trouvait **inversé** — le CPU y battait le GPU. Contrôler `forces!`, qui rend le
nombre de particules hors grille : 635 sur 800 000 est sain, 784 579 ne l'est pas.

## La contrainte : pas de double précision## La contrainte : pas de double précision

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

## Où on en est — le profil de la configuration optimale

Accelerate + forces sur GPU + bilan d'énergie un pas sur dix. Pas = **133,3 ms**,
contre 307,9 au départ.

| étage | ms | % |
|---|---|---|
| **dépôt lissé (fine)** | **39,1** | **29,4 %** |
| forces (GPU) | 28,4 | 21,3 % |
| poisson! | 21,5 | 16,1 % |
| forces du projectile | 14,3 | 10,7 % |
| dépôt (grossière) | 13,2 | 9,9 % |
| champ moyen | 11,2 | 8,4 % |
| Verlet | 2,8 | 2,1 % |
| coefficients spline | 2,5 | 1,9 % |
| recopie φ ← csol | 0,3 | 0,2 % |

⚠️ **Le classement dépend de la configuration**, et l'annoncer sans le dire
induit en erreur. Sur le chemin **CPU** (Accelerate, sans GPU) les forces mènent
encore largement — 66,1 ms contre 38,0 pour le dépôt. C'est le portage GPU qui
les ramène à 28,4 et fait passer le dépôt devant.

Les deux dépôts cumulent **52,4 ms, soit 39 % du pas** : c'est le premier poste,
et de loin.

## Le dépôt — les deux voies, mesurées

Prototype dans [`scripts/depot_gpu.jl`](../scripts/depot_gpu.jl), pas encore
branché dans `update_forces!`.

Le dépôt est un **scatter** : chaque particule écrit dans 8³ = 512 points, et
les particules voisines écrivent aux mêmes endroits.

| version | ms | écart à la référence |
|---|---|---|
| CPU, ordre courant | 35,9 | — |
| CPU, **ordre trié** | 27,3 | 2,2e-15 |
| **GPU atomique** | **117,8** | 6,0e-06 |
| **GPU trié** | **8,7** | 1,5e-07 |
| *tri (préparation)* | *6,7* | |

**La voie atomique perd, et largement** — trois fois plus lente que le CPU.
800 000 × 512 = 410 millions d'additions atomiques en conflit : le GPU passe son
temps à sérialiser.

**La voie triée gagne ×4 sur le noyau**, et ×2,3 en comptant la préparation. Le
principe est un renversement de boucle : un groupe de 512 fils par maille
occupée, et **chaque fil possède un des 512 points du pochoir**. Il parcourt
toutes les particules de la maille en accumulant dans un registre, et ne fait
qu'**une seule** atomique à la fin. Les atomiques sont divisées non par deux ou
trois mais par le nombre de particules par maille — ici **106**.

Elle est aussi plus **juste** : 1,5e-07 contre 6,0e-06, parce que l'accumulation
se fait en registre et non par additions atomiques successives en `Float32`.

Ce chiffre de 106 tient à une observation qui ne se devine pas : sur
91 125 mailles, **7 525 seulement sont occupées**. L'agrégat (rayon 40) n'occupe
qu'une fraction de la boîte (±78), et les particules s'y entassent.

### Le tri profite à tout le reste

C'est le tri de la thèse, et pour la même raison qu'en 1997 : la localité des
données. Mesuré sur le chemin **CPU**, sans une ligne de GPU :

| étage | ordre courant | ordre trié | gain |
|---|---|---|---|
| dépôt lissé | 37,3 | 27,8 | ×1,34 |
| dépôt grossier | 13,0 | 10,1 | ×1,28 |
| énergie d'interaction | 35,4 | 31,9 | ×1,11 |
| forces | 64,5 | 60,7 | ×1,06 |

Soit 19,7 ms économisées par pas pour un tri à 6,7 ms.

### Deux pièges du tri par comptage

**Remettre les compteurs à zéro.** `count_cells!` accumule ; une mesure répétée
les cumulait, les décalages devenaient faux et le placement écrivait hors
bornes.

**Ne pas écrire les totaux dans `partial[1]`.** `total = partial[1]` puis
`total .+= partial[t]` détruit les compteurs de la première tranche, dont le
calcul des décalages a besoin. Coût de l'oubli : une faute de segmentation.

Et une optimisation qui compte : le placement utilisait un `Dict` interrogé par
particule — **10,6 ms à lui seul**. Un `Vector` indexé par maille le ramène à
**0,5 ms**, vingt fois moins.

## Ce qu'il reste, par ordre de rendement

1. ~~Apple Accelerate~~ — **fait**, ×1,31 pour une ligne.
2. ~~Rendre le bilan d'énergie périodique~~ — **fait**, ×1,32 à lui seul.
   `interaction_energy` sort du même coup de la liste GPU : appelée un pas sur
   dix, elle ne vaut plus la peine d'être portée.
3. ~~Le dépôt~~ — **prototypé et mesuré** (ci-dessus) : la voie triée donne ×4
   sur le noyau. Reste à la brancher dans `update_forces!`, ce qui suppose de
   porter le tri dans le paquet — il profite aussi au CPU. Nos
   tampons par fil (5,6 Mo chacun) ne passent pas à l'échelle GPU. Deux voies :
   des atomiques, ou **trier les particules par cellule** pour en faire une
   réduction segmentée. Le tri de la thèse revient ici, pour exactement la même
   raison qu'en 1997 : la localité des données.
4. **Les forces du projectile** (4 %) — une réduction sur toutes les
   particules, triviale à porter.
5. **Les GEMM sur GPU** — sans objet : Accelerate les fait déjà en `Float64`,
   et le GPU ne saurait pas.

Une remarque de conception au passage : `ParticleCloud` range les positions en
`Vector{NTuple{3,T}}`, qu'il faut réempaqueter en matrice `3×N` à chaque appel.
Un rangement par composantes supprimerait cet empaquetage **et** aiderait la
vectorisation du chemin CPU.

## Comment l'exécuter

`Metal` est une dépendance **faible** : le paquet doit rester utilisable sans
GPU Apple. L'environnement `gpu/` la porte.

    julia --project=gpu -t auto scripts/bench_gpu.jl

L'environnement `gpu/` porte aussi `AppleAccelerate`, qui n'a rien d'un backend
GPU mais relève du même chantier : aller plus vite sans changer les résultats.
