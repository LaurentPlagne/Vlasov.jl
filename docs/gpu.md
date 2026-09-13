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

**Intégré** : `CellSort` dans [`src/sorting.jl`](../src/sorting.jl), le noyau
dans l'extension, et `update_forces!` y va dès qu'on lui passe un accélérateur.
Le prototype qui a servi à comparer les deux voies reste dans
[`scripts/depot_gpu.jl`](../scripts/depot_gpu.jl).

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

### Les colonnes de table : le `Float32` bute ici

Le plus gros morceau restant de la préparation est le calcul des colonnes de
table — l'indice, pour chaque particule et chaque direction, de l'échantillon
de gaussienne à employer. Purement particulaire, donc porté sur GPU sans peine :

| | ms | écart sur la densité |
|---|---|---|
| colonnes sur CPU (`Float64`) | 4,82 | 1,5e-07 |
| **colonnes sur GPU (`Float32`)** | **0,76** | **9,0e-05** |

Six fois plus rapide, six cents fois moins juste. La cause n'est pas une
maladresse d'écriture, elle est **structurelle** :

| | |
|---|---|
| largeur d'une colonne | 0,00355 a₀ |
| ULP de `Float32` à 78 a₀ | 7,6e-06 |
| rapport | **0,22 %** |

Une particule sur cinq cents est donc à moins d'un ULP d'une frontière de
colonne ; mesuré, **0,052 % basculent** sur la voisine. Ce n'est pas une erreur
d'arrondi qui se moyenne, c'est un **choix discret faux** : la particule reçoit
le mauvais échantillon de gaussienne.

Aucune réécriture ne le corrige — diviser par le pas avant de soustraire donne
la même précision relative. Ce qu'il faudrait, c'est ne jamais former la
différence en `Float32` : calculer sur l'hôte, en `Float64`, l'indice de nœud
`k` et l'écart `δ = u − knot` (lequel, majoré par 1,8, se code en `Float32` avec
une résolution de 1,2e-07), et ne monter que ceux-là.

Cela vaudrait d'être fait pour une autre raison : **le noyau des forces a besoin
des deux mêmes quantités**, et les recalcule aujourd'hui à partir de la position
absolue, donc avec la même faiblesse. Monter `(k, δ)` plutôt que `(x, y, z)`
servirait les deux noyaux et rendrait les deux plus justes.

En attendant, les colonnes restent sur CPU : 9,0e-05 sur la densité se propage
en ~1e-04 sur les forces, davantage que les 3,7e-05 déjà consentis, et pour
gagner 4 ms sur un pas de 133.

### PSRS : pourquoi la thèse en avait besoin, et pas nous

Le tri par échantillonnage régulier est plus rapide sur une séquence **déjà
presque triée**, et les particules d'un pas à l'autre le sont : mesuré,
**14,4 % seulement changent de maille par pas**, très stablement — une
particule traverse une maille de 3,55 a₀ en environ sept pas à la vitesse de
Fermi.

Mais le tri employé ici est un **tri par comptage**, donc `O(N)` et
**insensible à l'ordre par construction**. Décomposition de ses 6,7 ms
d'origine :

| étage | ms | dépend de l'ordre ? |
|---|---|---|
| indices de maille + comptage | 0,49 | non, `O(N)` |
| fusion des compteurs | 0,26 | non |
| décalages | 1,86 | non, `O(mailles)` |
| **placement** | **0,41** | **oui** |
| colonnes de table | 3,63 | non, `O(N)` |

**Seules 0,41 ms dépendent de l'ordre.** Un tri adaptatif optimiserait la part
la moins chère. Ce qui coûte, c'est le travail par particule — indices de
maille et colonnes de table — qu'aucun algorithme de tri ne touche, puisqu'il
dépend des positions exactes, lesquelles changent à chaque pas.

**Là où PSRS gagne vraiment, c'est en mémoire distribuée.** Sur le T3E de la
thèse, trier les particules est un problème de *communication* : il faut les
redistribuer entre processeurs, et l'échantillonnage régulier sert précisément
à équilibrer l'échange. Sur une machine à mémoire partagée il n'y a pas
d'échange à équilibrer — le placement écrit directement à sa place. La méthode
répondait à une contrainte que nous n'avons pas encore. Elle redeviendra
pertinente le jour où la simulation passera sur plusieurs nœuds.

Cette vérification a tout de même rapporté : le calcul des décalages balayait
les 91 125 mailles alors que 7 413 sont occupées. Restreint, il passe de
**1,86 à 0,13 ms** (plus 0,06 pour dresser la liste), et la préparation
complète de **6,7 à 5,2 ms**. Le dépôt trié sur GPU revient donc à
**8,7 + 5,2 = 13,9 ms** contre 35,7 sur CPU, soit ×2,6.

Le prochain morceau est le calcul des colonnes de table, 3,63 ms : il est
particulaire et parallèle, donc il a sa place sur le GPU, où les positions sont
déjà montées pour les forces.

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

## Bout en bout, une fois le dépôt intégré

Accelerate partout, minimum sur deux tours alternés.

| configuration | ms/pas |
|---|---|
| CPU, bilan à chaque pas | 234,2 |
| GPU, bilan à chaque pas | 176,4 |
| CPU, bilan 1 pas sur 10 | 179,3 |
| **GPU (forces + dépôt), bilan 1/10** | **120,5** |

Le dépôt intégré fait passer le meilleur de 140,2 à **120,5 ms**. Depuis le
point de départ — OpenBLAS, tout sur CPU, bilan à chaque pas, 307,9 ms — cela
fait **×2,6**.

⚠️ **Piège rencontré à l'intégration.** La première version de `_fill_columns!`
appelait `nearest_knot`, qui fait une **dichotomie**. Sur 800 000 particules et
trois directions, cela coûtait plus que tout le reste du dépôt réuni : le gain
tombait à ×1,26. La grille fine étant uniforme, l'indice se calcule. C'est la
deuxième fois que ce piège coûte dans ce chantier.

## Le projectile, fusionné dans le noyau des forces

Plutôt qu'un second passage sur 800 000 particules, l'interaction
projectile ↔ pseudo-électron est calculée **dans le noyau des forces**, sur des
positions déjà chargées : quelques opérations de plus par particule, aucune
lecture supplémentaire. La force que subit le projectile et l'énergie
d'interaction sortent par une réduction en arbre dans le groupe, puis une seule
atomique par groupe.

`projectile_forces!` sur accélérateur ne calcule donc plus rien sur les
particules — il ne lui reste que la part jellium, qui est un scalaire.

Validé sur vingt pas, contre le chemin CPU `Float64` :

| | CPU | GPU |
|---|---|---|
| position du projectile | −21,601650366 | −21,601650355 |
| perte d'énergie | 8,663683 eV | **8,663671 eV** |

Soit **1,4e-06** sur l'observable qui compte. La réduction en arbre tient bien
en `Float32` — une sommation naïve de 800 000 termes ne l'aurait pas fait.

⚠️ **Deux pièges de ce noyau.** Tous les fils doivent atteindre chaque
`threadgroup_barrier` : plus de `return` anticipé, seulement des drapeaux — un
fil qui sort tôt laisse les autres attendre indéfiniment. Et la part projectile
se calcule pour **toutes** les particules, y compris celles que le CPU reprend
au bord, sans quoi la réduction en oublierait ; la reprise CPU doit alors leur
réinjecter la réaction, que le noyau n'a pas pu ajouter à une force qu'il n'a
pas calculée.

## Où on en est

Accelerate partout, minimum sur deux tours alternés, 800 000 particules.

| configuration | ms/pas |
|---|---|
| CPU, bilan à chaque pas | 230,7 |
| GPU, bilan à chaque pas | 165,3 |
| CPU, bilan 1 pas sur 10 | 175,7 |
| **GPU, bilan 1 pas sur 10** | **104,4** |

Puis, le second membre de Poisson fusionné (ci-dessous) :

| configuration | ms/pas |
|---|---|
| CPU, bilan à chaque pas | 229,7 |
| GPU, bilan à chaque pas | 162,1 |
| CPU, bilan 1 pas sur 10 | 172,1 |
| **GPU, bilan 1 pas sur 10** | **94,8** |

Depuis le point de départ — OpenBLAS, tout CPU, bilan à chaque pas, 307,9 ms —
cela fait **×3,25**.

## Le second membre de Poisson : sept passes au lieu d'une

En décomposant `poisson!` — 23 % du pas, jamais réexaminé depuis le début — le
coupable n'était pas là où on l'attendait :

| étage | ms |
|---|---|
| multipôles | 0,56 |
| potentiel de bord | 0,33 |
| **`poisson_rhs!`** | **5,44** |
| solve tensoriel (6 GEMM + division) | 3,46 |
| recopie vers φ | 0,52 |

**Le second membre coûtait plus cher que le solveur qu'il alimente.** Écrit sous
sa forme naturelle — une diffusion pour la densité, puis six pour le relèvement
des faces — il faisait sept parcours de 681 000 points. Fusionné en une seule
passe : **4,63 → 0,29 ms, ×16**, résultat identique **au bit près**.

Deux vérifications faites au passage, et toutes deux négatives — ce qui valait
mieux que de les supposer :

* la forme transposée du produit (`mul!(C, Aᵀ, Bᵀ)`) que le solveur tensoriel
  emploie ne coûte que **16 %** de plus que la forme directe, et toutes les
  formes tournent à 400–470 GFLOPS sous Accelerate ;
* les six produits d'un solve ne font que **1,7 ms** — ce n'étaient jamais eux
  le problème.

`poisson!` passe de 21,4 à **12,1 ms**.

## Deux postes CPU, sans une ligne de GPU

### Le champ moyen : une décision de performance devenue fausse

`effective_potential!` portait ce commentaire : « délibérément séquentielle,
mesuré, la version parallèle est 0,93 fois plus rapide, c'est-à-dire plus
lente ». C'était **vrai au moment de la mesure**, et ça ne l'est plus : le
ralentissement venait du pool de fils d'OpenBLAS, qui disputait le processeur
aux boucles `Threads.@threads`. Sous Accelerate ce pool n'existe pas, et la
même boucle gagne **×5**, à résultat identique au bit près.

`effective_potential!` passe de 6,52 à **2,73 ms** par grille.

⚠️ **Une décision de performance n'est valable que dans l'environnement où elle
a été mesurée.** Celle-ci était consignée, argumentée, chiffrée — et périmée.

### Le dépôt grossier : une table au lieu d'une dichotomie

`locate` cherchait la cellule par dichotomie sur les points de collocation.
Mesuré seul, sur 800 000 particules et trois directions : **54,5 ms**, soit la
moitié du dépôt grossier une fois réparti sur huit fils.

[`LocateTable`](../src/splines.jl) le remplace par une lecture de table : un
découpage uniforme assez fin pour qu'aucun intervalle de collocation n'en
contienne moins d'un, puis au plus un cran de correction. C'est rentable parce
que la grille grossière, bien qu'**étirée**, ne l'est pas beaucoup — un facteur
trois entre son plus petit et son plus grand pas, d'où 288 entrées.

**×14,6** sur `locate`, et le dépôt grossier passe de 13,9 à **6,74 ms**. Le
résultat est **exactement** le même — c'est un test, sur mille points de trois
grilles différentes.

## Où on en est

| configuration | ms/pas |
|---|---|
| CPU, bilan à chaque pas | 218,4 |
| GPU, bilan à chaque pas | 147,0 |
| CPU, bilan 1 pas sur 10 | 160,3 |
| **GPU, bilan 1 pas sur 10** | **81,7** |

Depuis le point de départ — OpenBLAS, tout CPU, bilan à chaque pas, 307,9 ms —
**×3,77**.

## `(k, δ)` plutôt que la position absolue — un gain de justesse, pas de vitesse

Les deux noyaux recevaient les positions absolues et en tiraient eux-mêmes
l'indice de nœud et la colonne de table. Le second calcul forme `x − knot`, une
soustraction de grands nombres : à 78 a₀ l'ULP de `Float32` vaut 7,6e-06, soit
**0,22 % de la largeur d'une colonne**. Une particule sur cinq cents prenait
donc la colonne voisine — un échantillon de gaussienne faux, pas un arrondi qui
se moyenne.

L'hôte calcule désormais `(k, δ)` en `Float64` et ne monte que cela. `δ` est
majoré par un demi-pas (1,8 a₀) : en `Float32` sa résolution est 1,2e-07, trente
mille fois plus fine qu'une colonne. La position absolue se reconstruit par
`x₀ + (k−1)h + δ` là où le projectile en a besoin.

| | avant | après |
|---|---|---|
| forces, écart en norme | 3,7e-05 | **8,3e-06** |
| forces, écart médian | 1,9e-06 | 1,9e-06 |
| perte d'énergie du projectile (20 pas) | 1,4e-06 | **4,8e-07** |
| densité | 1,5e-07 | 6,5e-07 |
| pas | 81,7 ms | 80,8 ms |

**L'écart en norme des forces divise par 4,5, le médian ne bouge pas** — ce qui
confirme le diagnostic : la médiane mesurait la contraction en `Float32`
(irréductible), la norme était dominée par les quelques particules à colonne
fausse.

⚠️ **Aucun gain de vitesse.** J'en attendais quatre à cinq millisecondes, du
calcul des colonnes qui devait disparaître de l'hôte ; il n'en disparaît que la
moitié, et l'empaquetage ajoute ce qu'il économise. À porter au compte des
attentes non tenues, pas des résultats.

La densité se dégrade légèrement (1,5e-07 → 6,5e-07) : `δ` est arrondi en
`Float32`, ce qui fait encore basculer trois particules sur cent mille. C'est
cent quarante fois mieux que les 9,0e-05 qu'aurait donnés le calcul direct en
`Float32`.

## Ce qu'il reste, par ordre de rendement

1. ~~Apple Accelerate~~ — **fait**, ×1,31 pour une ligne.
2. ~~Rendre le bilan d'énergie périodique~~ — **fait**, ×1,32 à lui seul.
   `interaction_energy` sort du même coup de la liste GPU : appelée un pas sur
   dix, elle ne vaut plus la peine d'être portée.
3. ~~Le dépôt~~ — **fait**, ×1,94 sur la routine (38,5 → 19,8 ms). Nos
   tampons par fil (5,6 Mo chacun) ne passent pas à l'échelle GPU. Deux voies :
   des atomiques, ou **trier les particules par cellule** pour en faire une
   réduction segmentée. Le tri de la thèse revient ici, pour exactement la même
   raison qu'en 1997 : la localité des données.
4. ~~Les forces du projectile~~ — **fait**, fusionnées dans le noyau des
   forces : 16 ms de moins sur le pas.
5. **Les GEMM sur GPU** — sans objet : Accelerate les fait déjà en `Float64` à
   400–470 GFLOPS, et le GPU ne saurait pas. Ils ne font que 1,7 ms par solve.

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
