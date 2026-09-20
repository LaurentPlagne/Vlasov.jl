# Reprise — où en est le projet

Point de départ pour une session fraîche. Les détails sont dans les documents
cités ; ce fichier dit seulement **où regarder** et **quoi faire ensuite**.

## En une phrase

Le code Fortran de la thèse (1996-1998) est porté en Julia, validé contre un
oracle reconstruit, **les figures 5.2 et 5.3 de la thèse sont reproduites**, le
pas de temps a été accéléré **×3,95** (307,9 → 77,9 ms à 800 000 particules), et
le tout est documenté dans un site Documenter en anglais.

## Par où entrer

**Le site Documenter est désormais la porte d'entrée** — huit pages en anglais,
huit figures calculées à la construction :

    julia --project=docs docs/make.jl && open docs/build/index.html

Les documents ci-dessous restent le carnet de laboratoire : ils portent les
mesures brutes et les raisonnements, en français, et le site en cite la
substance sans les remplacer.

| document | ce qu'il contient |
|---|---|
| [`AGENTS.md`](../AGENTS.md) | conventions, disciplines de mesure, pièges transverses |
| [`chronologie-versions-fortran.md`](chronologie-versions-fortran.md) | les 43 versions Fortran datées, la cible du portage, où sont les runs de production |
| [`coquilles-fortran.md`](coquilles-fortran.md) | 10 anomalies du code d'origine, dont la n°10 qui change une conclusion physique |
| [`validation-chapitre6.md`](validation-chapitre6.md) | la figure reproduite, et comment |
| [`gpu.md`](gpu.md) | tout le travail de performance, mesures à l'appui |
| [`campagne-80M.md`](campagne-80M.md) | simulations ultra-convergées à 80M de particules, mesures et films |

## Le résultat physique

La courbe `dE/dx` de la thèse (Na₁₀₀₀, σ_ion = 1) est reproduite à ±5 % sur
quatre points sur cinq — c'est-à-dire dans l'écart que les deux colonnes
publiées ont **entre elles**.

Ce qui l'a permis : **le Fortran n'applique pas la force de sa propre thèse**
(anomalie 10). La thèse pose une interaction gaussienne, le code fait une boule
uniforme, dans ses 43 versions. Avec la boule on était 40 % trop haut.

La **figure 5.2** est reproduite aussi : les coupes de densité aux quatre
énergies, avec le sillage de plasmon qui apparaît à 9 keV et devient net à 16.
Elle teste la *structure spatiale* de la réponse là où la 5.3 n'en teste que
l'intégrale — `scripts/figure52.jl`.

Données publiées de la thèse : [`ref/these/`](../ref/these/), retrouvées dans le
répertoire de travail xmgr.

## Comment exécuter

    julia --project=.   -e 'include("test/runtests.jl")'   # 4750 tests, ~20 s
    julia --project=gpu -t auto scripts/profil_pas.jl      # profil d'un pas
    julia --project=gpu -t auto scripts/bench_gpu.jl       # CPU vs GPU
    julia --project=gpu -t auto scripts/figure52.jl        # figure 5.2, ~2 min 30
    julia --project=gpu -t auto scripts/figure53.jl        # figure 5.3, ~8 min

    julia --project=gpu -t auto scripts/film_images.jl --kev=16 --particules=8000000 --finesse=6
    julia --project=viz -t auto scripts/film.jl --video --fps=20

⚠️ **`figure53.jl` prend Accelerate mais PAS le GPU** : c'est la figure comparée
aux valeurs publiées, et le chemin GPU est en `Float32`.

**Deux environnements.** Le principal n'a pas de GPU ; `gpu/` porte `Metal`
(dépendance **faible**) et `AppleAccelerate`. Pour le REPL kaimon, activer
`gpu/` dans la session.

⚠️ **Sur Apple Silicon, charger `AppleAccelerate` — ×1,31 pour une ligne**, et
pas seulement sur les GEMM : les boucles particulaires gagnent 15 à 25 % parce
que le pool de fils d'OpenBLAS cesse de leur disputer le processeur. Les scripts
le chargent seuls sous `--project=gpu` et **annoncent au démarrage** sur quel
chemin ils tournent — pendant des mois aucun ne le faisait, et le ×1,31 mesuré
n'avait jamais quitté une session REPL.

## Les pièges qui ont coûté

Tous consignés dans les documents ci-dessus, mais voici ceux qui reviendraient :

* **Un profil dont les postes ne somment pas au total est faux.** Mesurer chaque
  étage dans sa propre boucle lui laisse ses données chaudes. Mesurer *en
  séquence* — `scripts/profil_pas.jl`.
* **Chauffer avant de chronométrer** : le premier appel GPU paie la compilation
  du noyau Metal, et donnait un GPU plus lent que le CPU.
* **Vérifier l'état avant de mesurer** : un banc qui réutilise la même
  `Simulation` la fait vieillir. `forces!` rend le nombre de particules hors
  grille — quelques centaines sur 800 000 est sain.
* **Une décision de performance n'est valable que dans l'environnement où elle a
  été mesurée.** Un commentaire disait de ne pas paralléliser le champ moyen ;
  c'était vrai sous OpenBLAS et faux sous Accelerate (×5).
* **Jamais `BLAS.lbt_forward(libacc)` nu** : cela lie l'ancien LAPACK
  d'Accelerate, `inv` rend du charbon et le nuage explose. Utiliser
  `AppleAccelerate.load_accelerate()`.
* **Le GPU Apple n'a pas de `Float64`.** Le chemin CPU reste la référence, seule
  à pouvoir se comparer à l'oracle à `1e-13`.
* **Un défaut réglé sur une prédiction, jamais corrigé après la mesure.**
  `--epaisseur` valait 40 parce que j'avais *prédit* qu'épaissir la tranche
  diviserait le bruit par √46. La mesure a dit le contraire — le sillage tient
  dans une maille en z — j'ai corrigé le discours et laissé le défaut. Mes
  propres essais passaient la valeur en ligne de commande, donc je n'ai jamais
  retraversé ce chemin ; l'auteur, lui, a lancé la commande documentée. **Quand
  une mesure contredit une prédiction, changer aussi ce que la prédiction avait
  réglé.**
* **Ne pas lisser pour rattraper du bruit.** Un flou d'une maille ne le divise
  que par 1,4 et fabrique de fausses structures cohérentes. Plus joli, moins
  vrai.

## Fait : traduction des commentaires en anglais

Demandé par l'auteur. ~7200 lignes sur 29 fichiers, et **rien n'était
mécanique** : presque chaque commentaire porte un fait mesuré ou un piège, à
rendre avec sa nuance.

**Tout est fait** — `src/`, `ext/`, `scripts/`, `test/`. Contrôle :

    grep -n "[éèêëàâçùûôîïÉÈÊÀÂÇÙÛÔÎÏ]" src/*.jl ext/*.jl scripts/*.jl test/*.jl

Les 4750 tests passent après la traduction, sans changement de compte. Deux
choses volontairement laissées en français, parce qu'elles sont des interfaces
et non de la prose :

* les noms d'options de `scripts/traversee.jl`, `figure53.jl`, `film*.jl`
  (`--profil`, `--pas`, `--graine`, `--champ`, `--sortie`) — une ligne le dit
  dans chaque docstring concernée ;
* les noms de fichiers (`traversee.jl`, `profil_pas.jl`, `depot_gpu.jl`,
  `film.jl`), que `docs/` et l'historique git citent.

Deux corrections que la relecture a imposées : la docstring d'`energy_budget`
présentait encore la coquille n°8 comme une question ouverte (elle est
élucidée), et celle de `GaussianSoftening` affirmait qu'`erfsr` n'est appelée
dans aucune version — elle l'est dans celle de juillet 1996.

## Affichage : ce qui a été mesuré

Trois réglages du film et des figures, tous établis par la mesure et tous
contre-intuitifs. Ils sont dans les docstrings de `film_images.jl` et `film.jl`
avec leurs chiffres.

* **`ρ`, pas `δρ`.** À 8 M de particules une maille fine en contient ~1330, soit
  2,8 % de bruit de tirage — l'ordre de grandeur de la déformation. Sur `δρ` le
  rapport signal/bruit tombe à 3 par maille et l'image se lit comme du poivre.
* **Échelle resserrée, vide masqué.** De 0 au maximum, le cœur occupe 89 % de la
  plage et sort uni. Sommet à `1,45 ρ_bulk`, palette arc-en-ciel : c'est ce que
  fait la thèse.
* **Évaluer la coupe depuis la spline** (`--finesse`), pas aux points de
  collocation. La surface ne fait qu'une ou deux mailles : affichée brute, elle
  sort en escalier. Vérifié — le champ le long d'un rayon est lisse et le
  contour a 0,185 a₀ d'écart-type, donc ni les données ni le rendu n'étaient en
  cause.
* **L'énergie compte.** Le sillage est un effet de vitesse : rien à 1 et 4 keV,
  net à 16. Chercher la structure à 4 keV, c'est regarder le panneau le moins
  intéressant des quatre.
* Le bruit suit `1/√N`, sans recours : ×10 en particules ne donne que ×3,2.

## Ce qui reste, par ordre

### Performance — voir la branche `gpu-portable`

Les trois points qui étaient ici sont **faits** (forces en ordre trié, dépôt
grossier sur device, champ moyen sur device). L'état courant et ce qui reste
sont dans la section [Branche `gpu-portable`](#branche-gpu-portable) ci-dessous.

### Physique

1. **Les courbes Na₄₀ et Na₂₅₀** compléteraient la figure — `desdx.dat.40` et
   `desdx.dat.250` sont dans `ref/these/`. Il faut un profil initial pour chaque
   taille ; voir comment `rhorad.Na1000.dat` a été utilisé.
2. **Les diagnostics de 1998** non portés : `mkpotrad`, `distene`, `multrcmax`,
   `denseta`, `sortietest`. Aucun ne change la trajectoire.
3. **`pspech3`** — la correction de la coquille n°8, écrite par l'auteur et
   **jamais branchée**. La porter veut dire la laisser morte, pour rester fidèle.

## Deux réserves à ne pas oublier

* `ref/fortran98/pot.dat` est **reconstruit** : l'original n'a pas survécu.
  Tout ce qui passe par `initialise4` porte cette hypothèse.
* Le chemin GPU travaille en `Float32`. Écarts constatés : forces 8,3e-06 en
  norme, densité 6,3e-07, perte d'énergie du projectile 3,9e-07 sur vingt pas.
  Loin sous la dispersion physique (2 à 4 %), mais ce n'est plus l'oracle.

## Branche `gpu-portable`

**29 commits, non fusionnée, suite verte (4777 tests) à chaque commit.**
`git log --oneline master..gpu-portable`.

Le portage est entièrement en `KernelAbstractions` : les mêmes noyaux servent
`CPU()`, Metal, et — non validé, faute de matériel — CUDA/ROCm/oneAPI.
L'extension Metal est passée de 414 à 56 lignes. Le pas de temps à **8×10⁷
particules sur 258³** est passé de **4213 à 1398 ms, ×3,01**.

### Où passe le temps aujourd'hui

| étage | ms | % | où |
|---|---:|---:|:--|
| forces + projectile | 447 | 32 % | device |
| dépôt fin | 250 | 18 % | device |
| packing + tri | 205 | 15 % | HÔTE |
| dépôt grossier (CIC) | 180 | 13 % | device |
| poisson! (2 niveaux) | 69 | 5 % | device |
| Verlet | 54 | 4 % | HÔTE |
| le reste | ~90 | 6 % | device |

Deux étages sont **finis**, au sens où ils touchent un plafond mesuré de la
machine : le dépôt grossier (87 % du débit d'atomiques) et le Verlet (95 % de la
bande passante hôte).

### Ce qui reste, par ordre de certitude

1. **Le budget énergétique**, encore particule par particule sur l'hôte. Pris un
   pas sur dix, jamais profilé à 8×10⁷.
2. **Disposition par composante** pour `ParticleCloud`, qui aiderait le packing
   hôte (19 % de la bande passante).

~~Recouvrir les deux dépôts (~180 ms)~~ — **mesuré et écarté** cette session.
L'hypothèse semblait ne dépendre d'aucun noyau : le dépôt fin et le CIC
grossier écrivent deux grilles distinctes, rien ne les lie en sortie. Mais
`global_queue` de Metal.jl est **task-local** — deux `Threads.@spawn` donnent
bien deux `MTLCommandQueue` distinctes, confirmé par `Metal.@profile`
(`[MTLDevice newCommandQueue]` ×2. Malgré ça, deux lancements du noyau de dépôt
(80M particules, 122 636 cellules occupées) prennent 634 → 624 → 593 ms en
A-B-A — aucun gain. Le profil dit pourquoi : les deux appels durent 306,89 ms
± 8,82 **chacun**, et leur somme colle au temps GPU occupé total : ils
s'exécutent bout à bout, pas ensemble. Deux files ne donnent pas deux noyaux
concurrents sur ce GPU, et les deux dépôts sont de toute façon liés par
atomiques — il n'y avait pas de débit de reste à prendre. Détail dans
`docs/src/performance.md`, section « Dead ends, measured ».

### Le noyau des forces : 447 ms, et je ne sais pas ce qui le borne

Il fait **680 GFLOP/s** contre 8849 pour un GEMM carré sur la même machine, avec
**16,2 % d'occupation**. Sept leviers testés, un seul a payé :

| levier | effet |
|---|---:|
| stencil 10³ en mémoire de groupe | **×1,20** |
| hisser `grad[:,cx]`/`ovl[:,cx]` en registres | ×1,00 |
| quatre chaînes d'accumulation (ILP) | ×1,01 |
| lectures `csol` par paires alignées | ×0,99 |
| supprimer l'indirection `perm` | ×1,001 |
| ranger physiquement les particules | ×1,02 |
| taille de groupe, 32 → 320 | ×1,006 |

Et les compteurs matériels **contredisent le code** : `Buffer Read Limiter` à
99,6 % alors qu'après hissage par le compilateur il ne reste qu'une soixantaine
de lectures buffer par particule contre mille en mémoire de groupe — laquelle
est mesurée à 7 %. ALU à 12 %, DRAM à 0,8 GB/s. Soit `Buffer Read Limiter` ne
compte pas ce que je crois, soit il reste un accès non identifié.

**Ne pas repartir sur une hypothèse de plus sans instrument neuf** : sur sept
hypothèses formulées avant mesure dans cette session, six étaient fausses.

### Impasses mesurées, à ne pas repayer

Détail chiffré dans `docs/src/performance.md`, section « Dead ends, measured ».
En bref : le tri sur device (×1,17), ranger physiquement le nuage (net −234 ms),
et ne suivre que les particules changeant de cellule (**38,3 % le font à chaque
pas** — déplacement moyen 0,356 a₀ pour une maille de 1,219).

Le fait qui les explique toutes : **le parcours trié avait déjà pris toute la
localité**, et les deux gros noyaux lisent 12 à 24 octets de donnée-particule
pour 1500 à 4900 flops. Aucune disposition mémoire ne peut les toucher.

### Discipline de mesure — quatre corrections imposées par l'auteur

Chacune a changé un résultat, et chacune est consignée en mémoire persistante.

1. **Mesurer sur le vrai nuage**, jamais synthétique : le nombre de particules
   par cellule occupée (428 contre 28) a inversé deux conclusions.
2. **Le profileur du pilote** (`Metal.@profile`), pas un découpage en étages
   fait main : il a montré 61 % d'inactivité GPU et 49,6 ms de broadcasts que
   mes étages ne comptaient nulle part.
3. **BenchmarkTools**, avec `samples`/`seconds` bornés — mais sur des tailles
   **réalistes** : à 128³ le lancement domine et masque l'effet cherché.
4. **Le REPL persistant de kaimon** (`ex(e=…, ses="…")`), pas un processus
   julia par mesure. Un protocole **A-B-A** dans un seul processus distingue un
   vrai écart d'une dérive machine ; sans lui, j'ai pris un profil dégradé par
   l'état thermique pour une régression de mon code.

Reproduire le profil du pas :

    julia --project=<env avec Metal> -t auto scripts/profil_pas.jl
