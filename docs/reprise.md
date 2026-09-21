# Reprise — où en est le projet

Point de départ pour une session fraîche. Les détails sont dans les documents
cités ; ce fichier dit seulement **où regarder** et **quoi faire ensuite**.

## En une phrase

Le code Fortran de la thèse (1996-1998) est porté en Julia, validé contre un
oracle reconstruit, **les figures 5.2 et 5.3 de la thèse sont reproduites**, le
pas de temps a été accéléré **×3,95** (307,9 → 77,9 ms à 800 000 particules), et
le tout est documenté dans un site Documenter en anglais. À l'échelle de
production — 8×10⁷ particules — le pas est passé de 4213 à **1090 ms** sur la
branche `gpu-portable`, où le nuage vit désormais sur le device. Les trois
gros noyaux ont ensuite été repris le même jour, chacun mesuré en A-B-A : les
forces 502 → 209 ms, le dépôt grossier 198 → 93, le dépôt fin 197 → 59. Le pas
de fin de journée fait **606 ms**. Le dépôt grossier a été **réécrit** depuis —
une tuile privée par work-item au lieu de huit atomiques par particule — et fait
48,0 ms contre 187,8, ×3,9 à l'échelle de production et **×8,5** à celle du film
sur une RTX 4070, où le pas entier passe de 57,3 à **38,0 ms**. Le chemin CUDA
est validé sur deux cartes, et **une seule commande** fait désormais la
simulation et son film.

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

    julia scripts/setup.jl                                # construit run/ pour cette machine
    julia --project=run -t auto scripts/xenon.jl          # premiere simulation + film, ~2 min 10
    julia --project=.   -e 'include("test/runtests.jl")'   # 4824 tests, ~80 s
    julia --project=gpu -t auto scripts/profil_pas.jl      # profil d'un pas, chemin hote
    julia --project=gpu -t auto scripts/profil_device.jl   # profil d'un pas, chemin device
    julia --project=cuda -t auto scripts/profil_device.jl  # le meme, sur NVIDIA
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

**69 commits, non fusionnée, suite verte (4824 tests) à chaque commit.**
`git log --oneline master..gpu-portable`. ⚠️ **Non poussée** au 21/09 au soir :
les quatre derniers commits sont locaux.

Le portage est entièrement en `KernelAbstractions` : les mêmes noyaux servent
`CPU()`, Metal et **CUDA — validé sur deux cartes**, une RTX 5060 louée et une
RTX 4070 locale ; ROCm et oneAPI restent non essayés. **Le nuage vit sur le
device**, et le pas à 8×10⁷ particules sur 222³ est passé de 4213 ms à
**1090**. Les trois gros noyaux ont ensuite été repris — les forces
502 → **209** ms, le dépôt grossier 198 → **93**, le dépôt fin 197 → **59**,
chacun mesuré en A-B-A — et le pas de fin de journée fait **606 ms**.

⚠️ Les chiffres du pas relevés ce jour-là (1291, 842, 746, 606) ne se déduisent
pas les uns des autres par les gains : les étages **hôtes** dérivent de 50 % au
cours de la journée. Ce sont les paires entrelacées qui valent, pas la
soustraction.

### Où passe le temps aujourd'hui

Profil mesuré en séquence, la somme fermant à 99,9 % :

| étage | ms | % | où |
|---|---:|---:|---|
| **forces + projectile** | **198** | **33 %** | device |
| tri (histogramme + placement) | 94 | 15 % | device |
| dépôt grossier (CIC) | 92 | 15 % | device |
| dépôt fin | 62 | 10 % | device |
| poisson! (2 niveaux) | 58 | 10 % | device |
| Verlet | 41 | 7 % | device |
| champ moyen | 34 | 6 % | device |
| `_fill_columns!` | 13 | 2 % | device |
| csolc | 12 | 2 % | device |
| le reste | 2 | 0 % | |

⚠️ **Les étiquettes de ce tableau ont été vérifiées le 20/09**, une à une : la
ligne à 94 ms avait été notée « `_pack!` + tri (hôte) » et c'est faux. Sur le
chemin résident `_pack!` ne fait **rien** — le nuage est déjà à sa place — et
les 94 ms sont l'histogramme et le placement, deux noyaux. Ce qui reste
vraiment sur l'hôte à chaque pas se compte en millisecondes : le balayage des
mailles (0,9 ms sur 1,37 M de mailles), la relecture des coefficients
grossiers (5,4 ms, 44 Mo) et l'avance du projectile (0,0). Le bilan
énergétique, lui, est un pas sur dix.

Pas = **606 ms**, contre 1291 le matin même avec les trois noyaux d'avant.
Les forces mènent de nouveau, et de loin.

⚠️ **Ce tableau précède le dépôt grossier tuilé** (21/09). Cette ligne-là vaut
maintenant **48,0 ms contre 187,8** pour le noyau qu'elle mesurait, A-B
entrelacé à 8×10⁷ sur 258³ — le dépôt grossier n'est donc plus le troisième
poste mais l'avant-dernier. Le reste du tableau n'a pas été repris : le refaire
demande de remesurer *toute* la séquence dans un seul processus, une ligne
corrigée dans un profil pris ailleurs ne voulant rien dire.

⚠️ **Ne pas comparer ce tableau ligne à ligne avec le précédent** (forces 458,
dépôt fin 199, grossier 194, tri 123, pas à 1090 ms) : il a été mesuré dans une
autre session, sur un nuage plus jeune. Les postes qui tournent sur l'**hôte**
varient de 50 % d'un quart d'heure à l'autre sur cette machine — `_pack!` + tri
a donné 95,7 et 164,3 ms le même jour sans qu'on y touche. Seules valent les
comparaisons **entrelacées dans un seul processus** ; les deux qui comptent
aujourd'hui sont plus bas (forces 502 → 209, dépôt grossier 198 → 93).

### L'empreinte mémoire, et jusqu'où on peut monter

Mesurée sur les objets vivants, à deux échelles : **128 octets par particule**,
au byte près aux deux tailles, plus 249 par point de grille (les deux niveaux
réunis).

    M ≈ 128·N + 249·n³ + 24·(n/2)³ + 25 Mo        modèle
    12,13 Gio prédits, 12,15 mesurés à 8×10⁷ sur 222³
    688 Mio prédits, 686 mesurés à 4×10⁶ sur 90³

Le détail, les plafonds par machine (≈340 M de particules sur un Mac 64 Go,
118 M sur une carte 16 Go) et les trois réserves sont dans
[`docs/src/device.md`](src/device.md), section « What it costs in memory ».

⚠️ **Le pic est à la construction, pas dans la boucle** : `sample_thomas_fermi`
alloue positions *et* impulsions en triplets `Float64` hôtes avant que le nuage
empaqueté n'existe — 48 octets de plus par particule. C'est lui qui borne, pas
le pas.

⚠️ **Un quart de l'empreinte était morte** avant la mesure du 20/09 : 4,1 Gio
sur 16,3, alloués pour des routines hôtes que le chemin device avait remplacées
(les forces du nuage, les tampons de dépôt par fil, les tampons du tri hôte).
Rien ne les avait suivies quand le travail a déménagé sur le device. **À
vérifier après chaque déménagement de ce genre.**

### Ce qui a changé, et qu'il faut savoir avant de toucher au code

**Le nuage est un tableau de [`PackedParticle`](@ref)**, 48 octets `isbits` :
`(k, δ)` courant *et* précédent dans un seul enregistrement. `positions` et
`previous` sont **deux vues du même stockage**, d'où `prev` en champ et non en
paramètre de type.

**Le tri déplace les particules**, il ne produit plus de permutation. Trois
conséquences, dont deux ont coûté un bug :

* l'indice d'une particule **n'est plus stable** d'un pas à l'autre — sans
  effet physique, les pseudo-particules étant indiscernables, mais tout code
  qui suivrait une particule par son indice serait faux ;
* **tout ce qui est tenu hors du nuage perd sa correspondance avec lui** au
  premier tri. L'amorçage fait donc voyager `q(0)` dans la moitié `previous`
  du même enregistrement ;
* le dépôt **grossier** lit dans l'ordre trié, par paquets de 64 particules
  qu'un work-item accumule dans une tuile privée avant de toucher la grille.
  Il lisait auparavant dans le désordre, par un pas premier avec le nombre de
  particules, parce que trier par maille fine met huit atomiques par particule
  en collision frontale — 4874 ms contre 193. La collision était réelle ; ce
  sont des additions à faire au même endroit, pas des voisines à séparer.

**Le placement du tri est une seule passe**, dans l'ordre du tableau — le
nuage étant déjà presque trié (dérive médiane : 2230 places sur 8×10⁷), un
premier étage qui « fabrique » de la localité la détruit : 151 ms contre 35,5.

### Le noyau des forces : fait, et pas par le chemin prévu

**×2,4 sans toucher à un seul calcul.** Les colonnes `y` et `z` du lissage sont
recopiées une fois par particule en mémoire de groupe, et la contraction les y
lit. Quarante valeurs, 10 Ko par groupe, **bit à bit identique** sur les 13,6
millions de fentes vérifiées.

| ce qui est mis en mémoire de groupe | ms |
|---|---:|
| rien — les tables, lues où elles sont | 505,9 |
| `x` seul | 505,6 |
| les trois directions | 260,8 |
| **`y` et `z`** | **208,5** |

Les cinq variantes entrelacées dans un même processus, témoin répété en
dernier (506,1). Sur le pas entier, A-B-A : **1291,2 → 988,1 → 1294,7 ms**.

**Ce qui coûtait n'était pas ce qu'on croyait.** La boucle interne lit
`ovl[ii, cx]` mille fois par particule et le compilateur la hisse tout seul —
la mettre en mémoire de groupe ne rend rien (505,6). Ce sont les boucles du
milieu et du dehors, `ovl[jj, cy]` deux cents fois et `ovl[kk, cz]` vingt,
qu'il refuse de hisser : il faudrait vingt registres vivants sur toute la
contraction. Mettre les trois directions est déjà moins bon (260,8) — les
valeurs de `x` font alors un aller-retour que le fichier de registres faisait
gratuitement.

**Et la forme fermée est une impasse.** [`smoothing_columns`](@ref), écrite et
testée pour remplacer ces lectures, greffée dans le noyau mesure **697,1 ms** —
pire que les tables, et plus de trois fois le coût de leur mise en cache.
Quinze `erf` et quinze `exp` par particule coûtent plus cher que soixante
lectures servies par le cache. Le micro-banc qui annonçait 90 ms mesurait
l'arithmétique seule et taisait ce qu'elle fait à la contraction autour d'elle.
La fonction reste : elle est exacte, c'est la référence contre laquelle les
tables sont vérifiées.

### Le dépôt fin : ×3,3 en donnant une colonne à chaque work-item

Aux compteurs, le noyau était **borné par l'ALU à 85,9 %** avec seulement
**12,8 % de F32** : six instructions émises pour une seule qui compte. La
mémoire, elle, dormait (lectures 1,9 %, MMU 1 %).

Un work-item possédait **un point** du stencil 8³ ; il possède désormais une
**colonne** de huit points, et le groupe passe de 512 à 64 :

    w = vy[jj] * vz[kk]                 # une fois pour huit points
    aᵢ = fma(vx[i], w, aᵢ)   i = 1…8

Neuf opérations flottantes au lieu de vingt-quatre, dix lectures en mémoire de
groupe au lieu de vingt-quatre, une itération de boucle au lieu de huit.

| | ms |
|---|---:|
| un work-item par point (512) | 196,6 |
| le même, boucle interne déroulée par quatre | 156,0 |
| un work-item par colonne (64) | 134,0 |
| **et la mise en scène par particules entières** | **58,9** |

⚠️ **La moitié du gain vient de la mise en scène, pas de la boucle interne.**
Elle parcourait des *valeurs* et retrouvait la particule et l'axe par quatre
divisions entières. Inoffensif quand 512 work-items en font six chacun ; ruineux
quand 64 en font quarante-huit. **Une taille de groupe n'est pas une décision
locale** : elle retarife tout ce qui se paie par work-item.

⚠️ Et l'étage de mise en scène suit : 128 particules étaient l'optimum à 512
work-items, **32** le sont à 64 (59,0 ms contre 84,4). Le tableau complet est
dans la docstring de `DEPOSIT_STAGE`.

Sur le pas entier, A-B entrelacé : **844,9 → 705,8 ms**, l'étage du dépôt
224,1 → **68,7**, et celui des forces immobile à 222,4 contre 223,0 — témoin
interne que rien d'autre n'a bougé. La densité s'accorde à 5,2e-07 en ponctuel,
soit l'ordre des atomiques.

### Le dépôt grossier : ×8,5 en cessant de fuir les collisions

**Fait le 21/09.** Le noyau fuyait les particules voisines depuis deux ans
parce que leurs huit atomiques se percutaient — trier par maille fine met des
particules consécutives dans la même maille grossière, 4874 ms contre 193. La
collision était réelle ; la conclusion ne l'était qu'à moitié.

Comptées plutôt que redoutées : **64 particules consécutives tombent dans cinq
mailles grossières et touchent 23,5 points de grille**. Leurs 512 atomiques sont
512 additions sur 23,5 adresses — une raison de les additionner avant de toucher
la grille, pas de les séparer. Un work-item prend donc 64 particules d'affilée,
les accumule dans une fenêtre 4³ de mémoire de groupe **qu'il possède seul**, et
la vide avec une atomique par point touché : 0,37 atomique par particule au lieu
de huit. Rien n'est atomique dans l'accumulation — la contention ne déménage pas
en mémoire de groupe, elle cesse d'exister, et le noyau n'a besoin d'aucun
support atomique là-bas, que Metal n'offre pas.

A-B entrelacé, chaque noyau portant ses propres constantes :

| dépôt grossier seul | une particule / work-item | tuile privée | |
|---|---:|---:|---:|
| RTX 4070, 8×10⁶ sur 130³ | 37,7 ms | **4,4** | **×8,5** |
| M1 Max, 8×10⁶ sur 130³ | 29,1 ms | **5,2** | **×5,6** |
| M1 Max, 8×10⁷ sur 258³ | 187,8 ms | **48,0** | **×3,9** |

et le pas entier sur la 4070, **57,3 → 38,0 ms**.

⚠️ **Ce sont les ablations qui ont tranché, pas les compteurs.** Remplacer les
atomiques par des additions simples — faux résultat, bonne forme — laissait
3,1 ms sur 29,2 sur la 4070 et 4,7 sur 29,0 sur le M1 Max : 89 % et 84 %, même
cause sur les deux fondeurs, donc **un noyau et non deux**. Les compteurs
nommaient la MMU parce que c'est ce que dépensait le noyau qu'on leur montrait ;
ils ne pouvaient pas nommer celui qui n'était pas écrit.

⚠️ **Le pas de dispersion s'est inversé, et j'ai failli livrer l'ancien.** Il
survit un étage plus haut — les *blocs* sont parcourus par un pas premier avec
leur nombre, sans quoi les vidanges voisines se percutent — mais son optimum
passe de 509 à **7**. Sept blocs de 64, c'est 448 particules : la même distance,
une autre unité. Garder le 509 hérité coûtait un facteur 2,2 sur Apple, sur un
noyau qu'on venait d'accélérer six fois. **Quand un noyau change, remesurer les
constantes dont il hérite.**

Les trois constantes (tuile 4³, bloc 64, groupe 32) sont les mêmes sur les deux
fondeurs. La tuile est une falaise et non une pente : un bloc de 64 tient dans
3³ dans 75,8 % des cas et dans 4³ dans 99,7 %, et le quart qui s'échappe coûte
19,3 ms contre 4,5 sur la 4070.

### L'histoire d'avant : ×2,1 en changeant un pas de 7919 à 509

Le noyau lit le nuage **dans le désordre**, par un pas premier avec le nombre
de particules, pour que ses huit atomiques ne tombent pas toutes sur la même
maille grossière (trié : 4874 ms). Le pas valait 7919, choisi « le plus grand
premier sous la main » sur l'idée que plus c'est dispersé, moins ça collisionne.

C'est vrai, et ce n'est pas le seul coût. Aux compteurs, le limiteur du noyau
est **la MMU à 70,7 %** — la traduction d'adresses — avec 15 % de défauts de
TLB : à cette distance, chaque work-item lit ses 48 octets dans sa propre page.
Les atomiques, elles, ne se voient pas (Buffer Write Limiter 0 %).

| pas | 1 | 31 | 127 | 251 | **383** | **509** | 1009 | 2003 | 7919 | 65537 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ms (2×10⁷ work-items) | 1148 | 132 | 42,7 | 24,2 | **22,9** | 23,4 | 26,5 | 34,5 | 48,9 | 57,9 |

Le fond est plat et les deux bords sont raides : trop près, les atomiques se
percutent ; trop loin, la MMU. À 8×10⁷ le noyau passe de **195,0 à 89,7 ms**
(A-B-A), et le pas de 841,9 à 745,9. 509 plutôt que 383 parce qu'au dixième du
nuage — 8×10⁶ particules, l'échelle des films — le creux se déplace un peu et
509 en est le fond.

⚠️ **Confiner la dispersion à une fenêtre** de particules consécutives, la
parade évidente au coût de traduction, est **bien pire** : 154,8 ms pour une
fenêtre de 4×10⁶ et 1009 ms pour une de 65 536, contre 89,7 pour le pas simple.
Dans une petite fenêtre les mêmes mailles grossières reviennent sans cesse, et
les collisions que le pas existe pour casser reviennent avec elles.

### Profiler un noyau : une minute, pas une heure

La recette est dans `docs/src/performance.md`. En bref : boucler **le seul
noyau visé** sur une fraction de ses cellules qui préserve le régime (20 000
sur 109 054 → 93 %), attacher `xctrace` **100 ms**, parser en flux **par
position**. Les compteurs sont stables à 1 % ; enregistrer plus ne donne rien
et coûte des gigaoctets.

⚠️ La boucle doit durer **bien plus** que l'enregistrement. Un premier essai a
capturé un GPU au repos : les compteurs lisent alors zéro à la médiane avec des
valeurs absurdes en queue.

### Impasses mesurées, à ne pas repayer

Détail chiffré dans `docs/src/performance.md`, section « Dead ends, measured ».

| tentative | mesure |
|---|---|
| recouvrir les deux dépôts sur deux files Metal | aucun recouvrement, le GPU les exécute bout à bout |
| permuter le nuage en deux tableaux séparés | 429,7 ms — deux lignes de cache par particule |
| lire `cols` au lieu de le recalculer dans les forces | 0 ms, entièrement recouvert |
| hisser les colonnes en registres | 570 ms, pire que l'original |
| déséquilibre de charge entre groupes | les cellules peu peuplées portent 0,09 % du travail |
| tri sur device **des clés** | ×1,17 — mais trier les *particules* gagne |
| **les colonnes en forme fermée** | **697,1 ms** contre 505,9 — l'arithmétique coûte plus que les lectures |
| mettre `x` aussi en mémoire de groupe | 505,6 puis 260,8 ms — le compilateur le faisait déjà |
| disperser le dépôt grossier **par fenêtres** | 154,8 ms contre 89,7 — les mailles reviennent dans la fenêtre |

⚠️ **Deux de ces mesures étaient justes et sont devenues fausses** parce que
leur prémisse avait bougé : le tri device « à ×1,17 » triait des clés, et le
placement à deux étages était l'optimum pour un nuage en ordre aléatoire. Avant
de rouvrir une impasse, vérifier ce qui a changé sous elle.

### Discipline de mesure

1. **Sur le vrai nuage**, jamais synthétique : les particules par cellule
   occupée (670 contre 28) ont inversé deux conclusions.
2. **Un accès mesuré isolément surestime ce que sa suppression rapporte** — le
   gather de `delta` coûtait 132 ms seul, sa suppression n'en a rendu que 98
   sur deux accès réunis, le reste étant recouvert par le calcul. Les coûts se
   mesurent isolément, les gains non.
3. **A-B-A entrelacé dans un seul processus.** Les comparaisons entre sessions
   ne valent rien : poisson et champ moyen ont varié d'un facteur 2 et 4 d'une
   session à l'autre sans qu'on y touche.
4. **Le REPL persistant de kaimon**, et `run_tests` pour la suite.

### Où passe le temps sur chaque carte, mesuré le 21/09

`scripts/profil_device.jl` — écrit ce jour-là, il manquait — rejoue le corps de
`_update_forces_resident!` et de `step!` appel par appel, avec un `synchronize`
entre les étages, et imprime un `step!` non synchronisé à côté de la somme pour
qu'on voie ce que la sérialisation coûte : 1 % sur Metal, −1 % sur CUDA, donc
les tables ferment.

Même nuage, même instant, 8×10⁶ particules sur 130³ :

| étage | RTX 4070 | % | M1 Max | % |
|---|---:|---:|---:|---:|
| empaquetage + tri | **14,2** | 29,8 | 13,2 | 18,2 |
| forces + projectile | 14,0 | 29,4 | **20,0** | 27,7 |
| Verlet | **6,9** | 14,4 | 3,8 | 5,2 |
| dépôt grossier (CIC) | 4,6 | 9,5 | 6,1 | 8,5 |
| dépôt fin | 2,8 | 5,9 | 8,0 | 11,0 |
| poisson! | 2,0 | 4,2 | 9,9 | 13,7 |
| champ moyen | 0,8 | 1,6 | 4,8 | 6,6 |
| csolf + csolc | 0,6 | 1,2 | 3,1 | 4,3 |
| csolc → hôte (PCIe) | 0,7 | 1,5 | 1,2 | 1,6 |
| **pas complet** | **48,1** | | **71,6** | |

**Le travail de grille est écrasé par la 4070** — `poisson!` ×4,9, le champ
moyen ×6,4, le dépôt fin ×2,8 : du FP32 dense, ce que cette carte fait de mieux.
**Le travail particulaire ne l'est pas** : le tri est *plus lent* sur la 4070 que
sur un M1 Max, et le Verlet 1,8× plus lent. Deux étages qui ne sont ni du calcul
ni de la grille, mais du déplacement de mémoire et des atomiques — exactement là
où la mémoire unifiée ne perd rien et où une carte discrète ne gagne rien.

### Le script d'entrée : un seul, trois options

**Fait le 21/09.** Le README proposait deux commandes qui se recouvraient —
`xenon.jl` qui simulait et imprimait une table, et `film_xenon.jl` qui simulait
de nouveau et dessinait. C'est un seul script : il fait la collision, imprime ce
qui se passe, et écrit le film. Options : `--particules`, `--nfine`,
`--cpu`/`--gpu`. Le fondeur est trouvé seul.

**Il dessine en GLMakie + `heatmap`** dès qu'un écran et le paquet le
permettent, ce que le script du film ne faisait pas : il dessinait en Cairo +
`contourf` sur un rééchantillonnage 350×350, et le chemin rapide n'existait que
dans un second script qui redessinait depuis un cache. **3,9 s contre 31,0**
pour les mêmes 176 images. C'est la primitive qui compte, pas le moteur : un
heatmap est un envoi de texture, une tessellation de contours est du travail
processeur dans Makie quel que soit le moteur — et Cairo ne sait pas dessiner
celui-ci, les points de collocation n'étant pas équidistants.

Trois défauts corrigés au passage :

* la commande « sans GPU » du README **ne pouvait pas marcher** (`--project=.`
  ne porte aucun Makie, volontairement) ; c'est `--project=viz`, dont le
  manifeste n'avait jamais reçu le paquet ;
* le cache était **relu dès que le fichier existait**, ce qui ignorait en
  silence `--particules` et redessinait le nuage de quelqu'un d'autre ;
* tous les temps du README **précédaient le dépôt tuilé**.

Remesuré machine au repos, la traversée complète à 8×10⁶ sur 130³ : **69,5 s**
sur M1 Max contre les 86,4 documentés, **53,5 s** sur la 4070. La commande
entière, film compris, fait 2 min 12 sur Apple et 3 min 31 sur dix cœurs.

⚠️ **La relecture des images est la seule ligne où les deux machines diffèrent
vraiment** : 176 coupes de densité et 176 lectures du nuage coûtent 6,8 s en
mémoire unifiée et **15,2 s** sur PCIe. Le pas lui-même, où rien ne traverse,
reste à l'avantage de la carte discrète.

### Ce qui reste, par ordre de rendement

1. **Le tri** — 30 % du pas sur la 4070, 18 % sur le M1 Max, et le premier
   poste de la carte NVIDIA. Il y est *plus lent en absolu* que sur un M1 Max
   (14,2 contre 13,2 ms) alors que la carte écrase l'Apple partout ailleurs :
   c'est du déplacement de mémoire, pas du calcul. L'histogramme puis le
   placement ; `docs/src/device.md` explique pourquoi le placement tient en une
   passe. L'empaquetage `(k, δ)` en `Float64` ne coûte plus rien sur ce chemin.
2. **Le noyau des forces**, 14,0 ms sur la 4070 et 20,0 sur le M1 Max — un bon
   quart du pas des deux côtés. Il n'a **jamais été repassé aux compteurs depuis
   la mise en mémoire de groupe** : ceux d'avant (Buffer Read 99 %) décrivent un
   noyau qui n'existe plus. La leçon du dépôt fin s'y applique peut-être telle
   quelle — sa contraction 10³ relit `cols[·, cx]` dix fois par point, exactement
   le motif que le blocage par colonne a supprimé ailleurs.
3. **Le Verlet, et c'est une anomalie** : 6,9 ms sur la 4070 contre 3,8 sur un
   M1 Max, soit 1,8× *plus lent* sur la carte deux fois plus rapide partout
   ailleurs. Une carte n'a pas de raison d'être deux fois plus lente sur une
   boucle aussi simple ; quelque chose n'y est pas mesuré. Peu de lignes à lire,
   donc bon rapport.
4. Sur la 4070, **la relecture des images du film coûte 15,2 s contre 6,8** en
   mémoire unifiée. Hors du pas, mais c'est un cinquième de la commande que le
   README propose, et personne n'a regardé si les 176 transferts peuvent être
   groupés.
3. ~~Le dépôt grossier, à 92 ms, est toujours borné par la MMU~~ — **fait le
   21/09**, et pas par le chemin annoncé. « Il faudrait changer la disposition
   du nuage, pas un paramètre » était faux des deux côtés : le nuage est
   disposé exactement comme avant, c'est le *noyau* qui a changé. Une tuile
   privée par work-item, 0,37 atomique par particule au lieu de huit :
   **48,0 ms contre 187,8** à 8×10⁷, **×8,5** à 8×10⁶ sur une RTX 4070. Les
   compteurs avaient nommé la MMU parce que c'est ce que le noyau qu'on leur
   montrait dépensait ; l'ablation, elle, a dit que les atomiques valaient
   84 à 89 % — même cause sur les deux fondeurs, donc un noyau et non deux.
5. Le budget énergétique, encore particule par particule sur l'hôte et jamais
   profilé à 8×10⁷.

⚠️ L'ablation qui donnait 110 ms « sans aucune lecture de table » date du noyau
d'avant, et ne borne plus rien : le noyau actuel fait 198 ms avec ses lectures
en mémoire de groupe. Re-mesurer avant de raisonner dessus.

⚠️ Les rangs ci-dessus viennent du profil à **8×10⁶** particules, l'échelle du
film. À 8×10⁷ les proportions bougent — c'est là que le dépôt grossier ne rend
plus que ×3,9 au lieu de ×8,5 — donc reprendre `scripts/profil_device.jl` à la
taille visée avant de choisir.
