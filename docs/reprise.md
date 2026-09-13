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

### Performance (rendement décroissant)

1. **Le noyau des forces (≈13 ms)** — contraction 10³ bornée par la mémoire. Les
   particules sont **déjà triées** pour le dépôt ; les faire lire `csol` dans cet
   ordre donnerait la localité qui vaut ×1,34 au CPU. Le tri est là, il suffit de
   s'en servir. C'est la piste la plus prometteuse.
2. Écrire directement dans les tampons partagés (`unsafe_wrap`) — gain mince,
   mais supprime les tampons hôtes et la moitié du code de transfert.
3. Le dépôt grossier (6,7 ms) et le champ moyen (5,5 ms), encore sur CPU.

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
