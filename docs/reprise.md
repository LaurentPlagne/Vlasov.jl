# Reprise — où en est le projet

Point de départ pour une session fraîche. Les détails sont dans les documents
cités ; ce fichier dit seulement **où regarder** et **quoi faire ensuite**.

## En une phrase

Le code Fortran de la thèse (1996-1998) est porté en Julia, validé contre un
oracle reconstruit, **la courbe de freinage de la thèse est reproduite**, et le
pas de temps a été accéléré **×3,95** (307,9 → 77,9 ms à 800 000 particules).

## Les quatre documents à lire

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

Données publiées de la thèse : [`ref/these/`](../ref/these/), retrouvées dans le
répertoire de travail xmgr. Rejouer : `julia --project=. -t auto scripts/figure53.jl`
(~11 min, 800 000 particules).

## Comment exécuter

    julia --project=. -e 'include("test/runtests.jl")'     # 4750 tests, ~20 s
    julia --project=. -t auto scripts/profil_pas.jl        # profil d'un pas
    julia --project=gpu -t auto scripts/bench_gpu.jl       # CPU vs GPU

**Deux environnements.** Le principal n'a pas de GPU ; `gpu/` porte `Metal`
(dépendance **faible**) et `AppleAccelerate`. Pour le REPL kaimon, activer
`gpu/` dans la session.

⚠️ **Sur Apple Silicon, charger `AppleAccelerate` — ×1,31 pour une ligne**, et
pas seulement sur les GEMM : les boucles particulaires gagnent 15 à 25 % parce
que le pool de fils d'OpenBLAS cesse de leur disputer le processeur.

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

## En cours : traduction des commentaires en anglais

Demandé par l'auteur. ~6760 lignes sur 29 fichiers, et **rien n'est mécanique** :
presque chaque commentaire porte un fait mesuré ou un piège, à rendre avec sa
nuance. Procéder fichier par fichier, en committant chacun.

**Fait** : `gpu.jl`, `threading.jl`, `random.jl`, `collocation.jl`, `energy.jl`,
`sorting.jl`, `meanfield.jl`, `particles.jl`, `mesh.jl`, `tensorsolver.jl`, `deposition.jl`, `initial.jl`, `projectile.jl`, `simulation.jl`, `poisson.jl`.

**Reste**, par taille croissante (`src/` puis `ext/`) :

`splines.jl` 385, `VlasovMetalExt.jl` 523.
Puis `scripts/` et `test/`.

Vérifier après chaque fichier dans la session kaimon (Revise recharge, pas de
démarrage Julia) ; lancer la suite complète tous les quelques fichiers.

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
