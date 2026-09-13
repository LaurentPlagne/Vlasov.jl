# Chronologie des versions du code Fortran

Une quarantaine de `vlas.f` coexistent dans `~/these_postdoc/these`, sans historique de
version. Ce document les ordonne et dit **vers laquelle le portage Julia doit converger**.

## Comment les dates ont été retrouvées

L'arborescence a été restaurée en janvier 2025 : presque tous les fichiers portent cette
date, qui ne dit rien. **Une seule sous-arborescence a conservé ses `mtime` d'origine** —
`temp/home/sauron2/plagne/` — parce que ses fichiers y sont restés compressés (`.gz`) et
n'ont pas été réécrits. Elle fournit 43 versions datées de 1996-07 à 1998-01.

Les copies non datées (`arkonnen/…`) ont ensuite été rattachées par **empreinte du contenu
normalisé** (espaces retirés), puis, à défaut, par similarité de diff ligne à ligne. Deux
tombent exactement sur un original daté ; les autres se placent à ±2 semaines.

Trois sources indépendantes recoupent les dates tardives, que `temp/` ne couvre pas :
les `mtime` internes des archives `.tar` (que `tar` préserve), les mesures de
J.-Y. Berthou (`arkonnen/jyb/`, juillet 1997) et la date de soutenance — **18 décembre 1998**.

> ⚠️ Piège de détection : le motif `subroutine\s+(…|sort|tri)` capture `sortie`,
> `sortierho`, `sortietemps` — les routines de **sortie de fichiers**. Un premier passage
> a conclu « tri parallèle présent dans les 43 versions », ce qui était faux dans toutes.
> Le tri se détecte par les noms réels : `tri`, `rajtri`, `sort4b`, `indexxb`,
> `grostri2g`, `petittri2g`.

## La lignée principale (physique)

Versions datées, sous `temp/home/sauron2/plagne/it8/ttt/` sauf mention contraire.

| Date | Lignes | Sub. | Chemin | Apport |
|---|---:|---:|---|---|
| 1996-07-02 | 3439 | 82 | `vlas.f` | plus ancienne conservée |
| 1996-08-09 | 2674 | 63 | `multip/sphere/vlas.f` | multipôles, sphère |
| 1996-09-06 | 2734 | 63 | `majrel/vlas.f.sodium` | sodium |
| 1996-09-18 | 2782 | 63 | `majrel/vlas.f.c60` | branche C₆₀ |
| 1996-10-27 | 3622 | 75 | `majrel2/double/vlas.f` | passage double précision |
| 1996-12-11 | 3604 | 74 | `mkflux/vlas.f` | flux |
| 1997-01-31 | 4294 | 78 | `majrel2/lucifer/keep/vlas.f.test` | |
| 1997-02-21 | 4053 | 77 | `majrel2/vlas.f` | |
| **1997-06-06** | **4133** | **75** | **`majrel2/sauron/vlas.f.keep`** | ⭐ **version portée en Julia** |
| 1997-06-17 | 4746 | 82 | `statistique/vlas.f` | statistiques |
| 1997-09-17 | 4600 | 83 | `majrel2/lucifer/self/vlas.f.ng` | énergie propre (`enertot2gi`) |
| 1997-11-06 | 5045 | 87 | `majrel2/lucifer/vlas.f.echcorr` | échange-corrélation, potentiel radial |
| 1997-12-29 | 5345 | 90 | `majrel2/lucifer/vlas.f.29-12-97` | |
| 1998-01-05 | 5345 | 90 | `majrel2/lucifer/vlas.f` | |
| **1998-01-07** | **5297** | **91** | **`majrel2/lucifer/initial/vlas.f`** | ⭐ **dernière séquentielle** |

### Rattachement des copies `arkonnen/`

| Copie | Rattachée à | Preuve |
|---|---|---|
| `vlasov/vlas.f` **(portée)** | 1997-06-06 `sauron/vlas.f.keep` | diff 0.999 |
| `lindhard/vlas.f` | idem (octet pour octet) | empreinte identique |
| `vlasov/lu/vlas.f` | 1997-03-05 `lucifer/vlas.f.3g` | **empreinte identique** |
| `vlasov/pghpf2/vlas.f` | 1997-06-17 `statistique/vlas.f` | **empreinte identique** |
| `t3e/new/vlas.f` | ≈ 1997-06-17 | diff 0.984, même jeu de routines |
| `vlasov/pghpf/vlas.f` | ≈ 1997-06/08 | diff 0.976, +2 routines |

## Les branches parallèles — et le fait central

**Aucune branche parallèle n'a été fusionnée dans la lignée séquentielle.** La version
séquentielle la plus récente (1998-01-07) ne contient ni tri, ni HPF, ni MPI. Trois
tentatives ont divergé puis ont été abandonnées comme branches :

* **`lu/` — 3 grilles, mars 1997.** 5631 lignes, 92 routines, des variantes `…3g` et des
  routines `…p`. Jamais reprise : la lignée de juin 1997 repart de 4133 lignes.
* **`sauron/sort/` — août 1997.** Premier tri séquentiel (`tri`, `rajtri`, `sort4b`,
  `indexxb`, `makerhogt`). Disparaît totalement des versions de décembre 1997 et janvier 1998.
* **`pghpf/`, `t3e/` — 1997.** Portages Cray T3E et PGI-HPF du **seul solveur de Poisson**
  (`poisson2.hpf`, `tensrus.f`), pas du code complet.

### La vraie version finale : `arkonnen/mystuffgz/vlas.hpf`

C'est **l'aboutissement du travail de parallélisation**, et elle n'est pas dans `temp/` :

* `program poisson`, `use mpi_library`, `USE HPF_LOCAL_LIBRARY` ;
* **217 directives HPF** (`distribute (block,*,*)` sur `rho`, `(*,*,block)` sur `phi`,
  `(*,block)` sur les particules `qp`/`qpold`) et 14 appels MPI ;
* des enveloppes écrites à la main : `mpi_tri`, `mpi_force`, `mpi_potentiel`,
  `mpi_incproj`, `mpi_rajoute{,b,g}` ;
* **le tri parallèle à deux niveaux** : `grostri2g`/`grostrib` (grille grossière) puis
  `petittri2g`/`petittrib` (grille fine) — le tri pour la localité des données ;
* 54 routines seulement : c'est un code **réduit au noyau**, sans les diagnostics.

Sa datation : elle contient `mkpotrad` et `pspech3`, apparus en novembre 1997 / janvier
1998, et appelle `mpi_tri`, qui s'appuie sur le `sort4b.f` daté du **15 décembre 1997**
dans `sort.tar` (venu d'un collègue, `requena`). Elle est donc **postérieure à
1998-01-07**, dans l'année de la soutenance. `speedup.txt`, à côté, donne la mesure :
**16 PE, 2471 MFLOPS**, soit 154 MFLOPS/PE.

`arkonnen/stuff/vlas.hpf` (2447 lignes, 30 routines) en est une variante antérieure et
plus courte ; `mpi_stuff.hpf` y isole les enveloppes MPI.

> Une différence à ne pas manquer : `vlas.hpf` déclare 581 tableaux en `real` **par défaut**
> et zéro en `real*8`. La branche parallèle est en simple précision (ou compilée avec une
> promotion implicite), là où la lignée séquentielle est passée en double dès octobre 1996.

## Chronologie synthétique

```
1996-07 ─ 1996-10   mise en place, simple précision
1996-10            double précision
1996-12 ─ 1997-02   flux, multipôles
        1997-03    ├─ branche « 3 grilles » (lu/)            ✗ abandonnée
1997-05 ─ 1997-06   ★ version portée en Julia (1997-06-06)
        1997-06    ├─ branche T3E / PGI-HPF (solveur seul)   ✗ partielle
        1997-08    ├─ branche tri séquentiel (sauron/sort/)  ✗ abandonnée
1997-09 ─ 1997-11   énergie propre, échange-corrélation, potentiel radial
1998-01-07          ★ dernière version séquentielle complète
   1998            ★★ vlas.hpf : HPF + MPI + tri parallèle 2 niveaux (16 PE)
1998-12-18          soutenance
```

## Conséquence pour le portage Julia

La cible n'est **pas une version unique** — les deux dernières sont sur des branches
disjointes, et il faut prendre à chacune ce qu'elle apporte :

1. **Physique → `lucifer/initial/vlas.f` (1998-01-07).** Elle ajoute 16 routines à la
   version portée : énergie propre (`enertot2gi`, `enertot2gix`), potentiel radial
   (`mkpotrad`, `mkpotradx`, `mkrhoradx`), distributions en énergie (`distene`,
   `mkdensene`, `denseta`), `pspech3`, et un jeu d'initialisation refait (`initialise4`,
   `testinit`, `litpotexa`). C'est là que se trouve l'état de la physique au moment de la
   rédaction — donc l'état auquel correspondent les figures de la thèse. **Piste à
   instruire pour l'écart `dE/dx`** : `pspech3` et `multrcmax` touchent au rayon de coupure
   du projectile, que vous identifiez comme le paramètre clé du pouvoir d'arrêt.
2. **Parallélisme → `mystuffgz/vlas.hpf`.** Non pas à porter, mais à lire comme
   spécification : le découpage `(block,*,*)` sur la densité contre `(*,*,block)` sur le
   potentiel dit quelles transpositions étaient jugées nécessaires, et le tri à deux
   niveaux `grostri` → `petittri` donne la granularité retenue pour la localité.
   L'équivalent Julia du tri n'existe pas encore ; il n'est pas urgent tant que le dépôt
   reste seul (`Threads.@threads` + un tableau de densité par fil suffit), mais c'est la
   référence quand la grille et le nombre de particules augmenteront.

Ordre raisonnable : d'abord (1), qui peut expliquer un écart physique réel ; (2) ne
change aucun résultat, seulement le temps de calcul.
