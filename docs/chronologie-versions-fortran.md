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
| **1998-01-05** | **5345** | **90** | **`majrel2/lucifer/vlas.f`** | ⭐ **cible : dernière de production** |
| 1998-01-07 | 5297 | 91 | `majrel2/lucifer/initial/vlas.f` | branche `initial/` — **sans projectile** |

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

## Quelle version est la cible

La plus récente par la date n'est pas la bonne. **`lucifer/initial/vlas.f` (1998-01-07)
n'appelle pas `initpro`** : son programme principal n'a plus de projectile. C'est une
expérience de relaxation — d'où le répertoire `initial/`, un `qpold=0.0` après
l'initialisation, un terme de friction `kconv = -1e-5` ajouté aux forces dans `move`, et
un terme cinétique de Thomas-Fermi `½(3π²)^{2/3}ρ^{2/3}` ajouté à `pspech`.

**La cible est `lucifer/vlas.f` (1998-01-05)** : dernière version de la lignée principale
qui fasse encore une collision (`initpro`, `incproj`, `force2g`, `enerele2g`).

### Ce qui sépare la version portée (1997-06-06) de la cible (1998-01-05)

61 routines identiques, 15 modifiées, 16 ajoutées, **aucune supprimée**. Le détail :

**Change la physique**

| Quoi | Effet |
|---|---|
| `makeinit` appelle **`initialise4`** au lieu de `initialise` | échantillonnage par **rejet dans l'espace des phases 6D** : on tire `r = rmax·x₁^{1/3}`, `p = pmax·x₄^{1/3}` et on accepte si `p²/2 + V(r) < E_F`. C'est la distribution de Thomas-Fermi exacte, là où l'ancienne inversait un profil radial tabulé (`hm1.dat`, `rhoinit.dat`). |
| `griech` : `nbprem` **10 → 2** | change la grille d'échantillonnage `gtech` |
| `initialise` : `integer rmax` → **`real*8 rmax`** | corrige la coquille n°4 |
| `initialise` : `sqrt`/`cos`/`sin` → `dsqrt`/`dcos`/`dsin` | ⚠️ **aucun effet** : `SQRT` est générique en F77, donc identique sur `real*8` (vérifié). Style seul. |
| `pspech2` : `rr` sort du `if (rho > 1e-7)` | avant, `rr` gardait la valeur du point précédent quand la densité était négligeable |

**Nouveau paramètre d'entrée `rcmax`**, lu en fin de `vlas.inp` (deux lignes de plus) et
passé à `move`, `enerele2g`, `enertot2g`, où il remplace le `100.d0` codé en dur : c'est
le rayon au-delà duquel un électron est compté comme sorti. `vlas.inp` de production le
documente comme « rayon considéré comme inner cluster ».

**Diagnostics seuls** — `multrcmax` (nombre d'électrons dans des sphères de 50 à 100 a₀,
écrit dans `rcm.dat` : *malgré son nom, rien à voir avec le rayon de coupure du
projectile*), `mkpotrad`/`mkpotradx`/`mkrhoradx` (profils radiaux), `mkdensene`/
`distene`/`denseta` (distributions en énergie), `sortietest`/`sortietest2`, et le couple
`litpotexa` + `enertot2gix` qui rejoue le bilan d'énergie avec un potentiel radial
externe. `angular`, `echanti2`, `griech` passent de `0:100` à `0:NBGEM` (constante).

**Code mort** — `pspech3` n'est appelé nulle part. Il calcule pourtant la **densité
d'énergie** d'échange-corrélation (Dirac avec son facteur ¾, et l'expression complète de
Gunnarsson-Lundqvist) plutôt que le **potentiel**, et corrige le double comptage de
Hartree par `csol ← ½csol + echsol`. C'est exactement la correction que réclame la
coquille n°8 : l'auteur l'avait écrite sans la brancher.

`ceq3d.f` change aussi : `NHFX` passe de **28 à 32**, `npartmax` à 3 000 000, et `NBGEM`
apparaît.

### Le verrou : `pot.dat`

`initialise4` et `litpotexa` lisent un fichier `pot.dat` — en-tête `nbgrid`, puis
`rmax pmax Ef`, puis `(r, ·, V(r))` — qu'**aucune version de `vlas.f` n'écrit**. Il est
produit par `mkpotradx` (sous le nom `potrad.dat`) lors d'un run précédent : c'est une
boucle d'amorçage auto-cohérente. **Aucun exemplaire n'a survécu dans l'archive.**

`ref/fortran98/pot.dat` est donc une **reconstruction**, pas l'original. Elle est exacte
pour ce que le fichier sert à faire : le test de rejet `p²/2 + V(r) < E_F` équivaut à
`p < p_F(r)`, donc poser `V(r) = E_F − p_F(r)²/2` avec `p_F = (3π²ρ)^{1/3}` reproduit la
distribution de Thomas-Fermi voulue, et `E_F` s'élimine. `ρ(r)` vient de `rhoinit.dat`,
le profil d'équilibre que l'ancienne initialisation utilisait déjà.

## Où sont les runs de production

`arkonnen/vlasov/vlas.inp` — à côté de la source portée — est le fichier **de production
proton** : il ne diffère de `ref/fortran/vlas.inp` que par deux lignes, **800 000
pseudo-particules** (au lieu de 20 000) et **4000 pas** (au lieu de 2). Grille, domaines,
cutoff, énergie, pas de temps sont identiques. Le nôtre était un fichier de test.

`arkonnen/launch/` contient un balayage de douze entrées au **format à 71 lignes**, celui
de la version parallèle : projectile de masse 236 864 u.a. (≈ Xe), charge 25, 1 310 720
particules, grille 32/63, domaines 120/300, `cutoff = 5.0`, `rcmax = 45`. Les suffixes
`02` et `08` sont les vitesses : √(2E/m) = 0,197 et 0,788 u.a. Les paramètres d'impact
balayent 30 à 80 a₀. **Les runs Xe²⁵⁺ de la thèse ont donc tourné sur le code HPF.**

`temp/…/lucifer/Eloss/` conserve les **sorties archivées** : `Em1q1e002i000.dat` se lit
masse 1, charge 1, 2 keV, impact 0 — exactement le nom que produit notre oracle 98. Voir
[`validation-chapitre6.md`](validation-chapitre6.md) : ces trajectoires archivées sont
une référence bien plus sûre qu'une lecture de figure.

## Ordre de travail

1. **`rcmax` en paramètre d'entrée** — mécanique, sans risque.
2. **`initialise4`** — le vrai changement de physique, et le seul qui déplace `dE/dx`.
3. **Les corrections silencieuses** (`dsqrt`, `rr` dans `pspech2`, `nbprem`) — à mesurer
   une par une contre l'oracle 98.
4. **`pspech3`**, à brancher ou non : c'est la correction de la coquille n°8, mais
   l'auteur ne l'a pas branchée. La reproduire veut dire la laisser morte.
5. **Le parallélisme**, depuis `mystuffgz/vlas.hpf` lu comme spécification : découpage
   `(block,*,*)` sur la densité contre `(*,*,block)` sur le potentiel, tri à deux niveaux
   `grostri` → `petittri`. Ne change aucun résultat, seulement le temps de calcul.
