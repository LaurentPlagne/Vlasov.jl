# Oracle « version tardive » — `vlas.f` du 1998-01-05

Second oracle, à côté de [`../fortran/`](../fortran/) qui reste celui de la version
portée (1997-06-06). Source :

    temp/home/sauron2/plagne/it8/ttt/majrel2/lucifer/vlas.f.gz

C'est **la dernière version séquentielle de la lignée principale qui fasse encore une
collision**. La plus récente par la date (1998-01-07, `lucifer/initial/`) n'appelle pas
`initpro` : c'est une expérience de relaxation sans projectile. Voir
[`../../docs/chronologie-versions-fortran.md`](../../docs/chronologie-versions-fortran.md).

## Construire et lancer

    make          # -> vlas98
    ./vlas98      # lit vlas.inp, écrit Eloss/Em1q1e002i000.dat

`modernize.patch` (12 lignes) montre exactement ce qui a été changé pour que gfortran
accepte le source : descripteurs de format `I` et `3I` sans largeur, et deux chemins
absolus `/home/tempo3/plagne/` devenus `out/`. Cinq lignes de moins que le patch de 1997,
parce que la version 1998 corrige elle-même le `integer rmax` (coquille n°4) — voir
[`../../docs/coquilles-fortran.md`](../../docs/coquilles-fortran.md).

`ceq3d.f` est celui de 1998 : `NHFX = 32` (au lieu de 28), `npartmax = 3 000 000`, et la
constante `NBGEM = 1000` qui remplace les `0:100` codés en dur.

## ⚠️ `pot.dat` est une reconstruction

`initialise4` et `litpotexa` lisent un `pot.dat` qu'**aucune version de `vlas.f`
n'écrit** : il est produit par `mkpotradx` (sous le nom `potrad.dat`) lors d'un run
précédent, dans une boucle d'amorçage auto-cohérente. Aucun exemplaire n'a survécu dans
l'archive — et sans lui le programme s'arrête au démarrage.

Celui d'ici est reconstruit, et exact pour l'usage qu'en fait le code : le test de rejet

    p²/2 + V(r) < E_F

équivaut à `p < p_F(r)` dès lors que `V(r) = E_F − p_F(r)²/2`. Avec
`p_F = (3π²ρ)^{1/3}` tiré de `rhoinit.dat` — le profil d'équilibre que l'ancienne
initialisation utilisait déjà — on retrouve la distribution de Thomas-Fermi voulue, et
`E_F` disparaît de l'inégalité (on le pose à zéro). Le script tient en quelques lignes ;
il est dans [`make_pot.py`](make_pot.py).

**Ne pas traiter `pot.dat` comme une donnée d'origine.** Toute comparaison qui en dépend
— c'est-à-dire tout ce qui passe par `initialise4` — porte cette hypothèse.

## `vlas.inp`

Celui de `../fortran/`, plus deux lignes : `rcmax`, nouveau paramètre de 1998, le rayon
au-delà duquel un électron compte comme sorti (auparavant `100.d0` en dur).
