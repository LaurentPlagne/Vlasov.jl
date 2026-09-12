# Le code Fortran de référence — l'oracle

Le code de simulation de la thèse (Fortran 77, 1996-1999), remis en état de
marche sur macOS arm64. Il ne sert pas à produire de la physique : il sert
d'**oracle** au portage Julia. Chaque étage porté est comparé, sur les mêmes
entrées, aux tableaux que celui-ci produit.

Sans lui le portage se ferait à l'aveugle, et une erreur de signe dans un
opérateur spline rendrait des potentiels faux mais parfaitement plausibles.

## Utilisation

```
make            # construit `vlas`, le code de référence tel quel
make oracle     # construit la version instrumentée et produit les dump_*.bin
```

`make oracle` écrit des tableaux binaires bruts (un `Int32` de taille, puis
des `Float64`). Convention de nommage : `dump_*` pour la **grille fine**,
`dumpb_*` pour la **grille grossière étirée**.

| Fichier | Contenu |
|---|---|
| `dump_gx.bin`, `dump_gtx.bin` | nœuds, points de collocation |
| `dump_sx.bin`, `dump_s2x.bin` | matrices de collocation `S`, `S″` |
| `dump_dex.bin` | opérateur `S″·S⁻¹` après conditions de Dirichlet |
| `dump_lxr.bin`, `dump_mx.bin` | valeurs et vecteurs propres |
| `dump_psx.bin`, `dump_pxx.bin`, `dump_px2.bin` | moments `∫φ`, `∫xφ`, `∫x²φ` |

⚠️ `static` est appelée **deux fois** (grille fine puis grossière). Sans la
distinction `dump_`/`dumpb_`, le second appel écrase le premier — piège dans
lequel ce portage est tombé une fois.

## Ce qui a été changé, et pourquoi

### `modernize.patch` — 17 lignes, pour que ça compile et tourne

Appliqué en amont ; `vlas.f` ici est déjà corrigé. Le patch documente l'écart
avec l'original de la thèse.

| Problème | Correctif |
|---|---|
| `force2gi` appelée avec un argument de trop (code mort, via `incproj2`) | retrait de `liste2` |
| Formats `'(I,3e15.7)'` et `'(3I)'` refusés par gfortran | `I10`, `3I10` |
| `rmax` déclaré `integer` alors que `rhoinit.dat` contient `35.0000` | `real*8` |
| Chemins de sortie codés en dur `/home/tempo3/plagne/` | `out/` |

⚠️ Le troisième est un **bug latent** que les compilateurs de 1996 toléraient.
D'autres dorment sans doute encore : tout écart oracle/Julia doit être arbitré
— bug d'origine, ou erreur de portage ? — jamais corrigé en silence.

### `f02agf_shim.f` — la seule dépendance NAG

Le code liait `f047.a`, un extrait de la bibliothèque NAG. Une seule routine
en était réellement utilisée, `f02agf` (valeurs et vecteurs propres d'une
matrice réelle générale) ; `inverse` appelait déjà LAPACK directement.

L'archive livrée est un `.a` i386 de 1996, inutilisable sur arm64. Le shim
réexprime `f02agf` en `dgeev`, en dépliant les paires conjuguées au format
attendu (parties réelle et imaginaire séparées).

En pratique le spectre de l'opérateur spline est **réel** : le code d'origine
calculait `mxi` et `lxi` puis les jetait. Le portage Julia vérifie cette
propriété à la construction plutôt que de la supposer.

### `instrument.patch` — les dumps

Appliqué par le `Makefile` pour produire `vlas_dump.f`. Sonde `static` juste
après la diagonalisation et après le calcul des moments. Séparé du reste pour
qu'on voie exactement ce que l'instrumentation ajoute — et qu'on puisse en
ajouter d'autres sans confondre avec les corrections de portage.

## Le binaire d'origine

`vlas`, tel que livré dans l'arborescence de la thèse, est un ELF 32 bits
i386 de 1996. Il n'est pas exécutable ici et n'a pas été conservé.
