# Coquilles et anomalies du code Fortran d'origine

Relevé des écarts constatés dans `vlas.f` pendant le portage. **Aucun n'est
corrigé** : le portage les reproduit à l'identique, faute de quoi la
comparaison à l'oracle perdrait sa valeur — c'est elle qui détecte les erreurs
de portage, et elle ne le peut que si les deux codes calculent la même chose.

Ils sont arbitrés **au fur et à mesure que les runs deviennent
reproductibles**, en mesurant ce que chaque correction change sur les
observables — jamais en discutant ce qu'elle devrait changer.

**État de l'arbitrage :** les anomalies 1 et 2 sont tranchées, sur l'agrégat
isolé (voir la section suivante) — aucune n'est distinguable du bruit
d'échantillonnage. Les 3, 6 et 8 restent ouvertes ; la 8 demande d'abord
d'élucider l'enchaînement des appels du Fortran.

> ⚠️ Un écart entre l'oracle et le portage doit toujours être arbitré — bug
> d'origine, ou erreur de portage ? — jamais corrigé en silence. Ce fichier
> est la trace de cet arbitrage.

## Arbitrage des anomalies 1 et 2 — mesuré

Depuis que la simulation de l'agrégat isolé tourne, les deux coquilles qui ont
une variante corrigée peuvent être **mesurées** au lieu d'être discutées.

**Protocole.** Agrégat de 196 électrons, 10 000 pseudo-particules, 30 pas à
`dt = 1 u.a.`. On compare quatre configurations — référence, `moveback2`
corrigé, `ran2` corrigé, les deux — sur l'énergie cinétique initiale,
l'énergie totale initiale et l'amplitude de dérive de l'énergie.

**La précaution qui compte.** Corriger `ran2` change le tirage : l'écart
observé pourrait n'être que du bruit d'échantillonnage. On établit donc
d'abord cette barre, en relançant la configuration de référence sur **cinq
graines** :

| grandeur | moyenne | σ (5 graines) |
|---|---|---|
| énergie cinétique initiale | 13,09 | 0,121 |
| énergie totale initiale | −23,75 | 0,160 |
| amplitude de dérive | 5,5e−5 | 1,1e−5 |

**Résultat.** Chaque correction, rapportée à cette dispersion :

| effet | en unités de σ |
|---|---|
| `moveback2` sur l'énergie cinétique | 0,06 σ |
| `moveback2` sur l'énergie totale | 0,18 σ |
| `moveback2` sur l'amplitude de dérive | 0,71 σ |
| `ran2` sur l'énergie cinétique | 0,27 σ |
| `ran2` sur l'amplitude de dérive | 0,33 σ |

**Conclusion.** Aucune des deux ne produit un effet distinguable du bruit
d'échantillonnage. Les résultats de la thèse ne sont pas remis en cause par
ces deux coquilles.

Pour `moveback2`, c'était prévisible une fois chiffré : l'écart de coefficient
ne déplace les particules que d'environ `0,03 u.a.` à l'amorçage, contre un
rayon d'agrégat de `23 u.a.`

⚠️ **Portée de cette conclusion.** Elle vaut pour *cette* configuration :
agrégat isolé, 30 pas, 10 000 particules. Elle ne dit rien des runs du
chapitre 6, où un projectile unique suit **une** trajectoire — le bruit
d'échantillonnage n'y joue pas le même rôle, et une erreur d'amorçage peut s'y
voir davantage. Ni des runs longs, où un biais sous le bruit peut s'accumuler.
À refaire sur ces cas-là le moment venu.

---

## Vue d'ensemble

| # | Où | Nature | Portée | Correction disponible |
|---|---|---|---|---|
| 1 | `moveback2` | `dltt*2` pour `dltt**2` | **mesurée : 0,06–0,71 σ** | `consistent = true` |
| 2 | `ran2` | `IQ1 = 3668` pour `53668` | **mesurée : 0,27–0,33 σ** | `consistent = true` |
| 3 | `maketaint` | supports d'intégration tronqués | champ lissé, ~1e-5 | — |
| 4 | `initialise` | `rmax` entier lisant un réel | lecture de `rhoinit.dat` | corrigée (obligatoire) |
| 5 | `force2gi` | appel avec un argument de trop | code mort | corrigée (obligatoire) |
| 6 | `makerhsf` | multipôles calculés puis jetés | temps de calcul, bruit | — |
| 7 | `ceq3d.f` | `π` tronqué à 12 décimales | ~1e-12 partout | non reproduite |
| 8 | `pspech2` | `rr` périmé dans la branche `ρ ≤ 1e-7` | **mesurée : ‖csol‖ 6,3 → 704,4** | corrigée en 1998-01-05 |
| 9 | `docapture` | adoucissement différent d'`incproj` | énergie de compte rendu | — |
| **10** | **`forceproji`** | **boule uniforme là où la thèse pose une gaussienne** | **`dE/dx` ×1,3** | `GaussianSoftening` |

Les points 4 et 5 sont corrigés dans `modernize.patch`, sans quoi le code ne
compile pas ; voir [`ref/fortran/README.md`](../ref/fortran/README.md).

Les points 1, 2 et 3 sont présents **à l'identique dans les cinq versions** du
code de la thèse (`arkonnen/vlasov`, `lu`, `pghpf`, `pghpf2`, `t3e/new`,
`lindhard`), et survivent jusqu'à la dernière (1998-01-05). Ce ne sont donc pas
des accidents de copie : ils ont traversé tout le développement sans être vus.

Le point 8 fait exception : l'auteur l'a corrigé début 1998. C'est la seule
anomalie de cette liste dont on sache qu'elle a été vue.

---

## 1. `moveback2` — formule non homogène

```fortran
coef2 = 0.5d0*dltt*2*dfloat(npart)/(mel*nbelec)      ! = dt/M
qpold(1,i) = -qp(1,i) + 2.d0*qpold(1,i) + coef2*fp(1,i)
```

**Le problème.** `coef2·F` a la dimension d'une *vitesse*, ajoutée à des
longueurs. Un développement de Taylor donne `dt²/4M` :

```
q(−dt) = 2·q(−dt/2) − q(0) + (dt²/4M)·F
```

Tout indique une coquille `dltt*2` pour `dltt**2` — d'autant que la routine
voisine `move`, qui fait le même genre de calcul, écrit bien `dltt**2`. Même
ainsi, il resterait un facteur 2 d'écart avec Taylor.

**Portée.** L'amorçage seul. `moveback2` ne sert qu'une fois, pour fabriquer
la position à `t = −dt` qui démarre le schéma de Verlet. L'effet est celui
d'une erreur sur la vitesse initiale, pas d'un biais entretenu. Avec
`dt = 1 u.a.`, le coefficient vaut `1/M` au lieu de `0.25/M`, soit un facteur
4.

**Ce que fait le portage.** `full_step_back` reproduit `dt/M`.
`full_step_back(…; consistent = true)` applique `dt²/4M`, et un test vérifie
que cette variante-là reproduit exactement le mouvement uniformément accéléré.

**Comment trancher.** Lancer la même simulation avec les deux coefficients et
comparer les observables du chapitre 6. Si l'écart est sous le bruit
statistique du tirage, la question est close.

---

## 2. `ran2` — débordement entier dans le générateur

```fortran
PARAMETER (IM1=2147483563, IA1=40014, IQ1=3668, IR1=12211, …)
```

**Le problème.** *Numerical Recipes* donne `IQ1 = 53668` : un chiffre a été
perdu. La méthode de Schrage n'évite le débordement qu'à la condition
`IR1 < IQ1` ; ici `12211 > 3668`, et le produit `k*IR1` atteint **7,1e9** pour
une limite entière 32 bits à **2,1e9**.

Ce n'est donc pas le générateur de L'Ecuyer, mais une variante repliée par le
débordement — et c'est elle qui a tiré toutes les pseudo-particules de la
thèse.

**Portée.** Toute l'initialisation de l'espace des phases.

**Ce que fait le portage.** `Ran2` reproduit la suite exacte, débordement
compris : 20 000 valeurs sur 20 000 identiques bit à bit. `Ran2(seed;
consistent = true)` rétablit `IQ1 = 53668`.

**Ce qui a déjà été mesuré.** Sur 10⁶ tirages, rien d'alarmant :

| | thèse | corrigé | attendu |
|---|---|---|---|
| moyenne | 0,50021 | 0,50023 | 0,5 |
| variance | 0,08338 | 0,08328 | 1/12 ≈ 0,08333 |
| corrélation lag-1 | 3,5e−4 | 1,3e−3 | 0 |
| khi², 100 casiers | 110 | 92 | ≈ 99 ± 14 |

La variante buggée se comporte aussi bien que la correcte sur ces tests. Ils
restent **faibles** : ils ne verraient pas une corrélation à longue portée ni
une période raccourcie. Une batterie sérieuse (TestU01) serait nécessaire
pour conclure.

---

## 3. `maketaint` — noyau de lissage non normalisé

Les tables de convolution intègrent chaque fonction de base contre une
gaussienne, sur des bornes fixées à la main :

```fortran
call intvg1(gx,nx,gx(0),gx(1),0,…)   ! nœud 0 : support [g0, g1]
…
call intvg1(gx,nx,gx(3),gx(4),8,…)   ! nœud 4 : support [g3, g4]
```

**Le problème.** Ces bornes sont celles des fonctions de base **du bord**, pas
d'un nœud intérieur générique. Or la table est ensuite appliquée par
translation autour de n'importe quel nœud, où les fonctions ont leur support
complet `[g(k−1), g(k+1)]`. Les deux fonctions extrêmes de la fenêtre sont
donc intégrées sur un support tronqué. S'y ajoute le fait qu'une fenêtre de
10 fonctions ne capte pas toute la gaussienne.

**Conséquence, mesurée.**

| grandeur | valeur attendue | valeur obtenue |
|---|---|---|
| `Σ recouvrements` | 1 | 1 à 3,4e−6 près |
| dérivée d'une constante | 0 | jusqu'à 1,3e−5 |

Le champ lissé porte donc une erreur relative de l'ordre de **1e-5**, avec une
composante **transverse** : un potentiel ne dépendant que de `x` produit un
champ en `y` non nul. Vérifié sur `Φ = x`, où le champ non lissé est exact à
1e−15 et le lissé se trompe de 2,2e−5.

**À noter — la méthode se protège pour la densité, pas pour le champ.**
`makerhog` renormalise explicitement la densité déposée après coup :

```fortran
coef = dfloat(npart-nbout)*charge/qtot
```

L'oracle en donne la mesure : `qtot = 196,080` avant correction pour 196
attendus, soit **4e-4**. Rien d'équivalent ne protège `champsg`.

**Ce que fait le portage.** Reproduit les tables à l'identique (1,3e−13 contre
l'oracle), documente la limite dans `GaussianSmoothing`, et **borne l'erreur
par un test** qui échouerait si elle s'aggravait.

**Piste de correction.** Intégrer chaque fonction de base sur son support
réel, élargir la fenêtre, ou normaliser les tables après coup comme le fait
`makerhog` pour la densité. Les trois changent les valeurs de l'oracle : à ne
faire qu'une fois les runs de référence reproduits.

---

## 4. `initialise` — un entier pour lire un réel

```fortran
integer i,npart,nbgrid,nbgrid2,rmax
…
read (2,*) rmax          ! rhoinit.dat contient « 35.0000000000000 »
```

`rmax` n'est ensuite utilisé que dans des divisions réelles. Les compilateurs
de 1996 toléraient la lecture ; gfortran la refuse
(*Bad integer for item 1 in list input*). **Corrigé** en `real*8` dans
`modernize.patch` — sans quoi le code ne tourne pas.

C'est le rappel le plus net que d'autres bugs latents dorment peut-être :
celui-ci n'a été révélé que par un compilateur plus strict, trente ans après.

---

## 5. `force2gi` — appel avec un argument de trop

`incproj2` appelle `force2gi` avec `liste2` en plus de sa signature. gfortran
le refuse. Le code est **mort** : tous les appels à `incproj2` sont commentés
dans le programme principal. **Corrigé** dans `modernize.patch` par retrait de
l'argument surnuméraire.

---

## 6. `makerhsf` — quarante lignes de calcul inutilisé

`makerhsf` calcule charge totale, barycentre et tenseur quadrupolaire de la
densité — puis **ne s'en sert pas** : les valeurs de bord viennent de
`potentiel`, c'est-à-dire de l'interpolation de la grille grossière. Les
multipôles ne sont qu'imprimés.

Ce n'est pas une erreur de résultat, mais :

* un coût inutile — 10 contractions sur 195 000 points, deux fois par pas de
  temps ;
* du bruit en sortie — la boucle d'impression du quadrupôle, commentée dans
  `makerh2`, est restée **active** ici : 9 lignes par appel.

Le portage ne les calcule pas : `boundary_from_coarse!` pose les faces, un
point c'est tout.

---

## 7. `ceq3d.f` — π tronqué

```fortran
parameter (pi=3.141592653589d0)
```

Il manque trois décimales : `π = 3.141592653589793…`. L'écart relatif est de
`2.5e-13`, et il se propage partout où `pi` intervient — angles du tirage
initial, normalisation gaussienne, facteur `−4π` du second membre.

**C'est la seule anomalie que le portage ne reproduit pas.** Julia utilise `π`
en pleine précision : la valeur tronquée n'apporte rien, et la recopier
figerait une imprécision gratuite.

**Conséquence sur les comparaisons.** Les grandeurs qui passent par `pi` ne
collent donc à l'oracle qu'à ~`1e-12`, et non `1e-13` comme le reste. Vérifié
sur le tirage initial :

| | écart sur les positions |
|---|---|
| avec π de Julia | 7,5e−13 |
| avec le π tronqué | **4,1e−17**, dont 95,5 % de valeurs bit-exactes |

L'attribution est donc sans ambiguïté : tout l'écart vient de là, et le reste
du tirage est exact. Même raisonnement que pour la tolérance de `findacc` dans
`stretched_axis` — on garde la version juste, on desserre la comparaison, et
on écrit pourquoi.

---

## 8. `pspech2` / `enertot2g` — quel potentiel voit le bilan ?

**Élucidé.** L'énoncé initial partait d'une lecture fausse : `pspech2` n'est
**pas** l'opposée de `pspech`. Elle contient **deux** boucles, et ajoute donc
deux fois à `csol` :

1. `ech = −(V_xc + V_jel)` — celle-là, oui, défait `pspech` ;
2. `ech = ε_xc + V_jel` — l'énergie d'échange-corrélation, avec le facteur ¾
   de Dirac et l'expression **intégrée** de Gunnarsson-Lundqvist
   `εc = −0,0333·[(1+x³)ln(1+1/x) + x/2 − x² − ⅓]`, `x = rs/11,4`.

`csol` n'est donc pas censé revenir à Hartree : `pspech2` le convertit du
**potentiel** vers l'**énergie**, ce que `enertot2g` demande. Le tableau
d'origine se relit alors sans mystère, et les deux premières lignes se
reproduisent **exactement** dans le portage :

| point de la boucle | ‖csol‖ Fortran | portage |
|---|---|---|
| Hartree seul | 737,4 | **737,4** |
| après `pspech` (+ V_xc + V_jel) | 7,8 | **7,8** |
| après `pspech2` (+ ε_xc + V_jel) | 322,2 | *voir ci-dessous* |

La troisième ne se reproduit pas, et c'est là que le vrai défaut se loge. La
seconde boucle est écrite ainsi :

```fortran
if (rho(i,j,k).gt.1.d-7) then
   rsrm1=rho(i,j,k)**us3
   rr=dsqrt(gtx(i)**2+gty(j)**2+gtz(k)**2)   ! ← calculé seulement ici
   ...
else
   ech(i,j,k)=potjel(rr,nbion)                ! ← rr du point PRÉCÉDENT
end if
```

**`rr` n'est calculé que dans la branche dense.** Sur la grille fine de l'état
initial, seuls 9,5 % des points dépassent `1e-7` : les 90,5 % restants
évaluent le potentiel de jellium à un rayon hérité d'un point antérieur,
choisi par l'ordre de parcours. Mesuré dans le portage, l'effet est massif —
`‖csol‖` passe de **6,3** (rayon correct) à **704,4** (rayon périmé). La
valeur 322,2, intermédiaire, n'a pas été reproduite au chiffre près : elle a
été relevée à un autre instant de la boucle, sur un autre état de densité.

**La version 1998-01-05 corrige le défaut** en hissant `rr` hors du `if`.
C'est la seule des corrections silencieuses de 1998 dont l'effet soit
quantifié ici.

**Ce que le portage fait.** Il ne reproduit pas la conversion en deux temps :
la boucle Julia alimente le bilan avec une définition explicite — `½∫ρΦ_H`
pour Hartree, `∫ρΦ_total` pour l'interaction — et `energy_budget` retrouve les
nombres de l'oracle à `6e-15`. La paire `xc_potential` / `xc_energy_density`
existe désormais dans `meanfield.jl` pour que la distinction soit nommée
plutôt que subie.

---

## 9. `docapture` — deux adoucissements pour un seul projectile

Le projectile est une boule uniformément chargée de rayon `cutoff`. Deux
routines évaluent son potentiel au contact, et elles ne s'accordent pas :

| routine | terme constant | terme en `r²` |
|---|---|---|
| `incproj` | `1,5·q/c` | `−0,5·q/c³` |
| `docapture` | `2·q/c` | `−q/c³` |

Seule la première est le potentiel d'une boule uniformément chargée,
`−q(3 − (r/c)²)/2c`. La seconde reste continue au raccord `r = c` — les deux
y valent `−q/c` — mais vaut `4/3` de l'autre au centre.

**Portée.** Faible : `einterne` ne sert qu'au compte rendu, et `docapture`
n'est appelée qu'une fois, quand le projectile a quitté la boîte. Le portage
reproduit les deux formes, en les signalant l'une à l'autre.

---

## 10. `forceproji` — le code n'applique pas la force de la thèse

**La plus lourde du relevé, et la seule qui change une conclusion physique.**

La thèse pose l'interaction projectile ↔ pseudo-particule comme une charge
ponctuelle contre une **gaussienne** de largeur `σ_ion` (chapitre 2,
éq. `Eforceproj2`) :

    V(r) = Erf(r / (√2 σ)) / r
    f⃗   = −Q·Q′ · [Erf(r/(√2σ)) − 2 g(r) r] / r³ · r⃗,
    g(r) = exp(−r²/2σ²) / (√(2π) σ)        (gaussienne normalisée à 1D)

Le Fortran, lui, fait une **boule uniformément chargée** de rayon `cutoff` :
Coulomb au-delà, force linéaire en deçà. C'est le cas dans toutes les
versions, `incproj` comme `forceproji`. (L'adoucissement de `docapture` diffère
encore des deux — voir anomalie 9.)

### Ce que `erfsr` raconte

`erfsr(r, sigr)`, qui est *exactement* `Erf(r/(√2σ))/r`, existe dans chaque
fichier source. Elle n'est appelée que dans **une** version sur 49 : la plus
ancienne conservée, `it8/ttt/vlas.f` du 2 juillet 1996. Et elle y était **déjà
tabulée** — l'erf ne se recalcule pas par paire :

| routine | rôle |
|---|---|
| `maketaerf` → `erftab` | potentiel `Erf(r/√2σ)/r` tabulé sur `[0, 5σ]` |
| `maketafor` → `fortab` | la **force**, tabulée sur `[0, 10σ]` |
| `forcerad` | la calcule, par quadrature 2D sur la gaussienne — pas par la formule analytique |
| `potdirect` | consomme `erftab` : Coulomb au-delà de `5σ`, table en deçà, par indice au plus proche |

Ce n'était donc pas une intention non réalisée mais une **implémentation
complète**, celle de la sommation directe paire à paire. À la date même du
2 juillet 1996 elle est déjà commentée, remplacée par la voie sur grille
(`maketable` / `maketaint`, qui tabulent la gaussienne contre les splines) —
l'approche du chapitre TBSCM, en `O(N + grille)` au lieu de `O(N²)`.

`erfsr` survit ensuite comme code mort dans les 48 versions suivantes. Quand
le projectile est ajouté, plus tard, il reçoit une boule — pas la table.

**L'anomalie n'est donc pas un oubli d'écriture, c'est un changement de voie
dont le projectile n'a pas hérité.**

### Pourquoi cela compte

À rayon égal (`σ = cutoff = 1`), les deux forces ne se ressemblent qu'au loin :

| r | boule | gaussienne | rapport |
|---|---|---|---|
| 0,25 | 0,250 | 0,065 | 0,26 |
| 0,50 | 0,500 | 0,123 | 0,25 |
| 1,00 | 1,000 | 0,199 | **0,20** |
| 2,00 | 0,250 | 0,185 | 0,74 |
| 3,00 | 0,111 | 0,108 | 0,97 |

La gaussienne est **cinq fois plus faible au contact**, et c'est là que naît
le freinage. Le chapitre le dit sans détour : « la perte d'énergie des ions
traversant l'agrégat **dépend fortement de ce lissage** », et `σ_ion = 1` a
été choisi « afin de reproduire les résultats de ce modèle [Lindhard] ».

Mesuré sur Na₁₉₆, proton 2 keV (`v = 0,283`), même graine, même tout :

| adoucissement | `dE/dx` (Δx = 4, centre) |
|---|---|
| boule, `cutoff = 1` (le Fortran) | **1,01 eV/a₀** |
| gaussienne, `σ_ion = 1` (la thèse) | **0,78 eV/a₀** |
| thèse, Na₁₀₀₀, interpolé à `v = 0,283` | ~0,70 |
| Lindhard | 0,64 |

### Ce que le portage fait

Les deux formes coexistent, en types : `BallSoftening` reproduit le Fortran —
c'est ce contre quoi l'oracle valide — et `GaussianSoftening` applique la
thèse. Le choix se fait à la construction du `Projectile`, et le noyau de
force se spécialise à la compilation.

⚠️ **Aucune des deux n'est « la bonne » par défaut.** Comparer à l'oracle
demande la boule ; comparer aux figures demande la gaussienne. Dire lequel on
emploie fait partie du résultat.

---

## Ce que la version de 1998 corrige — et ne corrige pas

Vérifié en comparant la version portée (1997-06-06) à la cible de production
(1998-01-05, voir [`chronologie-versions-fortran.md`](chronologie-versions-fortran.md)) :

| # | Anomalie | Dans 1998-01-05 |
|---|---|---|
| 1 | `moveback2` : `dltt*2` pour `dltt**2` | **survit à l'identique** |
| 2 | `ran2` : `IQ1=3668` au lieu de `53668` | **survit à l'identique** |
| 3 | `maketaint` : supports d'intégration tronqués | **survit** |
| 4 | `initialise` : `integer rmax` lisant un réel | ✅ **corrigée** — `real*8 rmax` |
| 5 | `force2gi` : un argument de trop | **survit** (code mort) |
| 9 | `docapture` : adoucissement divergent | **survit à l'identique** |

Deux corrections silencieuses s'ajoutent, non relevées jusqu'ici parce qu'elles
n'existaient pas dans la version portée :

* **`initialise`** troque `sqrt`/`cos`/`sin` contre `dsqrt`/`dcos`/`dsin`. ⚠️ **Sans
  effet numérique** : en FORTRAN 77 `SQRT` est un intrinsèque *générique*, résolu sur le
  type de l'argument, donc `sqrt(x)` sur un `real*8` **est** `dsqrt(x)` — vérifié avec
  gfortran. C'est un changement de style, pas une correction. `initialise4` garde
  d'ailleurs un `sqrt` pour l'angle polaire des impulsions à côté d'un `dsqrt` pour celui
  des positions : l'incohérence est visuelle seulement.
* **`pspech2`** sort le calcul de `rr` du `if (rho > 1e-7)`. Avant, aux points de densité
  négligeable, `rr` gardait la valeur du **point précédent** : le potentiel de jellium y
  était évalué au mauvais rayon.

L'anomalie 8 mérite un mot à part. La version 1998 contient une routine **`pspech3`** qui
calcule la densité d'énergie d'échange-corrélation (Dirac avec son facteur ¾, expression
complète de Gunnarsson-Lundqvist) au lieu du potentiel, et corrige le double comptage de
Hartree par `csol ← ½csol + echsol`. C'est précisément la correction que réclame
l'anomalie 8 — **mais elle n'est appelée nulle part**. L'auteur l'avait écrite sans la
brancher. Reproduire fidèlement la cible veut donc dire la porter et la laisser morte.

## Ce qui reste à examiner

Le portage n'a couvert que 60 % du code vivant. Les étages non encore lus en
détail — initialisation Thomas-Fermi, projectile, observables — n'ont pas été
audités. Ce fichier est à compléter au fur et à mesure.
