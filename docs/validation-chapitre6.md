# Validation sur une trajectoire complète — traversée du chapitre 6

Toutes les validations précédentes portaient sur des **appels isolés** :
une matrice, un dépôt, un pas de Verlet. Aucune ne disait ce que devient
l'accord sur des centaines de pas enchaînés, où un écart minuscule peut
s'amplifier. Celle-ci comble ce trou.

## Protocole

Proton de 2 keV traversant un agrégat Na₁₉₆, paramètre d'impact nul,
600 pas de `dt = 1 u.a.`, 20 000 pseudo-particules — exactement le
`ref/fortran/vlas.inp`, avec `nbt = 600`.

Le Fortran tourne à ~0,4 s/pas (4 min au total) une fois les dumps de
visualisation coupés (`nbphot = 0`) : ils écrivaient 14 641 lignes par pas et
représentaient 23 s sur 23 s. Le portage Julia tourne à 27 ms/pas.

## Résultat

| grandeur | Julia | Fortran | écart |
|---|---|---|---|
| position finale | 98,50 | 98,81 | 0,3 % |
| perte totale | 49,08 eV | 47,57 eV | 3,2 % |
| **dE/dx dans l'agrégat** | **1,0012** | **0,9961 eV/a₀** | **0,5 %** |

Et pas à pas, sur les 257 premiers :

| pas | x (Julia) | x (Fortran) | perte Julia | perte Fortran |
|---|---|---|---|---|
| 1 | −69,717 | −69,717 | 0,0001 | 0,0001 |
| 100 | −41,706 | −41,706 | −0,0903 | −0,0926 |
| 200 | −13,432 | −13,432 | 13,839 | 13,771 |
| 257 | 2,602 | 2,604 | 30,981 | 29,896 |

Les trajectoires se superposent à `2e-3` sur 257 pas. L'écart sur la perte
croît lentement — comportement attendu de deux systèmes chaotiques partant du
même point, dont les arithmétiques diffèrent au dernier bit.

**Le portage reproduit le code d'origine sur une trajectoire entière.**

## Ce que cela ne règle pas

Les trois codes ne s'accordent pas avec la figure publiée :

| source | dE/dx (eV/a₀) à v = 0,283 u.a. |
|---|---|
| TRIM (référence, 2000 eV) | 0,809 |
| Thèse, figure `Fperte2`, σ_ion = 1,0 a₀ | 0,69 – 0,74 |
| **Fortran d'origine, avec `vlas.inp`** | **0,996** |
| **Portage Julia, mêmes paramètres** | **1,001** |

Puisque le Fortran lui-même donne 0,996 avec ce fichier de paramètres,
**l'écart avec la figure ne vient pas du code mais des conditions du run**.
Le `vlas.inp` présent dans l'arborescence est un fichier de travail ; rien ne
garantit qu'il ait produit une figure publiée.

Écartée par la mesure : la **granularité du plasma**. De 20 000 à 200 000
pseudo-particules, `dE/dx` reste entre 0,985 et 1,006 — convergé à ±1 %, sans
la moindre dérive vers 0,72.

## Figure reproduite — Na₁₀₀₀, σ_ion = 1

Le balayage en vitesse de la thèse est retrouvé, avec la force gaussienne, sur
l'agrégat et à la statistique de la production.

Grille `nfine = 44`, `rcluster = 78`, `rbox = 235` (h = 3,55 a₀, la résolution
validée sur Na₁₉₆, étendue pour contenir Na₁₀₀₀ et le projectile dès son
entrée). **800 000 pseudo-particules**, départ à `x₀ = −65` comme les
trajectoires archivées. Onze minutes de calcul pour les cinq points.

| keV | v | portage `Δx=4` | ajustement ±10 | thèse | écart |
|---|---|---|---|---|---|
| 1 | 0,200 | 0,541 | 0,532 | 0,526 / 0,518 | **+3,6 %** |
| 4 | 0,400 | 0,999 | 1,061 | 0,961 / 0,995 | **+2,2 %** |
| 9 | 0,600 | 1,388 | 1,444 | 1,422 / 1,480 | **−4,4 %** |
| 16 | 0,800 | 1,608 | 1,572 | 1,587 / 1,586 | **+1,4 %** |
| 25 | 1,000 | 1,410 | 1,504 | 1,529 / 1,590 | −9,6 % |

Quatre points sur cinq tombent à ±5 %, c'est-à-dire dans l'écart que les deux
colonnes publiées ont **entre elles** (2 à 4 %). Le portage reproduit donc la
courbe au niveau de sa propre dispersion, y compris le maximum vers v = 0,8 et
la décroissance au-delà.

Le point à 25 keV est le moins bon. L'estimateur de la thèse mesure une pente
sur quatre bohrs : à v = 1, le projectile les franchit en quatre pas, et deux
points suffisent à porter tout le résultat. L'ajustement sur ±10 a₀ y ramène
l'écart à −3,5 %, ce qui dit que c'est l'estimateur qui plafonne, pas la
physique.

⚠️ **Il a fallu 800 000 pseudo-particules.** À 20 000, la trajectoire complète
reste juste (perte totale 73,7 eV contre 76,1 pour l'archive, à 4 keV) mais la
pente sur quatre bohrs devient **négative** : la fenêtre est trop étroite pour
le bruit de tirage. C'est pourquoi la production employait ce nombre-là, et
c'est le genre de chose qu'on ne devine pas — on la mesure.

L'état initial vient de `rhorad.Na1000.dat`, la densité radiale d'équilibre
archivée (octobre 1998, 998,7 électrons intégrés), convertie en profil de
tirage par `PotentialProfile(grid, density)`. Le `pot.dat` qu'attendait
`initialise4` n'a pas survécu, mais la densité suffit : à l'équilibre de
Thomas-Fermi, poser `V = −p_F²/2` rend le critère de rejet équivalent à
`p < p_F(r)`.

Rejouable par [`scripts/figure53.jl`](../scripts/figure53.jl) ; sortie brute
dans [`ref/these/resultat-portage.txt`](../ref/these/resultat-portage.txt).

## Figure 5.2 reproduite — le sillage

La 5.3 teste l'**intégrale** de la réponse ; la 5.2 en teste la **structure
spatiale** : le sillage de plasmon que laisse l'ion, de longueur d'onde
`2πv/ω_p`. Un champ moyen faux pourrait encore s'intégrer en un `dE/dx`
plausible ; il ne mettrait pas les nœuds du sillage au bon endroit.

Même progression que les panneaux publiés : à 1 et 4 keV l'ion traîne un amas
compact et rien d'autre ; le sillage s'amorce à 9 keV et il est net à 16. C'est
un effet de **vitesse** — d'où le fait qu'il soit invisible dans le panneau
qu'on choisirait spontanément.

Trois choix d'affichage comptent, et ils ont été **mesurés**, pas devinés :

* **`ρ`, pas `δρ`.** À 3,2 M de pseudo-particules une maille fine en contient
  ~530, donc 4,4 % de bruit de tirage — l'ordre de grandeur de la déformation
  elle-même. Sur `δρ` cela donne un S/B de 3 par maille, illisible ; sur `ρ` le
  même grain ne se voit pas.
* **Vide masqué, sommet d'échelle à `1,45 ρ_bulk`.** De 0 au maximum, le cœur de
  l'agrégat occupe 89 % de la plage et sort uni : tout le contraste est dépensé
  sur du vide.
* **Aucun lissage.** Un flou d'une maille ne divise le bruit que par 1,4 et
  fabrique de fausses structures cohérentes. Moyenner selon `z` échoue
  symétriquement : le sillage tient dans **une** maille en z, donc élargir la
  tranche dilue le signal plus vite qu'elle ne tue le bruit — optimum mesuré à
  `|z| ≤ 2 a₀`, qui vaut 23 %, et la projection complète est *pire* qu'un plan
  unique (S/B 1,24 contre 3,02).

Rejouable par [`scripts/figure52.jl`](../scripts/figure52.jl) — ~2 min 30 sur le
chemin GPU.

## Ce qui a permis d'y arriver — la force

Les points **effectivement tracés** des figures du chapitre ont été retrouvés,
dans le répertoire de travail xmgr de la thèse, et versionnés sous
[`ref/these/`](../ref/these/) — voir son README pour le détail. L'axe porte
`dE/dx (eV/a₀)`, ce qui lève l'ambiguïté d'unité, et la recette est explicite :

    dE/dx ≃ [E_k(+2) − E_k(−2)] / 4      (pente locale au centre)

Appliquée aux trajectoires archivées `Ekproj.dat.N`, elle redonne
`desdx.dat.1000` à 1–3 % près. La chaîne figure → données → recette est donc
fermée.

**Et l'écart s'explique par la force.** La thèse pose une interaction
projectile ↔ pseudo-particule gaussienne (éq. `Eforceproj2`) ; le Fortran
implémente une boule uniformément chargée, dans ses 43 versions. La routine `erfsr`, qui est exactement le potentiel gaussien, n'est appelée
que dans la version la plus ancienne (juillet 1996) — où elle était déjà
**tabulée**, avec sa force, pour une sommation directe paire à paire
abandonnée le mois même au profit de la voie sur grille. Le projectile,
ajouté plus tard, n'a pas hérité de cette table. Détail en [anomalie 10](coquilles-fortran.md).

Mesuré sur Na₁₉₆, proton 2 keV (`v = 0,283`), tout le reste égal :

| | `dE/dx` (eV/a₀) |
|---|---|
| Portage, boule `cutoff = 1` (le Fortran) | **1,01** |
| Portage, gaussienne `σ_ion = 1` (la thèse) | **0,78** |
| Thèse, Na₁₀₀₀, interpolé à `v = 0,283` | ~0,70 |
| Lindhard | 0,64 |

Le facteur 1,4 qui séparait le portage des figures vient donc de là, et non
d'un défaut de portage : le portage reproduisait fidèlement un Fortran qui ne
suivait pas sa propre thèse.

⚠️ **Ce n'est pas encore une reproduction de la figure.** Un point, sur Na₁₉₆
là où la figure trace Na₁₀₀₀, avec 20 000 pseudo-particules là où la
production en utilisait 800 000. L'accord à 12 % est encourageant et rien de
plus. Reproduire la courbe demande Na₁₀₀₀ aux cinq vitesses (1, 4, 9, 16,
25 keV), ce qui suppose un profil initial pour Na₁₀₀₀ — `rhorad.dat` existe
dans l'archive (`…/majrel2/initial/1000/`).

## Sur l'initialisation de 1998

Mesuré, trois graines par variante, traversée ±R sur Na₁₉₆ :

| graine | `initialise` (1997) | `initialise4` (1998) |
|---|---|---|
| −1 | 1,010 | 0,972 |
| −2 | 0,971 | 0,949 |
| −3 | 1,021 | 0,969 |

Le tirage par rejet abaisse `dE/dx` de ~3,7 %, pour une dispersion entre
graines de ±2,5 %. L'effet existe mais reste du même ordre que le bruit : il
ne pesait rien face au facteur 1,4 de la force.

## Étape antérieure — les trajectoires archivées

`temp/…/lucifer/Eloss/` conserve les **sorties de production de l'époque**, sous le nom
que le code fabrique lui-même : `Em1q1e002i000.dat` = masse 1, charge 1, 2 keV, impact 0.
Six énergies existent à impact nul : 2, 5, 7, 10, 19 et 50 keV.

Ces fichiers donnent `x(t)` tous les pas de temps, donc `v(t)` par différences finies —
et `½mv²` au premier pas redonne 73,502 u.a. contre 73,498 annoncés en en-tête, ce qui
confirme `dt = 1,0` et valide la lecture.

**La colonne 6 vaut `−N/|x|`**, ce qui identifie le système : `−15,680 × 99,717 = 1564` et
`−5,0911 × 308,266 = 1569`. Le run archivé est donc **Na₁₅₆₈**, lancé depuis `x₀ = −100`
— et non Na₁₉₆ depuis −70. (Le même calcul sur notre sortie donne 195,2 : Na₁₉₆.)

| Source | `dE/dx` (eV/a₀) |
|---|---|
| Archive `Em1q1e002i000.dat`, Na₁₅₆₈, sur la traversée ±R | **1,09** |
| Archive, plateau dans le cœur (moyenne glissante) | **1,15 – 1,20** |
| **Portage Julia, Na₁₉₆** | **1,00** |
| Fortran d'origine, Na₁₉₆ | 0,996 |
| Ma lecture de la figure `Fperte2` | 0,69 – 0,74 |

Le profil local le confirme : `dE/dx` est nul jusqu'à `x ≈ −55`, monte à un plateau de
1,15 – 1,20 sur tout le cœur, et retombe à zéro après `x ≈ +55` — un rayon de 46 a₀, qui
est exactement `r_s·N^{1/3} = 4 × 1568^{1/3}` pour Na₁₅₆₈.

**C'est donc ma lecture de la figure qui était fausse, pas le portage.** Les deux tailles
de cluster partagent la même densité de cœur (`r_s = 4`), et le pouvoir d'arrêt suit la
densité : 1,00 pour Na₁₉₆ et 1,15 pour Na₁₅₆₈ sont cohérents entre eux. Aucune des
valeurs produites par le code — d'époque ou portée — n'approche 0,72. La figure `Fperte2`
porte des courbes Na₄₀/Na₂₅₀/Na₁₀₀₀ ; ni Na₁₉₆ ni Na₁₅₆₈ n'y figurent, et la grandeur
tracée n'est vraisemblablement pas la perte totale divisée par le diamètre.

> 🗣️ **L'auteur, de mémoire : « le cutoff du projectile était le paramètre clé pour le
> pouvoir d'arrêt ».** Cohérent avec le reste : le `vlas.inp` de production proton garde
> `cutoff = 1.0`, tandis que le balayage Xe²⁵⁺ de `arkonnen/launch/` monte à `5.0`. C'est
> bien un paramètre d'entrée que l'on fait varier, et `σ_ion = 1,0 a₀` de la figure
> désigne `cutoff = 1.0`.

### Ce qui reste ouvert

Non pas un écart code/thèse — il n'y en a plus — mais la **définition** portée par la
figure `Fperte2`. Pour la trancher il faudrait rejouer un Na₄₀ ou un Na₂₅₀ et comparer
courbe à courbe, ce que le portage sait faire.

Reste aussi à vérifier si `initialise4` (échantillonnage par rejet, version 1998-01-05)
déplace `dE/dx` : c'est le seul changement de physique entre la version portée et la
cible. Voir [`chronologie-versions-fortran.md`](chronologie-versions-fortran.md).

<!-- ancienne section, conservée pour mémoire -->

> 🗣️ **L'auteur, de mémoire : « le cutoff du projectile était le paramètre
> clé pour le pouvoir d'arrêt ».** Cela oriente fortement la recherche : le
> `cutoff` **est** bien le paramètre pertinent, et `σ_ion = 1,0 a₀` de la
> figure désigne vraisemblablement `cutoff = 1,0`. La piste d'un σ_ion
> « conséquence du maillage » perd donc son crédit, et l'écart 0,72 → 1,00
> doit s'expliquer à `cutoff` identique.

Restent à élucider, par ordre de vraisemblance :

1. **La résolution de grille.** `n1xyz = 28` fixe `h = 3,57`, donc le second
   lissage `sigr = h/3 ≈ 1,19` de `maketable`. Avec `cutoff = 1,0` par
   ailleurs correct, c'est `sigr` — non paramétrable, subi — qui devient le
   suspect : il dépasse le cutoff et dominerait alors l'interaction.
2. **La définition de `dE/dx` dans la figure** — pente ajustée sur la partie
   linéaire, ou rapportée à la densité du cœur, plutôt que perte totale
   divisée par le diamètre.
3. **La version du code.** Le portage part du 1997-06-06 ; la cible de
   production est le 1998-01-05, qui change l'initialisation (`initialise4`).
   ⚠️ J'avais écrit ici que `multrcmax` touchait au rayon de coupure du
   projectile : c'est faux. Cette routine compte les électrons dans des
   sphères de 50 à 100 a₀ et écrit `rcm.dat` — un diagnostic d'évaporation.
