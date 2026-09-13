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
3. **La version du code.** Les figures de la thèse ont été produites près de
   la soutenance (déc. 1998), alors que le portage part de la version du
   1997-06-06. La dernière séquentielle (1998-01-07) ajoute `pspech3` et
   `multrcmax`, qui touchent précisément au rayon de coupure du projectile.
   Voir [`chronologie-versions-fortran.md`](chronologie-versions-fortran.md).
