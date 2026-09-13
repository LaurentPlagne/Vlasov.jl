# Directives projet Julia — Vlasov.jl

Ce fichier est le **noyau** : ce qui vaut à chaque tour, quelle que soit la tâche.
Règle de maintenance : **n'écrire ici que ce qui n'est pas dérivable du code** —
l'intention, les invariants, les alternatives rejetées, ce qui doit rester numériquement
identique. La liste des fichiers, les symboles, les appelants se retrouvent avec
l'outillage.

⚠️ **Ne consigner ici qu'un constat mesuré sur CE dépôt.** Une règle héritée d'un autre
projet, si vraie soit-elle ailleurs, n'y a pas sa place tant qu'elle n'a pas été
re-vérifiée ici.

## Objet du projet

Portage **idiomatique** en Julia du code de simulation Vlasov de la thèse (Fortran 77,
`vlas.f`). Physique : réponse non-linéaire des électrons d'un agrégat de sodium en
collision avec des ions ; équation de Vlasov résolue par méthode pseudo-particulaire,
Poisson par collocation sur splines cubiques d'Hermite et décomposition tensorielle
(méthode TBSCM, cf. Plagne & Berthou, *J. Comput. Phys.* **157**(2), 419-440, 2000).

**Idiomatique, pas littéral** : le Fortran est la référence *numérique*, jamais le modèle
de conception. Il encode la base spline dans un entier (`ido = 2k + isig`), duplique à la
main la grille grossière sur 41 variables suffixées `big`, et passe 47 arguments à
`static`. Reproduire ces structures serait un contresens.

## Environnement & versioning

- **Langage** : Julia v1.12.
- **Dépendances** : toujours se référer à `Project.toml`. Ne pas ajouter de package
  externe sans accord explicite.
- **Validation** : toute modification doit passer les tests avant validation.

## Carte du dépôt

`src/` — chaque sous-module est explicitement inclus par `include()` dans `src/Vlasov.jl`,
**dans un ordre qui compte** (les types de base avant ce qui les consomme).

Une fiche de sous-système (rôle, point d'entrée, contrat, invariants, frontière, ce qui
doit rester numériquement identique, pièges propres à la zone) est à écrire sous
`docs/architecture/` dès qu'un chantier ouvre une zone non triviale, et à ouvrir avant d'y
toucher. Aucune n'existe encore.

## Méthodologie : chemins numériques

Le code Fortran d'origine **compile et tourne** sur cette machine : il sert d'**oracle**.
La fidélité ne se démontre pas par raisonnement d'équivalence mais par **comparaison
numérique à l'oracle gelé**. Trois contrôles, dans cet ordre :

1. **Étalonner** — le harnais doit rendre zéro écart contre le code INCHANGÉ, sinon c'est
   lui qui est faux.
2. **Éprouver sa sensibilité** — introduire une mutation délibérée (1 ULP, un signe, une
   borne inversée) et vérifier qu'elle est vue.
3. **Mesurer la couverture des branches** — compter les tirages qui atteignent chaque
   branche réécrite, et écrire un générateur ciblé pour celles à zéro. Un tirage uniforme
   n'atteint jamais les configurations dégénérées, et ce sont exactement celles qui
   distinguent deux implémentations.

Le comparateur est du code, et il ment aussi : le refaire valider dès que la disposition
des types change.

⚠️ **Ici la comparaison est à la précision machine, pas bit-à-bit.** Le portage remplace
des briques (NAG `f02agf` → `eigen`, inversion explicite → factorisation bande) : l'égalité
exacte n'est pas attendue et l'exiger ferait rejeter du code correct. Les écarts constatés
sur la chaîne splines → opérateur → spectre → solveur tensoriel 3D sont de l'ordre de
`1e-16` à `6e-15` en norme relative. **Un écart qui sort de cet ordre est une régression**,
pas du bruit.

⚠️ **Tout écart oracle/Julia doit être arbitré, jamais corrigé en silence** : bug d'origine
ou erreur de portage ? Le Fortran contient des bugs latents que les compilateurs de 1996
toléraient (un `integer` relisant un réel, un appel avec un argument de trop en code mort).

📋 **La validation sur une trajectoire complète est dans
[`docs/validation-chapitre6.md`](docs/validation-chapitre6.md)** : le portage suit le
Fortran sur 600 pas, `dE/dx` à 0,5 %. Les comparaisons ponctuelles à `1e-13` ne disaient
rien de l'accord après des centaines de pas enchaînés ; celle-ci le dit.

📋 **La chronologie des ~43 versions Fortran est dans
[`docs/chronologie-versions-fortran.md`](docs/chronologie-versions-fortran.md)**. Le
portage part de la version du **1997-06-06** ; la dernière séquentielle est du
**1998-01-07** (16 routines de plus : énergie propre, potentiel radial, `pspech3`), et la
parallélisation vit dans une branche jamais fusionnée, `arkonnen/mystuffgz/vlas.hpf`
(HPF + MPI + tri à deux niveaux). Une seule sous-arborescence a gardé ses dates d'origine :
`temp/home/sauron2/plagne/` — partout ailleurs les `mtime` valent janvier 2025.

📋 **Les coquilles constatées sont relevées dans
[`docs/coquilles-fortran.md`](docs/coquilles-fortran.md)** — à consulter avant de
s'étonner d'un résultat, et à compléter dès qu'une nouvelle apparaît. Aucune n'est
corrigée : le portage les reproduit, sans quoi la comparaison à l'oracle perdrait sa
valeur. Elles seront arbitrées une fois les runs de la thèse reproductibles, en mesurant
ce que chaque correction change. Deux ont déjà leur variante corrigée derrière un
`consistent = true`, prête pour cette mesure.

⚠️ **Dumper les entrées à l'entrée de la routine testée, jamais ailleurs.** Ce piège a
coûté trois fois : une routine appelée **deux fois** sur deux grilles (`static`), une
appelée **un pas sur dix** quand ses voisines tournent à chaque pas (`enertot2g`), une
appelée **avant et après** une transformation du même tableau (`solve`). À chaque fois
les instantanés venaient d'instants différents et la comparaison ne voulait rien dire —
avec des écarts de 26 %, 38 % et 4105 % qui ressemblaient à des bugs de portage.

Trois parades, dans cet ordre : dumper **tout** ce que la routine consomme, à son entrée,
y compris sa grille ; apparier deux routines par un drapeau en `COMMON` quand l'une
produit ce que l'autre consomme ; et reconstruire ce qui est reconstructible —
[`axis_from_collocation`](src/splines.jl) retrouve les nœuds depuis les seuls points de
collocation, ce qui dispense d'emprunter une grille à une autre routine.

⚠️ **Comparer deux configurations sans barre de bruit ne prouve rien.** Avant de conclure
qu'une correction change quelque chose, mesurer la dispersion de l'observable sur
plusieurs graines. Sur l'arbitrage des coquilles 1 et 2, les effets valaient 0,06 à
0,71 σ — indiscernables du bruit, ce qu'aucune comparaison à une seule graine n'aurait
pu dire.

⚠️ **Un nom réutilisé entre une fermeture et sa fonction englobante devient UNE SEULE
variable, boxée — donc partagée par tous les fils.** Constaté, et coûteux à trouver :
`deposit!` nommait `lx, ly, lz` les résultats de `locate` dans son `do`-block, et les
longueurs duales dans la fonction. Les huit fils écrivaient dans la même case. Le
symptôme est sournois — résultat **juste à un fil**, faux à deux, de plus en plus faux à
huit, **somme totale conservée** (rien n'est perdu, tout est redistribué), et **non
déterministe** d'une exécution à l'autre.

Deux réflexes : vérifier qu'un résultat parallèle est **déterministe** sur plusieurs
exécutions, et qu'il **coïncide avec le séquentiel** — c'est le seul test qui distingue
une course d'une erreur de calcul. Un test par valeur attendue ne l'aurait pas vu.

⚠️ **Une bibliothèque tierce n'est pas un oracle.** Constaté sur `BandedMatrices` v1.12.0 :
la division à droite entre deux `BandedMatrix` rend un résultat **faux sans rien signaler**
(résidu ≈ 0.3 pour `cond(S) ≈ 4`), et `BandedMatrix / BandedLU` **ne termine pas**. Seul
`Matrix(A) / lu(B)` est correct et terminant ; la division à gauche, elle, va bien.
Corollaire de méthode : **un test de compatibilité sur une matrice aléatoire « gentille »
ne prouve rien.** Celui-ci est passé au vert parce que la matrice était à diagonale
dominante — la vraie `S` ne l'est pas, ses lignes de bord n'ayant qu'une seule entrée.
Éprouver une brique tierce sur **les données réelles**, et sur l'équation qu'elle prétend
résoudre (`‖A·X − B‖`), jamais sur sa seule absence d'erreur.

## Style de code & idiomes Julia

1. **Dispatch multiple** : privilégier le dispatch sur les types plutôt que des `if/else`
   ou une approche orientée objet.
2. **Stabilité des types** : pas de global non `const` ; type de retour prévisible (éviter
   les `Union` inutiles et `Any`) ; `@inferred` dans les tests.
3. **Performance & mémoire** : convention `!` pour les mutations ; `@views` pour le
   slicing ; vectorisation par l'opérateur point ; `Tuple`/`NTuple` dès que la taille est
   fixe (pile, immuable, zéro allocation tas).
4. **Style compact et fonctionnel** : `map`, `filter`, `any`, `all`, `reduce`,
   `findfirst`, générateurs et compréhensions plutôt que des boucles impératives à
   accumulateurs ou drapeaux. Fonctions courtes en affectation directe (`f(x) = …`) pour
   prédicats, accesseurs et transformations simples. La compacité doit préserver
   `@inferred` et l'absence d'allocations.
   - ⚠️ **Le fonctionnel est un moyen, pas un but : la lisibilité tranche.** Là où il
     masque un état séquentiel, écrire la boucle. Deux formes à proscrire : un `foldl`
     dont on jette le résultat et qui mute des accumulateurs extérieurs (c'est une boucle
     déguisée), et `Iterators.peel(Iterators.filter(…))` sur un générateur imbriqué pour
     trouver le premier élément. Une boucle avec un drapeau **nommé** se lit mieux.
   - Pas de `let` dans une définition en forme d'affectation pour se donner une variable
     locale : c'est exactement ce que fournit un corps de `function … end`.
   - Pas d'arithmétique d'indices déguisée : `t[3-i]` pour « l'autre extrémité » devient
     `reverse(t)[i]`. Et `x / 2` plutôt que `0.5 * x`.
5. **Conception des types.**
   - **Réifier ce que le Fortran encodait dans un entier.** `BasisIndex(knot, kind)` porte
     le couple (nœud, nature) que l'original écrasait en `2k + isig`. Conséquence voulue :
     plus aucun code ne manipule d'indice linéaire brut, et **le problème du 0-based
     disparaît** sans recourir à `OffsetArrays`.
   - **Le nom du champ EST le nom de l'accesseur.** N'écrire aucune fonction qui ne fait
     qu'écho au champ. Une fonction d'accès ne se justifie que si elle **abstrait** ou
     **assemble** quelque chose, ou si elle sert en broadcast — ce qu'un champ ne sait pas
     faire.
   - **Ne pas homogénéiser une asymétrie** : deux grandeurs de natures différentes gardent
     deux champs. Les fondre dans un tuple homogène masquerait l'information au lieu de la
     nommer.
   - **Envelopper un `NTuple` dans un type nommé, jamais un alias** : surcharger `==` sur
     un alias changerait l'égalité de *tout* tuple de cette forme dans le module.
   - **Pas de splat inutile** : `f(map(g, t)...)` qui reconstruit aussitôt un tuple est un
     aller-retour. Le splat ne sert qu'à **concaténer**.
   - Préférer le vocabulaire de `Base` à un nom inventé : `reverse(t)` plutôt qu'un
     `swap_xxx` maison.
   - **Paramétrer plutôt que dupliquer** : la hiérarchie de grilles (fine ⊂ grossière ⊂ …)
     est un paramètre de type, pas une copie manuelle de chaque tableau.
6. **Vérifier les propriétés constatées, ne pas les supposer.** Le spectre réel et négatif
   de l'opérateur spline est une propriété *observée*, pas une garantie structurelle :
   elle est contrôlée à la construction de `DiagonalizedOperator`, pour qu'une régression
   s'arrête là plutôt que se propager silencieusement.

## Exécution de code et tests

- **Session interactive** : utiliser l'outil MCP `ex` (Kaimon) plutôt que des commandes de
  terminal. Le REPL est **partagé avec l'utilisateur**, qui voit tout en direct. Revise
  recharge `src/` automatiquement — ne jamais appeler `Revise.revise()`.
  ⚠️ `println`/`print` sont dépouillés du code agent : terminer par une expression et
  utiliser `q=false` pour voir une valeur.
  - Démarrage : `investigate_environment()`. Pas de session ? `start_session(project_path=…)`
    la crée — ne pas attendre l'utilisateur. Le projet doit figurer dans la liste autorisée
    (`~/.config/kaimon/projects.json`) ; s'il en est absent, seul l'utilisateur peut l'ajouter
    depuis le TUI (onglet Config, `[p]`).
  - Paquets : `pkg_add(packages=[…])`, jamais `Pkg.add`. **Ne jamais changer de projet**
    avec `Pkg.activate`.
  - Mise en forme : `format_code(path)` (JuliaFormatter).
- ⚠️ **Tout `ex` dépassant ~30 s bascule en tâche de fond** et rend un `eval_id`.
  Le relever avec `check_eval` — attendre 30 s, puis ~60 s entre deux appels ; sonder plus
  vite ne le fait pas finir plus tôt. Un calcul long doit être écrit **coopératif**, sinon
  il n'est ni observable ni interruptible : `KaimonGate.is_cancelled()` dans la boucle (que
  `cancel_eval` déclenche), `KaimonGate.progress("…")`, `KaimonGate.stash(:clé, v)`. Ces
  noms sont `public` mais **non exportés** : les qualifier, sous peine d'`UndefVarError`.
  ⚠️ La compilation compte dans ces 30 s — un premier appel touchant une nouvelle
  bibliothèque peut basculer en tâche de fond sans rien calculer de lourd.
- ⚠️ **Un eval parti en boucle dans du code de bibliothèque bloque toute la session**, et
  la chaîne de secours ne suffit pas : `cancel_eval` ne fait que lever un drapeau que ce
  code ne lit pas, `manage_repl(command="restart")` échoue sur une session `stalled`, et
  même `1 + 1` reste en attente. Seule issue mesurée : relever le PID (`ping(extended=true)`),
  `kill -9`, puis `start_session`. Le processus reste `<defunct>` un moment — c'est normal,
  il n'est pas encore récolté. **Corollaire : tester les variantes une par une**, pas six
  dans un même eval, sinon on sait seulement que « quelque chose » pend.
- **Les outils Kaimon avant le shell, pour TOUT** — pas seulement pour évaluer du Julia.
  La dérive vers `grep`/`cat`/`find`/`rg` en session longue est le travers récurrent, et
  elle coûte : ces outils-là sont limités au dépôt, respectent `.gitignore`, et rendent le
  symbole englobant de chaque occurrence.
  - concept qu'on sait **décrire** → `search_code(query="…")` ;
  - **token exact** (symbole, appel, chaîne, TODO, regex) → `grep_code(pattern="…")`,
    `no_ignore=true` pour couvrir aussi les logs et le généré ;
  - **toutes les méthodes d'une fonction générique**, avec fichier:ligne → `search_methods`.
    C'est l'outil du dispatch multiple ; y penser avant de grep un nom de fonction ;
  - **champs, hiérarchie, sous-types d'un type** → `type_info` (sur un type CONCRET : sur un
    `UnionAll` il ne rend que le paramètre) ;
  - symboles d'un fichier → `document_symbols` ; lire un fichier ou un log → `ex`.
  - ⚠️ Le code Fortran de référence vit **hors du dépôt** : lui seul échappe à ces outils.
- ⚠️ **Angle mort commun à tous ces outils** : `goto_definition`, `workspace_symbols` et
  `search_methods` passent par la réflexion Julia, donc ne voient que ce qui est lié au
  module. Une fermeture (`map(xs) do x … end`) est liée comme un TYPE, pas comme une
  fonction : les appels émis depuis un `do`-block leur échappent.
- **Qui appelle ceci ?** Aucun outil ne répond exactement. `grep_code` reste le **filet de
  sécurité** : il SUR-déclare (docstrings, commentaires), ce qui est le bon sens de
  l'erreur — on relit des faux positifs, on ne casse pas un appelant manqué. Une absence de
  référence n'est JAMAIS une preuve de code mort : les appels par valeur
  (`f = cond ? g : h ; f(x)`) échappent à toute analyse statique.
- **Suppression de sortie verbeuse** : point-virgule **systématique** en fin
  d'expression/bloc envoyé à `ex` (sauf résultat ciblé explicitement voulu avec `q=false`).
- **Tests** : les exécuter **dans la session vivante**, pas via `run_tests` :

      using Test
      ts = include(joinpath(pkgdir(Vlasov), "test", "runtests.jl"))

  `run_tests` lance un sous-processus qui **recompile tout** — 35 s par tour contre 16 s
  en session, où le code est déjà chaud. Sur une boucle correction/vérification la
  différence est celle entre deux essais et cinq.

  ⚠️ La sortie du testset est dépouillée comme tout `println` : récupérer l'objet rendu
  par `include` et le résumer (`Test.get_test_counts`), ou attraper la
  `Test.TestSetException` que `@testset` lève en cas d'échec.

  ⚠️ `run_tests(pattern=…)` n'a de toute façon aucun effet ici : le filtre passe par
  `ARGS` et n'est honoré que par les suites ReTest. `test/runtests.jl` est du `Test.jl`
  simple. (`test/retest.jl` est un vestige du template : il appelle `Retest`, qui n'est
  pas dans `Project.toml`.)

- **Vérification de type** : `@code_warntype ma_fonction(args...)` dans `ex`
  (`using InteractiveUtils`).

## Méthodologie d'analyse et de performance

1. **Benchmarks rapides (< 10 s)** : sous-ensembles représentatifs (petit nombre de
   pseudo-particules, quelques pas de temps) plutôt que la simulation complète.
2. **Zéro hypothèse non mesurée** : ne jamais affirmer qu'une étape est un goulet sans
   profilage chiffré (`@time`, chronométrage pas à pas).
3. ⚠️ **Relever la charge machine AVANT de chronométrer** (`uptime`). Cette machine
   sert aussi à autre chose : à `load 14` pour 10 cœurs, les temps absolus ne valent
   rien. Une conclusion de perf a déjà dû être retirée pour cette raison.
   - **Préférer ce qui ne dépend pas de la charge.** Un compteur d'allocations
     (`@allocated`), un décompte de passes mémoire, un nombre d'opérations : ces
     preuves-là tiennent quelle que soit la charge, et c'est ainsi qu'a été établi le
     facteur 11 de l'instabilité de type.
   - **Quand il faut du temps, l'A/B entrelacé** : alterner les deux variantes au même
     instant et prendre le minimum de chacune, ce qui annule les dérives lentes.
     `Vlasov.PARALLEL[]` existe pour ça.
   - **Chercher une corroboration structurelle.** « Paralléliser les contractions
     ralentit » ne vaut que parce qu'un second argument, indépendant du chronomètre,
     l'explique : dix lectures concurrentes du même tableau saturent la bande passante.
     Un chiffre seul, sur machine chargée, ne prouve rien.
3. **Zéro spéculation sur le code de référence** : ne rien affirmer sur ce que fait le
   Fortran sans vérification. Si la lecture ne suffit pas à trancher, **instrumenter et
   recompiler** plutôt que spéculer — l'oracle est reconstructible.
4. **Calculs longs en tâche de fond**, sans sonder l'avancement à intervalles courts.
5. **Portée minimale** : tester une hypothèse sur un seul pas de temps ; n'élargir qu'une
   fois l'hypothèse confirmée. Ne jamais lancer la suite complète pendant les itérations de
   travail — c'est pour la validation finale.
6. **Traçabilité** : toute modification proposée est présentée sous forme de **diff**, pas
   seulement décrite.
7. **Tester avant de proposer** : toute commande ou snippet recommandé à l'utilisateur est
   d'abord exécuté, pour garantir qu'il tourne sans erreur.

## Communication

- Répondre de manière concise, en privilégiant le résultat, les mesures utiles et les
  éventuels blocages.
- **Ne pas affirmer sans mesurer**, y compris sur le langage lui-même : une propriété de
  Julia qu'on croit connaître (ordre d'itération, égalité de flottants, comportement du
  broadcast) se vérifie en trois lignes. Plusieurs « évidences » se sont révélées fausses
  ainsi — y compris sur des bibliothèques tierces réputées interchangeables.
- **Se corriger platement** quand une objection de l'utilisateur est fondée, sans la
  défendre.
- **Commenter ce qui est cryptique**, en particulier les gardes sur les cas dégénérés :
  dire ce qu'elles écartent, et pas seulement ce qu'elles testent.
