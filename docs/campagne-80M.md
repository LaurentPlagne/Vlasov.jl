# Campagne de simulations ultra-convergées à 80 millions de particules ("25 ans plus tard")

Réf. historique : Thèse de doctorat (1998), Chapitres 4, 5 et 6 ; Plagne & Berthou, *J. Comput. Phys.* **157**(2), 419-440 (2000) ; Springer (1997).

---

## 1. Contexte et motivation

En 1998, sur supercalculateur vectoriel et massivement parallèle (Cray C98 / T3E), les simulations Vlasov-Poisson en méthode particulaire traitaient couramment entre **$50\,000$ et $400\,000$ macro-particules** réparties sur des maillages de $20^3$ à $33^3$ points. Avec de tels effectifs :
* Le nombre de particules par maille spatiale centrale était de l'ordre de 10 à 30.
* Le bruit de grenaille statistique de Poisson ($\sigma_\rho / \rho \sim 1/\sqrt{N_{\text{cell}}}$) atteignait **18 à 30 %**, imposant un lissage artificiel important pour éviter la thermalisation prématurée vers la distribution de Maxwell-Boltzmann (cf. Chapitre 4 de la thèse).
* Les visualisations 2D étaient limitées à des isolignes discrètes fragmentées par le bruit.

Vingt-cinq ans plus tard, l'implémentation Julia / Metal GPU sur station Apple Silicon (M1 Max, mémoire unifiée) permet de franchir un cap d'échelle : **80 millions de macro-particules ($8\times 10^7$)** sur maillages fins $130^3$ à $222^3$.

À cette échelle, la statistique dépasse **$800$ macro-particules par maille** dans le cœur du cluster. Le bruit de Poisson tombe sous les **2 %**, révélant le comportement fluide continu de l'équation de Vlasov sans aucun artefact de discrétisation.

---

## 2. Budget matériel et empreinte mémoire (Apple Silicon M1 Max)

Sur une machine dotée de 64 Go de mémoire unifiée (UMA, bande passante de 400 Go/s) :

| Composant mémoire | Précision & Type | Taille mémoire | Rôle & Implémentation |
|---|---|:---:|---|
| **Positions et vitesses particulaires** | `Float64` (`ParticleCloud`) | **$5,76\text{ Go}$** | $80\times 10^6 \times 6 \times 8\text{ octets}$ (invariance symplectique CPU) |
| **Buffers d'accélération GPU Metal** | `Float32` (`MetalForceAccelerator`) | **$3,84\text{ Go}$** | $80\times 10^6 \times 12 \times 4\text{ octets}$ (partagés sans copie `SharedStorage`) |
| **Index et tri par cellules** | `Int32` (`CellSort`) | **$0,80\text{ Go}$** | Accélération du dépôt et de l'interpolation spline |
| **Tampon de tranches 2D en vol** | `Float32` (175 à 240 coupes) | **$0,40\text{ Go}$** | Stockage en RAM des plans de densité $z \approx 0$ pour post-traitement |
| **Total mémoire vive active** | — | **$10,4\text{ Go}$** | **16 % de la RAM totale (aucun swap, empreinte plate)** |

L'allocation de ces 10,4 Go est réalisée en **1,4 seconde** à l'initialisation du calcul.

---

## 3. Résultats des trois systèmes physiques à 80 millions

### 3.1. Traversée axiale de proton : $\text{Na}_{1000} + \text{H}^+$ ($16\text{ keV}$, $b = 0$)

* **Physique** : Traversée centrale au sommet du pic de Bragg ($v = 0,80\text{ u.a.}$).
* **Paramètres** : 80 M particules, Grille 110 (maillage spatial $222^3$ points, pas $h = 1,42\text{ a}_0$).
* **Durée de calcul** : **13,5 minutes** (812,8 s pour 200 pas de temps, $\Delta t = 1,0\text{ u.a.}$).
* **Observables mesurées** :
  - Pouvoir d'arrêt au centre : **$dE/dx = 1,541\text{ eV}/\text{a}_0$** (oracle thèse : $1,587\text{ eV}/\text{a}_0$, accord à 2,9 %).
  - Perte cinétique totale du projectile : **$\Delta E_k = 125,14\text{ eV}$**.
  - Excitation interne résiduelle du cluster : **$E_{\text{exc}} = 133,56\text{ eV}$**.
* **Phénoménologie révélée** :
  - Le sillage plasmonique oscillatoire déploie deux nœuds de surdensité/dépression parfaitement symétriques et lisses.
  - Absence totale d'îlots de discrétisation.
  - Visibilité des oscillations radiales de Friedel dans la queue Thomas-Fermi de l'agrégat.
* **Artefacts produits** :
  - Film dynamique : `film_proton_80M.mp4` et `docs/src/assets/film_proton_80M.gif`.
  - Planche 4 panneaux : `docs/src/assets/proton_snapshots_80M.png`.
  - Données brutes : `proton_converged_80M_data_cache.jls`.

---

### 3.2. Collision périphérique Xénon : $\text{Na}_{196} + \text{Xe}^{25+}$ ($500\text{ keV}$, $b = 45\text{ a}_0$)

* **Physique** : Interaction rasante ($b \approx 2\,R_{\text{jel}}$) reproduisant l'article Springer 1997 (`xenon40.ps`).
* **Paramètres** : 80 M particules, Grille 64 (maillage spatial $130^3$ points, pas $h = 2,45\text{ a}_0$).
* **Durée de calcul** : **48,4 minutes** (2 905,8 s pour 700 pas de temps, $\Delta t = 0,5\text{ u.a.}$, 176 frames).
* **Observables mesurées** :
  - Ionisation nette de l'agrégat $\text{Na}_{196}$ : **$Q_{\text{net}} = +12,87\,e$** (expulsion de 13 électrons).
  - Charge capturée par l'ion ("atome creux") : **$Q_{\text{cap}} = 0,81\,e$** (stabilisée en orbite de Rydberg).
  - Perte d'énergie cinétique du projectile : **$\Delta E_k = 19,7\text{ eV}$**.
* **Phénoménologie révélée** :
  - Déformation hydrodynamique continue de la surface métallique sous l'effet du puits de potentiel coulombien de $\text{Xe}^{25+}$.
  - Formation sans bruit du pont électronique entre $t = 2,5$ et $4,5\text{ fs}$.
  - Relaxation post-collision par oscillation plasmonique dipolaire collective.
* **Artefacts produits** :
  - Film dynamique : `film_xenon_80M.mp4` et `docs/src/assets/film_xenon_80M.gif`.
  - Planche chronologique 6 panneaux : `docs/src/assets/xenon_snapshots_80M.png`.
  - Données brutes : `xenon_converged_80M_data_cache.jls`.

---

### 3.3. Collision périphérique Argon : $\text{Na}_{40} + \text{Ar}^{8+}$ ($80\text{ keV}$, $b = 20\text{ a}_0$)

* **Physique** : Collision périphérique de référence du Chapitre 4 de la thèse (Fig. 4.1 / `Fsnap1.ps`).
* **Paramètres** : 80 M particules, Grille 64 (maillage spatial $130^3$ points, pas $h = 1,56\text{ a}_0$).
* **Durée de calcul** : **69,9 minutes** (4 196,9 s pour 960 pas de temps, $\Delta t = 0,5\text{ u.a.}$, 240 frames).
* **Observables mesurées** :
  - Ionisation résiduelle de $\text{Na}_{40}$ : **$Q_{\text{net}} = +7,97\,e$** (arrachage net de 8 électrons).
  - Charge capturée et emportée par $\text{Ar}^{8+}$ : **$Q_{\text{cap}} = 2,14\,e$**.
  - Perte d'énergie cinétique de l'ion : **$\Delta E_k = 22,8\text{ eV}$**.
* **Phénoménologie révélée** :
  - L'ion multichargé arrache un paquet dense de charge qui s'enroule autour de sa trajectoire.
  - Concordance parfaite de la séquence temporelle avec les 12 instantanés de la thèse de $T = 3,6\text{ fs}$ à $11,5\text{ fs}$.
* **Artefacts produits** :
  - Film dynamique : `film_argon_80M.mp4` et `docs/src/assets/film_argon_80M.gif`.
  - Planche chronologique 12 panneaux : `docs/src/assets/argon_snapshots_80M.png`.
  - Données brutes : `argon_converged_80M_data_cache.jls`.

---

## 4. Tableau comparatif : 1998 vs Étape 8M vs Étape 80M

| Critère | Thèse (1998, Fortran 77) | Étape intermédiaire 8M (2026) | Étape ultra-convergée 80M (2026) |
|---|:---:|:---:|:---:|
| **Nombre de macro-particules** | $400\,000$ | $8\,000\,000$ ($\times 20$) | **$80\,000\,000$ ($\times 200$)** |
| **Résolution spatiale (Proton)** | Grille 66 ($134^3$, $h = 2,36\text{ a}_0$) | Grille 88 ($178^3$, $h = 1,77\text{ a}_0$) | **Grille 110 ($222^3$, $h = 1,42\text{ a}_0$)** |
| **Bruit de Poisson par cellule** | $\approx 18 - 25 \%$ | $\approx 7,8 \%$ | **$< 2,0 \%$** |
| **Temps d'exécution Proton** | Plusieurs heures (batch C98) | $1,2\text{ min}$ | **$13,5\text{ min}$** |
| **Temps d'exécution Xénon** | N/A (non convergé) | $5,5\text{ min}$ | **$48,4\text{ min}$** |
| **Temps d'exécution Argon** | N/A (batch partiel) | $7,8\text{ min}$ | **$69,9\text{ min}$** |
| **Rendu vidéo d'un film (200 frames)** | Hors de portée | $8\text{ min}$ (`contourf!`) | **$7,8\text{ s}$ (`heatmap!` interpolée)** |
| **Aspect visuel** | Isolignes discontinues | Texture légèrement granuleuse | **Fluide continu sans grain** |

---

## 5. Pipeline d'automatisation et conservation des résultats

1. **Scripts autonomes de production** :
   - `scripts/film_proton_80M.jl`
   - `scripts/film_xenon_80M.jl`
   - `scripts/film_argon_80M.jl`
   - `scripts/pipeline_80M_night.jl` (orchestrateur global avec reprise automatique sur cache).
2. **Préservation des étapes intermédiaires** :
   - Les fichiers intermédiaires `_converged` (8 millions de particules) sont conservés dans `docs/src/assets/` et utilisables pour l'analyse de convergence.
3. **Documentation officielle** :
   - Pages mises à jour sous le label *"25 ans plus tard"* :
     * [`docs/src/chapter5_multicharged.md`](src/chapter5_multicharged.md)
     * [`docs/src/chapter6_stopping.md`](src/chapter6_stopping.md)
   - Compilation Documenter.jl validée : `julia --project=docs docs/make.jl` (exit code 0).
