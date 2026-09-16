#!/bin/bash
set -e

DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DIR"

echo "=== Démarrage des gros calculs batch Vlasov.jl ==="
date

# 1. Figure 5.2 : Coupes de densité (GPU)
echo "--- Lancement de Figure 5.2 (GPU, 3.2M particules) ---"
julia --project=gpu -t auto scripts/figure52.jl --particules=3200000 --sortie=figure52.png > fig52.log 2>&1
echo "✓ Figure 5.2 terminée (voir fig52.log et figure52.png)"

# 2. Film : Génération des coupes puis encodage vidéo/GIF (GPU)
echo "--- Lancement du Film 16 keV (GPU, 1.6M particules, 300 images) ---"
julia --project=gpu -t auto scripts/film_images.jl --kev=16 --images=300 --particules=1600000 > film_images.log 2>&1
echo "--- Rendu vidéo MP4 et GIF ---"
julia --project=gpu scripts/make_film.jl --champ=rho --fps=25 > make_film.log 2>&1
echo "✓ Film terminé (voir film.mp4 et film.gif)"

echo "=== Calculs GPU batch terminés avec succès ==="
date
