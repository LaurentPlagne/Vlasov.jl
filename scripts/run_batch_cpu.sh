#!/bin/bash
set -e

DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DIR"

echo "=== Démarrage du calcul Figure 5.3 très convergé (CPU Float64 + Accelerate) ==="
date

# Figure 5.3 : Courbe de pouvoir d'arrêt dE/dx (CPU Float64, 1.6M particules, grille 66, 12 énergies)
echo "--- Lancement de Figure 5.3 (Grille 66, 1.6M particules, 12 énergies avec Pic de Bragg) ---"
julia --project=gpu -t auto scripts/figure53.jl \
    --particules=1600000 \
    --nfine=66 \
    --kev=1,4,9,12,14,16,18,20,25,36,50,64 \
    --sortie=figure53.png > fig53.log 2>&1

echo "✓ Figure 5.3 terminée (voir fig53.log et figure53.png)"
echo "=== Calcul CPU Figure 5.3 terminé avec succès ==="
date
