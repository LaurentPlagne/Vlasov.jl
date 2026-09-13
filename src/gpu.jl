"""
Interface du calcul des forces sur accélérateur.

Le portage GPU vise les **boucles sur les particules**, qui font 79 % d'un pas
à l'échelle de production — et non le solveur tensoriel, qui n'en fait que 14 %
et plafonnerait donc à ×1,16.

Rien ici ne dépend d'un backend : les méthodes sont fournies par une extension
de paquet (`ext/VlasovMetalExt.jl` pour Metal), chargée seulement si l'utilisateur
charge `Metal`. Sans backend, ces fonctions lèvent une erreur explicite et le
chemin CPU reste le seul.

⚠️ **Le GPU travaille en `Float32`.** Les GPU Apple n'ont pas de double
précision — Metal n'a pas de type `double`. Le chemin CPU `Float64` reste donc
la référence, celle qui se compare à l'oracle Fortran à `1e-13` ; le chemin GPU
se valide contre lui, au niveau où la physique le demande.
"""

"""
    ForceAccelerator

Ce qu'un backend doit fournir pour prendre en charge l'évaluation du champ
lissé. Le type est déclaré ici pour que [`forces!`](@ref) puisse le nommer ;
les réalisations vivent dans les extensions.

    ForceAccelerator(MtlArray, fine_axes, smoothing, npart, n)

Prépare le calcul pour une grille et un nombre de particules donnés : tables et
tampons sont alloués **une fois**, pas à chaque pas. Les allocations tuent le
parallélisme, sur GPU plus encore.
"""
abstract type ForceAccelerator end

"""
    forces!(cloud, acc, csol_fine, coarse, csol_coarse, sm; escaped) -> Int

Même contrat que la méthode CPU de [`forces!`](@ref), l'évaluation du champ
lissé en moins : elle part sur l'accélérateur `acc`.

Les particules trop près du bord pour que le pochoir 10³ tienne dans la grille
sont **repassées au CPU** : elles sont rares, et les traiter sur GPU
demanderait des branches là où l'intérêt est justement de n'en avoir aucune.
"""
function forces! end

"""
    deposit_smoothed!(ρ, acc, mesh, sm, positions; charge) -> nout

Même contrat que la méthode CPU de [`deposit_smoothed!`](@ref), le *scatter* en
moins : il part sur l'accélérateur.

Le dépôt est la partie difficile à porter — chaque particule écrit dans 8³
points, et les voisines écrivent aux mêmes. La voie naïve, une addition
atomique par point et par particule, est **trois fois plus lente que le CPU** :
410 millions d'atomiques en conflit, que le GPU sérialise. La voie retenue
range d'abord les particules par maille ([`CellSort`](@ref)) et confie une
maille à un groupe de fils, chacun propriétaire d'un point du pochoir — une
atomique par point et par **maille**, soit cent fois moins.

Voir `docs/gpu.md` pour les mesures.
"""
function deposit_smoothed! end
