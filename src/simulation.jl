"""
Paramètres de simulation et boucle en temps.

Le Fortran lisait ses paramètres dans `vlas.inp`, un fichier positionnel où
chaque valeur est précédée de son commentaire : intervertir deux lignes passe
inaperçu et change la physique. Ici ils sont nommés.
"""

"""
    SimulationParameters(; …)

Paramètres d'une simulation. Les noms du Fortran sont rappelés en regard.

  * `nfine` (`n1xyz`) — intervalles de la grille fine, sur `[-rcluster, rcluster]` ;
  * `ninner`, `nouter` (`n1big`, `n2big`) — découpage de la grille grossière ;
  * `rcluster`, `rbox` (`xclu`, `xboite`) — rayons des deux domaines ;
  * `nions`, `nelectrons` (`nbion`, `nbelec`) ;
  * `nparticles` (`npart`) — pseudo-particules ;
  * `nsteps`, `dt` (`nbt`, `dltt`) ;
  * `rcmax` — rayon au-delà duquel un électron est compté comme **sorti** de
    l'agrégat. Codé en dur à `100.d0` jusqu'en 1997 ; devenu la 19ᵉ valeur de
    `vlas.inp` dans la version 1998-01-05, où `vlas.inp` de production le
    documente « rayon considéré comme inner cluster ». La valeur par défaut
    reproduit donc le comportement d'avant.
"""
Base.@kwdef struct SimulationParameters{T<:AbstractFloat}
    nfine::Int = 28
    ninner::Int = 14
    nouter::Int = 14
    rcluster::T = 50.0
    rbox::T = 150.0
    nions::T = 196.0
    nelectrons::T = 196.0
    nparticles::Int = 20_000
    nsteps::Int = 10
    dt::T = 1.0
    rcmax::T = 100.0
end

"""
    read_parameters(path) -> SimulationParameters

Lit un `vlas.inp` du code d'origine : une ligne de commentaire, une ligne de
valeur, en alternance. Les paramètres de projectile qui suivent sont ignorés —
ils ne concernent pas l'agrégat isolé.
"""
function read_parameters(path::AbstractString)
    vals = String[]
    open(path) do io
        for (i, line) in enumerate(eachline(io))
            isodd(i) || push!(vals, strip(line))   # lignes paires = valeurs
        end
    end
    length(vals) >= 10 ||
        throw(ArgumentError("$path : 10 valeurs attendues, $(length(vals)) trouvées"))
    num(s) = parse(Float64, replace(s, "d" => "e", "D" => "e"))
    # `rcmax` est la 19ᵉ valeur, absente des `vlas.inp` d'avant 1998 : on
    # retombe alors sur le `100.d0` que le Fortran codait en dur.
    rcmax = length(vals) >= 19 && !isempty(vals[19]) ? num(vals[19]) : 100.0
    SimulationParameters(
        nfine = Int(num(vals[1])), ninner = Int(num(vals[2])), nouter = Int(num(vals[3])),
        rcluster = num(vals[4]), rbox = num(vals[5]),
        nions = num(vals[6]), nelectrons = num(vals[7]),
        nparticles = Int(num(vals[8])), nsteps = Int(num(vals[9])), dt = num(vals[10]),
        rcmax = rcmax)
end

"""
    Simulation(params, profile)

Tout ce qui est constant au cours d'une simulation — maillages emboîtés,
tables de lissage, fond de jellium — plus l'état qui évolue : le nuage de
pseudo-particules.

Construire une `Simulation` fait le travail lourd une fois : assemblage des
matrices, diagonalisations, tables de convolution. Les pas qui suivent ne
réutilisent que des multiplications.
"""
struct Simulation{T<:AbstractFloat,P}
    params::SimulationParameters{T}
    meshes::NestedMeshes{2,3,T,BandedMatrix{T,Matrix{T},Base.OneTo{Int}}}
    smoothing::GaussianSmoothing{T}
    jellium::Jellium{T}
    cloud::ParticleCloud{T}
    """Projectile, ou `nothing` pour un agrégat isolé. Le type le porte plutôt
       qu'un champ `Union` : la boucle reste spécialisée dans les deux cas."""
    projectile::P
    ρ::NTuple{2,Array{T,3}}
    φ::NTuple{2,Array{T,3}}
    """Coefficients spline du potentiel, un jeu par niveau. Ils vivent le temps
       d'un pas entier : leur donner des tampons propres évite 2,8 Mo
       d'allocations par pas, et le ramasse-miettes qui va avec."""
    csol::NTuple{2,Array{T,3}}
    """Tampons de diffusion, un par fil et par niveau, pour le dépôt parallèle.
       Voir `ScatterBuffers` : leur coût mémoire croît comme le cube de la
       grille, d'où leur présence explicite ici plutôt qu'une création en
       douce à chaque dépôt."""
    scatter::NTuple{2,ScatterBuffers{T,3}}
end

function Simulation(p::SimulationParameters{T}, profile::PhaseSpaceProfile{T};
                    rng::Ran2 = Ran2(-1), consistent_startup::Bool = false,
                    projectile = nothing) where {T}
    fine = uniform_axis(-p.rcluster, p.rcluster, p.nfine)
    coarse = stretched_axis(p.rcluster, p.rbox, p.ninner ÷ 2, (p.nouter + 2) ÷ 2)
    meshes = NestedMeshes(SplineMesh(fine, fine, fine),
                          SplineMesh(coarse, coarse, coarse))

    weight = p.nelectrons / p.nparticles
    positions, momenta = sample_thomas_fermi(profile, p.nparticles, weight; rng)
    n = nbasis(fine)
    sim = Simulation(p, meshes, GaussianSmoothing(fine), Jellium(p.nions),
                     ParticleCloud(positions, weight), projectile,
                     (zeros(T, n, n, n), zeros(T, n, n, n)),
                     (zeros(T, n, n, n), zeros(T, n, n, n)),
                     (zeros(T, n, n, n), zeros(T, n, n, n)),
                     (ScatterBuffers(meshes[1]), ScatterBuffers(meshes[2])))
    prime_leapfrog!(sim, positions, momenta; consistent = consistent_startup)
end

"""
    prime_leapfrog!(sim, positions, momenta; consistent) -> sim

Amorce le schéma de Verlet, qui a besoin de **deux** positions et non d'une
position et d'une vitesse.

En deux temps, comme `moveback1` puis `moveback2` du Fortran : un demi-pas en
arrière depuis les impulsions, puis le pas complet, qui demande les forces —
donc un potentiel, donc un dépôt et une résolution de Poisson.

⚠️ Sauter le second temps ne laisse pas le nuage « approximativement » amorcé :
`previous` vaudrait `q(−dt/2)` là où le schéma attend `q(−dt)`, soit une
vitesse initiale fausse d'un facteur deux. Le symptôme est une énergie
cinétique qui monte dès les premiers pas.
"""
function prime_leapfrog!(sim::Simulation{T}, positions, momenta;
                         consistent::Bool = false) where {T}
    M = mass(sim.cloud)
    dt = sim.params.dt
    half = half_step_back(positions, momenta, M, dt)

    # Les forces s'évaluent en q(−dt/2), pas en q(0). Le projectile, lui, ne
    # bouge pas : le Fortran n'appelle `incproj` qu'une fois l'amorçage fini,
    # et l'avancer ici le ferait entrer dans l'agrégat avec un pas d'avance.
    copyto!(sim.cloud.positions, half)
    update_forces!(sim; advance = false)

    copyto!(sim.cloud.previous,
            full_step_back(positions, half, sim.cloud.forces, M, dt; consistent))
    copyto!(sim.cloud.positions, positions)
    sim
end

"""
    update_forces!(sim) -> T

Dépose les particules, résout Poisson sur les deux grilles, ajoute le champ
moyen et remplit les forces du nuage. Renvoie l'énergie de Hartree, mesurée
**avant** l'ajout du champ moyen — c'est le seul moment où le potentiel nu est
disponible.

`advance = false` calcule les forces sans faire avancer le projectile, ce dont
l'amorçage a besoin.
"""
function update_forces!(sim::Simulation{T}; advance::Bool = true) where {T}
    fine, coarse = sim.meshes[1], sim.meshes[2]
    ρf, ρc = sim.ρ
    w = sim.cloud.weight

    deposit_smoothed!(ρf, fine, sim.smoothing, sim.cloud.positions;
                      charge = w, buffers = sim.scatter[1])
    deposit!(ρc, coarse, sim.cloud.positions; charge = w, buffers = sim.scatter[2])
    poisson!(sim.φ, sim.ρ, sim.meshes)

    csolf = spline_coefficients!(sim.csol[1], sim.φ[1], fine)
    csolc = spline_coefficients!(sim.csol[2], sim.φ[2], coarse)
    hartree = interaction_energy(sim.cloud, fine.axes, csolf,
                                 coarse.axes, csolc, sim.smoothing) / 2

    effective_potential!(csolf, ρf, fine, sim.jellium)
    effective_potential!(csolc, ρc, coarse, sim.jellium)
    forces!(sim.cloud, fine.axes, csolf, coarse.axes, csolc, sim.smoothing)
    advance && advance_projectile!(sim)

    # Le potentiel total sert ensuite au bilan : on le garde sous la main.
    sim.φ[1] .= csolf
    sim.φ[2] .= csolc
    hartree
end

"""
    step!(sim) -> EnergyBudget

Un pas de temps complet, dans l'ordre du code d'origine :

 1. dépôt des particules sur les deux grilles — lissé sur la fine ;
 2. Poisson emboîté, du grossier au fin, qui donne le potentiel de Hartree ;
 3. énergie de Hartree, mesurée **sur ce potentiel-là**, avant qu'on y ajoute
    quoi que ce soit ;
 4. ajout de l'échange-corrélation et du jellium ;
 5. forces, puis avance de Verlet ;
 6. bilan d'énergie, mesuré sur le potentiel **total**.

L'ordre des points 3 et 6 n'est pas un détail de commodité : les deux termes
du bilan se réfèrent à des potentiels différents, et les intervertir rendrait
le total silencieusement faux.
"""
function step!(sim::Simulation{T}) where {T}
    hartree = update_forces!(sim)
    diag = step!(sim.cloud, sim.params.dt; rcmax = sim.params.rcmax)
    total = interaction_energy(sim.cloud, sim.meshes[1].axes, sim.φ[1],
                               sim.meshes[2].axes, sim.φ[2], sim.smoothing)
    energy_budget(sim.jellium, diag.kinetic, hartree, total, diag.escaped)
end

"""
    run!(sim; nsteps, callback) -> Vector{EnergyBudget}

Enchaîne `nsteps` pas et rend l'historique du bilan d'énergie — l'observable
de stabilité du chapitre 4. `callback(i, budget)` est appelée après chaque pas.
"""
function run!(sim::Simulation; nsteps::Integer = sim.params.nsteps,
              callback = (i, b) -> nothing)
    history = EnergyBudget{Float64}[]
    for i in 1:nsteps
        b = step!(sim)
        push!(history, b)
        callback(i, b)
    end
    history
end

"""
    advance_projectile!(sim)

Ajoute l'interaction du projectile — force sur lui, réaction sur les
pseudo-particules — puis l'avance d'un pas.

Sans projectile, ne fait rien : le corps est éliminé à la compilation, la
boucle de l'agrégat isolé n'en paie pas le prix.
"""
advance_projectile!(::Simulation{T,Nothing}) where {T} = nothing

function advance_projectile!(sim::Simulation{T,Projectile{T}}) where {T}
    force, _, _ = projectile_forces!(sim.cloud, sim.projectile, sim.jellium)
    step!(sim.projectile, force, sim.params.dt)
    nothing
end
