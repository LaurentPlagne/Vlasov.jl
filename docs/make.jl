using Documenter, Vlasov

# The architecture diagrams, rendered by `dot` into `src/assets/diagrams/`.
# Their sources are DOT, so they diff like code; the SVG is produced **here**
# rather than in the reader's browser.
#
# ⚠️ Not a client-side renderer. Mermaid and its kin import their script as an
# ES module, which a page opened over `file://` refuses to load — the diagram
# then shows as its own source text, silently. This site is read locally
# (`open docs/build/index.html`), so nothing may depend on the network or on
# JavaScript.
include(joinpath(@__DIR__, "diagrams", "make_diagrams.jl"))

# The figures are generated at build time by the `@example` blocks; CairoMakie
# is loaded there, not here. Loading it once up front only warms the cache.
using CairoMakie
CairoMakie.activate!(type = "png")

makedocs(
    modules = [Vlasov],
    # Stated explicitly rather than read from `git remote`: this repository has
    # no origin configured, and Documenter would refuse to build without it.
    repo = Documenter.Remotes.GitHub("laurentplagne", "Vlasov.jl"),
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://laurentplagne.github.io/Vlasov.jl",
        assets = String[],
        size_threshold = 500 * 1024,
    ),
    authors = "Laurent Plagne",
    sitename = "Vlasov.jl",
    pages = [
        "Home" => "index.md",
        "Principles" => "principles.md",
        "Architecture" => "architecture.md",
        "The device path" => "device.md",
        "TBSCM Poisson Solver" => "solver_tbscm.md",
        "Numerics" => "numerics.md",
        "Chapter 4: Stability & Entropy" => "chapter4_stability.md",
        "Chapter 5: Multicharged Collisions" => "chapter5_multicharged.md",
        "Chapter 6: Proton Stopping & Wakes" => "chapter6_stopping.md",
        "25-Year Computing Leap" => "computing_evolution.md",
        "Validation" => "validation.md",
        "Performance" => "performance.md",
        "The original code" => "history.md",
        "API reference" => "reference.md",
    ],
    checkdocs = :none,
)

deploydocs(
    repo = "github.com/laurentplagne/Vlasov.jl.git",
)
