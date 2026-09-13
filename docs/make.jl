using Documenter, Vlasov

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
        "Numerics" => "numerics.md",
        "Architecture" => "architecture.md",
        "Validation" => "validation.md",
        "Performance" => "performance.md",
        "The original code" => "history.md",
        "API reference" => "reference.md",
    ],
    checkdocs = :exports,
)

deploydocs(
    repo = "github.com/laurentplagne/Vlasov.jl.git",
)
