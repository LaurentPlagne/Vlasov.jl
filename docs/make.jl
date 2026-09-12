push!(LOAD_PATH,joinpath(@__DIR__, ".."))
using Documenter, Vlasov

makedocs(
    modules = [Vlasov],
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    authors = "Laurent Plagne",
    sitename = "Vlasov.jl",
    pages = Any["index.md"]
    # strict = true,
    # clean = true,
    # checkdocs = :exports,
)

deploydocs(
    repo = "github.com/laurentplagne/Vlasov.jl.git",
)
