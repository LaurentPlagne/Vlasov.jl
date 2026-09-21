#!/usr/bin/env julia
"""
Builds `run/`, the one environment this machine needs.

    julia scripts/setup.jl
    julia --project=run -t auto scripts/xenon.jl

Two commands, the same on every machine — which is the point. `Metal` and
`CUDA` cannot share a `Project.toml`: each installs artifacts the other
platform has no build for, so an environment carrying both is an environment
that instantiates nowhere. The choice has to be made *somewhere*; this makes it
here, by looking, rather than in the README by asking the reader to know.

What it looks at, and what it adds:

  * **Apple Silicon** (`Sys.isapple()` and an ARM64 processor) → `Metal`, and
    `AppleAccelerate`, which is worth ×1.31 on the host loops for one line;
  * **an NVIDIA card** (`nvidia-smi` answers) → `CUDA`;
  * **neither** → nothing, and the run takes the processor.

  * **a display** → `GLMakie`, which draws the film in 3.9 s;
  * **none** → `CairoMakie`, which needs no window and takes 31 s.

⚠️ **`run/` is generated, never committed** — neither its `Project.toml` nor
its `Manifest.toml`. A manifest resolved on one machine pins versions and
artifacts another cannot replay, and the failure it causes names nothing. Run
this again whenever the machine changes.

⚠️ Nothing here is fatal. Each `add` reports rather than throwing, so a vendor
package that will not install leaves an environment that still runs — on the
processor, saying so.

The repository's other environments (`gpu/`, `cuda/`, `viz/`, `docs/`) are the
ones its own scripts and its CI use; they are not what a first run needs.
"""

using Pkg

const ROOT = dirname(@__DIR__)
const ENVDIR = joinpath(ROOT, "run")

apple_silicon() = Sys.isapple() && Sys.ARCH === :aarch64
nvidia() = Sys.which("nvidia-smi") !== nothing
has_display() = Sys.isapple() || haskey(ENV, "DISPLAY") || haskey(ENV, "WAYLAND_DISPLAY")

"""Adds one package, and says so rather than throwing: a vendor stack that
refuses to install must not cost the user the environment."""
function try_add(name)
    print("  $name ... ")
    flush(stdout)
    try
        Pkg.add(name; io = devnull)
        println("ok")
        true
    catch e
        println("FAILED")
        @warn "could not add $name — the run will do without it" exception = (e, catch_backtrace())
        false
    end
end

function main()
    println("Looking at this machine:")
    println("  operating system   $(Sys.KERNEL), $(Sys.ARCH)")
    println("  Apple Silicon      $(apple_silicon() ? "yes" : "no")")
    println("  nvidia-smi         $(nvidia() ? "yes, $(Sys.which("nvidia-smi"))" : "not found")")
    println("  display            $(has_display() ? "yes" : "none — headless")")
    println()

    mkpath(ENVDIR)
    Pkg.activate(ENVDIR)
    println("Building $ENVDIR:")
    print("  Vlasov ... ")
    flush(stdout)
    Pkg.develop(path = ROOT; io = devnull)
    println("ok")

    if apple_silicon()
        try_add("Metal")
        try_add("AppleAccelerate")
    elseif nvidia()
        try_add("CUDA")
    end
    try_add(has_display() ? "GLMakie" : "CairoMakie")

    print("\nResolving ... ")
    flush(stdout)
    Pkg.instantiate(; io = devnull)
    println("ok\n")

    println("Done. The one command from here, on any machine:\n")
    println("    julia --project=run -t auto scripts/xenon.jl\n")
    if !apple_silicon() && !nvidia()
        println("No GPU was found, so that will run on the processor — about")
        println("three and a half minutes. `--particules=100000 --nfine=28` is a")
        println("minute, and tells the same story.")
    end
end

main()
