```@meta
CurrentModule = Vlasov
```

# API reference

Everything the module exports, grouped by the layer it belongs to. The layers
are those of [Architecture](@ref); within each, the order is the source order.

```@index
```

## Driver

The time loop and its parameters.

```@autodocs
Modules = [Vlasov]
Pages   = ["simulation.jl"]
```

## Projectile

```@autodocs
Modules = [Vlasov]
Pages   = ["projectile.jl"]
```

## Entropy and Thermodynamics

```@autodocs
Modules = [Vlasov]
Pages   = ["entropy.jl"]
```

## Energy

```@autodocs
Modules = [Vlasov]
Pages   = ["energy.jl"]
```

## Initial state

```@autodocs
Modules = [Vlasov]
Pages   = ["initial.jl"]
```

## Mean field

```@autodocs
Modules = [Vlasov]
Pages   = ["meanfield.jl"]
```

## Fields and forces

```@autodocs
Modules = [Vlasov]
Pages   = ["fields.jl"]
```

## Particles

```@autodocs
Modules = [Vlasov]
Pages   = ["particles.jl"]
```

## Sorting

```@autodocs
Modules = [Vlasov]
Pages   = ["sorting.jl"]
```

## Accelerators

```@autodocs
Modules = [Vlasov]
Pages   = ["gpu.jl"]
```

## The device path

The portable kernels, the accelerator that drives them, and the device-side
mirror of a mesh. Narrated in [The device path](device.md); the kernels are listed
here because the prose refers to them by name.

`devicemesh.jl` is deliberately not auto-documented: most of what it holds are
device methods of functions already listed above — `poisson!`, `deposit_cic!` —
and listing the file again would document them twice.

```@autodocs
Modules = [Vlasov]
Pages   = ["accelerator.jl", "kernels.jl"]
```

## Poisson

```@autodocs
Modules = [Vlasov]
Pages   = ["poisson.jl"]
```

## Deposition

```@autodocs
Modules = [Vlasov]
Pages   = ["deposition.jl"]
```

## Meshes

```@autodocs
Modules = [Vlasov]
Pages   = ["mesh.jl"]
```

## Tensor solver

```@autodocs
Modules = [Vlasov]
Pages   = ["tensorsolver.jl"]
```

## Collocation

```@autodocs
Modules = [Vlasov]
Pages   = ["collocation.jl"]
```

## Splines

```@autodocs
Modules = [Vlasov]
Pages   = ["splines.jl"]
```

## Threading

```@autodocs
Modules = [Vlasov]
Pages   = ["threading.jl"]
```

## Random numbers

```@autodocs
Modules = [Vlasov]
Pages   = ["random.jl"]
```
