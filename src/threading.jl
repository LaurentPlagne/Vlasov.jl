"""
Découpage en tranches pour les boucles parallèles.

Les boucles qui pèsent dans un pas de temps sont toutes de la même forme : un
parcours indépendant des pseudo-particules ou des points de grille, suivi
éventuellement d'une somme. On les découpe en tranches **contiguës** — et non
entrelacées — pour que chaque fil travaille sur une zone de mémoire continue.
"""

using LinearAlgebra: BLAS

"""
Interrupteur global du parallélisme.

Sert à deux choses : comparer en **alternance** une version parallèle et une
version séquentielle sur la même machine au même instant — seule façon de
mesurer quoi que ce soit quand la charge varie — et retrouver un
comportement déterministe quand on débogue.

    Vlasov.PARALLEL[] = false
"""
const PARALLEL = Ref(true)

"""Nombre de tranches à utiliser : 1 si le parallélisme est coupé."""
@inline nchunks_now() = PARALLEL[] ? Threads.nthreads() : 1

"""
    configure_blas!(; threads = 2)

Limite le nombre de fils d'OpenBLAS.

⚠️ **Le défaut d'OpenBLAS est mauvais ici.** Il prend autant de fils que de
cœurs, en plus de ceux de Julia, et les deux se disputent la machine. Mesuré
sur `poisson!`, 8 fils Julia sur 10 cœurs :

| fils BLAS | médiane | étendue (max/min) |
|---|---|---|
| 1 | 11,1 ms | 1,1× |
| 2 | 9,6 ms | 1,4× |
| 4 | 8,6 ms | 2,6× |
| 8 *(défaut)* | **20,5 ms** | **5,4×** |

Le défaut n'est pas seulement deux fois plus lent en médiane : il rend les
temps **imprévisibles**, du simple au quintuple. Deux fils donnent le meilleur
compromis entre vitesse et régularité.

N'est appelée nulle part automatiquement : changer un réglage global à l'insu
de l'appelant serait discourtois. À invoquer avant une campagne de calcul.
"""
function configure_blas!(; threads::Integer = 2)
    BLAS.set_num_threads(threads)
    threads
end

"""
    chunks(n, nchunks = nchunks_now()) -> Vector{UnitRange}

Découpe `1:n` en tranches contiguës de tailles aussi égales que possible.
Rend une tranche vide de moins que demandé plutôt que des tranches vides.
"""
function chunks(n::Integer, nchunks::Integer = nchunks_now())
    nchunks = max(1, min(nchunks, n))
    base, reste = divrem(n, nchunks)
    stop = 0
    map(1:nchunks) do c
        start = stop + 1
        stop += base + (c <= reste)
        start:stop
    end
end

"""
    tmapreduce(f, n) -> T

Somme `f(range)` sur les tranches de `1:n`, en parallèle. `f` reçoit une
**tranche** et non un indice : c'est à elle de boucler, ce qui lui laisse
accumuler dans une variable locale plutôt que dans un tableau partagé.

Retombe sur un appel direct à un seul fil, pour que le résultat soit
identique — à l'ordre de sommation près — et que le surcoût disparaisse.
"""
function tmapreduce(f, n::Integer)
    parts = chunks(n)
    length(parts) == 1 && return f(parts[1])
    results = Vector{Any}(undef, length(parts))
    Threads.@threads for c in eachindex(parts)
        results[c] = f(parts[c])
    end
    reduce(_addall, results)
end

"""Addition terme à terme, qui accepte aussi bien des nombres que des tuples."""
@inline _addall(a::Number, b::Number) = a + b
@inline _addall(a::Tuple, b::Tuple) = map(_addall, a, b)

"""
    tforeach(f, n)

Applique `f` à chaque tranche de `1:n`, en parallèle, sans rien collecter.
Pour les boucles qui écrivent chacune dans leur propre case.
"""
function tforeach(f, n::Integer)
    parts = chunks(n)
    length(parts) == 1 && return f(parts[1])
    Threads.@threads for c in eachindex(parts)
        f(parts[c])
    end
    nothing
end
