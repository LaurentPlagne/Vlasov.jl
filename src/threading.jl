"""
Découpage en tranches pour les boucles parallèles.

Les boucles qui pèsent dans un pas de temps sont toutes de la même forme : un
parcours indépendant des pseudo-particules ou des points de grille, suivi
éventuellement d'une somme. On les découpe en tranches **contiguës** — et non
entrelacées — pour que chaque fil travaille sur une zone de mémoire continue.
"""

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
