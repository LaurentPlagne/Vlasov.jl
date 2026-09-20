# Generates the documentation's diagrams as SVG, at build time.
#
# ⚠️ Generated, not rendered in the browser. A client-side renderer (mermaid and
# the like) imports its script as an ES module, which a page opened over
# `file://` may not load: the diagram then shows as its own source text. The
# site is read locally here — `open docs/build/index.html` — so the pictures are
# produced by `dot` while the docs are built, and the pages carry plain SVG.

using Graphviz_jll

include(joinpath(@__DIR__, "common.dot.jl"))

"""Renders one DOT source to `docs/src/assets/diagrams/<name>.svg`."""
function render(name::AbstractString, src::AbstractString)
    out = joinpath(@__DIR__, "..", "src", "assets", "diagrams")
    mkpath(out)
    path = joinpath(out, name * ".svg")
    # `dot` reports a syntax error on stderr and exits non-zero; without this
    # the failure arrives as a bare `ProcessExited(1)` naming no line.
    err = IOBuffer()
    svg = IOBuffer()
    ok = Graphviz_jll.dot() do exe
        success(pipeline(pipeline(IOBuffer(src), `$exe -Tsvg`); stdout = svg, stderr = err))
    end
    ok || error("dot failed on $name:\n" * String(take!(err)))
    write(path, take!(svg))
    path
end

# --- the boundary a step crosses -------------------------------------------

render("step-boundary", """
digraph {
$(preamble(rankdir = "TB"))
  subgraph cluster_dev {
    $(cluster("<B>device</B> — the cloud never leaves"))
    cloud [$(role(:device, shape = "cylinder")), label=<cloud<BR/><FONT POINT-SIZE="9">(k, δ) · previous · forces</FONT>>];
    hist  [$(role(:device)), label=<histogram<BR/><FONT POINT-SIZE="9">_hist_cells_kernel!</FONT>>];
    sort  [$(role(:device)), label=<placement, two stages<BR/><FONT POINT-SIZE="9">coarse → fine</FONT>>];
    dep   [$(role(:device)), label=<deposition<BR/><FONT POINT-SIZE="9">fine 8³ · coarse CIC</FONT>>];
    pois  [$(role(:device)), label=<poisson!<BR/><FONT POINT-SIZE="9">coarse, then fine</FONT>>];
    mf    [$(role(:device)), label=<effective_potential!>];
    force [$(role(:device)), label=<forces<BR/><FONT POINT-SIZE="9">projectile fused in</FONT>>];
    verl  [$(role(:device)), label=<Verlet<BR/><FONT POINT-SIZE="9">_verlet_packed_kernel!</FONT>>];
    cloud -> hist -> sort -> dep -> pois -> mf -> force -> verl;
    verl -> cloud [style=dashed, constraint=false, label="in place"];
  }
  subgraph cluster_host {
    $(cluster("<B>host</B> — what is left"))
    scan    [$(role(:host)), label=<scan of the counts<BR/><FONT POINT-SIZE="9">O(cells), sequential</FONT>>];
    outside [$(role(:host)), label=<out-of-stencil forces<BR/><FONT POINT-SIZE="9">a few hundred particles</FONT>>];
    budget  [$(role(:host)), label=<energy budget<BR/><FONT POINT-SIZE="9">one step in ten</FONT>>];
  }
  hist  -> scan    [style=dotted, label="counts"];
  scan  -> sort    [style=dotted, label="offsets"];
  mf    -> outside [style=dotted, label="csol coarse"];
  outside -> force [style=dotted];
  force -> budget  [style=dotted, label="φ"];
}
""")

# --- splitting every function in two ---------------------------------------

render("split-in-two", """
digraph {
$(preamble(rankdir = "TB"))
  mesh [$(role(:object)), label=<<B>SplineMesh</B><BR/><FONT POINT-SIZE="9">banded factorisations, eigenbasis,<BR/>locator tables — none of it for a GPU</FONT>>];
  dm   [$(role(:device)), label=<<B>DeviceMesh</B><BR/><FONT POINT-SIZE="9">the short list the loop reads,<BR/>copied once, in E</FONT>>];
  m1   [$(role(:object)), label=<poisson_rhs!(rhs, ρ, <B>mesh</B>, φ)>];
  m2   [$(role(:device)), label=<poisson_rhs!(rhs, ρ, <B>dm</B>, φ)>];
  core [$(role(:good)), label=<<B>_poisson_rhs!</B>(rhs, ρ, φ, laplacians, interior)<BR/><FONT POINT-SIZE="9">never asks where its arrays live</FONT>>];
  mesh -> dm [label="mirrored once per run"];
  mesh -> m1 [style=invis];
  { rank=same; m1; m2; }
  m1 -> core [label="supplies tables"];
  m2 -> core [label="supplies tables"];
  mesh -> m1 [constraint=false];
  dm -> m2 [constraint=false];
}
""")

# --- DualBuffer: the one thing that is genuinely hardware -------------------

render("dual-buffer", """
digraph {
$(preamble(rankdir = "TB"))
  subgraph cluster_u {
    $(cluster("<B>unified memory</B> — Apple Silicon"))
    uh [$(role(:host)), label=<host face<BR/><FONT POINT-SIZE="9">unsafe_wrap(Array, mtl)</FONT>>];
    ub [$(role(:good, shape = "cylinder")), label=<<B>one allocation</B>>];
    ud [$(role(:device)), label=<device face<BR/><FONT POINT-SIZE="9">MtlArray</FONT>>];
    uh -> ub [dir=none];
    ub -> ud [dir=none];
  }
  subgraph cluster_d {
    $(cluster("<B>discrete GPU</B>"))
    dh  [$(role(:host)), label=<host face>];
    dhb [$(role(:host, shape = "cylinder")), label=<host allocation>];
    ddb [$(role(:device, shape = "cylinder")), label=<device allocation>];
    dd  [$(role(:device)), label=<device face>];
    dh -> dhb [dir=none];
    dhb -> ddb [dir=both, label="a real copy", color="#a44a4a", fontcolor="#a44a4a"];
    ddb -> dd [dir=none];
  }
  note [shape=plaintext, fontcolor="$EDGE", label=<the same call sites — <B>upload!</B> / <B>download!</B><BR/><FONT POINT-SIZE="9">free on the left, a transfer on the right</FONT>>];
  ud -> note [style=dotted];
  dd -> note [style=dotted];
}
""")

# --- why the cloud changed shape -------------------------------------------

render("packed-positions", """
digraph {
$(preamble(rankdir = "TB"))
  subgraph cluster_bad {
    $(cluster("<B>deriving δ from an absolute coordinate</B>"))
    a1 [$(role(:bad)), label=<x = 78.000<B>4213</B> a₀<BR/><FONT POINT-SIZE="9">Float32 resolves 7.6e-6 at this magnitude</FONT>>];
    a2 [$(role(:bad)), label=<− knot = 77.99929…>];
    a3 [$(role(:bad)), label=<δ = 0.00113…<BR/><FONT POINT-SIZE="9"><B>the leading digits cancelled</B><BR/>what survives is the resolution of 78, not of δ</FONT>>];
    av [shape=plaintext, fontcolor="#a44a4a", label=<<B>0.54 %</B> of a table column<BR/>0.05 % of particles pick the wrong sample>];
    a1 -> a2 -> a3 -> av;
  }
  subgraph cluster_good {
    $(cluster("<B>holding (k, δ)</B>"))
    b1 [$(role(:good)), label=<k = 111<BR/><FONT POINT-SIZE="9">exact, Int32</FONT>>];
    b2 [$(role(:good)), label=<δ = 0.00113…<BR/><FONT POINT-SIZE="9">bounded by h/2 — Float32 resolves 6e-8 here</FONT>>];
    b3 [$(role(:good)), label=<x rebuilt only when someone asks<BR/><FONT POINT-SIZE="9">x = x0 + (k−1)·h + δ</FONT>>];
    bv [shape=plaintext, fontcolor="#4aa46a", label=<<B>0.004 %</B> of a column>];
    b1 -> b3;
    b2 -> b3 -> bv;
  }
}
""")

# --- the two-stage placement ------------------------------------------------

render("two-stage-sort", """
digraph {
$(preamble(rankdir = "LR"))
  subgraph cluster_one {
    $(cluster("<B>one pass</B> — 118 ms"))
    i1 [$(role(:bad)), label=<particle i>];
    i2 [$(role(:bad)), label=<particle i+1>];
    i3 [$(role(:bad)), label=<particle i+2>];
    o1 [$(role(:bad, shape = "cylinder")), label=<perm<BR/><FONT POINT-SIZE="9">305 MB</FONT>>];
    i1 -> o1 [label="a page"];
    i2 -> o1 [label="another"];
    i3 -> o1 [label="another again"];
  }
  subgraph cluster_two {
    $(cluster("<B>two stages</B> — 17.5 + 29.4 ms"))
    s1  [$(role(:good)), label=<<B>stage 1</B> — bucket = cell ≫ 9<BR/><FONT POINT-SIZE="9">2 672 destinations, each advancing in order</FONT>>];
    mid [$(role(:good, shape = "cylinder")), label=<permtmp<BR/><FONT POINT-SIZE="9">grouped by region</FONT>>];
    s2  [$(role(:good)), label=<<B>stage 2</B> — exact cell, walking that order<BR/><FONT POINT-SIZE="9">neighbouring work-items → neighbouring cells</FONT>>];
    out [$(role(:good, shape = "cylinder")), label=<perm<BR/><FONT POINT-SIZE="9">identical to the host sort</FONT>>];
    s1 -> mid -> s2 -> out;
  }
}
""")

# --- the cloud's two forms --------------------------------------------------

render("cloud-forms", """
digraph {
$(preamble(rankdir = "TB"))
  readers [$(role(:object)), label=<every reader of <B>cloud.positions</B><BR/><FONT POINT-SIZE="9">interaction_energy · projectile_forces! · entropy · diagnostics</FONT>>];
  iface   [shape=plaintext, fontcolor="$EDGE", label=<<B>AbstractVector{NTuple{3,T}}</B><BR/><FONT POINT-SIZE="9">one interface, two storages</FONT>>];
  plain   [$(role(:object)), label=<<B>Vector{NTuple{3,T}}</B><BR/><FONT POINT-SIZE="9">the Fortran's (3, npartmax) layout<BR/>the reference path</FONT>>];
  packed  [$(role(:device)), label=<<B>PackedPositions</B><BR/><FONT POINT-SIZE="9">knode::Int32 · delta::E<BR/>rebuilt on getindex</FONT>>];
  kern    [$(role(:good)), label=<the kernels<BR/><FONT POINT-SIZE="9">they want (k, δ) and nothing else</FONT>>];
  readers -> iface [dir=none];
  iface -> plain;
  iface -> packed;
  packed -> kern [label="already in the right form"];
  plain -> kern [style=dashed, label="packed every step, in Float64"];
}
""")
