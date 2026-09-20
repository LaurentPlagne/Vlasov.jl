# Shared styling for the documentation's diagrams.
#
# The SVGs are generated at build time and read on a page whose theme the
# reader chooses, so nothing may depend on the page background: every box is
# filled opaquely, and edges and their labels use a mid grey that stays legible
# on white and on the dark theme alike.

const BG = "transparent"
const EDGE = "#8a96a0"

const PALETTE = (
    device = ("#1f3a4d", "#e8f1f5", "#4a90a4"),   # fill, font, border
    host   = ("#3d3320", "#f5efe0", "#9a7b3f"),
    object = ("#2f2a3d", "#efe8f5", "#7a6b9a"),
    good   = ("#1f4d2f", "#e8f5ec", "#4aa46a"),
    bad    = ("#4d2020", "#f5e8e8", "#a44a4a"),
)

"""Node attributes for one of the palette's roles."""
function role(name::Symbol; shape = "box")
    fill, font, border = getfield(PALETTE, name)
    """shape=$shape, style="filled,rounded", fillcolor="$fill", fontcolor="$font", color="$border\""""
end

"""The preamble every diagram shares."""
function preamble(; rankdir = "TB", fontsize = 11)
    """
      bgcolor="$BG";
      rankdir=$rankdir;
      fontname="Helvetica"; fontsize=$fontsize; fontcolor="$EDGE";
      node [fontname="Helvetica", fontsize=$fontsize, margin="0.14,0.08"];
      edge [fontname="Helvetica", fontsize=$(fontsize - 1), color="$EDGE", fontcolor="$EDGE"];
    """
end

"""
A cluster's frame: no fill, so the reader's background shows through.

⚠️ Semicolons, not commas. Inside a `subgraph` body the attributes are separate
statements; a comma is only valid between brackets, and `dot` reports it as a
syntax error on the line of the *next* token.
"""
cluster(label) =
    """style="rounded"; color="$EDGE"; fontcolor="$EDGE"; label=<$label>; labeljust="l";"""
