#!/usr/bin/env python3
"""Apple GPU performance counters, from an `xctrace` recording to a table.

    xcrun xctrace record --template "Metal System Trace" \
         --instrument "Metal GPU Counters" --no-prompt \
         --output k.trace --attach <pid> --time-limit 100ms
    scripts/compteurs_gpu.py k.trace

`--attach`, never `--launch`: recording from launch captures the fifty seconds
of construction, which at 8×10⁷ particles is fifty million samples and a 7.6 GB
export, none of it about the kernel. And the loop under measurement must last
**much longer** than the recording — a recording that outlives it reads a GPU
at rest, which looks like zeros at the median with absurd values in the tail.

See `docs/src/performance.md`, "Profiling a kernel", for how to shrink the work
without leaving the regime — and for what the counters said about the two
kernels they have been pointed at so far.

Three things make this export unlike ordinary XML, and each one silently
produces plausible nonsense if ignored:

  * it **compresses by reference** — a value appears once as `id="N"` and every
    repetition is `<tag ref="N"/>`, so a dictionary of definitions has to be
    carried along as the rows stream past;
  * the columns are identified **by position**, not by tag name. Matching on
    names is what gives occupancies above 100 %;
  * the counter itself is an integer id, and its name lives in a second table,
    `gpu-counter-info`, laid out at the same two positions.
"""
import statistics
import subprocess
import sys
import xml.etree.ElementTree as ET

VALUES = '/trace-toc/run[@number="1"]/data/table[@schema="gpu-counter-value"]'
NAMES = '/trace-toc/run[@number="1"]/data/table[@schema="gpu-counter-info"]'


def export(trace, xpath, out):
    """One table of the trace, as XML, into `out`."""
    with open(out, "wb") as f:
        subprocess.run(["xcrun", "xctrace", "export", "--input", trace,
                        "--xpath", xpath], stdout=f, stderr=subprocess.DEVNULL,
                       check=True)
    return out


def rows(path):
    """Each row as the list of its resolved cells, in column order."""
    defs = {}
    for _, elem in ET.iterparse(path, events=("end",)):
        if elem.tag != "row":
            continue
        cells = []
        for child in elem:
            ref = child.get("ref")
            if ref is not None:
                cells.append(defs.get(ref, ""))
                continue
            val = (child.text or "").strip()
            if child.get("id") is not None:
                defs[child.get("id")] = val
            cells.append(val)
        yield cells
        elem.clear()


def main(trace):
    names = {c[1]: c[2] for c in rows(export(trace, NAMES, trace + ".info.xml"))
             if len(c) >= 3}
    per_counter = {}
    total = 0
    for c in rows(export(trace, VALUES, trace + ".values.xml")):
        if len(c) < 3:
            continue
        total += 1
        try:
            per_counter.setdefault(names.get(c[1], c[1]), []).append(float(c[2]))
        except ValueError:
            pass

    print(f"{total} rows, {len(per_counter)} counters\n")
    print(f"{'counter':<46} {'samples':>8} {'median':>10} {'mean':>10} {'max':>10}")
    for name in sorted(per_counter, key=lambda k: -statistics.median(per_counter[k])):
        v = per_counter[name]
        print(f"{name:<46} {len(v):>8} {statistics.median(v):>10.2f} "
              f"{statistics.fmean(v):>10.2f} {max(v):>10.2f}")
    # ⚠️ The medians are rates over the recorded window, not per unit of work.
    # Two runs of the same kernel at different speeds are therefore not
    # comparable line by line: a rate that holds while the kernel does twice
    # the work means the absolute activity doubled.


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
