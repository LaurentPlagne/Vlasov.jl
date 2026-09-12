# Code Fortran de référence (thèse, 1996-1999), remis en état de marche.
#
#   make            construit `vlas`, le code de référence
#   make dump       construit `vlas_dump`, la version instrumentée
#   make oracle     lance `vlas_dump` et produit les fichiers dump_*.bin
#   make clean
#
# Les dumps binaires produits par `make oracle` sont ce contre quoi le portage
# Julia est validé (voir README.md).

FC      = gfortran
FCFLAGS = -O2 -std=legacy
# gfortran de Homebrew ne trouve pas la libSystem du SDK courant sans ceci.
export SDKROOT := $(shell xcrun --show-sdk-path)
# f02agf est la seule routine NAG utilisée ; le shim la rend à LAPACK.
LDLIBS  = -framework Accelerate

COMMON  = libspl.o f02agf_shim.o

.PHONY: all dump oracle clean
all: vlas
dump: vlas_dump

vlas: vlas.o $(COMMON)
	$(FC) $(FCFLAGS) -o $@ $^ $(LDLIBS)

vlas_dump: vlas_dump.o dumputil.o $(COMMON)
	$(FC) $(FCFLAGS) -o $@ $^ $(LDLIBS)

# La version instrumentée est dérivée de vlas.f par un patch, pour qu'on voie
# exactement ce que l'instrumentation ajoute.
vlas_dump.f: vlas.f instrument.patch
	cp vlas.f $@ && patch -s $@ < instrument.patch

%.o: %.f ceq3d.f
	$(FC) $(FCFLAGS) -c $< -o $@

# Le programme écrit dans ces répertoires et échoue s'ils n'existent pas.
oracle: vlas_dump
	mkdir -p Eloss out && cp -n data/*.dat . 2>/dev/null || true
	./vlas_dump > oracle.log 2>&1 || (tail -20 oracle.log; exit 1)
	@ls dump_*.bin dumpb*.bin

clean:
	rm -f *.o vlas vlas_dump vlas_dump.f *.bin *.log
	rm -rf Eloss out
	rm -f $(notdir $(wildcard data/*.dat))
