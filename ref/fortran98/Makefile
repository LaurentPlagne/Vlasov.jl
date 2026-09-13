# Oracle « version tardive » : vlas.f du 1998-01-05
# (temp/home/sauron2/plagne/it8/ttt/majrel2/lucifer/vlas.f.gz)
#
# Dernière version séquentielle complète de la lignée principale : elle appelle
# encore `initpro`/`incproj`, contrairement à celle du 1998-01-07 qui est une
# expérience de relaxation sans projectile.
#
#   make          construit `vlas98`
#   make clean

FC      = gfortran
FCFLAGS = -O2 -std=legacy
export SDKROOT := $(shell xcrun --show-sdk-path)
LDLIBS  = -framework Accelerate

.PHONY: all clean
all: vlas98

vlas98: vlas.o libspl.o f02agf_shim.o
	$(FC) $(FCFLAGS) -o $@ $^ $(LDLIBS)

%.o: %.f ceq3d.f
	$(FC) $(FCFLAGS) -c $< -o $@

clean:
	rm -f *.o vlas98 *.bin *.log
	rm -rf Eloss out
