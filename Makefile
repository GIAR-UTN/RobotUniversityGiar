# SIMULATOR only needs SOME valid value to get `rugiar` past legged_gym's own
# import-time gate (legged_gym/__init__.py) -- it does not decide which
# backend actually launches, `rugiar drive <system>` does that on its own via
# its own subprocess env. 'mjlab' is picked here specifically because that
# branch imports nothing extra at gate time (unlike 'genesis', which eagerly
# imports the genesis package) -- see legged_gym/__init__.py.
RUGIAR := SIMULATOR=mjlab .venv/bin/rugiar

.PHONY: help drive-genesis drive-mjlab

help:
	@echo "make drive-genesis [ARGS='--task g1_target']  — launch the control web on Genesis"
	@echo "make drive-mjlab   [ARGS='--task g1_target']  — launch the control web on mjlab"
	@echo "(both stop whatever's already on the control port first — see 'rugiar drive --help')"

drive-genesis:
	$(RUGIAR) drive genesis $(ARGS)

drive-mjlab:
	$(RUGIAR) drive mjlab $(ARGS)
