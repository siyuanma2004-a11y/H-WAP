H_WAP_PYTHON ?= .venv/bin/python
H_WAP_VIEWER_PYTHON ?= $(if $(filter Darwin,$(shell uname -s)),.venv/bin/mjpython,$(H_WAP_PYTHON))

.PHONY: setup check test demo viewer pilot collect smoke sonic harness

setup:
	bash scripts/setup_env.sh

check:
	"$(H_WAP_PYTHON)" scripts/check_env.py

test:
	"$(H_WAP_PYTHON)" -m pytest tests -q

demo:
	"$(H_WAP_PYTHON)" scripts/run_sim.py --demo reach

viewer:
	"$(H_WAP_VIEWER_PYTHON)" scripts/run_sim.py --demo reach --viewer --gl glfw

pilot:
	"$(H_WAP_PYTHON)" scripts/run_experiment.py

collect:
	"$(H_WAP_PYTHON)" scripts/collect_data.py --episodes 10

smoke:
	bash scripts/smoke_test.sh

sonic:
	"$(H_WAP_PYTHON)" scripts/download_sonic.py

harness:
	"$(H_WAP_PYTHON)" -m hwap harness configs/harness/smoke_oracle.json --output runs/harness_smoke_$$(date -u +%Y%m%dT%H%M%S)
