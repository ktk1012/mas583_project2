PROJECT_DIR := $(shell pwd)
REINFORCEFLOW_DIR := lib/reinforceflow

bootstrap:
	pip install -r requirements.txt
	git submodule update --init --checkout $(REINFORCEFLOW_DIR)

cpu_bootstrap: bootstrap
	cd $(REINFORCEFLOW_DIR); pip install -e .[tf]

gpu_bootstrap:
	cd $(REINFORCEFLOW_DIR); pip install -e .[tf-gpu]

.PHONY: bootstrap cpu_bootstrap gpu_bootstrap
