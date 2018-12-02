PROJECT_DIR := $(shell pwd)
REINFORCEFLOW_DIR := lib/reinforceflow

bootstrap:
	pip install numpy gym[atari] six opencv-python
	git submodule update --init --checkout $(REINFORCEFLOW_DIR)

cpu_bootstrap: bootstrap
	pip install tensorflow
	cd $(REINFORCEFLOW_DIR); pip install -e .[tf]

gpu_bootstrap: bootstrap
	pip install tensorflow-gpu
	cd $(REINFORCEFLOW_DIR); pip install -e .[tf-gpu]

clean:
	rm -r $(REINFORCEFLOW_DIR)

.PHONY: bootstrap cpu_bootstrap gpu_bootstrap
