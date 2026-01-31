IMAGE_NAME = s1-point-transformer
HOST_DATA_DIR = $(PWD)/data

build:
	podman build -t $(IMAGE_NAME) -f Containerfile .

dev:
	podman run -it --rm \
		--device nvidia.com/gpu=all \
		--security-opt=label=disable \
		--shm-size=32g \
		-v $(PWD):/workspace/project \
		-v $(HOST_DATA_DIR):/workspace/project/data \
		$(IMAGE_NAME) /bin/bash

clean:
	podman container prune -f