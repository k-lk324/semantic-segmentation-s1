IMAGE_NAME = s1-segmentation
DATA_DIR = $(PWD)/data
FEATURE_MODE ?= zero
BLEND_ALPHA ?= 0.5
SRC_LAS ?= data/raw_las/Fh_parking_outside_2025-09-19-17-08-32_Section_Section_normal.las
TILE_MIN_POINTS ?= 200

# Host paths
HOST_LIBCUDA = /lib/x86_64-linux-gnu/libcuda.so.1
HOST_NVML = /lib/x86_64-linux-gnu/libnvidia-ml.so.1

build:
	docker build -t $(IMAGE_NAME) -f Containerfile .

dev:
	docker run -it --rm --user 1003:1003 \
		--device /dev/nvidia0 \
		--device /dev/nvidiactl \
		--device /dev/nvidia-uvm \
		--device /dev/nvidia-modeset \
		-v $(HOST_LIBCUDA):/usr/lib/x86_64-linux-gnu/libcuda.so.1:ro \
		-v $(HOST_NVML):/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1:ro \
		--security-opt=label=disable \
		--shm-size=32g \
		-v $(PWD):/workspace/project \
		-v $(DATA_DIR):/workspace/project/data \
		$(IMAGE_NAME) /bin/bash

clean:
	docker container prune -f

tile:
	rm -rf data/processed_tiles
	mkdir -p data/processed_tiles
	docker run --rm --user 1003:1003 \
		-v $(shell pwd):/workspace/project \
		$(IMAGE_NAME) \
		python src/preprocessing/tile_s1.py \
		--src $(SRC_LAS) \
		--dst data/processed_tiles \
		--min_points $(TILE_MIN_POINTS)

infer:
	docker run --rm -it --user 1003:1003 \
		--security-opt=label=disable \
		--device /dev/nvidia0 \
		--device /dev/nvidiactl \
		--device /dev/nvidia-uvm \
		--device /dev/nvidia-modeset \
		-v /usr/lib/x86_64-linux-gnu/libcuda.so.1:/usr/lib/x86_64-linux-gnu/libcuda.so.1:ro \
		-v /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1:ro \
		-v /usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1:/usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1:ro \
		-v $(shell pwd):/workspace/project \
		s1-segmentation \
		python scripts/run_inference.py \
		--config configs/s1_inference.py \
		--weights weights/ptv3_nuscenes.pth \
		--data_dir data/processed_tiles \
		--feature_mode $(FEATURE_MODE) \
		--blend_alpha $(BLEND_ALPHA) \
		--output_dir data/predictions

reconstruct:
	docker run --rm --user 1003:1003 \
		-v $(shell pwd):/workspace/project \
		$(IMAGE_NAME) python scripts/vote_and_reconstruct.py \
		--src_las $(SRC_LAS) \
		--tiles_dir data/processed_tiles \
		--pred_dir data/predictions \
		--num_classes 16 \
		--output_las data/reconstructed.las
