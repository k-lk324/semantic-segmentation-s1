IMAGE_NAME = s1-segmentation
DATA_DIR = $(PWD)/data

# Host paths
HOST_LIBCUDA = /lib/x86_64-linux-gnu/libcuda.so.1
HOST_NVML = /lib/x86_64-linux-gnu/libnvidia-ml.so.1

build:
	podman build -t $(IMAGE_NAME) -f Containerfile .

dev:
	podman run -it --rm \
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
	podman container prune -f


infer:
	podman run --rm -it \
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
		--output_dir data/predictions

reconstruct:
	podman run --rm -it \
		-v $(PWD):/workspace/project \
		-v $(DATA_DIR):/workspace/project/data \
		$(IMAGE_NAME) python scripts/vote_and_reconstruct.py \
		--src_las data/raw_las/fh_parking.las \
		--tiles_dir data/processed_tiles \
		--pred_dir data/predictions \
		--num_classes 16 \
		--output_las data/final_scene_segmented.las
