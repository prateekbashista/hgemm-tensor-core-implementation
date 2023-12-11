all: tensor_hgemm.bin tensor_hgemm-debug.bin tensor_hgemm-profile.bin

SOURCE_FILE=tensor_hgemm.cu

# optimized binary
tensor_hgemm.bin: $(SOURCE_FILE)
	nvcc -std=c++17 --generate-code=arch=compute_75,code=[compute_75,sm_75] $^ -lcublas -o $@

# debug binary without optimizations
tensor_hgemm-debug.bin: $(SOURCE_FILE)
	nvcc -g -G -src-in-ptx -std=c++17 --generate-code=arch=compute_75,code=[compute_75,sm_75] $^ -lcublas -o $@

# optimized binary with line number information for profiling
tensor_hgemm-profile.bin: $(SOURCE_FILE)
	nvcc -g --generate-line-info -src-in-ptx -std=c++17 --generate-code=arch=compute_75,code=[compute_75,sm_75] $^ -lcublas -o $@

# NB: make sure you change the --algo flag here to profile the one you care about. 
# You can change the --export flag to set the filename of the profiling report that is produced.
profile: tensor_hgemm-profile.bin
	sudo /usr/local/cuda-11.8/bin/ncu --export my-profile --set full ./cugemm-profile.bin --size=1024 --reps=1 --algo=1 --validate=false

clean:
	rm -f tensor_hgemm*.bin
