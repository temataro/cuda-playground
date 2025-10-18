%: %.cu
	nvcc $< -o $@
	./$@

sharp:
	clang-format -i *cu *cpp
