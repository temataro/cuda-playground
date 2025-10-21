OPTS="-Wall -Wextra -Wno-format-extra-args"
%: %.cu
	nvcc -Xcompiler $(OPTS) $< -o $@
	./$@

sharp:
	clang-format -i *cu *cpp
