
run: output input.in
	./output < input.in

output: main.cu
	nvcc main.cu -o output

clear:
	rm output
