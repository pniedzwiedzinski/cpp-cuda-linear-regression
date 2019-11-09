
run: output input.in
	./output < input.in

output: main.cpp load.cpp
	g++ main.cpp load.cpp -o output

clear:
	rm output
