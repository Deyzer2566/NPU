all: out/Vtest_npu.exe
	out/Vtest_npu.exe
out/Vtest_npu.exe: gen.sv
	verilator -j --trace -CFLAGS "-D_GLIBCXX_USE_CXX11_ABI=0" --binary --top-module test_npu -Mdir out src/sim/*.sv src/syn/*.sv
lint:
	verilator -j -CFLAGS "-D_GLIBCXX_USE_CXX11_ABI=0" --lint-only --top-module test_npu src/sim/*.sv src/syn/*.sv
gen.sv: compiler.py
	python compiler.py
clean:
	rm -r out/