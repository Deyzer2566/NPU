`timescale 1ns / 1ps

module test_npu ;

logic clk=0, rst=0;

// Для работы с NPU, будто в нём только одно ядро, использовать .PE_COUNT(2),
// указав в компиляторе количество ядер равным 1
npu #(.PE_COUNT(3), .RAM_SIZE(64)) m(.clk_i(clk), .rst_i(rst));

task tick;
    begin
    clk=0;
    #1;
    clk=1;
    #1;
    clk=0;
    end
endtask
logic [31:0] inst_count;
logic [31:0] out_addr;
logic [31:0] out_len;
initial begin
    $dumpfile("dump.vcd");
    $dumpvars();
    clk = 0;
    rst = 0;
    m.pe_inst[1].pe_core.instruction_mem[0] = 41'h0;
    m.pe_inst[1].pe_core.instruction_mem[1] = 41'h1 << (30+4+3); // На случай, если должно быть только 1 ядро
    `include "gen.sv" // Импортируем скомпилированную программу
    for(int i = 0;i<inst_count+1;i++) begin // inst_count заполняется в gen.sv
        tick();
    end
    for(int i = out_addr; i < out_addr + out_len; i++)// out_addr и out_len заполняются в gen.sv
        $display("%x %x %x %x", m.shared_mem.ram[i][31:24], m.shared_mem.ram[i][23:16], m.shared_mem.ram[i][15:8], m.shared_mem.ram[i][7:0]);
    $finish;
end

endmodule
