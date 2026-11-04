`timescale 1ns / 1ps

interface memory_bus;
    
    logic [31:0]  addr;
    logic [31:0] wdata;
    logic [31:0] rdata;
    logic        we;     // Write Enable
    logic [31:0] fast_r_addr;
    logic [127:0] fast_r_rdata;

    modport user (
        output addr, wdata, we, fast_r_addr,
        input  rdata, fast_r_rdata
    );

    modport mem (
        input  addr, wdata, we, fast_r_addr,
        output rdata, fast_r_rdata
    );

endinterface

module memory
# (
    parameter RAM_SIZE=1024
) (
    memory_bus.mem m,
    input clk_i
);
    
    logic [31:0] ram[0:RAM_SIZE-1];

    always @* begin
        m.rdata = ram[m.addr];
        m.fast_r_rdata = {ram[m.fast_r_addr], ram[m.fast_r_addr+1], ram[m.fast_r_addr+2], ram[m.fast_r_addr+3]};
    end
    
    always @(negedge clk_i) begin
        if (m.we) begin
            ram[m.addr] <= m.wdata;
        end
    end
        
endmodule

module shared_memory_bus // переходник для подключения нескольких устройств к памяти
# (
    parameter SOURCES_COUNT = 4
) (
    memory_bus.mem bus_in[0:SOURCES_COUNT-1], // входы/выходы с устройств
    memory_bus.user mem_out // выход в/из памяти
);

    logic [31:0] bus_addr [0:SOURCES_COUNT-1];
    logic [31:0] bus_wdata [0:SOURCES_COUNT-1];
    logic bus_we [0:SOURCES_COUNT-1];
    logic [31:0] bus_rdata[0:SOURCES_COUNT-1];

    generate
        for(genvar i = 0;i<SOURCES_COUNT; i++) begin
            assign bus_addr[i] = bus_in[i].addr;
            assign bus_wdata[i] = bus_in[i].wdata;
            assign bus_we[i] = bus_in[i].we;
            assign bus_in[i].rdata = bus_rdata[i];
        end
    endgenerate

    always @* begin
        mem_out.addr = 0;
        mem_out.wdata = 0;
        mem_out.we = 0;
        for(int i = 0;i<SOURCES_COUNT; i++) begin
            mem_out.addr |= bus_addr[i];
            mem_out.wdata |= bus_wdata[i];
            mem_out.we |= bus_we[i];

            bus_rdata[i] = mem_out.rdata;
        end
    end

endmodule