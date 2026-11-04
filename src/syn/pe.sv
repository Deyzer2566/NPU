`timescale 1ns / 1ps

module systolic_array (
    input logic signed [7:0] x1 [0:3],
    input logic signed [7:0] x2 [0:3],
    output logic signed [7:0] out [0:3]
);

    logic signed [15:0] prod[0:3];

    always_comb begin
        for (int i = 0; i < 4; i++) begin
            prod[i] = (x1[i] * x2[i]);
        end
    end

    assign out = {prod[0][11:4], prod[1][11:4], prod[2][11:4], prod[3][11:4]};

endmodule

module sigmoid 
# (
    parameter DIGITS_AFTER_DOT_IN = 4,
    parameter DIGITS_AFTER_DOT_OUT = 4
) (
    input logic signed [7:0] x [0:3],
    output logic signed [7:0] out [0:3]
);
    
    logic signed [7:0] sig[-128:127];
    initial begin
        for(int i = -128; i < 128; i++) begin
            sig[i] = 8'($rtoi((1/(1+$exp(-$itor(i)/$itor(1<<DIGITS_AFTER_DOT_IN))))*(1<<DIGITS_AFTER_DOT_OUT)));
        end;
    end;
    
    always_comb begin
        for(int i =0;i<4;i++) begin
            out[i] = sig[x[i]];
        end
    end

endmodule

module summator (
    input logic signed [7:0] x[3:0],
    output logic signed [7:0] out
);

    logic signed [10:0] sum;

    always @* begin
        sum = 0;
        for(int i = 0;i<4;i++)
            sum += {{3{x[i][7]}}, x[i]};
        if (sum > 127) begin
            out = 8'd127;
        end else if (sum < -127) begin
            out = -8'd127;
        end else begin
            out = sum[7:0];
        end
    end

endmodule

module vector_builder (
    input wire clk_i,
    input logic [7:0] x,
    input logic [1:0] ind,
    input logic we,
    output logic [7:0] buff[0:3],
    input logic rst_i
);

    logic[1:0] prev_ind;
    always @(negedge clk_i, posedge rst_i) begin
        if(rst_i) begin
            buff <= {8'd0, 8'd0, 8'd0, 8'd0};
            prev_ind = 2'd0;
        end
        if(we) begin
            buff[ind] <= x;
            prev_ind <= ind;
            if(ind == 0 && prev_ind != 0) begin
                buff[1:3] = {8'd0,8'd0,8'd0};
            end
        end
    end
endmodule

interface dual_read_memory_bus();
    logic [31:0]  addr1;
    logic [31:0] rdata1;
    logic [31:0]  addr2;
    logic [31:0] rdata2;
    logic [31:0] wdata_addr;
    logic [31:0] wdata;
    logic        we;     // Write Enable

    modport user (
        output addr1, addr2, wdata_addr, wdata, we,
        input  rdata1, rdata2
    );

    modport mem (
        input addr1, addr2, wdata_addr, wdata, we,
        output  rdata1, rdata2
    );
endinterface

module local_memory
# (
    parameter MEM_SIZE = 32
) (
    input clk_i,
    dual_read_memory_bus.mem mem
);

    logic [31:0] ram[0:MEM_SIZE-1];

    always @* begin
        mem.rdata1 = ram[mem.addr1];
        mem.rdata2 = ram[mem.addr2];
    end

    always @(negedge clk_i) begin
        if (mem.we) begin
            ram[mem.wdata_addr] <= mem.wdata;
        end
    end
    
endmodule

module instruction_executor
# (
    parameter ADDR_WIDTH = 10,
    parameter SOURCES_COUNT = 9,
    parameter INSTRUCTION_WIDTH = ADDR_WIDTH*3 + $clog2(SOURCES_COUNT) + 7,
    parameter PE_COUNT = 3
) (
    input [INSTRUCTION_WIDTH-1:0] instruction,
    memory_bus.user sm,
    output disable_clock,
    dual_read_memory_bus.user lm,
    ring_buffer.saver rb_out,
    ring_buffer.reader rb_in[0:PE_COUNT-2],
    input logic clk_i
);
    logic [31:0] A;
    logic [31:0] B;
    logic [31:0] C;
    logic [$clog2(SOURCES_COUNT)-1:0] source_selector;
    logic save_to_shared;
    logic save_to_buffout;
    logic reset_vb;
    logic use_sm;
    logic we_vectorbuilder;
    logic sm_fastdata;

    always @* begin
        A = {{(32-ADDR_WIDTH){1'b0}}, instruction[ADDR_WIDTH-1:0]};
        B = {{(32-ADDR_WIDTH){1'b0}}, instruction[ADDR_WIDTH*2-1:ADDR_WIDTH]};
        C = {{(32-ADDR_WIDTH){1'b0}}, instruction[ADDR_WIDTH*3-1:2*ADDR_WIDTH]};
        source_selector = instruction[ADDR_WIDTH*3+$clog2(SOURCES_COUNT)-1:ADDR_WIDTH*3];
        save_to_shared = instruction[ADDR_WIDTH*3+$clog2(SOURCES_COUNT)];
        save_to_buffout = instruction[ADDR_WIDTH*3+$clog2(SOURCES_COUNT)+1];
        reset_vb= instruction[ADDR_WIDTH*3+$clog2(SOURCES_COUNT)+2];
        disable_clock= instruction[ADDR_WIDTH*3+$clog2(SOURCES_COUNT)+3];
        use_sm = instruction[ADDR_WIDTH*3+$clog2(SOURCES_COUNT)+4];
        we_vectorbuilder = instruction[ADDR_WIDTH*3+$clog2(SOURCES_COUNT)+5];
        sm_fastdata = instruction[ADDR_WIDTH*3+$clog2(SOURCES_COUNT)+6];
    end
    
    logic signed [7:0] vecA [0:3];
    logic signed [7:0] vecB [0:3];
    
    assign vecA = {
        lm.rdata1[31:24],lm.rdata1[23:16],lm.rdata1[15:8],lm.rdata1[7:0]
    };
    assign vecB = {
        lm.rdata2[31:24],lm.rdata2[23:16],lm.rdata2[15:8],lm.rdata2[7:0]
    };
    
    assign lm.addr1 = A;
    assign lm.addr2 = B;
    assign lm.wdata_addr = C;
    assign lm.we = 1;
    
    assign sm.addr = B & {32{use_sm | save_to_shared}};
    assign sm.we = save_to_shared;

    assign sm.fast_r_addr = B & {32{sm_fastdata}};
    logic signed [7:0] sm_long_buff[0:3][0:3];
    always @(negedge clk_i) begin
        if (sm_fastdata) begin
            for (int i = 0;i<4; i++) begin
                for (int j = 0;j < 4;j++) begin
                    sm_long_buff[i][j] <= sm.fast_r_rdata[127-(32*i+j*8)-:8];
                end
            end
        end
    end

    logic signed [7:0] sources_out [0:SOURCES_COUNT-1] [0:3];

    systolic_array sa(.x1(vecA), .x2(vecB), .out(sources_out[0]));
    sigmoid sig(.x(vecA), .out(sources_out[1]));
    logic signed[7:0] sum_out;
    summator sum(.x(vecA), .out(sum_out));
    vector_builder vb(.clk_i(clk_i), .x(sum_out), .ind(B[1:0]), .we(we_vectorbuilder), .buff(sources_out[2]), .rst_i(reset_vb & clk_i));
    assign sources_out[3] = {sm.rdata[31:24], sm.rdata[23:16], sm.rdata[15:8], sm.rdata[7:0]};

    logic signed [7:0] rb_ins[0:PE_COUNT-2][0:3];
    generate
        for(genvar i = 0;i<PE_COUNT-1; i++) begin
            assign rb_ins[i] = rb_in[i].out;
        end
    endgenerate
    always @* begin
        if (B < SOURCES_COUNT) begin
            sources_out[4] = rb_ins[B];
        end else begin
            sources_out[4] = {8'd0, 8'd0, 8'd0, 8'd0};
        end
    end

    always @* begin
        for (int i = 0;i<4;i++) begin
            sources_out[i+5] = sm_long_buff[i];
        end
    end

    logic signed [7:0] selected_source_val [0:3];
    
    always @* begin
        if(source_selector < SOURCES_COUNT) begin
            selected_source_val = sources_out[source_selector];
        end else begin
            selected_source_val = {8'd0, 8'd0, 8'd0, 8'd0};
        end
    end
    assign lm.wdata = {
            selected_source_val[0], selected_source_val[1],
            selected_source_val[2], selected_source_val[3]
        };
    assign sm.wdata = {
            selected_source_val[0], selected_source_val[1],
            selected_source_val[2], selected_source_val[3]
        } & {32{save_to_shared}};
    always @(negedge clk_i) begin
        if(save_to_buffout) begin
            rb_out.out = selected_source_val;
        end
    end

endmodule

interface ring_buffer;

    logic signed [7:0] out[0:3];

    modport reader(
        input out
    );

    modport saver(
        output out
    );
    
endinterface

module pe
# (
    parameter PE_COUNT=3
) (
        input clk_i,
        input rst_inst,
        memory_bus.user sm,
        ring_buffer.saver rb_out,
        ring_buffer.reader rb_in[0:PE_COUNT-2]
);

    dual_read_memory_bus lm_int();
    local_memory lm(.mem(lm_int.mem), .clk_i(clk_i));

    logic disable_clock;
    
    logic[ie.INSTRUCTION_WIDTH-1:0] instruction_mem[0:1000-1];
    logic [31:0] pc;
    logic [ie.INSTRUCTION_WIDTH-1:0] cur_instruction;
    assign cur_instruction = instruction_mem[pc];

    instruction_executor#(.PE_COUNT(PE_COUNT)) ie(.sm(sm), .lm(lm_int.user),
        .instruction(cur_instruction), .disable_clock(disable_clock),
        .rb_out(rb_out), .rb_in(rb_in),
        .clk_i(clk_i & !disable_clock));

    always @(posedge clk_i) begin
        if (rst_inst) begin
            pc <= 0;
        end
        if (!disable_clock) begin
            pc <= pc+1;
        end
    end
    
endmodule
