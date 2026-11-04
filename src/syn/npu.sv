`timescale 1ns / 1ps

module npu
# (
    parameter PE_COUNT = 3,
    parameter RAM_SIZE = 1024
) (
    input clk_i,
    input rst_i
);

    memory_bus sm_bus();
    memory #(.RAM_SIZE(RAM_SIZE)) shared_mem (
        .m(sm_bus.mem),
        .clk_i(clk_i)
    );

    ring_buffer rb_out[0:PE_COUNT-1]();

    memory_bus pe_sm_bus[0:PE_COUNT-1]();
    
    shared_memory_bus #(.SOURCES_COUNT(PE_COUNT)) sm_bus_mux (
        .bus_in(pe_sm_bus),
        .mem_out(sm_bus.user)
    );

    generate
        for (genvar i = 0; i < PE_COUNT; i++) begin : pe_inst
            ring_buffer rb_in[0:PE_COUNT-2]();
            for (genvar j = 0;j<PE_COUNT;j++) begin
                if(j < i) begin
                    assign rb_in[j].out = rb_out[j].out;
                end else if(j > i) begin
                    assign rb_in[j-1].out = rb_out[j].out;
                end
            end
            pe #(
                .PE_COUNT(PE_COUNT)
            ) pe_core (
                .clk_i(clk_i),
                .rst_inst(rst_i),
                .sm(pe_sm_bus[i].user),
                .rb_out(rb_out[i].saver),
                .rb_in(rb_in)
            );
        end
    endgenerate

endmodule