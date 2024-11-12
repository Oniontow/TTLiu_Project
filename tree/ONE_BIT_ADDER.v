// submodule: bit adders
module ONE_BIT_ADDER (
    input I1,
    input I2,
    inout VDD,
    inout VSS,
    output [1:0] O
);
FA fa1_0(
    .A(I1), 
    .B(I2),
    .Ci(1'b0), 
    .S(O[0]), 
    .Co(O[1]),
    .VDD(VDD),
    .VSS(VSS)
);
    
endmodule