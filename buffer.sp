* buffer.sp
.Lib "/usr/cad/designkit/analog/cic18/model/cic018.l" TT
*-------------------------------------------------------------------------
.subckt Buffer IN OUT VDD VSS
    XINV_1 IN TEMP VDD VSS inv wp=2u wn=1u
    XINV_2 TEMP OUT VDD VSS inv wp=8u wn=4u
.ends Buffer
*-------------------------------------------------------------------------
.subckt inv IN OUT VDD VSS wp=2u wn=1u
    M1 OUT IN VDD VDD P_18 L=0.18u W=wp
    M2 OUT IN VSS VSS N_18 L=0.18u W=wn
.ends
*-------------------------------------------------------------------------
XBUFFER A OUT VDD 0 Buffer
*-------------------------------------------------------------------------
VVDD VDD 0 DC=1.8
VVA  A 0 PULSE(0 1.8 0n 1n 1n 20n 40n)
*-------------------------------------------------------------------------
.TEMP 27
.options post

.TRAN 1p 1u

.meas tran Period_Rough TRIG V(OUT)=0.9V rise=2
    + TARG V(OUT)=0.9V rise=3
    
.end