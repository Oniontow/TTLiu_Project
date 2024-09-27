*sram.sp
.lib "/usr/cad/designkit/analog/cic18/model/cic018.l" TT
.param DCLEVEL=1.8
.param DCQ1='DCLEVEL/2'
.param DCQ2='DCLEVEL/2'

*-------------------------------------------------------------------------
.subckt inv IN OUT VDD VSS wp=0.30u wn=0.72u
    M1 OUT IN VDD VDD P_18 L=0.18u W=wp
    M2 OUT IN VSS VSS N_18 L=0.18u W=wn
.ends
*-------------------------------------------------------------------------

XINV1 VQ2 QOUT2 VDD VSS inv
XINV2 VQ1 QOUT1 VDD VSS inv
MA1 BL WL QOUT2 VSS N_18 L=0.18u W=0.48u
MA2 BLB WL QOUT1 VSS N_18 L=0.18u W=0.48u

*-------------------------------------------------------------------------
VVQ1    VQ1 0 DC=DCQ1
VVQ2    VQ2 0 DC=DCQ2
VVBL    BL  0 DC=DCLEVEL
VVBLB   BLB 0 DC=0
VVWL    WL  0 DC=DCLEVEL

VVDD VDD 0 DC=DCLEVEL
VVSS VSS 0 DC=0
*-------------------------------------------------------------------------
.TEMP 27
.options post

.dc DCQ1 0 DCLEVEL 0.001
.dc DCQ2 0 DCLEVEL 0.001

.end