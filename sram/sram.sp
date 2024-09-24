*sram.sp
.lib "/usr/cad/designkit/analog/cic18/model/cic018.l" TT

.param DCQ1=0.9
.param DCQ2=0.9
*-------------------------------------------------------------------------
.subckt inv IN OUT VDD VSS wp=0.3u wn=0.72u
    M1 OUT IN VDD VDD P_18 L=0.18u W=wp
    M2 OUT IN VSS VSS N_18 L=0.18u W=wn
.ends
*-------------------------------------------------------------------------
.subckt sram WL BL BLB VQ1 VQ2 VDD VSS
    INV1 QIN2 QOUT2 VDD VSS inv wp=0.3u wn=0.72u
    INV2 QIN1 QOUT1 VDD VSS inv wp=0.3u wn=0.72u
    MA1 BL WL QOUT2 VSS N_18 L=0.18u W=0.48u
    MA2 BLB WL QOUT1 VSS N_18 L=0.18u W=0.48u
.ends
*-------------------------------------------------------------------------
XSRAM WL BL BLB VQ1 VQ2 VDD VSS sram
*-------------------------------------------------------------------------
VVQ1 VQ1 0 DC=DCQ1
VVQ2 VQ2 0 DC=DCQ2
VVDD VDD 0 DC=1.8
VVSS VSS 0 DC=0
*-------------------------------------------------------------------------
.TEMP 27
.options post

.TRAN 1p 1u
.dc DCQ1 0 1.8 0.001
.dc DCQ2 0 1.8 0.001

.end