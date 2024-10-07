*sram.sp
.lib "/usr/cad/designkit/analog/cic18/model/cic018.l" TT
.param DCLEVEL=1.8
.param DCQ1='DCLEVEL/2'
.param DCQ2='DCLEVEL/2'

*-------------------------------------------------------------------------
.subckt and A B OUT VDD VSS wp=0.51u wn=0.42u
    M1 OUTB A VDD VDD P_18 L=0.18u W=wp
    M2 OUTB B VDD VDD P_18 L=0.18u W=wp
    M3 OUTB A D4 VSS N_18 L=0.18u W=wn
    M4 D4 B VSS VSS N_18 L=0.18u W=wn
    M5 OUT OUTB VDD VDD P_18 L=0.18u W=wp
    M6 OUT OUTB VSS VSS N_18 L=0.18u W=wn
.ends
*-------------------------------------------------------------------------
XAND1 A B O VDD VSS and
C1 O VSS 1f
*-------------------------------------------------------------------------

VVA A 0 pulse(0 1.8 10n 125p 125p 10n 20n)
VVB B 0 pulse(0 1.8 20n 125p 125p 10n 40n)

VVDD VDD 0 DC=DCLEVEL
VVSS VSS 0 DC=0

*-------------------------------------------------------------------------
.TEMP 27
.options post
.tran 1p 100n
.probe V(A) V(B) V(O)

.end