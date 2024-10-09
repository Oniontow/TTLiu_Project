************************************************************************
* auCdl Netlist:
* 
* Library Name:  FA
* Top Cell Name: FA
* View Name:     schematic
* Netlisted on:  Oct  7 15:55:27 2024
************************************************************************

*.BIPOLAR
*.RESI = 2000 
*.RESVAL
*.CAPVAL
*.DIOPERI
*.DIOAREA
*.EQUATION
*.SCALE METER
*.MEGA
.PARAM



************************************************************************
* Library Name: FA
* Cell Name:    FA
* View Name:    schematic
************************************************************************

.SUBCKT FA A B CI CO S VDD VSS
*.PININFO A:I B:I CI:I CO:O S:O VDD:B VSS:B
MM29 S net6 VSS VSS N_18 m=1 l=180.0n w=250.0n
MM27 net7 A VSS VSS N_18 m=1 l=180.0n w=250.0n
MM26 net8 B net7 net7 N_18 m=1 l=180.0n w=250.0n
MM25 net6 CI net8 net8 N_18 m=1 l=180.0n w=250.0n
MM18 net1 B VSS VSS N_18 m=1 l=180.0n w=250.0n
MM17 net1 CI VSS VSS N_18 m=1 l=180.0n w=250.0n
MM16 net1 A VSS VSS N_18 m=1 l=180.0n w=250.0n
MM15 net6 net16 net1 net1 N_18 m=1 l=180.0n w=250.0n
MM11 CO net16 VSS VSS N_18 m=1 l=180.0n w=250.0n
MM4 net10 B VSS VSS N_18 m=1 l=180.0n w=250.0n
MM3 net16 A net10 net10 N_18 m=1 l=180.0n w=250.0n
MM2 net9 B VSS VSS N_18 m=1 l=180.0n w=250.0n
MM1 net9 A VSS VSS N_18 m=1 l=180.0n w=250.0n
MM0 net16 CI net9 net9 N_18 m=1 l=180.0n w=250.0n
MM28 S net6 VDD VDD P_18 m=1 l=180.0n w=750.0n
MM24 net6 CI net5 net5 P_18 m=1 l=180.0n w=750.0n
MM23 net5 B net4 net4 P_18 m=1 l=180.0n w=750.0n
MM22 net4 A VDD VDD P_18 m=1 l=180.0n w=750.0n
MM21 net3 A VDD VDD P_18 m=1 l=180.0n w=750.0n
MM20 net3 B VDD VDD P_18 m=1 l=180.0n w=750.0n
MM19 net3 CI VDD VDD P_18 m=1 l=180.0n w=750.0n
MM14 net6 net16 net3 net3 P_18 m=1 l=180.0n w=750.0n
MM10 CO net16 VDD VDD P_18 m=1 l=180.0n w=750.0n
MM9 net2 A VDD VDD P_18 m=1 l=180.0n w=750.0n
MM8 net2 B VDD VDD P_18 m=1 l=180.0n w=750.0n
MM7 net11 A VDD VDD P_18 m=1 l=180.0n w=750.0n
MM6 net16 B net11 net11 P_18 m=1 l=180.0n w=750.0n
MM5 net16 CI net2 net2 P_18 m=1 l=180.0n w=750.0n
.ENDS

