$PROBLEM Friberg myelosuppression: the data and starts of figures/fig5_friberg.py
; Data: python checks/nonmem/friberg_nonmem.py data (writes friberg_nm.csv here).
; Model: models/friberg_maturation.tq with its three maturation cells written out as
; compartments T1-T3 (KTR = 4/MTT); drug in AC with CL = 1, V = 50 as in the TenQSP model.
$INPUT ID TIME AMT EVID MDV CMT DV
$DATA friberg_nm.csv IGNORE=@
$SUBROUTINES ADVAN13 TOL=6
$MODEL COMP=(PROL) COMP=(T1) COMP=(T2) COMP=(T3) COMP=(CIRC DEFOBS) COMP=(AC DEFDOSE)
$PK
 CIRC0 = THETA(1)*EXP(ETA(1))
 SLOPE = THETA(2)*EXP(ETA(2))
 MTT   = THETA(3)
 GAMMA = THETA(4)
 CL = 1
 V  = 50
 KTR = 4/MTT
 A_0(1) = CIRC0
 A_0(2) = CIRC0
 A_0(3) = CIRC0
 A_0(4) = CIRC0
 A_0(5) = CIRC0
$DES
 CONC = A(6)/V
 EDRUG = SLOPE*CONC
 CC = A(5)
 IF (CC.LT.1E-12) CC = 1E-12
 FB = (CIRC0/CC)**GAMMA
 DADT(1) = KTR*A(1)*(1-EDRUG)*FB - KTR*A(1)
 DADT(2) = KTR*A(1) - KTR*A(2)
 DADT(3) = KTR*A(2) - KTR*A(3)
 DADT(4) = KTR*A(3) - KTR*A(4)
 DADT(5) = KTR*A(4) - KTR*A(5)
 DADT(6) = -CL/V*A(6)
$ERROR
 IPRED = F
 Y = IPRED + IPRED*THETA(5)*EPS(1)
$THETA (0, 4) (0, 0.3) (0, 80) (0, 0.25) (0, 0.2)
$OMEGA 0.25 0.25
$SIGMA 1 FIX
$ESTIMATION METHOD=1 INTER MAXEVAL=9999 PRINT=1 NOABORT
$COVARIANCE PRINT=E MATRIX=R
