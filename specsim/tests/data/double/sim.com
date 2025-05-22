#!/bin/csh

simTimeND -ndim 2 -in double.tab -noise 0.0 -scale 1.0 -tdd -nots \
 -xT           512 -yT           128 \
 -xN          1024 -yN           256 \
 -xMODE    Complex -yMODE    Complex \
 -xSW      10000.0 -ySW       2000.0 \
 -xOBS       600.0 -yOBS        61.0 \
 -xCAR       4.745 -yCAR       118.0 \
 -xLAB          HN -yLAB           N \
 -xP0          0.0 -yP0          0.0 \
 -xP1          0.0 -yP1          0.0 \
 -aq2D States -out double.fid -verb -ov

basicFT2.com -in double.fid -ft1 double.ft1 -out double.ft2 -xEXTX1 0% -xEXTXN 100% -xSOL NULL -xBASEARG NULL
