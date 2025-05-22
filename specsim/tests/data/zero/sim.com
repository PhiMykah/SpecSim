#!/bin/csh

simTimeND -ndim 2 -in zero_freq.tab -rms 0.0 -scale 1.0 -nots -tdd \
 -xT           512 -yT           128 \
 -xN          1024 -yN           256 \
 -xMODE    Complex -yMODE    Complex \
 -xSW       1024.0 -ySW        256.0 \
 -xOBS         1.0 -yOBS         1.0 \
 -xCAR         0.0 -yCAR         0.0 \
 -xLAB          Hx -yLAB          Hy \
 -xP0          0.0 -yP0          0.0 \
 -xP1          0.0 -yP1          0.0 \
 -xExp             -yExp             \
 -aq2D States -out zero_freq_exp.fid -verb -ov

simTimeND -ndim 2 -in zero_freq.tab -rms 0.0 -scale 1.0 -nots -tdd \
 -xT           512 -yT           128 \
 -xN          1024 -yN           256 \
 -xMODE    Complex -yMODE    Complex \
 -xSW       1024.0 -ySW        256.0 \
 -xOBS         1.0 -yOBS         1.0 \
 -xCAR         0.0 -yCAR         0.0 \
 -xLAB          Hx -yLAB          Hy \
 -xP0          0.0 -yP0          0.0 \
 -xP1          0.0 -yP1          0.0 \
 -xGauss           -yGauss           \
 -aq2D States -out zero_freq_gauss.fid -verb -ov


# 
# No scaling

basicFT2.com -in zero_freq_exp.fid   -ft1 zero_freq_exp.ft1   -out zero_freq_exp.ft2
basicFT2.com -in zero_freq_gauss.fid -ft1 zero_freq_gauss.ft1 -out zero_freq_gauss.ft2 

exit 0

#
# Setting -Q2 0.5 makes a flat window function (i.e. no window, just first point scaling):

# basicFT2.com -in zero_freq_exp.fid   -ft1 zero_freq_exp.ft1   -out zero_freq_exp.ft2   -xQ2 0.5 -yQ2 0.5
# basicFT2.com -in zero_freq_gauss.fid -ft1 zero_freq_gauss.ft1 -out zero_freq_gauss.ft2 -xQ2 0.5 -yQ2 0.5


