awk '{print $1," ", " ", $2," ", $3/($1*$1)}' energy-vs-var-fantasy-cmap.dat > energy-vs-hc-fantasy-cmap.dat
awk '{print $1," ", " ", $2," ", $3/($1*$1)}' energy-vs-var-real-cmap.dat > energy-vs-hc-real-cmap.dat
