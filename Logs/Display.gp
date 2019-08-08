set key off
 #set xrange [-3:2]
 #set zrange [-2:3]
 set yrange [-0.5:1]
set view equal xyz
splot 'map.tsv' using 1:2:3 with points pointsize 0.25 pointtype 7, \
      'pose.tsv' using 2:3:4 with points pointsize 0.35 linecolor rgb "green" pointtype 1
pause -1
