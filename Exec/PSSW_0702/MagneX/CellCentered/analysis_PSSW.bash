# Read the user input   
rm -r raw_data
rm Mx_xface_center_32.txt

mkdir raw_data
touch Mx_xface_center_32.txt

# if ( n == 1 && i == 4 && j == 4 ){
#                         std::cout << i << " " << j << " " << k << " " << n << " " << mfdata(i,j,k,n) << "\n";
#                     }


#!/bin/bash
for timestep in $(seq 0 500 7300000); do
    newnumber='000000000'${timestep}      # get number, pack with zeros
    newnumber=${newnumber:(-8)}       # the last seven characters
    ./WritePlotfileToASCII3d.gnu.ex infile=diags/plt${newnumber} | tee raw_data/${newnumber}.txt
    bash analysis_getline.bash raw_data/${newnumber}.txt 32 | tee -a Mx_xface_center_32.txt
done