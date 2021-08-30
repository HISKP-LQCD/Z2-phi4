#!/bin/bash

check(){

#cd ../../Ranalysis

file=./data/G2t_T8_L4_msq0-4.900000_msq1-4.900000_l02.500000_l12.500000_mu5.000000_g0.000000_rep0 
ref=./data/G2t_T8_L4_msq0-4.900000_msq1-4.900000_l02.500000_l12.500000_mu5.000000_g0.000000_rep0

Rscript  diff_mesuraments.R   $file $ref

file=./data/checks_T8_L4_msq0-4.900000_msq1-4.900000_l02.500000_l12.500000_mu5.000000_g0.000000_rep0 
ref=./data/checks_T8_L4_msq0-4.900000_msq1-4.900000_l02.500000_l12.500000_mu5.000000_g0.000000_rep0

Rscript  diff_mesuraments.R   $file $ref
}

bad=()
./main/main -i infile.in || bad+=( "running main" ) 
check || bad+=( "main" ) 
./main/contraction -i infile.in || bad+=( "running contractions" ) 
check || bad+=( "contraction" ) 
./main/contraction -i infile_conf.in || bad+=( "running contractions with infile_conf.in" ) 
check || bad+=( "contraction conf" )  
./main/contraction -i infile_confFT.in || bad+=( "running contractions with infile_confFT.in" )
check || bad+=( "contraction confFT" ) 

if [[ -n "${bad-}" ]]; then
  echo -e "\n error in:\n"
  for path in "${bad[@]}"; do
    echo "  - $path"
  done

  exit 1
fi

echo "all good!"
