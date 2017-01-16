!Workspase 16384
File
 Animal * !P
 Haplotype * !A !SORT
 Pheno
pedigreefile.txt !ALPHA !SORT
TestDFa.txt !MAXIT 100 !EXTRA 2 !FCON !DDF 2
Pheno ~ mu Haplotype !r Animal
0 0 1
Animal 1
0 0 AINV 1 !GP 

predict Haplotype !TDIFF
