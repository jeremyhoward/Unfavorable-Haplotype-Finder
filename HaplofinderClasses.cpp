#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <iomanip>
#include <omp.h>
#include <cmath>
//#include <math.h>
#include <random>
#include "HaplofinderClasses.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Core>
#include <Eigen/LU>
#include <mkl.h>

using namespace std;

////////////////////////////////////////////////////////////////////////////////////
////////////////////////////      Class Functions       ////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
// Class to store index start and stop site for each chromosome //
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
////////////////////////////
// Constructors ROH_Index //
////////////////////////////
CHR_Index::CHR_Index(){Chromosome = 0; StartIndex = 0; EndIndex = 0; Num_SNP = 0;}
CHR_Index::CHR_Index(int chr, int stind, int enind, int numsnp){Chromosome = chr; StartIndex = stind; EndIndex = enind; Num_SNP = numsnp;}
////////////////////////////
//  Destructors ROH_Index //
////////////////////////////
CHR_Index::~CHR_Index(){}

//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
//         Class to store regions that passed Stage 1-2         //
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
/////////////////////////////////////////
// Constructors Unfavorable Haplotypes //
/////////////////////////////////////////
Unfavorable_Regions::Unfavorable_Regions()
{
    Chromosome_R = 0; StartPos_R = 0; EndPos_R = 0; StartIndex_R = 0; EndIndex_R = 0; Haplotype_R = "0"; Raw_Phenotype= 0.0; effect = 0.0; LS_Mean = 0; t_value = 0.0;
}
Unfavorable_Regions::Unfavorable_Regions(int chr, int st_pos, int en_pos, int st_ind, int en_ind, std::string hap, double pheno, double eff, double lsm, double tval)
{
    Chromosome_R = chr; StartPos_R = st_pos; EndPos_R = en_pos; StartIndex_R = st_ind; EndIndex_R = en_ind; Haplotype_R = hap; Raw_Phenotype = pheno; effect = eff; LS_Mean = lsm; t_value = tval;
}
/////////////////////////////////////////
// Destructors Unfavorable Haplotypes  //
/////////////////////////////////////////
Unfavorable_Regions::~Unfavorable_Regions(){};
/////////////////////////////////////////
//  Functions Unfavorable Haplotypes   //
/////////////////////////////////////////
void Unfavorable_Regions::Update_Haplotype(std::string temp){Haplotype_R = temp;}
void Unfavorable_Regions::Update_RawPheno(double temp){Raw_Phenotype = temp;}
void Unfavorable_Regions::Update_Effect(double temp){effect = temp;}
void Unfavorable_Regions::Update_LSM(double temp){LS_Mean = temp;}
void Unfavorable_Regions::Update_Tstat(double temp){t_value = temp;}
bool sortByStart(const Unfavorable_Regions &lhs, const Unfavorable_Regions &rhs) {return lhs.StartPos_R < rhs.StartPos_R;}
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
//         Class to store regions within each chromosome        //
//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
/////////////////////////////////////////////
// Constructors Unfavorable Haplotypes_sub //
/////////////////////////////////////////////
Unfavorable_Regions_sub::Unfavorable_Regions_sub()
{
    StartIndex_s = 0; EndIndex_s = 0; Haplotype_s = ""; Number_s = 0; Phenotype_s = 0.0; Animal_ID_s = "";
}
Unfavorable_Regions_sub::Unfavorable_Regions_sub(int st, int end, std::string hap_s, int num, double pheno_s, std::string animal_s)
{
    StartIndex_s = st; EndIndex_s = end; Haplotype_s = hap_s; Number_s = num; Phenotype_s = pheno_s; Animal_ID_s = animal_s;
}
/////////////////////////////////////////////
// Destructors Unfavorable Haplotypes_sub //
/////////////////////////////////////////////
Unfavorable_Regions_sub::~Unfavorable_Regions_sub(){};
/////////////////////////////////////////////
//   Functions Unfavorable Haplotypes_sub  //
/////////////////////////////////////////////
void Unfavorable_Regions_sub::Update_substart(int temp){StartIndex_s = temp;}
void Unfavorable_Regions_sub::Update_subend(int temp){EndIndex_s = temp;}
void Unfavorable_Regions_sub::Update_subHaplotype(std::string temp){Haplotype_s = temp;}
void Unfavorable_Regions_sub::Update_subNumber(int temp){Number_s = temp;}
void Unfavorable_Regions_sub::Update_subPhenotype(double temp){Phenotype_s = temp;}
void Unfavorable_Regions_sub::Update_subAnimal_IDs(std::string temp){Animal_ID_s = temp;}
bool sortByPheno(const Unfavorable_Regions_sub &lhs, const Unfavorable_Regions_sub &rhs) {return lhs.Phenotype_s < rhs.Phenotype_s;}


////////////////////////////////////////////////////////////////////////////////////
////////////////////////////     Catch all Functions    ////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
// Check if nan exists //
bool isNaN(double x) {
    return x != x;
}

////////////////////////////
///      Status Bar      ///
////////////////////////////
static inline void loadbar(unsigned int x, unsigned int n, unsigned int w = 50)
{
    if((x != n) && (x % (n/100+1) != 0)) return;
    float ratio_ =  x/(float)n;
    int c = int(ratio_ * w);
    cout << setw(3) << (int)(ratio_*100) << "% [";
    for (int x=0; x<c; x++) cout << "|";
    for (int x=c; x<w; x++) cout << " ";
    cout << "]\r" << flush;
}
////////////////////////////
///    Read in Map File  ///
////////////////////////////
void ReadMapFile_Index(string mapfile, vector < int > &chr, vector < int > &position, vector < int > &index, vector < CHR_Index> &chr_index,ostream& logfile)
{
    logfile << "\n==================================\n";
    logfile << "==\tReading in Map file \t==\n";
    logfile << "==================================\n";
    /* Read in file */
    vector <string> numbers; string line;
    ifstream infile;
    infile.open(mapfile.c_str());
    if(infile.fail()){cout << "Error Opening Map File \n"; exit (EXIT_FAILURE);}
    while (getline(infile,line)){numbers.push_back(line);}  /* Stores in vector and each new line push back to next space */
    for(int i = 0; i < numbers.size(); i++)
    {
        string temp = numbers[i]; size_t pos = temp.find(" ", 0); string tempa = temp.substr(0,pos);
        chr.push_back(atoi(tempa.c_str())); temp.erase(0, pos+1);     /* Grab chromosome number */
        position.push_back(atoi(temp.c_str()));                       /* Grab Position */
        index.push_back(i);                                           /* Make Index */
    }
    numbers.clear();                                            /* clear vector that holds each row */
    int snp = chr.size();
    //cout << chr.size() << " " << position.size() << " " << index.size() << endl;
    //cout << chr[0] << " " << position[0] << " " << index[0] << endl;
    //cout << chr[chr.size()-1] << " " << position[position.size()-1] << " " << index[index.size()-1] << endl;
    /* now index where chromosme starts and ends */
    vector < int > change_locations;                            /* used to track at which point the chromosome switches */
    change_locations.push_back(0);                              /* first SNP is in chromosome 1 */
    /* Create index to grab correct columns from genotype file when constructing ROH and Autozygosity*/
    for(int i = 1; i < snp; i++)
    {
        if((chr[i]-1) == chr[i-1])                      /* Know when it switch because previous one is one less than current one */
        {
            change_locations.push_back(index[i-1]);     /* end of chromosome */
            change_locations.push_back(index[i]);       /* beginning of next chromosome */
        }
    }
    change_locations.push_back(snp-1);                  /* Add the end of last chromosome */
    for(int i = 0; i < (change_locations.size()/2); i++)/* store in vector of CHR_index objects */
    {
        CHR_Index chr_temp((i+1),change_locations[((((i+1)*2)-1)-1)],change_locations[((((i+1)*2))-1)],(change_locations[((((i+1)*2))-1)]-change_locations[((((i+1)*2)-1)-1)]+1));
        chr_index.push_back(chr_temp);
    }
    change_locations.clear();                           /* Clear change_locations vector */
    logfile << "   - Index locations for Chromosome:" << endl;
    for(int i = 0; i < chr_index.size(); i++)
    {
        logfile << "        - Chromosome " << chr_index[i].getChr() << ": " << chr_index[i].getStInd() << " - " << chr_index[i].getEnInd();
        logfile << "; Number of SNP: " << chr_index[i].getNumSnp() << endl;
    }
    logfile << endl;
}
////////////////////////////
// Read in Phenotype File //
////////////////////////////
void ReadPhenoFile_Index(string phenofile, vector<string> &id, vector<double> &pheno, vector<int> &phenogenorownumber, vector<vector<string>> &FIXED_CLASS, vector<vector<string >> &uniqueclass,vector<vector<double>> &FIXED_COV, vector<int> const &fixed_class_col, vector<int> const &fixed_cov_col, int id_column, int phenocolumn,ostream& logfile)
{
    logfile << "\n==================================\n";
    logfile << "==\tReading in Pheno file\t==\n";
    logfile << "==================================" << endl;
    vector <string> numbers; string line;                                               /* Import file and put each row into a vector */
    ifstream infile1;
    infile1.open(phenofile.c_str());
    if(infile1.fail()){cout << "Error Opening Phenotype File \n"; exit (EXIT_FAILURE);}
    while (getline(infile1,line)){numbers.push_back(line);}     /* Stores in vector and each new line push back to next space */
    //cout << numbers.size() << endl;
    /* make size of FIXED_CLASS 2-D vector */
    for(int i = 0; i < numbers.size(); i++)
    {
        vector < string > temp;
        for(int j = 0; j < fixed_class_col.size(); j++){temp.push_back("");}
        FIXED_CLASS.push_back(temp);
        phenogenorownumber.push_back(-5);
    }
    //cout << FIXED_CLASS.size() << " " << FIXED_CLASS[0].size() << endl;
    for(int i = 0; i < numbers.size(); i++)
    {
        vector < double > temp;
        for(int j = 0; j < fixed_cov_col.size(); j++){temp.push_back(0.0);}
        FIXED_COV.push_back(temp);
    } 
    //cout << FIXED_COV.size() << " " << FIXED_COV[0].size() << endl;
    /* Put each column into a vector */
    for(int i = 0; i < numbers.size(); i++)
    {
        vector < string > col_variables;
        string quit = "NO";             /* Don't know how many their are, but seperated by "," */
        while(quit != "YES")
        {
            size_t pos = numbers[i].find(" ",0);
            if(pos > 0)                                                     /* hasn't reached last one yet */
            {
                col_variables.push_back(numbers[i].substr(0,pos)); numbers[i].erase(0, pos + 1);
            }
            if(pos == std::string::npos){quit = "YES";}                     /* has reached last one so now kill while loop */
        }
        if(col_variables.size() != (1 + fixed_class_col.size() + fixed_cov_col.size() + 1))
        {
            cout << endl << "Number of columns doesn't match up with way parameter file was specified.";
            cout << " Check row number " << i + 1 << ". Program Ended." << endl; exit (EXIT_FAILURE);
        }
        //cout << col_variables.size() << endl;
        //for(int j = 0; j < col_variables.size(); j++){cout << col_variables[j] << endl;}
        if(col_variables.size() == (1 + fixed_class_col.size() + fixed_cov_col.size() + 1))
        {
            id.push_back(col_variables[id_column-1]);                       /* Save ID */
            pheno.push_back(atof(col_variables[phenocolumn-1].c_str()));    /* Save Phenotype */
            /* Save cross-classified variables */
            for(int j = 0; j < fixed_class_col.size(); j++){FIXED_CLASS[i][j] = col_variables[fixed_class_col[j]-1];}
            for(int j = 0; j < fixed_cov_col.size(); j++){FIXED_COV[i][j] = atof((col_variables[fixed_cov_col[j]-1]).c_str());}
        }
        //if(i == 0)
        //{
        //    cout  << "    - This is what first line got partitioned into:" << endl;
        //    cout << "       - ID: " << id[i] << endl;
        //    cout << "       - Phenotype: " << pheno[i]  << endl;
        //    for(int j = 0; j < fixed_class_col.size(); j++)
        //    {
        //        cout << "       - Classified Variable " << j + 1 << ": " << FIXED_CLASS[i][j]  << endl;
        //    }
        //    for(int j = 0; j < fixed_cov_col.size(); j++)
        //    {
        //        cout << "       - Covariate Variable " << j + 1 << ": " << FIXED_COV[i][j] << endl;
        //    }
        //}
    }
    for(int i = 0; i < FIXED_CLASS[0].size(); i++)
    {
        vector < string > temp;
        for(int j = 0; j < FIXED_CLASS.size(); j++)
        {
            if(temp.size() > 0)
            {
                int num = 0;
                while(1)
                {
                    if(FIXED_CLASS[j][i] == temp[num]){break;}
                    if(FIXED_CLASS[j][i] != temp[num]){num++;}
                    if(num == temp.size()){temp.push_back(FIXED_CLASS[j][i]); break;}
                }
            }
            if(temp.size() == 0){temp.push_back(FIXED_CLASS[j][i]);}
        }
        uniqueclass.push_back(temp);
    }
    //cout << "    - Number of levels for each class fixed effect: " << endl;
    //for(int i = 0; i < uniqueclass.size(); i++){cout <<"        - Fixed Class Level "<<i+1<<" levels: "<<uniqueclass[i].size()<<"."<<endl;}
    logfile << "    - Total Number of observations in datafile is " << pheno.size() << endl;
    logfile << "    - This is what first line got partitioned into:" << endl;
    logfile << "       - ID: " << id[0] << "\n       - Phenotype: " << pheno[0]  << endl;
    for(int j = 0; j < fixed_class_col.size(); j++){logfile<<"       - Classified Variable "<<j + 1 <<": "<<FIXED_CLASS[0][j]<<endl;}
    for(int j = 0; j < fixed_cov_col.size(); j++){logfile<<"       - Covariate Variable "<<j + 1 <<": "<<FIXED_COV[0][j]<<endl;}
    logfile << "    - Number of levels for each class fixed effect: " << endl;
    for(int i = 0; i < uniqueclass.size(); i++)
    {
        logfile << "        - Fixed Class Level " << i + 1 << " levels: " << uniqueclass[i].size() << "." << endl;
    }
}
////////////////////////////
// Read in Genotype File  //
////////////////////////////
void ReadGenoFile_Index(string genofile, vector<string> &genotype, vector<string> &genotypeID, vector<int> &phenogenorownumber, vector<string> const &id, ostream& logfile)
{
    logfile << "\n==================================" << endl;
    logfile << "==\tReading in Geno file \t==" << endl;
    logfile << "==================================" << endl;
    vector <string> numbers; string line;                                               /* Import file and put each row into a vector */
    /* Import file and put each row into a vector */
    ifstream infile2;
    infile2.open(genofile.c_str());
    if(infile2.fail()){cout << "Error Opening Genotype File \n"; exit (EXIT_FAILURE);}
    while (getline(infile2,line)){numbers.push_back(line);}     /* Stores in vector and each new line push back to next space */
    int genorows = numbers.size();
    /* has to be two rows */
    for(int i = 0; i < genorows; i++)
    {
        vector < string > col_variables; string quit = "NO";
        while(quit != "YES")
        {
            size_t pos = numbers[i].find(" ",0);
            if(pos > 0) {col_variables.push_back(numbers[i].substr(0,pos)); numbers[i].erase(0, pos + 1);} /* hasn't reached last one yet */
            if(pos == std::string::npos){quit = "YES";}                     /* has reached last one so now kill while loop */
        }
        if(col_variables.size() != 2)
        {
            cout << endl << "Incorrect number of columns in genofile.";
            cout << " Check row number " << i + 1 << ". Program Ended." << endl; exit (EXIT_FAILURE);
        }
        if(col_variables.size() == 2){genotypeID.push_back(col_variables[0]); genotype.push_back(col_variables[1]);}
    }
    /* now go back and find ID in phenotype file and match it up with genotype file */
    #pragma omp parallel for
    for(int i = 0; i < id.size(); i++)
    {
        int currentgenoline = 0; string kill = "NO";
        while(kill == "NO")
        {
            if(genotypeID[currentgenoline] != id[i]){currentgenoline++;}
            if(genotypeID[currentgenoline] == id[i]){phenogenorownumber[i] = currentgenoline; kill = "YES";}
            if(currentgenoline > genorows){cout <<"\n Problem matching geno and pheno ID's; Check record "<<i+1<<endl; exit (EXIT_FAILURE);}
        }
    }
    /* Double check to make sure all match */
    int numbermismatch = 0;
    #pragma omp parallel for
    for(int i = 0; i < id.size(); i++){if(id[i] != genotypeID[phenogenorownumber[i]]){numbermismatch += 1;}}
    if(numbermismatch > 0){cout << endl << "    - Number of mismatched ID lines: " << numbermismatch << endl; exit (EXIT_FAILURE);}
    //cout << numbermismatch << endl;
    logfile << "    - Total Number of lines in genotype file is " << genotype.size() << endl;
}
////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
////////////////       Generate Mixed Model Equations        ///////////////////////
////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////
// Figure out A matrix size //
//////////////////////////////
void uniquephenotypeanimals(vector < string > &uniqueID, vector < string > const &id)
{
    for(int i = 0; i < id.size(); i++){uniqueID.push_back(id[i]);}
    //cout << uniqueID.size() << endl;
    int ROWS = id.size();
    int i = 0;                                                                          /* Start at first row and look forward */
    while(i < ROWS)
    {
        int j = i + 1;
        while(j < ROWS)
        {
            if(uniqueID[i] == uniqueID[j]){uniqueID.erase(uniqueID.begin()+j); ROWS = ROWS -1;}
            if(uniqueID[i] != uniqueID[j]){j++;}                                                /* not the same ID so move to next row */
        }
        i++;
    }
    //cout << uniqueID.size() << endl;          /* Size of A Matrix */
}
////////////////////////
// Generate A Inverse //
////////////////////////
void GenerateAinv(string pedigreefile,vector < string > const &uniqueID,vector < string > const &id, double* Relationshipinv_mkl)
{
    /* Read in pedigree file and create Ainv for animals that have a phenotype can be numeric or character,but parents come before progeny */
    time_t fullped_begin_time = time(0);
    vector < string > animal; vector < string > sire; vector < string > dam; string line;
    ifstream infile22;
    infile22.open(pedigreefile);                                                    /* This file has all animals in it */
    if(infile22.fail()){cout << "Error Opening File Pedigree File \n"; exit (EXIT_FAILURE);}
    while (getline(infile22,line))
    {
        /* Fill each array with correct number already in order so don't need to order */
        size_t pos = line.find(" ",0); animal.push_back(line.substr(0,pos)); line.erase(0, pos + 1);            /* Grab Animal ID */
        pos = line.find(" ",0); sire.push_back(line.substr(0,pos)); line.erase(0, pos + 1);                     /* Grab Sire ID */
        dam.push_back(line);                                                                                    /* Grab Dam ID */
    }
    vector < int > renum_animal(animal.size(),0);
    vector < int > renum_sire(animal.size(),0);
    vector < int > renum_dam(animal.size(),0);
    for(int i = 0; i < animal.size(); i++)
    {
        renum_animal[i] = i + 1;
        string temp = animal[i];
        for(int j = 0; j < animal.size(); j++)
        {
            /* change it if sire or dam */
            if(temp == sire[j]){renum_sire[j] = i + 1;}
            if(temp == dam[j]){renum_dam[j] = i + 1;}
        }
    }
    //cout << animal.size() << " " << sire.size() << " " << dam.size() << endl;
    //cout << renum_animal.size() << " " << renum_sire.size() << " " << renum_dam.size() << endl;
    using Eigen::MatrixXd;
    MatrixXd FullRelationship(renum_animal.size(),renum_animal.size());
    for(int i = 0; i < renum_animal.size(); i++)
    {
        if (renum_sire[i] != 0 && renum_dam[i] != 0)
        {
            for (int j = 0; j < i; j++)
            {
                FullRelationship(j,i) = FullRelationship(i,j) = 0.5 * (FullRelationship(j,(renum_sire[i]-1)) + FullRelationship(j,(renum_dam[i]-1)));
            }
            FullRelationship((renum_animal[i]-1),(renum_animal[i]-1)) = 1 + 0.5 * FullRelationship((renum_sire[i]-1),(renum_dam[i]-1));
        }
        if (renum_sire[i] != 0 && renum_dam[i] == 0)
        {
            for (int j = 0; j < i; j++)
            {
                FullRelationship(j,i) = FullRelationship(i,j) = 0.5 * (FullRelationship(j,(renum_sire[i]-1)));
            }
            FullRelationship((renum_animal[i]-1),(renum_animal[i]-1)) = 1;
        }
        if (renum_sire[i] == 0 && renum_dam[i] != 0)
        {
            for (int j = 0; j < i; j++)
            {
                FullRelationship(j,i) = FullRelationship(i,j) = 0.5 * (FullRelationship(j,(renum_dam[i]-1)));
            }
            FullRelationship((renum_animal[i]-1),(renum_animal[i]-1)) = 1;
        }
        if (renum_sire[i] == 0 && renum_dam[i] == 0)
        {
            FullRelationship((renum_animal[i]-1),(renum_animal[i]-1)) = 1;
        }
    }
    vector < int > renum_id (id.size(),0);              /* refers to phenotype row */
    #pragma omp parallel for
    for(int i = 0; i < id.size(); i++)
    {
        int j = 0;
        while(1)
        {
            if(id[i] == animal[j]){renum_id[i] = renum_animal[j]; break;}
            if(id[i] != animal[j]){j++;}
            if(j == animal.size()){cout << "Couldn't Find Animal" << endl; exit (EXIT_FAILURE);}
        }
    }
    // Tabulate animals with haplotypes in order to only use a subset of relationship matrix */
    vector < int > uniqueIDrenum;
    for(int i = 0; i < id.size(); i++){uniqueIDrenum.push_back(renum_id[i]);}
    int ROWS = id.size();
    int i = 0;                                                                          /* Start at first row and look forward */
    while(i < ROWS)
    {
        int j = i + 1;
        while(j < ROWS)
        {
            if(uniqueIDrenum[i] == uniqueIDrenum[j]){uniqueIDrenum.erase(uniqueIDrenum.begin()+j); ROWS = ROWS -1;}
            if(uniqueIDrenum[i] != uniqueIDrenum[j]){j++;}              /* not the same ID so move to next row */
        }
        i++;
    }
    int relsize = uniqueIDrenum.size();
    /*******************************************/
    /*** MKL's Cholesky Decomposition of A   ***/
    /*******************************************/
    // Set up variables to use for functions //
    unsigned long i_pa = 0, j_pa = 0;
    unsigned long na = relsize;
    long long int infoa = 0;
    const long long int int_na =(int)na;
    char lowera ='L';
    #pragma omp parallel for private(j_pa)
    for(i_pa = 0; i_pa < na; i_pa++)
    {
        for(j_pa=0; j_pa < na; j_pa++)
        {
            Relationshipinv_mkl[(i_pa*na) + j_pa] = FullRelationship((uniqueIDrenum[i_pa] - 1),(uniqueIDrenum[j_pa] - 1));
        }
    }
    //for(int i = 0; i < 5; i++)
    //{
    //    for(int j = 0; j < 5; j++){        cout << Relationshipinv_mkl[(i*na) + j] << "\t";}
    //    cout << endl;
    //}
    //cout << endl;
    dpotrf(&lowera, &int_na, Relationshipinv_mkl, &int_na, &infoa);    /* Calculate upper triangular L matrix */
    dpotri(&lowera, &int_na, Relationshipinv_mkl, &int_na, &infoa);    /* Calculate inverse of lower triangular matrix result is the inverse */
    /* Copy upper triangler part to lower traingular part and then you have the inverse ! */
    #pragma omp parallel for private(j_pa)
    for(j_pa = 0; j_pa < na; j_pa++)
    {
        for(i_pa = 0; i_pa <= j_pa; i_pa++)
        {
            Relationshipinv_mkl[(j_pa*na)+i_pa] = Relationshipinv_mkl[(i_pa*na)+j_pa];
        }
    }
    //for(int i = 0; i < 5; i++)
    //{
    //    for(int j = 0; j < 5; j++){cout << Relationshipinv_mkl[(i*na) + j] << "\t";}
    //    cout << endl;
    //}
}
////////////////////////////
// Generate Full Rank XtX //
////////////////////////////
void GenerateFullRankXtX(double* X_Dense, vector <int> &keepremove, vector <double> &MeanPerCovClass,  vector<double> const &pheno, vector<vector<string>> const &FIXED_CLASS, vector<vector<string >> const &uniqueclass,vector<vector<double>> const &FIXED_COV,ostream& logfile)
{
    using Eigen::MatrixXd;
    for(int i = 0; i < pheno.size(); i++)
    {
        X_Dense[(i*keepremove.size())+0] = 1; /* Intercept */
        int columnstart = 1;                                    /* as you loop across this gets added to based on number of levels */
        for(int j = 0; j < FIXED_CLASS[0].size(); j++)
        {
            int col = -5; int k = 0;
            while(k < uniqueclass[j].size())
            {
                if(FIXED_CLASS[i][j] == uniqueclass[j][k]){col = k; break;}
                if(FIXED_CLASS[i][j] != uniqueclass[j][k]){k++;}
                if(k == uniqueclass[j].size()){cout << "Can't Find" << endl; exit (EXIT_FAILURE);}
            }
            /* need to figure out where it should be taking into account intercept and previous class effects; */
            col = columnstart + col;
            X_Dense[(i*keepremove.size())+col] = 1;
            columnstart += uniqueclass[j].size();
        }
        /* Loop across covariate classes and add number to matrix */
        for(int j = 0; j < FIXED_COV[0].size(); j++)
        {
            X_Dense[(i*keepremove.size())+columnstart] = FIXED_COV[i][j]; MeanPerCovClass[j] += FIXED_COV[i][j]; columnstart += 1;
        }
        //for(int j = 0; j < keepremove.size(); j++){cout << X_Dense[(i*keepremove.size())+j] << " ";}
        //cout << endl;
        //if(i > 10){exit (EXIT_FAILURE);}
    }
    int keepremoveindex = 1;
    //for(int i = 0; i < keepremove.size(); i++){cout << keepremove[i] << " ";}
    //cout << endl;
    keepremove[0] = 1;  /* keep intercept */
    for(int i = 0; i < uniqueclass.size(); i++)
    {
        for(int j = 0; j < uniqueclass[i].size(); j++)
        {
            if(j > 0){keepremove[keepremoveindex] = 1;}             /* Zero out first level */
            keepremoveindex++;
        }
    }
    for(int i = 0; i < FIXED_COV[0].size(); i++){keepremove[keepremoveindex] = 1; keepremoveindex++;}
    int dimension = 0;
    for(int i = 0; i < keepremove.size(); i++){dimension += keepremove[i];}
    //for(int i = 0; i < keepremove.size(); i++){cout << keepremove[i] << " ";}
    //cout << endl;
    //cout << dimension << endl;
    /* First check if full rank by remove out first level of effect */
    MatrixXd check(pheno.size(),dimension);
    int checkcolumn = 0;
    for(int i = 0; i < keepremove.size(); i++)
    {
        if(keepremove[i] == 1)
        {
            #pragma omp parallel for
            for(int j = 0; j < pheno.size(); j++){check(j,checkcolumn) = X_Dense[(j*keepremove.size())+i];}
            checkcolumn++;
        }
    }
    //for(int i = 0; i < 10; i++)
    //{
    //    for(int j = 0; j < dimension; j++){cout << check(i,j) << " ";}
    //    cout << endl;
    //}
    Eigen::FullPivLU <MatrixXd> lu(check);
    int rank = lu.rank();
    if(rank != dimension)
    {
        MatrixXd X_dense(pheno.size(),keepremove.size());
        for(int i = 0; i < keepremove.size(); i++)
        {
            #pragma omp parallel for
            for(int j = 0; j < pheno.size(); j++){X_dense(j,i) = X_Dense[(j*keepremove.size())+i];}
        }
        logfile << "        - Not Full Rank for Fixed Effects (rank=" << rank << "; columns=" << dimension << ")!!" << endl;
        int columnstograb = 1; int columnstocheck = 1; int fullranksize = 0;                       /* intercept size */
        vector < int > numberzeroed(FIXED_CLASS[0].size(),0);
        for(int i = 0; i < FIXED_CLASS[0].size(); i++)
        {
            columnstograb += uniqueclass[i].size() - 1;
            columnstocheck += uniqueclass[i].size();
            check.resize(pheno.size(),columnstograb);
            checkcolumn = 0;
            for(int i = 0; i < columnstocheck; i++)
            {
                if(keepremove[i] == 1){check.col(checkcolumn) = X_dense.col(i); checkcolumn++;}
            }
            Eigen::FullPivLU <MatrixXd> lu(check);
            int rank = lu.rank();
            logfile << "            - Check Class Effect " << i + 1 << " (rank=" << rank << "; columns=" << columnstograb << ")!!" << endl;
            if(columnstograb == rank){numberzeroed[i] = 1;}
            if(columnstograb != rank){numberzeroed[i] = 1 + (columnstograb - rank);}
            
        }
        checkcolumn = 1; int columnstozero = 0; int numberlevels = 0;
        for(int i = 0; i < numberzeroed.size(); i++)
        {
            if(numberzeroed[i] == 1){checkcolumn += uniqueclass[i].size();}
            if(numberzeroed[i] != 1)
            {
                logfile << "            - Appears to be issues with fixed class " << i + 1 << "!!" << endl;
                for(int j = 0; j < (uniqueclass[i].size()); j++)
                {
                    if(j > 0){keepremove[checkcolumn] = -5;}
                    checkcolumn++;
                }
                columnstozero = (uniqueclass[i].size() - numberzeroed[i] + 1);
                numberlevels = uniqueclass[i].size() - 1;
            }
        }
        int loop = 0; string found = "NO";
        while(loop < 10)
        {
            vector < int > keepremovetemp (keepremove.size(),0);
            vector < int > zerooutcolumns (numberlevels,0);
            for(int i = 0; i < numberlevels; i++){if(i < (columnstozero-1)){zerooutcolumns[i] = 1;}}
            std::random_shuffle(zerooutcolumns.begin(),zerooutcolumns.end());
            int zerooutindex = 0;
            for(int i = 0; i < keepremove.size(); i++)
            {
                if(keepremove[i] == 1){keepremovetemp[i] = keepremove[i];}
                if(keepremove[i] == -5){keepremovetemp[i] = zerooutcolumns[zerooutindex]; zerooutindex++;}
            }
            dimension = 0;
            for(int i = 0; i < keepremovetemp.size(); i++){dimension += keepremovetemp[i];}
            /* check rank to see if correct */
            check.resize(pheno.size(),dimension);
            int checkcolumn = 0;
            for(int i = 0; i < keepremovetemp.size(); i++)
            {
                if(keepremovetemp[i] == 1){check.col(checkcolumn) = X_dense.col(i); checkcolumn++;}
            }
            Eigen::FullPivLU <MatrixXd> lu(check);
            int rank = lu.rank();
            logfile << "            - Zeroed Out Attemp " << loop + 1 << " (rank=" << rank << "; columns=" << check.cols() << ")!!" << endl;
            if(rank == check.cols())
            {
                logfile << "            -Generated a full rank fixed effect matrix." << endl; loop =9; found = "YES";
            }
            if(loop > 1){columnstozero -= 1;}
            if(loop == 9){for(int i = 0; i < keepremove.size(); i++){keepremove[i] = keepremovetemp[i];}}
            loop++;
        }
        if(found == "NO")
        {
            logfile << "            - Still couldn't Figure it out!! Zero out mean and add first level of largest variable!!" << endl;
            keepremove[0] = 0;
            /* figure out which one is CG; assumed to be largest */
            int largest = 0; int indexlargest = 0;
            for(int i = 0; i < uniqueclass.size(); i++)
            {
                if(uniqueclass[i].size() > largest){largest = uniqueclass[i].size(); indexlargest = i;}
            }
            int zeroesfound = 0;
            for(int i = 0; i < keepremove.size(); i++)
            {
                if(keepremove[i] == 0){zeroesfound += 1;}
                if(zeroesfound == (indexlargest + 1)){keepremove[i] = 1;}
            }
            dimension = 0;
            for(int i = 0; i < keepremove.size(); i++){dimension += keepremove[i];}
            /* check rank to see if correct */
            check.resize(pheno.size(),dimension);
            int checkcolumn = 0;
            for(int i = 0; i < keepremove.size(); i++)
            {
                if(keepremove[i] == 1){check.col(checkcolumn) = X_dense.col(i); checkcolumn++;}
            }
            Eigen::FullPivLU <MatrixXd> lu(check);
            int rank = lu.rank();
            if(rank == check.cols()){logfile << "            - Now fixed effects are full rank!!" << endl;}
            if(rank != check.cols())
            {
                logfile << "            - (rank=" << rank << "; columns=" << check.cols() << ")" <<endl;
                logfile << "            - Couldn't Find a way to make full rank. Alter model!!" << endl; exit (EXIT_FAILURE);
            }
        }
        X_dense.resize(0,0);
    }
}
//////////////////////////
// Generate LHS Reduced //
//////////////////////////
void GenerateLHSRed(vector <int> const &X_i, vector <int> const &X_j, vector <double> const &X_A, int dimension, double* Relationshipinv_mkl, vector <int> &ZW_i, vector <int> &ZW_j, vector <double> &ZW_A, vector <string> id, vector <string> uniqueID,vector <int> &LHSred_i, vector <int> &LHSred_j, vector <double> &LHSred_A, int dim_lhs, vector <double> const &lambda)
{
    using Eigen::MatrixXd;  using Eigen::SparseMatrix; using Eigen::VectorXd;
    double* LHS = new double[dim_lhs*dim_lhs];
    for(int i = 0; i < (dim_lhs*dim_lhs); i++){LHS[i] = 0.0;}
    int ipar, jpar;
    /* Generate Dense X */
    MatrixXd Xfullrank(id.size(),dimension); Xfullrank.setZero(id.size(),dimension);
    for(ipar = 0; ipar < X_i.size(); ipar++)
    {
        if(ipar < (X_i.size()-1))
        {
            for(int jpar = X_i[ipar]; jpar < X_i[ipar+1]; jpar++)
            {
                Xfullrank(ipar,X_j[jpar]) = X_A[jpar];
            }
        }
        if(ipar == (X_i.size()-1))
        {
            for(int jpar = X_i[ipar]; jpar < X_j.size(); jpar++)
            {
                Xfullrank(ipar,X_j[jpar]) = X_A[jpar];
            }
        }
        //for(int j = 0; j < dimension; j++){cout << Xfullrank(ipar,j) << " ";}
        //cout << endl;
        //if(ipar > 10){exit (EXIT_FAILURE);}
    }
    MatrixXd tempXtX(Xfullrank.cols(),Xfullrank.cols());
    tempXtX = Xfullrank.transpose() * Xfullrank;            /* Generate XtX */
    /* Put XtX in LHSinv vector */
    #pragma omp parallel for private(ipar)
    for(ipar = 0; ipar < Xfullrank.cols(); ipar++)
    {
        for(jpar = 0; jpar < Xfullrank.cols(); jpar++){LHS[(ipar*dim_lhs)+jpar] = tempXtX(ipar,jpar);}
    }
    //cout << tempXtX << endl;
    tempXtX.resize(0,0);
    //for(int i = 0; i < 15; i++)
    //{
    //    for(int j = 0; j < 10; j++){cout << LHSinv[(i*dim_lhs)+j]  << " ";}
    //    cout << endl;
    //}
    /* Make ZtZ and WtW */
    /* At this time Z and W are the same; ZtZ and WtW are just a diagonal matrix of the number of observations */
    SparseMatrix<double> Z(id.size(),uniqueID.size());
    vector < double > diagonalztz(uniqueID.size(), 0);
    for(int i = 0; i < id.size(); i++)
    {
        ZW_i.push_back(ZW_j.size());
        int col = -5; int j = 0;
        while(j < uniqueID.size())
        {
            if(id[i] == uniqueID[j])
            {
                ZW_j.push_back(j); ZW_A.push_back(1.0);
                Z.insert(i,j) = 1.0;
                diagonalztz[j] += 1;
                break;}
            if(id[i] != uniqueID[j]){j++;}
            if(j == uniqueID.size()){cout << "Can't Find" << endl; exit (EXIT_FAILURE);}
        }
        //cout << endl << ZW_i.size() << " " << ZW_j.size() << " " << ZW_A.size() << endl;
        //if(i > 2)
        //{
        //    for(int k = 0; k < ZW_j.size(); k++)
        //    {
        //        if(k < ZW_i.size()){cout << ZW_i[k] << "\t" << ZW_j[k] << "\t" << ZW_A[k] << endl;}
        //        if(k >= ZW_i.size()){cout << "--" << "\t" << ZW_j[k] << "\t" << ZW_A[k] << endl;}
        //    }
        //    exit (EXIT_FAILURE);
        //}
    }
    /* Put ZtZ in LHSinv vector */
    int rowadd = dimension; int coladd = dimension;
    //#pragma omp parallel for private(ipar)
    for(ipar = 0; ipar < uniqueID.size(); ipar++)
    {
        for(jpar = 0; jpar < uniqueID.size(); jpar++)
        {
            //cout << ipar << " " << jpar << " " << LHSinv[((ipar+rowadd)*dim_lhs)+(jpar+coladd)] << endl;
            if(ipar != jpar)
            {
                //cout << Relationshipinv_mkl[(ipar*uniqueID.size())+jpar] << endl;
                /* off-diagonal won't be a function of diagonals */
                double temp = (Relationshipinv_mkl[(ipar*uniqueID.size())+jpar] * double(lambda[0]));
                LHS[((ipar+rowadd)*dim_lhs)+(jpar+coladd)] = temp;
                //cout << ipar << " " << jpar << " " << LHSinv[((ipar+rowadd)*dim_lhs)+(jpar+coladd)] << endl;
                //exit (EXIT_FAILURE);
            }
            if(ipar == jpar)
            {
                double temp = diagonalztz[ipar] + (Relationshipinv_mkl[(ipar*uniqueID.size())+jpar] * double(lambda[0]));
                LHS[((ipar+rowadd)*dim_lhs)+(jpar+coladd)] = temp;
            }
            //cout << ipar << " " << jpar << " " << LHSinv[((ipar+rowadd)*dim_lhs)+(jpar+coladd)] << endl;
        }
        //exit (EXIT_FAILURE);
    }
    /* Generate XtZ */
    MatrixXd tempXtZ (Xfullrank.cols(),Z.cols()); tempXtZ = Xfullrank.transpose() * Z;        /* XsubtZ */
    //cout << X_sub.cols() << " " << Z.cols() << endl;
    rowadd = 0; coladd = Xfullrank.cols();
    //#pragma omp parallel for private(ipar)
    for(ipar = 0; ipar < tempXtZ.rows(); ipar++)
    {
        for(jpar = 0; jpar < tempXtZ.cols(); jpar++)
        {
            //cout << ipar+rowadd << "-" << jpar+coladd << "-" << LHSinv[((ipar+rowadd)*dim_lhs)+(jpar+coladd)] << endl;
            //cout << jpar+coladd << "-" << ipar+rowadd << "-" << LHSinv[((jpar+coladd)*dim_lhs)+(ipar+rowadd)] << endl;
            LHS[((ipar+rowadd)*dim_lhs)+(jpar+coladd)] = tempXtZ(ipar,jpar);
            LHS[((jpar+coladd)*dim_lhs)+(ipar+rowadd)] = LHS[((ipar+rowadd)*dim_lhs)+(jpar+coladd)];
            //cout << ipar+rowadd << " " << jpar+coladd << " " << LHSinv[((ipar+rowadd)*dim_lhs)+(jpar+coladd)] << endl;
            //cout << jpar+coladd << " " << ipar+rowadd << " " << LHSinv[((jpar+coladd)*dim_lhs)+(ipar+rowadd)] << endl;
            //if(jpar > 2){exit (EXIT_FAILURE);}
        }
        //exit (EXIT_FAILURE);
    }
    tempXtZ.resize(0,0);
    if(lambda[1] > 0.0)
    {
        MatrixXd tempXtW (Xfullrank.cols(),Z.cols()); tempXtW = Xfullrank.transpose() * Z;        /* XsubtW */
        rowadd = 0; coladd = (Xfullrank.cols()+uniqueID.size());
        for(ipar = 0; ipar < tempXtW.rows(); ipar++)
        {
            for(jpar = 0; jpar < tempXtW.cols(); jpar++)
            {
                //cout << ipar+rowadd << "-" << jpar+coladd << "-" << LHS[((ipar+rowadd)*dim_lhs)+(jpar+coladd)] << endl;
                //cout << jpar+coladd << "-" << ipar+rowadd << "-" << LHS[((jpar+coladd)*dim_lhs)+(ipar+rowadd)] << endl;
                LHS[((ipar+rowadd)*dim_lhs)+(jpar+coladd)] = tempXtW(ipar,jpar);
                LHS[((jpar+coladd)*dim_lhs)+(ipar+rowadd)] = LHS[((ipar+rowadd)*dim_lhs)+(jpar+coladd)];
                //cout << ipar+rowadd << " " << jpar+coladd << " " << LHS[((ipar+rowadd)*dim_lhs)+(jpar+coladd)] << endl;
                //cout << jpar+coladd << " " << ipar+rowadd << " " << LHS[((jpar+coladd)*dim_lhs)+(ipar+rowadd)] << endl;
                //if(jpar > 2){exit (EXIT_FAILURE);}
            }
            //exit (EXIT_FAILURE);
        }
        tempXtW.resize(0,0);
        rowadd = Xfullrank.cols(); coladd = (Xfullrank.cols()+uniqueID.size());
        for(ipar = 0; ipar < uniqueID.size(); ipar++)
        {
            //cout << ipar+rowadd << " " << ipar+coladd << " " << LHS[((ipar+rowadd)*dim_lhs)+(ipar+coladd)] << endl;
            //cout << ipar+coladd << "-" << ipar+rowadd << "-" << LHS[((ipar+coladd)*dim_lhs)+(ipar+rowadd)] << endl;
            LHS[((ipar+rowadd)*dim_lhs)+(ipar+coladd)] = diagonalztz[ipar];
            LHS[((ipar+coladd)*dim_lhs)+(ipar+rowadd)]= LHS[((ipar+rowadd)*dim_lhs)+(ipar+coladd)];
            //cout << ipar+rowadd << " " << ipar+coladd << " " << LHS[((ipar+rowadd)*dim_lhs)+(ipar+coladd)] << endl;
            //cout << ipar+coladd << "-" << ipar+rowadd << "-" << LHS[((ipar+coladd)*dim_lhs)+(ipar+rowadd)] << endl;
            //exit (EXIT_FAILURE);
        }
        rowadd = (Xfullrank.cols()+uniqueID.size()); coladd = (Xfullrank.cols()+uniqueID.size());
        for(ipar = 0; ipar < uniqueID.size(); ipar++)
        {
            //cout << diagonalztz[ipar] << " " << double(lambda[1]) << endl;
            //cout << ipar+rowadd << " " << ipar+coladd << " " << LHS[((ipar+rowadd)*dim_lhs)+(ipar+coladd)] << endl;
            LHS[((ipar+rowadd)*dim_lhs)+(ipar+coladd)] = diagonalztz[ipar] + (1*double(lambda[1]));
            //cout << ipar+rowadd << " " << ipar+coladd << " " << LHS[((ipar+rowadd)*dim_lhs)+(ipar+coladd)] << endl;
            //exit (EXIT_FAILURE);
        }
    }
    //cout << endl << endl;
    //for(int i = 0; i < 10; i++)
    //{
    //    for(int j = 0; j < 10; j++){cout << LHS[(i*dim_lhs) + j] << "\t";}
    //    cout << endl;
    //}
    for(int i = 0; i < dim_lhs; i++)
    {
        LHSred_i.push_back(LHSred_j.size());
        for(int j = i; j < dim_lhs; j++)
        {
            if(LHS[(i*dim_lhs)+j] != 0){LHSred_j.push_back(j); LHSred_A.push_back(LHS[(i*dim_lhs)+j]);}
        }
    }
    //cout << LHSred_i.size() << " " << LHSred_j.size() << " " << LHSred_A.size() << endl;
    //for(int i = 0; i < 10; i++){cout << i + 1 << "- " << LHSred_i[i] << " " << LHSred_j[i] << " " << LHSred_A[i] << endl;}
    Xfullrank.resize(0,0); Z.resize(0,0); delete [] LHS;
}
/////////////////////////////////////////
// Generate LHS with haplotype effects //
/////////////////////////////////////////
void updateLHSinv(vector <int> const &X_i, vector <int> const &X_j, vector <double> const &X_A, int dimension, vector <int> &ZW_i, vector <int> &ZW_j, vector <double> &ZW_A,vector <int> &LHSred_i, vector <int> &LHSred_j, vector <double> &LHSred_A,vector <string> uniqueID, vector <int> const &sub_genotype,float * LHSinvupdated,int dim_lhs, int upddim_lhs, float * solutions, vector < double > const &pheno)
{
    float* RHSupdated = new float[upddim_lhs];
    /* Update LHSinvupdated with old LHS */
    for(int i_p = 0; i_p < LHSred_i.size(); i_p++)
    {
        if(i_p < ( LHSred_i.size()-1))
        {
            for(int j_p = LHSred_i[i_p]; j_p < LHSred_i[i_p+1]; j_p++)
            {
                LHSinvupdated[(LHSred_j[j_p]*upddim_lhs)+i_p] = LHSinvupdated[(i_p*upddim_lhs)+LHSred_j[j_p]] = LHSred_A[j_p];
            }
        }
        if(i_p == ( LHSred_i.size()-1))
        {
            for(int j_p = LHSred_i[i_p]; j_p < LHSred_j.size(); j_p++)
            {
                LHSinvupdated[(LHSred_j[j_p]*upddim_lhs)+i_p] = LHSinvupdated[(i_p*upddim_lhs)+LHSred_j[j_p]] = LHSred_A[j_p];
            }
        }
        RHSupdated[i_p] = 0.0;
    }
    //cout << endl << endl;
    //for(int i = 0; i < 10; i++)
    //{
    //    for(int j = 0; j < 10; j++){cout << LHSinvupdated[(i*upddim_lhs) + j] << "\t";}
    //    cout << endl;
    //}
    int ipar, jpar;
    int rohclasses = upddim_lhs - dim_lhs;
    /* Get appropriate matrices constructed */
    using Eigen::MatrixXd; using Eigen::SparseMatrix;
    /* Generate X and Z */
    MatrixXd X(sub_genotype.size(),dimension); X.setZero(sub_genotype.size(),dimension);
    MatrixXd y(pheno.size(),1);
    for(int i = 0; i < X_i.size(); i++)
    {
        if(i < (X_i.size()-1)){for(int j = X_i[i]; j < X_i[i+1]; j++){X(i,X_j[j]) = X_A[j];}}
        if(i == (X_i.size()-1)){for(int j = X_i[i]; j < X_j.size(); j++){X(i,X_j[j]) = X_A[j];}}
        y(i,0) = pheno[i];
    }
    MatrixXd Z(sub_genotype.size(),uniqueID.size()); Z.setZero(sub_genotype.size(),uniqueID.size());
    for(int i = 0; i < ZW_i.size(); i++)
    {
        if(i < (ZW_i.size()-1)){for(int j = ZW_i[i]; j < ZW_i[i+1]; j++){Z(i,ZW_j[j]) = ZW_A[j];}}
        if(i == (ZW_i.size()-1)){for(int j = ZW_i[i]; j < ZW_j.size(); j++){Z(i,ZW_j[j]) = ZW_A[j];}}
    }
    /* Set up X_hap and non_roh is zeroed out */
    MatrixXd X_hap(sub_genotype.size(),rohclasses); X_hap.setZero(sub_genotype.size(),rohclasses);
    for(int i = 0; i < sub_genotype.size(); i++){if(sub_genotype[i] > 0){X_hap(i,(sub_genotype[i]-1)) = 1;}}
    /* Generate LHS_21 */
    MatrixXd LHS_21 (X_hap.cols(),dim_lhs); LHS_21.setZero(X_hap.cols(),dim_lhs);
    if((dimension+uniqueID.size()) == dim_lhs)      /* No permanent Environment */
    {
        LHS_21.block(0,0,rohclasses,dimension) = (X_hap.transpose() * X);
        LHS_21.block(0,dimension,rohclasses,(dim_lhs-dimension)) = (X_hap.transpose() * Z);
    }
    if((dimension+uniqueID.size()+uniqueID.size()) == dim_lhs)      /* permanent Environment */
    {
        //cout << "permanent" << endl;
        LHS_21.block(0,0,rohclasses,dimension) = (X_hap.transpose() * X);
        LHS_21.block(0,dimension,rohclasses,uniqueID.size()) = (X_hap.transpose() * Z);
        LHS_21.block(0,(dimension+uniqueID.size()),rohclasses,uniqueID.size()) = (X_hap.transpose() * Z);
    }
    /* Fill in LHSinvupdated */
    int row = 0;
    for(int i_p = dim_lhs ; i_p < upddim_lhs; i_p++)
    {
        for(int j_p = 0; j_p < dim_lhs; j_p++)
        {
            LHSinvupdated[(i_p*upddim_lhs)+j_p] = LHSinvupdated[(j_p*upddim_lhs)+i_p] = LHS_21(row,j_p);
        }
        row++;
    }
    LHS_21.resize(0,0);
    MatrixXd LHS_22 (X_hap.cols(),X_hap.cols()); LHS_22 = (X_hap.transpose() * X_hap);
    row = 0;
    for(int i_p = dim_lhs; i_p < upddim_lhs; i_p++)
    {
        int rowj = 0;
        for(int j_p = dim_lhs; j_p < upddim_lhs; j_p++){LHSinvupdated[(i_p*upddim_lhs)+j_p]  = LHS_22(row,rowj); rowj++;}
        row++;
    }
    LHS_22.resize(0,0);
    MatrixXd X_subty(X.cols(),1); MatrixXd Zty(Z.cols(),1);  MatrixXd Wty(Z.cols(),1); MatrixXd Xhapy(X_hap.cols(),1);
    MatrixXd RHS(upddim_lhs,1);
    if((dimension+uniqueID.size()) == dim_lhs)      /* No permanent Environment */
    {
        X_subty = X.transpose() * y; Zty = Z.transpose() * y; Xhapy = X_hap.transpose() * y;
        RHS << X_subty,
        Zty,
        Xhapy;
    }
    if((dimension+uniqueID.size()+uniqueID.size()) == dim_lhs)      /* permanent Environment */
    {
        X_subty = X.transpose() * y; Zty = Z.transpose() * y; Wty = Z.transpose() * y; Xhapy = X_hap.transpose() * y;
        RHS << X_subty,
        Zty,
        Wty,
        Xhapy;
    }
    for(int i = 0; i < upddim_lhs; i++){RHSupdated[i] = RHS(i,0);}
    //for(int i = 0; i < 10; i++)
    //{
    //    for(int j = 0; j < 10; j++){cout << LHSinvupdated[(i*upddim_lhs) + j] << "\t";}
    //    cout << endl;
    //}
    //for(int i = 0; i < upddim_lhs; i++)
    //{
    //    for(int j = 0; j < upddim_lhs; j++)
    //    {
    //        cout << LHSinvupdated[(i*upddim_lhs)+j] << " ";
    //    }
    //    cout << endl;
    //}
    //cout << endl;
    //for(int i = 0; i < upddim_lhs; i++){cout << RHSupdated[i] << " ";}
    //cout << endl;
    X.resize(0,0); Z.resize(0,0); X_hap.resize(0,0); y.resize(0,0); X_subty.resize(0,0);
    Zty.resize(0,0); Xhapy.resize(0,0); RHS.resize(0,0); Wty.resize(0,0);
    /*******************************************/
    /*** MKL's Cholesky Decomposition of LHS ***/
    /*******************************************/
    int N = (int)upddim_lhs;
    unsigned long i_p = 0, j_p = 0;
    unsigned long n = upddim_lhs;
    long long int info = 0;
    const long long int int_n =(int)n;
    const long long int int_na =(int)n * (int)n;
    const long long int increment = int(1);
    char lower='L';  char diag ='N';
    const long long int rowslhs = upddim_lhs; const long long int colsrhs = 1;
    //for(int i = 0; i < 15; i++)
    //{
    //    for(int j = 0; j < 10; j++){cout << LHSinvupdated[(i*upddim_lhs)+j]  << " ";}
    //    cout << endl << endl;;
    //}
    spotrf(&lower, &int_n, LHSinvupdated, &int_n, &info);          /* Calculate upper triangular L matrix */
    spotri(&lower, &int_n, LHSinvupdated, &int_n, &info);          /* Calculate inverse of lower triangular matrix */
    //cout << info << endl;
    /* Copy upper triangler part to lower traingular part and then you have the inverse ! */
    for(int jpar = 0; jpar < upddim_lhs; jpar++)
    {
        for(int ipar = 0; ipar <= jpar; ipar++){LHSinvupdated[(jpar*upddim_lhs)+ipar] = LHSinvupdated[(ipar*upddim_lhs)+jpar];}
    }
    //for(int i = 0; i < upddim_lhs; i++)
    //{
    //    for(int j = 0; j < upddim_lhs; j++){cout << LHSinvupdated[(i*upddim_lhs)+j] << " ";}
    //    cout << endl;
    //}
    //cout << endl;
    /* Generate solutions */
    cblas_sgemv(CblasRowMajor,CblasNoTrans,rowslhs,rowslhs,1.0,LHSinvupdated,rowslhs,RHSupdated,1.0,0.0,solutions,1.0);
    delete [] RHSupdated;
    /* Check if 'nan' exist in dataset; if so just skip over haplotype */
    int numnan = 0;
    for(int i = 0; i < upddim_lhs; i++){if(isNaN(solutions[i])){numnan++;}}
    if(numnan > 0)
    {
        for(int i = 0; i < upddim_lhs; i++){solutions[i] = -999;}
    }
}
//////////////////////////
// Generate ROH T-stats //
//////////////////////////
void estimateROHeffect(float * LHSinvupdated,float * solutions,int upddim_lhs,vector < string > const &factor_red, vector < int > const &zero_columns_red, vector <double> &LSM, vector <double> &T_stat, double resvar,vector<vector<string>> &FIXED_CLASS, vector<vector<string >> &uniqueclass,vector<vector<double>> &FIXED_COV,vector <double> const &MeanPerCovClass)
{
    using Eigen::MatrixXd;
    int fullsizedimen = zero_columns_red.size() + LSM.size()+ 1;
    vector < int > updated_zerocolumns; vector < string > updated_factor;
    for(int i = 0; i < zero_columns_red.size(); i++){updated_zerocolumns.push_back(zero_columns_red[i]); updated_factor.push_back(factor_red[i]);}
    updated_zerocolumns.push_back(0); updated_factor.push_back("haplotype");
    for(int i = 0; i < LSM.size(); i++){updated_zerocolumns.push_back(1); updated_factor.push_back("haplotype");}
    //cout << updated_zerocolumns.size() << " " << updated_factor.size() << endl;
    MatrixXd b_full(fullsizedimen,1);
    MatrixXd LHSinv_full(fullsizedimen,fullsizedimen); LHSinv_full.setZero(fullsizedimen,fullsizedimen);
    /********************************************/
    /* First Fill b_full with zeroed out levels */
    /********************************************/
    int locationred = 0; int locationfull = 0;
    for(int i = 0; i < fullsizedimen; i++)
    {
        if(updated_zerocolumns[i] == 1){b_full(locationfull,0) = solutions[locationred]; locationred++; locationfull++;}
        if(updated_zerocolumns[i] == 0){b_full(locationfull,0) = 0.0; locationfull++;}
    }
    /********************************************/
    /*   Now do LHSinv with zeroed out effects  */
    /********************************************/
    int where_at_in_reduced_i = 0;
    for(int i = 0; i < LHSinv_full.rows(); i++)
    {
        if(updated_zerocolumns[i] == 1)
        {
            int where_at_in_reduced_j = where_at_in_reduced_i;
            for(int j = i; j < LHSinv_full.cols(); j++)
            {
                if(updated_zerocolumns[j] == 1)
                {
                    LHSinv_full(j,i) = LHSinv_full(i,j) = LHSinvupdated[(where_at_in_reduced_i*upddim_lhs)+where_at_in_reduced_j] * double(resvar);
                    where_at_in_reduced_j++;
                }
            }
            where_at_in_reduced_i++;
        }
    }
    //for(int i = (fullsizedimen - (LSM.size()+ 1)); i < fullsizedimen; i++)
    //{
    //    for(int j = (fullsizedimen - (LSM.size()+ 1)); j < fullsizedimen; j++){cout << LHSinv_full(i,j) << " ";}
    //    cout << endl;
    //}
    //for(int i = 0; i < 100; i++){cout << b_full(i,0) << endl;}
    for(int j = 0; j < (LSM.size()+1); j++)
    {
        MatrixXd Lvec(1,b_full.rows());
        int current_haplotype = 0;
        for(int i = 0; i < b_full.rows(); i++)
        {
            if(updated_factor[i] == "int"){Lvec(0,i) = 1;}
            for(int k = 0; k < FIXED_CLASS[0].size(); k++)
            {
                stringstream ss; ss << (k + 1); string str = ss.str();
                string lookup = "Fixed_Class" + str;
                if(updated_factor[i] == lookup){Lvec(0,i) = 1 / double(uniqueclass[k].size());}
            }
            for(int k = 0; k < FIXED_COV[0].size(); k++)
            {
                stringstream ss; ss << (k + 1); string str = ss.str();
                string lookup = "Cov_Class" + str;
                if(updated_factor[i] == lookup){Lvec(0,i) = MeanPerCovClass[k];}
                
            }
            if(updated_factor[i] == "haplotype")
            {
                while(1)
                {
                    if(current_haplotype == j){Lvec(0,i) = 1; current_haplotype++; break;}
                    if(current_haplotype != j){Lvec(0,i) = 0; current_haplotype++; break;}
                }
            }
            if(updated_factor[i] == "Random"){Lvec(0,i) = 0;}
        }
        //for(int i = 0; i < 100; i++){cout << b_full(i,0) << " " << Lvec(0,i) << endl;}
        //for(int i = (fullsizedimen - (LSM.size()+ 1)); i < fullsizedimen; i++){cout << b_full(i,0) << " " << Lvec(0,i) << endl;}
        //exit (EXIT_FAILURE);
        if(j > 0){double temp = (Lvec * b_full).value(); LSM[j-1] = temp;}
    }
    for(int j = 1; j < (LSM.size()+1); j++)
    {
        MatrixXd Lvec(2,b_full.rows());
        int current_haplotype = 1; int baseline = 0;
        for(int i = 0; i < b_full.rows(); i++)
        {
            if(updated_factor[i] == "int"){Lvec(0,i) = 1;Lvec(1,i) = Lvec(0,i) ;}
            for(int k = 0; k < FIXED_CLASS[0].size(); k++)
            {
                stringstream ss; ss << (k + 1); string str = ss.str();
                string lookup = "Fixed_Class" + str;
                if(updated_factor[i] == lookup)
                {
                    Lvec(0,i) = 1 / double(uniqueclass[k].size()); Lvec(1,i) = Lvec(0,i);
                }
            }
            for(int k = 0; k < FIXED_COV[0].size(); k++)
            {
                stringstream ss; ss << (k + 1); string str = ss.str();
                string lookup = "Cov_Class" + str;
                if(updated_factor[i] == lookup)
                {
                    Lvec(0,i) = MeanPerCovClass[k]; Lvec(1,i) = Lvec(0,i);
                }
            }
            if(updated_factor[i] == "haplotype" && baseline > 0)
            {
                while(1)
                {
                    if(current_haplotype == j){Lvec(0,i) = 0; Lvec(1,i) = -1.0; current_haplotype++; break;}
                    if(current_haplotype != j){Lvec(0,i) = 0; Lvec(1,i) = 0; current_haplotype++; break;}
                }
            }
            if(updated_factor[i] == "haplotype" && baseline == 0){Lvec(0,i) = 1; Lvec(1,i) = 0.0; baseline++;}
            if(updated_factor[i] == "Random"){Lvec(0,i) = 0; Lvec(1,i) = 0;}
        }
        // SE is var(a) + var(b) - 2*cov(a,b)
        MatrixXd SE_Matrix(Lvec.rows(),Lvec.rows());
        SE_Matrix = (Lvec * LHSinv_full * Lvec.transpose());
        double SE = SE_Matrix(0,0) + SE_Matrix(1,1) - (2 * SE_Matrix(0,1));
        MatrixXd Means_Matrix (Lvec.rows(),1);
        Means_Matrix = (Lvec * b_full);
        double LSM_Diff = Means_Matrix(0,0) - Means_Matrix(1,0);
        double temp = LSM_Diff / double(sqrt(SE));
        T_stat[j-1] = temp;
    }
    //for(int i = 0; i < LSM.size(); i++){cout << LSM[i] << " " << T_stat[i] << endl;}
}
//////////////////////////////
// Double Check with ASReml //
//////////////////////////////
void doublecheckasreml(vector <CHR_Index> chr_index,int min_Phenotypes,vector <int> const &width,vector <string> const &genotype,vector < int > const &phenogenorownumber,vector <double> const &pheno,int dim_lhs,vector <int> const &X_i, vector <int> const &X_j, vector <double> const &X_A, int dimension, vector <int> &ZW_i, vector <int> &ZW_j, vector <double> &ZW_A,vector <string> const &id,vector <string> uniqueID,vector <int> &LHSred_i, vector <int> &LHSred_j, vector <double> &LHSred_A,vector < string > const &factor_red, vector < int > const &zero_columns_red,double res_var,vector<vector<string>> &FIXED_CLASS, vector<vector<string >> &uniqueclass,vector<vector<double>> &FIXED_COV,vector <double> const &MeanPerCovClass,string unfav_direc)
{
    mt19937 gen(1337);
    /********************************************************************************/
    /***                     Randomly Grab a chr and position                     ***/
    /********************************************************************************/
    /* randomly pick chromsome */
    std::uniform_real_distribution<double> distribution(0,1);                       /* Generate sample */
    double temp = (distribution(gen) * (chr_index.size()-1));
    int chromo = temp + 0.5;
    /* randomly window size */
    temp = (distribution(gen) * (width.size() -1));
    int width_index = temp + 0.5;
    /* randomly grab snp to start at */
    int totalsnp = (genotype[0].substr(chr_index[chromo].getStInd(),chr_index[chromo].getNumSnp()).size());
    int startsnp = (totalsnp-1);
    while((startsnp + width[width_index]) > totalsnp)
    {
        temp = (distribution(gen) * (totalsnp-1)); startsnp = temp + 0.5;   /* ensures window is actually possible */
    }
    //cout << chromo << " " << startsnp << " " << width_index << " " << width[width_index] << endl;
    vector < int > sub_genotype(pheno.size(),0);                                    /* haplotype for each individual */
    vector < string > ROH_haplotypes;                                               /* Tabulates all unique ROH haplotypes */
    vector < int > ROH_ID;                                                          /* Numeric ID for unique haplotype */
    vector < int > Haplo_number;                                                    /* Number of phenotypes that have haplotype */
    for(int i = 0; i < pheno.size(); i++)
    {
        string tempa = genotype[phenogenorownumber[i]].substr(chr_index[chromo].getStInd(),chr_index[chromo].getNumSnp());
        string temp = tempa.substr(startsnp,width[width_index]);                         /* Grab substring */
        /* check to see if a 1 exists; if so then not a ROH */
        size_t found =  temp.find("1");
        /* if not in ROH then replace it with a zero */
        if (found != string::npos){sub_genotype[i] = 0;}
        /* if both found3 and found4 weren't located then in an ROH and replace it with haplotype number  */
        if (found == string::npos)
        {
            /* place into corrent ROH haplotypes bin */
            if(ROH_haplotypes.size() > 0)
            {
                string stop = "GO"; int h = 0;
                while(stop == "GO")
                {
                    if(temp.compare(ROH_haplotypes[h]) == 0)                /* Is the same */
                    {
                        sub_genotype[i] = h + 1; Haplo_number[h] = Haplo_number[h] + 1; stop = "KILL";
                    }
                    if(temp.compare(ROH_haplotypes[h]) != 0){h++;}           /* Not the same */
                    if(h == ROH_haplotypes.size())                          /* If number not match = size of hapLibary then add */
                    {
                        ROH_haplotypes.push_back(temp); Haplo_number.push_back(1); ROH_ID.push_back((h+1));
                        sub_genotype[i] = ROH_haplotypes.size(); stop = "KILL";
                    }
                }
            }
            if(ROH_haplotypes.size() == 0)                                  /* Haplotype library will be empty for first individual */
            {
                ROH_haplotypes.push_back(temp); Haplo_number.push_back(1); ROH_ID.push_back((1));
                sub_genotype[i] = ROH_haplotypes.size();
            }
        }
    }
    /* All haplotypes tabulated now if below frequency threshold zero out and place in non-ROH category */
    int ROWS = ROH_haplotypes.size(); int i = 0;
    while(i < ROWS)
    {
        while(1)
        {
            if(Haplo_number[i] < min_Phenotypes)                     /* less than minimum number of phenotypes so remove from class */
            {
                /* loop through and replace haplotypes to 0 if below minimum */
                for(int h = 0; h < pheno.size(); h++){if(sub_genotype[h] == ROH_ID[i]){sub_genotype[h] = 0;}}
                ROH_haplotypes.erase(ROH_haplotypes.begin()+i);
                Haplo_number.erase(Haplo_number.begin()+i);
                ROH_ID.erase(ROH_ID.begin()+i);
                ROWS = ROWS -1; break;                               /* Reduce size of population so i stays the same */
            }
            if(Haplo_number[i] >= min_Phenotypes){i++; break;}      /* greater than minimum number of phenotypes so remove from class */
            cout << "Step 1 Broke" << endl; exit (EXIT_FAILURE);
        }
    }
    vector < int > old_ROH_ID;
    for(int i = 0; i < ROH_haplotypes.size(); i++){old_ROH_ID.push_back(ROH_ID[i]); ROH_ID[i] = i + 1;}
    /* Renumber genotype ID */
    for(int i = 0; i < sub_genotype.size(); i++)
    {
        if(sub_genotype[i] > 0)
        {
            int j = 0;
            while(1)
            {
                if(sub_genotype[i] == old_ROH_ID[j]){sub_genotype[i] = ROH_ID[j]; break;}
                if(sub_genotype[i] != old_ROH_ID[j]){j++;}
                if(j > old_ROH_ID.size()){cout << "Renumbering Failed " << endl; exit (EXIT_FAILURE);}
            }
        }
    }
    /* Tabulate Phenotypic mean for haplotype */
    vector < double > mean_ROH((ROH_haplotypes.size()+1),0);
    vector < double > number_ROH((ROH_haplotypes.size()+1),0);
    vector < int > category_ROH((ROH_haplotypes.size()+1),0);
    /* sub_genotype has already been binned into 0 (non_roh) and anything greater than 0 is an ROH */
    for(int i = 0; i < pheno.size(); i++)
    {
        mean_ROH[sub_genotype[i]] += pheno[i];
        number_ROH[sub_genotype[i]] += 1;
    }
    for(int i = 0; i < mean_ROH.size(); i++){mean_ROH[i] = mean_ROH[i] / number_ROH[i]; category_ROH[i] = i;}
    //for(int i = 0; i < mean_ROH.size(); i++)
    //{
    //    if(i == 0){cout << category_ROH[i] << "\t" << mean_ROH[i] << "\t" << number_ROH[i] << "\t" << "Non_ROH" << endl;}
    //    if(i >= 1){cout << category_ROH[i] << "\t" << mean_ROH[i] << "\t" << number_ROH[i] << "\t" << ROH_haplotypes[i-1] << endl;}
    //}
    //cout << endl;
    fstream checktestdf; checktestdf.open("ASREML_DF.txt", std::fstream::out | std::fstream::trunc); checktestdf.close();
    std::ofstream output5("ASREML_DF.txt", std::ios_base::app | std::ios_base::out);
    for(int i = 0; i < pheno.size(); i++)
    {
        output5 << id[i] << " ";
        for(int j = 0; j < FIXED_CLASS[0].size(); j++){output5 << FIXED_CLASS[i][j] << " ";}
        for(int j = 0; j < FIXED_COV[0].size(); j++){output5 << FIXED_COV[i][j]<< " ";}
        output5 << sub_genotype[i] << " " << pheno[i] << endl;
    }
    //////////////////////////////////////////////////////////////////
    ///                 Update Inverese LHS Matrix                 ///
    //////////////////////////////////////////////////////////////////
    int upddim_lhs = dim_lhs + mean_ROH.size() - 1;
    float* LHSinvupdated = new float[upddim_lhs*upddim_lhs];
    for(int i = 0; i < upddim_lhs*upddim_lhs; i++){LHSinvupdated[i] = 0.0;}
    float* solutions = new float[upddim_lhs];
    updateLHSinv(X_i,X_j,X_A,dimension,ZW_i,ZW_j,ZW_A,LHSred_i,LHSred_j,LHSred_A,uniqueID,sub_genotype,LHSinvupdated,dim_lhs,upddim_lhs,solutions,pheno);
    //for(int i = 0; i < 100; i++){cout << i + 1 << " " << solutions[i] << endl;}
    //for(int i = dim_lhs; i < upddim_lhs; i++){cout << i + 1 << " " << solutions[i] << endl;}
    vector < double > LSM((mean_ROH.size() - 1),0.0); vector < double > T_stat((mean_ROH.size() - 1),0.0);
    estimateROHeffect(LHSinvupdated,solutions,upddim_lhs,factor_red,zero_columns_red,LSM,T_stat,res_var,FIXED_CLASS,uniqueclass,FIXED_COV,MeanPerCovClass);
    vector < double > effect_beta;
    for(int i = dim_lhs; i < upddim_lhs; i++){effect_beta.push_back(double(solutions[i]));}
    delete [] LHSinvupdated; delete [] solutions;
    cout << "\nWanted to double check with external program: " << endl;
    cout << "Output of non-ROH and each unique ROH haplotype from complete model: " << endl;
    cout << " ROH ID -- Least Square Mean -- Beta Estimate -- T_stat of difference between non-ROH and haplotype " << endl;
    for(int i = 0; i < LSM.size(); i++){cout << i + 1 << " " << LSM[i] << " " << effect_beta[i] << " " << T_stat[i] << endl;}
    cout << endl;
    exit (EXIT_FAILURE);
}
////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
///////        Steps in the Unfavorable Haplotype Finder Algorithm          ////////
////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////
// Generate Phenotype Cutoff //
///////////////////////////////
double phenocutoff(vector <CHR_Index> chr_index,int null_samples,int min_Phenotypes,vector <int> const &width,vector <string> const &genotype,vector < int > const &phenogenorownumber,vector <double> const &pheno,int dim_lhs,vector <int> const &X_i, vector <int> const &X_j, vector <double> const &X_A, int dimension, vector <int> &ZW_i, vector <int> &ZW_j, vector <double> &ZW_A,vector <string> uniqueID,vector <int> &LHSred_i, vector <int> &LHSred_j, vector <double> &LHSred_A,vector < string > const &factor_red, vector < int > const &zero_columns_red,double res_var,vector<vector<string>> &FIXED_CLASS, vector<vector<string >> &uniqueclass,vector<vector<double>> &FIXED_COV,vector <double> const &MeanPerCovClass,string unfav_direc,ostream& logfile)
{
    vector < vector < double > > sample_raw_pheno;
    vector < vector < double > > sample_raw_adj_pheno;
    vector < vector < double > > sample_t_value;
    //make it so it is number of rows based on null_samples
    for(int i = 0; i < null_samples; i++)
    {
        vector < double > temp;
        sample_raw_pheno.push_back(temp); sample_raw_adj_pheno.push_back(temp); sample_t_value.push_back(temp);
    }
    mt19937 gen(1337);
    #pragma omp parallel for
    for(int sample = 0; sample < null_samples; sample++)
    {
        /********************************************************************************/
        /***                     Randomly Grab a chr and position                     ***/
        /********************************************************************************/
        /* randomly pick chromsome */
        std::uniform_real_distribution<double> distribution(0,1);                       /* Generate sample */
        double temp = (distribution(gen) * (chr_index.size()-1));
        int chromo = temp + 0.5;
        /* randomly window size */
        temp = (distribution(gen) * (width.size() -1));
        int width_index = temp + 0.5;
        /* randomly grab snp to start at */
        int totalsnp = (genotype[0].substr(chr_index[chromo].getStInd(),chr_index[chromo].getNumSnp()).size());
        int startsnp = (totalsnp-1);
        while((startsnp + width[width_index]) > totalsnp)
        {
            temp = (distribution(gen) * (totalsnp-1)); startsnp = temp + 0.5;   /* ensures window is actually possible */
        }
        //cout << chromo << " " << startsnp << " " << width_index << " " << width[width_index] << endl;
        vector < int > sub_genotype(pheno.size(),0);                                    /* haplotype for each individual */
        vector < string > ROH_haplotypes;                                               /* Tabulates all unique ROH haplotypes */
        vector < int > ROH_ID;                                                          /* Numeric ID for unique haplotype */
        vector < int > Haplo_number;                                                    /* Number of phenotypes that have haplotype */
        for(int i = 0; i < pheno.size(); i++)
        {
            string tempa = genotype[phenogenorownumber[i]].substr(chr_index[chromo].getStInd(),chr_index[chromo].getNumSnp());
            string temp = tempa.substr(startsnp,width[width_index]);                         /* Grab substring */
            /* check to see if a 1 exists; if so then not a ROH */
            size_t found =  temp.find("1");
            /* if not in ROH then replace it with a zero */
            if (found != string::npos){sub_genotype[i] = 0;}
            /* if both found3 and found4 weren't located then in an ROH and replace it with haplotype number  */
            if (found == string::npos)
            {
                /* place into corrent ROH haplotypes bin */
                if(ROH_haplotypes.size() > 0)
                {
                    string stop = "GO"; int h = 0;
                    while(stop == "GO")
                    {
                        if(temp.compare(ROH_haplotypes[h]) == 0)                /* Is the same */
                        {
                            sub_genotype[i] = h + 1; Haplo_number[h] = Haplo_number[h] + 1; stop = "KILL";
                        }
                        if(temp.compare(ROH_haplotypes[h]) != 0){h++;}           /* Not the same */
                        if(h == ROH_haplotypes.size())                          /* If number not match = size of hapLibary then add */
                        {
                            ROH_haplotypes.push_back(temp); Haplo_number.push_back(1); ROH_ID.push_back((h+1));
                            sub_genotype[i] = ROH_haplotypes.size(); stop = "KILL";
                        }
                    }
                }
                if(ROH_haplotypes.size() == 0)                                  /* Haplotype library will be empty for first individual */
                {
                    ROH_haplotypes.push_back(temp); Haplo_number.push_back(1); ROH_ID.push_back((1));
                    sub_genotype[i] = ROH_haplotypes.size();
                }
            }
        }
        /* All haplotypes tabulated now if below frequency threshold zero out and place in non-ROH category */
        int ROWS = ROH_haplotypes.size(); int i = 0;
        while(i < ROWS)
        {
            while(1)
            {
                if(Haplo_number[i] < min_Phenotypes)                     /* less than minimum number of phenotypes so remove from class */
                {
                    /* loop through and replace haplotypes to 0 if below minimum */
                    for(int h = 0; h < pheno.size(); h++){if(sub_genotype[h] == ROH_ID[i]){sub_genotype[h] = 0;}}
                    ROH_haplotypes.erase(ROH_haplotypes.begin()+i);
                    Haplo_number.erase(Haplo_number.begin()+i);
                    ROH_ID.erase(ROH_ID.begin()+i);
                    ROWS = ROWS -1; break;                               /* Reduce size of population so i stays the same */
                }
                if(Haplo_number[i] >= min_Phenotypes){i++; break;}      /* greater than minimum number of phenotypes so remove from class */
                cout << "Step 1 Broke" << endl; exit (EXIT_FAILURE);
            }
        }
        vector < int > old_ROH_ID;
        for(int i = 0; i < ROH_haplotypes.size(); i++){old_ROH_ID.push_back(ROH_ID[i]); ROH_ID[i] = i + 1;}
        /* Renumber genotype ID */
        for(int i = 0; i < sub_genotype.size(); i++)
        {
            if(sub_genotype[i] > 0)
            {
                int j = 0;
                while(1)
                {
                    if(sub_genotype[i] == old_ROH_ID[j]){sub_genotype[i] = ROH_ID[j]; break;}
                    if(sub_genotype[i] != old_ROH_ID[j]){j++;}
                    if(j > old_ROH_ID.size()){cout << "Renumbering Failed " << endl; exit (EXIT_FAILURE);}
                }
            }
        }
        /* Tabulate Phenotypic mean for haplotype */
        vector < double > mean_ROH((ROH_haplotypes.size()+1),0);
        vector < double > number_ROH((ROH_haplotypes.size()+1),0);
        vector < int > category_ROH((ROH_haplotypes.size()+1),0);
        /* sub_genotype has already been binned into 0 (non_roh) and anything greater than 0 is an ROH */
        for(int i = 0; i < pheno.size(); i++)
        {
            mean_ROH[sub_genotype[i]] += pheno[i];
            number_ROH[sub_genotype[i]] += 1;
        }
        for(int i = 0; i < mean_ROH.size(); i++){mean_ROH[i] = mean_ROH[i] / number_ROH[i]; category_ROH[i] = i;}
        for(int i = 1; i < mean_ROH.size(); i++){sample_raw_pheno[sample].push_back(mean_ROH[i]);}
        //if(mean_ROH.size() > 1)
        //{
        //    for(int i = 0; i < mean_ROH.size(); i++)
        //    {
        //        if(i == 0){cout << category_ROH[i] << "\t" << mean_ROH[i] << "\t" << number_ROH[i] << "\t" << "Non_ROH" << endl;}
        //        if(i >= 1){cout << category_ROH[i] << "\t" << mean_ROH[i] << "\t" << number_ROH[i] << "\t" << ROH_haplotypes[i-1] << endl;}
        //    }
        //    cout << endl;
        //}
        //cout << mean_ROH.size() << endl;
        //fstream checktestdf; checktestdf.open("TestDFa.txt", std::fstream::out | std::fstream::trunc); checktestdf.close();
        //std::ofstream output5("TestDFa.txt", std::ios_base::app | std::ios_base::out);
        //for(int i = 0; i < pheno.size(); i++)
        //{
        //    output5 << id[i] << " ";
        //    for(int j = 0; j < FIXED_CLASS[0].size(); j++){output5 << FIXED_CLASS[i][j] << " ";}
        //    for(int j = 0; j < FIXED_COV[0].size(); j++){output5 << FIXED_COV[i][j]<< " ";}
        //    output5 << sub_genotype[i] << " " << pheno[i] << endl;
        //}
        //////////////////////////////////////////////////////////////////
        ///                 Update Inverese LHS Matrix                 ///
        //////////////////////////////////////////////////////////////////
        if(mean_ROH.size() > 1)
        {
            int upddim_lhs = dim_lhs + mean_ROH.size() - 1;
            float* LHSinvupdated = new float[upddim_lhs*upddim_lhs];
            for(int i = 0; i < upddim_lhs*upddim_lhs; i++){LHSinvupdated[i] = 0.0;}
            float* solutions = new float[upddim_lhs];
            updateLHSinv(X_i,X_j,X_A,dimension,ZW_i,ZW_j,ZW_A,LHSred_i,LHSred_j,LHSred_A,uniqueID,sub_genotype,LHSinvupdated,dim_lhs,upddim_lhs,solutions,pheno);
            //for(int i = 0; i < 100; i++){cout << i + 1 << " " << solutions[i] << endl;}
            //for(int i = dim_lhs; i < upddim_lhs; i++){cout << i + 1 << " " << solutions[i] << endl;}
            int solutionsneg999 = 0;
            for(int i = 0; i < upddim_lhs; i++){if(solutions[i] == -999){solutionsneg999++;}}
            if(solutionsneg999 < upddim_lhs)
            {
                vector < double > LSM((mean_ROH.size() - 1),0.0); vector < double > T_stat((mean_ROH.size() - 1),0.0);
                estimateROHeffect(LHSinvupdated,solutions,upddim_lhs,factor_red,zero_columns_red,LSM,T_stat,res_var,FIXED_CLASS,uniqueclass,FIXED_COV,MeanPerCovClass);
                /* Save in 2-D vector */
                for(int i = 0; i < LSM.size(); i++)
                {
                    for(int i = 0; i < LSM.size(); i++){if(isNaN(T_stat[i])){T_stat[i] = 0.0;}}
                    //cout << LSM[i] << " " << T_stat[i] << endl;
                    sample_raw_adj_pheno[sample].push_back(LSM[i]); sample_t_value[sample].push_back(T_stat[i]);
                }
                delete [] LHSinvupdated; delete [] solutions;
            }
        }
        if(sample % 10 == 0){logfile << "      - " << sample << endl;}
    }
    vector < double > pheno_given_percentile;
    /* loop through samples and determine if value falls within given interval */
    for(int i = 0; i < sample_t_value.size(); i++)
    {
        for(int j = 0; j < sample_t_value[i].size(); j++)
        {
            if(unfav_direc == "low")
            {
                if(sample_t_value[i][j] >= -1.645 && sample_t_value[i][j] <= -1.282){pheno_given_percentile.push_back(sample_raw_pheno[i][j]);}
            }
            if(unfav_direc == "high")
            {
                if(sample_t_value[i][j] >= 1.282 && sample_t_value[i][j] <= 1.645){pheno_given_percentile.push_back(sample_raw_pheno[i][j]);}
            }
        }
    }
    /* delete 2-D vectors */
    for(int i = 0; i < sample_raw_pheno.size(); i++){sample_raw_pheno[i].clear();}
    for(int i = 0; i < sample_raw_adj_pheno.size(); i++){sample_raw_adj_pheno[i].clear();}
    for(int i = 0; i < sample_t_value.size(); i++){sample_t_value[i].clear();}
    sample_raw_pheno.clear(); sample_raw_adj_pheno.clear(); sample_t_value.clear();
    /* If observations within range are above 2 then can estimate mean. If not will give a -nan so exit out */
    double tempphenotype_cutoff;
    if(pheno_given_percentile.size() <= 2)
    {
        cout << endl << "No observations fell within given percentile to generate cutoff!! " << endl; exit (EXIT_FAILURE);
    }
    if(pheno_given_percentile.size() > 2)
    {
        /* Find mean */
        double sum = 0;
        for(int i = 0; i < pheno_given_percentile.size(); i++){sum += pheno_given_percentile[i];}
        tempphenotype_cutoff = sum / pheno_given_percentile.size();
    }
    pheno_given_percentile.clear();
    return tempphenotype_cutoff;
    
}
///////////////////////////
//// Step 1 of Algorithm //
///////////////////////////
void Step1(vector < Unfavorable_Regions_sub > regions_sub, double phenotype_cutoff, string unfav_direc, int chromo, vector <CHR_Index> chr_index,int min_Phenotypes,vector <int> const &width,vector <string> const &genotype,vector < int > const &phenogenorownumber,vector <double> const &pheno, vector <string> const &id,vector<int> const &chr, vector<int> const &position, vector<int> const &index,vector < Unfavorable_Regions > &regions)
{
    for(int increment = 0; increment < width.size(); increment++)              /* Loop through and reduce window size 10 */
    {
        int totalsnp = (genotype[0].substr(chr_index[chromo].getStInd(),chr_index[chromo].getNumSnp()).size());
        vector < Unfavorable_Regions_sub > regions_subinc;                 /* vector of objects to store everything about unfavorable region */
        /* first initialize to length equal to totalsnp which is max length */
        for(int i = 0; i < totalsnp; i++)
        {
            Unfavorable_Regions_sub regionsub_temp(-5,0,"",0,0.0,"");
            regions_subinc.push_back(regionsub_temp);
        }
        /* Start to move sliding haplotype window by one; can do in parrallel since already initialized */
        #pragma omp parallel for
        for(int scan = 0; scan < totalsnp; scan++)
        {
            /* First determine if length goes past longest; if so then stops */
            if((scan+width[increment]) < (totalsnp-1))
            {
                /* create vectors to hold things pertaining to haplotypes and animals */
                vector < int > sub_genotype(pheno.size(),0);                                    /* haplotype for each individual */
                vector < string > ROH_haplotypes;                                               /* Tabulates all unique ROH haplotypes */
                vector < int > ROH_ID;                                                          /* Numeric ID for unique haplotype */
                vector < int > Haplo_number;                                                    /* Number of phenotypes that have haplotype */
                //if(start_pos.size() > 0)
                //{
                //    if(start_pos[start_pos.size()-1] == 91 && end_pos[end_pos.size()-1] == 105)
                //    {
                //        logfile<<scan<<" "<<start_pos.size()-1 << " " << start_pos[start_pos.size()-1] << " " << end_pos[end_pos.size()-1] << endl;
                //        exit (EXIT_FAILURE);
                //    }
                //}
                //if(scan >= 9 && increment == 7 ){logfile << scan << " " << haplotype_worst[21] << " " << start_pos[21] << " " << end_pos[21] << endl;}
                //if(scan >= 122 && increment == 0){cout << scan << " " << haplotype_worst[0] << " " << start_pos[0] << " " << end_pos[0] << endl;}
                for(int i = 0; i < pheno.size(); i++)
                {
                    string tempa = genotype[phenogenorownumber[i]].substr(chr_index[chromo].getStInd(),chr_index[chromo].getNumSnp());
                    string temp = tempa.substr(scan,width[increment]);                         /* Grab substring */
                    /* check to see if a 1 exists; if so then not a ROH */
                    size_t found =  temp.find("1");
                    /* if not in ROH then replace it with a zero */
                    if (found != string::npos){sub_genotype[i] = 0;}
                    /* if both found3 and found4 weren't located then in an ROH and replace it with haplotype number  */
                    if (found == string::npos)
                    {
                        /* place into corrent ROH haplotypes bin */
                        if(ROH_haplotypes.size() > 0)
                        {
                            string stop = "GO"; int h = 0;
                            while(stop == "GO")
                            {
                                if(temp.compare(ROH_haplotypes[h]) == 0)                /* Is the same */
                                {
                                    sub_genotype[i] = h + 1; Haplo_number[h] = Haplo_number[h] + 1; stop = "KILL";
                                }
                                if(temp.compare(ROH_haplotypes[h]) != 0){h++;}           /* Not the same */
                                if(h == ROH_haplotypes.size())                          /* If number not match = size of hapLibary then add */
                                {
                                    ROH_haplotypes.push_back(temp); Haplo_number.push_back(1); ROH_ID.push_back((h+1));
                                    sub_genotype[i] = ROH_haplotypes.size(); stop = "KILL";
                                }
                            }
                        }
                        if(ROH_haplotypes.size() == 0)                                  /* Haplotype library will be empty for first individual */
                        {
                            ROH_haplotypes.push_back(temp); Haplo_number.push_back(1); ROH_ID.push_back((1));
                            sub_genotype[i] = ROH_haplotypes.size();
                        }
                    }
                }
                //if(scan >= 122 && increment == 0){cout << scan << " " << haplotype_worst[0] << " " << start_pos[0] << " " << end_pos[0] << endl;}
                //cout << ROH_haplotypes.size() << endl;
                //for(int i = 0; i < ROH_haplotypes.size(); i++){cout << ROH_haplotypes[i] << "\t" << ROH_ID[i] << "\t" << Haplo_number[i] << endl;}
                //cout << endl;
                /* All haplotypes tabulated now if below frequency threshold zero out and place in non-ROH category */
                int ROWS = ROH_haplotypes.size(); int i = 0;
                while(i < ROWS)
                {
                    while(1)
                    {
                        if(Haplo_number[i] < min_Phenotypes)                     /* less than minimum number of phenotypes so remove from class */
                        {
                            /* loop through and replace haplotypes to 0 if below minimum */
                            for(int h = 0; h < pheno.size(); h++){if(sub_genotype[h] == ROH_ID[i]){sub_genotype[h] = 0;}}
                            ROH_haplotypes.erase(ROH_haplotypes.begin()+i);
                            Haplo_number.erase(Haplo_number.begin()+i);
                            ROH_ID.erase(ROH_ID.begin()+i);
                            ROWS = ROWS -1; break;                               /* Reduce size of population so i stays the same */
                        }
                        if(Haplo_number[i] >= min_Phenotypes){i++; break;}      /* greater than minimum number of phenotypes so remove from class */
                        cout << "Step 1 Broke" << endl; exit (EXIT_FAILURE);
                    }
                }
                vector < int > old_ROH_ID;
                for(int i = 0; i < ROH_haplotypes.size(); i++){old_ROH_ID.push_back(ROH_ID[i]); ROH_ID[i] = i + 1;}
                /* Renumber genotype ID */
                for(int i = 0; i < sub_genotype.size(); i++)
                {
                    if(sub_genotype[i] > 0)
                    {
                        int j = 0;
                        while(1)
                        {
                            if(sub_genotype[i] == old_ROH_ID[j]){sub_genotype[i] = ROH_ID[j]; break;}
                            if(sub_genotype[i] != old_ROH_ID[j]){j++;}
                            if(j > old_ROH_ID.size()){cout << "Renumbering Failed " << endl; exit (EXIT_FAILURE);}
                        }
                    }
                }
                /* Tabulate Phenotypic mean for haplotype */
                vector < double > mean_ROH;
                vector < double > number_ROH;
                vector < int > category_ROH;
                vector < string > haplo_string;
                for(int i = 0; i < (ROH_haplotypes.size()+1); i++)
                {
                    mean_ROH.push_back(0);
                    number_ROH.push_back(0);
                    category_ROH.push_back(0);
                    if(i == 0){haplo_string.push_back("Non_ROH");}
                    if(i > 0){haplo_string.push_back(ROH_haplotypes[i-1]);}
                }
                for(int i = 0; i < pheno.size(); i++)
                {
                    mean_ROH[sub_genotype[i]] += pheno[i];
                    number_ROH[sub_genotype[i]] += 1;
                }
                for(int i = 0; i < mean_ROH.size(); i++){mean_ROH[i] = mean_ROH[i] / number_ROH[i]; category_ROH[i] = i;}
                //for(int i = 0; i < mean_ROH.size(); i++)
                //{
                //    cout << category_ROH[i] << "\t" << mean_ROH[i] << "\t" << number_ROH[i] << "\t" << haplo_string[i] << endl;
                //}
                //cout << endl << endl;
                int temp; double tempa; double tempb; string tempc;
                for(int i = 0; i < (mean_ROH.size() - 1); i++)
                {
                    for(int j = i+1; j < mean_ROH.size(); j++)
                    {
                        if(unfav_direc == "low")
                        {
                            if(mean_ROH[i] > mean_ROH[j])
                            {
                                temp = category_ROH[i]; tempa = mean_ROH[i]; tempb = number_ROH[i]; tempc = haplo_string[i];
                                category_ROH[i] = category_ROH[j]; mean_ROH[i] = mean_ROH[j];
                                number_ROH[i] = number_ROH[j]; haplo_string[i] = haplo_string[j];
                                category_ROH[j] = temp; mean_ROH[j] = tempa; number_ROH[j] = tempb; haplo_string[j] = tempc;
                            }
                        }
                        if(unfav_direc == "high")
                        {
                            if(mean_ROH[i] < mean_ROH[j])
                            {
                                temp = category_ROH[i]; tempa = mean_ROH[i]; tempb = number_ROH[i]; tempc = haplo_string[i];
                                category_ROH[i] = category_ROH[j]; mean_ROH[i] = mean_ROH[j];
                                number_ROH[i] = number_ROH[j]; haplo_string[i] = haplo_string[j];
                                category_ROH[j] = temp; mean_ROH[j] = tempa; number_ROH[j] = tempb; haplo_string[j] = tempc;
                            }
                        }
                    }
                }
                //for(int i = 0; i < mean_ROH.size(); i++)
                //{
                //    cout << category_ROH[i] << "\t" << mean_ROH[i] << "\t" << number_ROH[i] << "\t" << haplo_string[i] << endl;
                //}
                //cout << endl << endl;
                /* only save if below minimum phenotype threshold */
                if(unfav_direc == "low")
                {
                    if(mean_ROH[0] < phenotype_cutoff && haplo_string[0] != "Non_ROH")
                    {
                        string stringedid = "";
                        for(int i = 0; i < id.size(); i++)
                        {
                            string tempa = genotype[phenogenorownumber[i]].substr(chr_index[chromo].getStInd(),chr_index[chromo].getNumSnp());
                            string temp = tempa.substr(scan,width[increment]);                         /* Grab substring */
                            if(temp == ROH_haplotypes[(category_ROH[0]-1)]){stringedid = stringedid + "_" + id[i];}
                        }
                        int endindex = (scan + width[increment]);
                        regions_subinc[scan].Update_substart(scan);
                        regions_subinc[scan].Update_subend(endindex);
                        regions_subinc[scan].Update_subHaplotype(haplo_string[0]);
                        regions_subinc[scan].Update_subNumber(number_ROH[0]);
                        regions_subinc[scan].Update_subPhenotype(mean_ROH[0]);
                        regions_subinc[scan].Update_subAnimal_IDs(stringedid);
                        //cout << regions_subinc[scan].getStartIndex_s() << " " << regions_subinc[scan].getEndIndex_s() << " ";
                        //cout << regions_subinc[scan].getHaplotype_s() << " " << regions_subinc[scan].getNumber_s() << " ";
                        //cout << regions_subinc[scan].getPhenotype_s() << " " << regions_subinc[scan].getAnimal_ID_s() << endl;
                    }
                }
                if(unfav_direc == "high")
                {
                    if(mean_ROH[0] > phenotype_cutoff && haplo_string[0] != "Non_ROH")
                    {
                        string stringedid = "";
                        for(int i = 0; i < id.size(); i++)
                        {
                            string tempa = genotype[phenogenorownumber[i]].substr(chr_index[chromo].getStInd(),chr_index[chromo].getNumSnp());
                            string temp = tempa.substr(scan,width[increment]);                         /* Grab substring */
                            if(temp == ROH_haplotypes[(category_ROH[0]-1)]){stringedid = stringedid + "_" + id[i];}
                        }
                        int endindex = (scan + width[increment]);
                        regions_subinc[scan].Update_substart(scan);
                        regions_subinc[scan].Update_subend(endindex);
                        regions_subinc[scan].Update_subHaplotype(haplo_string[0]);
                        regions_subinc[scan].Update_subNumber(number_ROH[0]);
                        regions_subinc[scan].Update_subPhenotype(mean_ROH[0]);
                        regions_subinc[scan].Update_subAnimal_IDs(stringedid);
                        //cout << regions_subinc[scan].getStartIndex_s() << " " << regions_subinc[scan].getEndIndex_s() << " ";
                        //cout << regions_subinc[scan].getHaplotype_s() << " " << regions_subinc[scan].getNumber_s() << " ";
                        //cout << regions_subinc[scan].getPhenotype_s() << " " << regions_subinc[scan].getAnimal_ID_s() << endl;
                    }
                }
            }
        }
        /* remove ones that were below threshold which are ones that are still -5 */
        int i = 0;
        while(i < regions_subinc.size())
        {
            if(regions_subinc[i].getStartIndex_s() == -5){regions_subinc.erase(regions_subinc.begin()+i);}
            if(regions_subinc[i].getStartIndex_s() != -5){i++;}
        }
        /* now add to regions_sub */
        for(int i = 0; i < regions_subinc.size(); i++)
        {
            Unfavorable_Regions_sub regionsub_temp(regions_subinc[i].getStartIndex_s(),regions_subinc[i].getEndIndex_s(),regions_subinc[i].getHaplotype_s(),regions_subinc[i].getNumber_s(),regions_subinc[i].getPhenotype_s(),regions_subinc[i].getAnimal_ID_s());
            regions_sub.push_back(regionsub_temp);
        }
        //for(int i = 0; i < regions_sub.size(); i++)
        //{
        //    if(haplotype_worst[i] == ""){logfile << start_pos[i] << " " << end_pos[i] << endl; exit (EXIT_FAILURE);}
        //}
        //cout << regions_sub.size() << endl;
        //for(int i = 0; i < regions_sub.size(); i++)
        //{
        //    cout << regions_sub[i].getStartIndex_s() << " " << regions_sub[i].getEndIndex_s() << " " << regions_sub[i].getNumber_s() << " ";
        //    cout <<  regions_sub[i].getHaplotype_s() << " " << regions_sub[i].getPhenotype_s() << endl;
        //}
        //cout << endl;
        int ROWS = regions_sub.size();                           /* Current Size of summary statistics */
        i = 1;
        while(i < ROWS)
        {
            while(1)
            {
                string kill = "YES";                        /* used to kill the program if doesn't pass at least if statement */
                /* if have same exact same animals contained within haplotype i & i -1 and have a different end position of +1 then */
                /* recombination hasn't broken it down at this point and all individuals have same haplotype and therefore can be seen */
                /* as nested haplotypes therefore lump them together */
                if(regions_sub[i].getAnimal_ID_s()==regions_sub[i-1].getAnimal_ID_s() && (regions_sub[i].getEndIndex_s()-regions_sub[i-1].getEndIndex_s()==1))
                {
                    /* Double Check to see if exactly the same except for first and last one */
                    //cout << regions_sub[i-1].getStartIndex_s() << " " << regions_sub[i-1].getEndIndex_s() << " " << regions_sub[i-1].getHaplotype_s() << " ";
                    //cout << regions_sub[i-1].getNumber_s() << " " << regions_sub[i-1].getPhenotype_s() << endl;
                    //cout << regions_sub[i].getStartIndex_s() << " " << regions_sub[i].getEndIndex_s() << " " << regions_sub[i].getHaplotype_s() << " ";
                    //cout << regions_sub[i].getNumber_s() << " " << regions_sub[i].getPhenotype_s() << endl;
                    int start_previous = (regions_sub[i-1].getHaplotype_s()).size() - (width[increment]) + 1;
                    /* Grab substring */
                    string temp_1 = regions_sub[i-1].getHaplotype_s().substr(start_previous,(regions_sub[i-1].getHaplotype_s()).size());
                    string temp_2 = regions_sub[i].getHaplotype_s().substr(0,((regions_sub[i].getHaplotype_s()).size()-1));
                    //cout << temp_1 << endl;
                    //cout << temp_2 << endl;
                    if(temp_1 == temp_2)
                    {
                        /* update haplotype by adding new part and then delete it */
                        temp_1 = regions_sub[i].getHaplotype_s().substr(((regions_sub[i].getHaplotype_s()).size()-1),(regions_sub[i].getHaplotype_s()).size());
                        int new_end_pos = regions_sub[i].getEndIndex_s();
                        string new_haplotype = regions_sub[i-1].getHaplotype_s() + temp_1;
                        //cout << new_haplotype << " " << regions_sub[i-1].getHaplotype_s() << " " << temp_1 << endl;;
                        regions_sub[i-1].Update_subHaplotype(new_haplotype); regions_sub[i-1].Update_subend(new_end_pos);
                        //cout << regions_sub[i-1].getStartIndex_s() << " " << regions_sub[i-1].getEndIndex_s() << " " << regions_sub[i-1].getHaplotype_s() << " ";
                        //cout << regions_sub[i-1].getNumber_s() << " " << regions_sub[i-1].getPhenotype_s() << endl;
                        regions_sub.erase(regions_sub.begin()+i); ROWS = ROWS -1; kill = "NO"; break; /* Reduce size of population so i stays the same */
                    }
                    if(temp_1 != temp_2)
                    {
                        cout <<regions_sub[i-1].getStartIndex_s()<<" "<<regions_sub[i-1].getEndIndex_s()<<" "<<regions_sub[i-1].getHaplotype_s()<<endl;
                        cout <<regions_sub[i].getStartIndex_s()<<" "<<regions_sub[i].getEndIndex_s()<<" "<<regions_sub[i].getHaplotype_s()<< endl;
                        cout << "KILLED at step 1a" << endl; exit (EXIT_FAILURE);
                    }
                }
                /* if at least one not pass then keeps this one and doesn't combine; should be the rest */
                if(regions_sub[i].getAnimal_ID_s()!=regions_sub[i-1].getAnimal_ID_s() || (regions_sub[i].getEndIndex_s()-regions_sub[i-1].getEndIndex_s()!=1))
                {
                    i++; kill = "NO"; break;
                }
                /* Should pass at least one previous if statement if not kill program */
                if(kill == "YES"){cout << "KILLED at step 1b" << endl; exit (EXIT_FAILURE);}
            }
        }
        //cout << regions_sub.size() << endl;
        //for(int i = 0; i < regions_sub.size(); i++)
        //{
        //    cout << regions_sub[i].getStartIndex_s() << " " << regions_sub[i].getEndIndex_s() << " " << regions_sub[i].getNumber_s() << " ";
        //    cout << regions_sub[i].getHaplotype_s() << " " << regions_sub[i].getPhenotype_s() << endl;
        //}
        //cout << endl;
        /************************************************************************************/
        /* Tabulated Unique Phenotypic Means Now add to Statistics across all increments.   */
        /* As you keep reducing the size you either narrow the region and keep same mean.   */
        /************************************************************************************/
    }
    sort(regions_sub.begin(), regions_sub.end(), sortByPheno);
    //cout << regions_sub.size() << endl;
    //for(int i = 0; i < regions_sub.size(); i++)
    //for(int i = 0; i < 20; i++)
    //{
    //    cout << regions_sub[i].getStartIndex_s() << " " << regions_sub[i].getEndIndex_s() << " " << regions_sub[i].getHaplotype_s() << " ";
    //    cout << regions_sub[i].getNumber_s() << " " << regions_sub[i].getPhenotype_s() << endl;
    //}
    //cout << endl;
    /* Double check to make sure nothing got changed if so changed it back; this is a BUG; for some reason a value will randomly get changed to 0*/
    for(int i = 0; i < regions_sub.size(); i++)
    {
        if((regions_sub[i].getHaplotype_s()).size() != (regions_sub[i].getEndIndex_s() - regions_sub[i].getStartIndex_s()))
        {
            int tempstart = regions_sub[i].getEndIndex_s() - (regions_sub[i].getHaplotype_s()).size();
            regions_sub[i].Update_substart(tempstart);
        }
    }
    int ROWS = regions_sub.size();                                              /* Current Size of summary statistics */
    int i = 1;                                                                  /* Start at one because always look back at previous one */
    while(i < ROWS)
    {
        while(1)
        {
            string kill = "YES";                                                    /* Should pass at least one if statement; if not kill */
            /* if have same exact same animals contained within haplotype i & i -1 then only keep shortest one and check to see if it matches up */
            /* Ex. 1322 1373 100 2220022222000202222222002000000200000222222220220000 9.67 */
            /* Ex. 1327 1368 100 222220002022222220020000002000002222222202 9.67 */
            if(regions_sub[i].getAnimal_ID_s() == regions_sub[i-1].getAnimal_ID_s() && regions_sub[i-1].getStartIndex_s() <= regions_sub[i].getStartIndex_s() && regions_sub[i-1].getEndIndex_s() >= regions_sub[i].getEndIndex_s())
            {
                int max_start = regions_sub[i].getStartIndex_s();
                int prev_start = max_start - regions_sub[i-1].getStartIndex_s();
                int curr_start = max_start - regions_sub[i].getStartIndex_s();
                int length = regions_sub[i].getEndIndex_s() - regions_sub[i].getStartIndex_s();
                //cout << regions_sub[i-1].getStartIndex_s() << " " << regions_sub[i-1].getEndIndex_s() << " " << regions_sub[i-1].getHaplotype_s() << " ";
                //cout << regions_sub[i-1].getNumber_s() << " " << regions_sub[i-1].getPhenotype_s() << endl;
                //cout << regions_sub[i].getStartIndex_s() << " " << regions_sub[i].getEndIndex_s() << " " << regions_sub[i].getHaplotype_s() << " ";
                //cout << regions_sub[i].getNumber_s() << " " << regions_sub[i].getPhenotype_s() << endl;
                string temp_1 = regions_sub[i-1].getHaplotype_s().substr(prev_start,length);
                string temp_2 = regions_sub[i].getHaplotype_s().substr(curr_start,length);
                if(temp_1 == temp_2)
                {
                    regions_sub.erase(regions_sub.begin()+(i-1)); ROWS = ROWS -1; kill = "NO"; break;
                }
                if(temp_1 != temp_2)
                {
                    int max_start = regions_sub[i].getStartIndex_s();
                    int prev_start = max_start - regions_sub[i-1].getStartIndex_s();
                    int curr_start = max_start - regions_sub[i].getStartIndex_s();
                    int length = regions_sub[i].getEndIndex_s() - regions_sub[i].getStartIndex_s();
                    string temp_1 = regions_sub[i-1].getHaplotype_s().substr(prev_start,length);
                    string temp_2 = regions_sub[i].getHaplotype_s().substr(curr_start,length);
                    cout << temp_1 << endl;
                    cout << temp_2 << endl;
                    cout << regions_sub[i-1].getStartIndex_s() << " " << regions_sub[i-1].getEndIndex_s() << " " << regions_sub[i-1].getHaplotype_s() << " ";
                    cout << regions_sub[i-1].getNumber_s() << " " << regions_sub[i-1].getPhenotype_s() << endl;
                    cout << regions_sub[i].getStartIndex_s() << " " << regions_sub[i].getEndIndex_s() << " " << regions_sub[i].getHaplotype_s() << " ";
                    cout << regions_sub[i].getNumber_s() << " " << regions_sub[i].getPhenotype_s() << endl;
                    cout << "Killed Step 2a" << endl; exit (EXIT_FAILURE);
                }
            }
            /* Ex. 260 324 95 20002020002002222002200220202220002220002200200200220020202000222 10.3158 */
            /* Ex. 258 339 95 2220002020002002222002200220202220002220002200200200220020202000222220002200002202 10.3158 */
            if(regions_sub[i].getAnimal_ID_s() == regions_sub[i-1].getAnimal_ID_s() && regions_sub[i-1].getStartIndex_s() >= regions_sub[i].getStartIndex_s() && regions_sub[i-1].getEndIndex_s() <= regions_sub[i].getEndIndex_s())
            {
                int max_start = regions_sub[i-1].getStartIndex_s();
                int prev_start = max_start - regions_sub[i-1].getStartIndex_s();
                int curr_start = max_start - regions_sub[i].getStartIndex_s();
                int length = regions_sub[i-1].getEndIndex_s() - regions_sub[i-1].getStartIndex_s();
                //cout << regions_sub[i-1].getStartIndex_s() << " " << regions_sub[i-1].getEndIndex_s() << " " << regions_sub[i-1].getHaplotype_s() << " ";
                //cout << regions_sub[i-1].getNumber_s() << " " << regions_sub[i-1].getPhenotype_s() << endl;
                //cout << regions_sub[i].getStartIndex_s() << " " << regions_sub[i].getEndIndex_s() << " " << regions_sub[i].getHaplotype_s() << " ";
                //cout << regions_sub[i].getNumber_s() << " " << regions_sub[i].getPhenotype_s() << endl;
                string temp_1 = regions_sub[i-1].getHaplotype_s().substr(prev_start,length);
                string temp_2 = regions_sub[i].getHaplotype_s().substr(curr_start,length);
                if(temp_1 == temp_2)
                {
                    regions_sub.erase(regions_sub.begin()+i); ROWS = ROWS -1; kill = "NO"; break;
                }
                if(temp_1 != temp_2)
                {
                    int max_start = regions_sub[i-1].getStartIndex_s();
                    int prev_start = max_start - regions_sub[i-1].getStartIndex_s();
                    int curr_start = max_start - regions_sub[i].getStartIndex_s();
                    int length = regions_sub[i-1].getEndIndex_s() - regions_sub[i-1].getStartIndex_s();
                    string temp_1 = regions_sub[i-1].getHaplotype_s().substr(prev_start,length);
                    string temp_2 = regions_sub[i].getHaplotype_s().substr(curr_start,length);
                    cout << temp_1 << endl;
                    cout << temp_2 << endl;
                    cout << regions_sub[i-1].getStartIndex_s() << " " << regions_sub[i-1].getEndIndex_s() << " " << regions_sub[i-1].getHaplotype_s() << " ";
                    cout << regions_sub[i-1].getNumber_s() << " " << regions_sub[i-1].getPhenotype_s() << endl;
                    cout << regions_sub[i].getStartIndex_s() << " " << regions_sub[i].getEndIndex_s() << " " << regions_sub[i].getHaplotype_s() << " ";
                    cout << regions_sub[i].getNumber_s() << " " << regions_sub[i].getPhenotype_s() << endl;
                    cout << "KILLED at step 2b" << endl; exit (EXIT_FAILURE);
                }
            }
            /* Need to skip the not nested one (i.e. cross each other but not within each other)
             /* Same phenotype and number of haplotypes but not within each other */
            /* Ex. 1295 1351 100 9.67 */
            /* Ex. 1312 1377 100 9.67 */
            if(regions_sub[i].getAnimal_ID_s() == regions_sub[i-1].getAnimal_ID_s() && regions_sub[i].getStartIndex_s() > regions_sub[i-1].getStartIndex_s() && regions_sub[i].getEndIndex_s() > regions_sub[i-1].getEndIndex_s())
            {
                i++; break; kill = "NO";
            }
            /* Ex. 1312 1377 100 9.67 */
            /* Ex. 1295 1351 100 9.67 */
            if(regions_sub[i].getAnimal_ID_s() == regions_sub[i-1].getAnimal_ID_s() && regions_sub[i-1].getStartIndex_s() > regions_sub[i].getStartIndex_s() && regions_sub[i-1].getEndIndex_s() > regions_sub[i].getEndIndex_s())
            {
                i++; break; kill = "NO";
            }
            /* Ex. 1295 1354 100 9.67 */
            /* Ex. 1280 1340 100 9.67 */
            if(regions_sub[i].getAnimal_ID_s() == regions_sub[i-1].getAnimal_ID_s() && regions_sub[i].getStartIndex_s() < regions_sub[i-1].getStartIndex_s() && regions_sub[i].getEndIndex_s() < regions_sub[i-1].getEndIndex_s())
            {
                i++; break; kill = "NO";
            }
            /* Ex. 1280 1340 100 9.67 */
            /* Ex. 1295 1354 100 9.67 */
            if(regions_sub[i].getAnimal_ID_s() == regions_sub[i-1].getAnimal_ID_s() && regions_sub[i-1].getStartIndex_s() < regions_sub[i].getStartIndex_s() && regions_sub[i-1].getEndIndex_s() < regions_sub[i].getEndIndex_s())
            {
                i++; break; kill = "NO";
            }
            /* different animals across haplotypes */
            if(regions_sub[i].getAnimal_ID_s() != regions_sub[i-1].getAnimal_ID_s())
            {
                i++; break; kill = "NO";                         /* greater than minimum number of phenotypes so remove from class */
            }
            /* Should pass at least one previous if statement if not kill program */
            if(kill == "YES"){cout << "KILLED at step 2c" << endl; exit (EXIT_FAILURE);}
        }
    }
    //cout << regions_sub.size() << endl;
    //for(int i = 0; i < regions_sub.size(); i++)
    //for(int i = 0; i < 20; i++)
    //{
    //    cout << regions_sub[i].getStartIndex_s() << " " << regions_sub[i].getEndIndex_s() << " " << regions_sub[i].getHaplotype_s() << " ";
    //    cout << regions_sub[i].getNumber_s() << " " << regions_sub[i].getPhenotype_s() << endl;
    //}
    //cout << endl;
    /* Save ones in passed ones in regions */
    vector < int > positionbp;                                  /* Vector to store position in bp */
    vector < int > colid_full;                                  /* column in regards to full snp matrix it is in */
    vector < int > colid_sub;                                   /* column in regards to chromosome level it is in */
    for(int i = 0; i < chr.size(); i++)
    {
        if(chr[i] == chromo + 1){positionbp.push_back(position[i]); colid_full.push_back(index[i]);}
    }
    for(int i = 0; i < positionbp.size(); i++){colid_sub.push_back(i);}
    /* Loop through and grab region and full col size haplotype and raw phenotype to place in Unfavorable ROH class */
    for(int i = 0; i < regions_sub.size(); i++)
    {
        vector < int > region_position;                         /* Stores position that matches up */
        vector < int > region_full;                             /* Stores column that matches up */
        
        for(int j = 0; j < positionbp.size(); j++)
        {
            if(colid_sub[j] >= regions_sub[i].getStartIndex_s() && colid_sub[j] <= regions_sub[i].getEndIndex_s())
            {
                region_position.push_back(positionbp[j]); region_full.push_back(colid_full[j]);
            }
        }
        Unfavorable_Regions region_temp(chromo+1,region_position[0],region_position[region_position.size()-2],(region_full[0]),region_full[region_full.size()-1],regions_sub[i].getHaplotype_s(),regions_sub[i].getPhenotype_s(),0,0,0);
        regions.push_back(region_temp);
    }
}
///////////////////////////////////////
// Output Regions that Passed Step l //
///////////////////////////////////////
void OutputStage1(string Stage1loc,vector < Unfavorable_Regions > &regions)
{
    for(int i = 0; i < regions.size(); i++)
    {
        std::ofstream output5(Stage1loc, std::ios_base::app | std::ios_base::out);
        output5 << regions[i].getChr_R() << " " << regions[i].getStPos_R() << " " << regions[i].getEnPos_R() << " ";
        output5 << (regions[i].getStartIndex_R()+1) << " " << regions[i].getEndIndex_R() << " " << regions[i].getHaplotype_R() << " ";
        output5 << regions[i].getRawPheno_R() << endl;
    }
}
///////////////////////////
//// Step 2 of Algorithm //
///////////////////////////
void Step2(vector < Unfavorable_Regions > &regions,int min_Phenotypes,vector <string> const &genotype,vector < int > const &phenogenorownumber,vector <double> const &pheno,int dim_lhs,vector <int> const &X_i, vector <int> const &X_j, vector <double> const &X_A, int dimension, vector <int> &ZW_i, vector <int> &ZW_j, vector <double> &ZW_A,vector <string> uniqueID,vector <int> &LHSred_i, vector <int> &LHSred_j, vector <double> &LHSred_A,vector < string > const &factor_red, vector < int > const &zero_columns_red,double res_var,vector<vector<string>> &FIXED_CLASS, vector<vector<string >> &uniqueclass,vector<vector<double>> &FIXED_COV,vector <double> const &MeanPerCovClass,string unfav_direc,double one_sided_t,double phenotype_cutoff,ostream& logfile)
{
    vector < vector < string > > ROH_haplotypes_region;
    vector < vector < double > > raw_pheno_region;
    vector < vector < double > > beta_roh_effect_region;
    vector < vector < double > > lsm_region;
    vector < vector < double > > t_stat_region;
    /* add rows to vector of dimension region */
    for(int i = 0; i < regions.size(); i++)
    {
        vector < string > temp; vector < double > tempa;
        ROH_haplotypes_region.push_back(temp); raw_pheno_region.push_back(tempa); beta_roh_effect_region.push_back(tempa);
        lsm_region.push_back(tempa); t_stat_region.push_back(tempa);
    }
    logfile << "                 - Begin Step 2:" << endl;
    #pragma omp parallel for
    for(int loopregions = 0; loopregions < regions.size(); loopregions++)
    {
        vector < int > sub_genotype(pheno.size(),0);                                    /* haplotype for each individual */
        vector < string > ROH_haplotypes;                                               /* Tabulates all unique ROH haplotypes */
        vector < int > ROH_ID;                                                          /* Numeric ID for unique haplotype */
        vector < int > Haplo_number;                                                    /* Number of phenotypes that have haplotype */
        vector < double > effect_beta;
        for(int i = 0; i < pheno.size(); i++)
        {
            int length = regions[loopregions].getEndIndex_R() - regions[loopregions].getStartIndex_R();             /* Get length of haplotype */
            string temp = genotype[phenogenorownumber[i]].substr(regions[loopregions].getStartIndex_R(),length);    /* Grab substring */
            /* check to see if a 1 exists; if so then not a ROH */
            size_t found =  temp.find("1");
            /* if not in ROH then replace it with a zero */
            if (found != string::npos){sub_genotype[i] = 0;}
            /* if both found3 and found4 weren't located then in an ROH and replace it with haplotype number  */
            if (found == string::npos)
            {
                /* place into corrent ROH haplotypes bin */
                if(ROH_haplotypes.size() > 0)
                {
                    string stop = "GO"; int h = 0;
                    while(stop == "GO")
                    {
                        if(temp.compare(ROH_haplotypes[h]) == 0)                /* Is the same */
                        {
                            sub_genotype[i] = h + 1; Haplo_number[h] = Haplo_number[h] + 1; stop = "KILL";
                        }
                        if(temp.compare(ROH_haplotypes[h]) != 0){h++;}           /* Not the same */
                        if(h == ROH_haplotypes.size())                          /* If number not match = size of hapLibary then add */
                        {
                            ROH_haplotypes.push_back(temp); Haplo_number.push_back(1); ROH_ID.push_back((h+1));
                            sub_genotype[i] = ROH_haplotypes.size(); stop = "KILL";
                        }
                    }
                }
                if(ROH_haplotypes.size() == 0)                                  /* Haplotype library will be empty for first individual */
                {
                    ROH_haplotypes.push_back(temp); Haplo_number.push_back(1); ROH_ID.push_back((1));
                    sub_genotype[i] = ROH_haplotypes.size();
                }
            }
        }
        /* All haplotypes tabulated now if below frequency threshold zero out and place in non-ROH category */
        int ROWS = ROH_haplotypes.size(); int i = 0;
        while(i < ROWS)
        {
            while(1)
            {
                if(Haplo_number[i] < min_Phenotypes)                     /* less than minimum number of phenotypes so remove from class */
                {
                    /* loop through and replace haplotypes to 0 if below minimum */
                    for(int h = 0; h < pheno.size(); h++){if(sub_genotype[h] == ROH_ID[i]){sub_genotype[h] = 0;}}
                    ROH_haplotypes.erase(ROH_haplotypes.begin()+i); Haplo_number.erase(Haplo_number.begin()+i); ROH_ID.erase(ROH_ID.begin()+i);
                    ROWS = ROWS -1; break;                               /* Reduce size of population so i stays the same */
                }
                if(Haplo_number[i] >= min_Phenotypes){i++; break;}      /* greater than minimum number of phenotypes so remove from class */
                cout << "Step 1 Broke" << endl; exit (EXIT_FAILURE);
            }
        }
        /* Renumber so goes for 1 to haplotype size; this is important for indexing later on */
        vector < int > old_ROH_ID;
        for(int i = 0; i < ROH_haplotypes.size(); i++){old_ROH_ID.push_back(ROH_ID[i]); ROH_ID[i] = i + 1;}
        /* Renumber genotype ID */
        for(int i = 0; i < sub_genotype.size(); i++)
        {
            if(sub_genotype[i] > 0)
            {
                int j = 0;
                while(1)
                {
                    if(sub_genotype[i] == old_ROH_ID[j]){sub_genotype[i] = ROH_ID[j]; break;}
                    if(sub_genotype[i] != old_ROH_ID[j]){j++;}
                    if(j > old_ROH_ID.size()){cout << "Renumbering Failed " << endl; exit (EXIT_FAILURE);}
                }
            }
        }
        /* Tabulate Phenotypic mean for haplotype */
        vector < double > mean_ROH((ROH_haplotypes.size()+1),0);
        vector < double > number_ROH((ROH_haplotypes.size()+1),0);
        vector < int > category_ROH((ROH_haplotypes.size()+1),0);
        /* sub_genotype has already been binned into 0 (non_roh) and anything greater than 0 is an ROH */
        for(int i = 0; i < pheno.size(); i++){mean_ROH[sub_genotype[i]] += pheno[i]; number_ROH[sub_genotype[i]] += 1;}
        for(int i = 0; i < mean_ROH.size(); i++){mean_ROH[i] = mean_ROH[i] / number_ROH[i]; category_ROH[i] = i;}
        //for(int i = 0; i < mean_ROH.size(); i++)
        //{
        //    if(i == 0){cout << category_ROH[i] << "\t" << mean_ROH[i] << "\t" << number_ROH[i] << "\t" << "Non_ROH" << endl;}
        //    if(i >= 1){cout << category_ROH[i] << "\t" << mean_ROH[i] << "\t" << number_ROH[i] << "\t" << ROH_haplotypes[i-1] << endl;}
        //}
        //cout << endl;
        //////////////////////////////////////////////////////////////////
        ///                 Update Inverese LHS Matrix                 ///
        //////////////////////////////////////////////////////////////////
        int upddim_lhs = dim_lhs + mean_ROH.size() - 1;
        float* LHSinvupdated = new float[upddim_lhs*upddim_lhs];
        for(int i = 0; i < upddim_lhs*upddim_lhs; i++){LHSinvupdated[i] = 0.0;}
        float* solutions = new float[upddim_lhs];
        updateLHSinv(X_i,X_j,X_A,dimension,ZW_i,ZW_j,ZW_A,LHSred_i,LHSred_j,LHSred_A,uniqueID,sub_genotype,LHSinvupdated,dim_lhs,upddim_lhs,solutions,pheno);
        //for(int i = 0; i < 100; i++){cout << i + 1 << " " << solutions[i] << endl;}
        //for(int i = dim_lhs; i < upddim_lhs; i++){cout << i + 1 << " " << solutions[i] << endl;}
        int solutionsneg999 = 0;
        for(int i = 0; i < upddim_lhs; i++){if(solutions[i] == -999){solutionsneg999++;}}
        if(solutionsneg999 < upddim_lhs)
        {
            //for(int i = 0; i < upddim_lhs; i++){cout << i + 1 << " " << solutions[i] << endl;}
            vector < double > LSM((mean_ROH.size() - 1),0.0); vector < double > T_stat((mean_ROH.size() - 1),0.0);
            estimateROHeffect(LHSinvupdated,solutions,upddim_lhs,factor_red,zero_columns_red,LSM,T_stat,res_var,FIXED_CLASS,uniqueclass,FIXED_COV,MeanPerCovClass);
            for(int i = dim_lhs; i < upddim_lhs; i++){effect_beta.push_back(double(solutions[i]));}
            delete [] LHSinvupdated; delete [] solutions;
            //for(int i = 0; i < ROH_haplotypes.size(); i++)
            //{
            //    cout << regions[loopregions].getHaplotype_R() << " " << ROH_haplotypes[i] << " " << T_stat[i] << " " << effect_beta[i] << endl;
            //}
            //cout << endl;
            for(int i = 0; i < ROH_haplotypes.size(); i++)
            {
                if(isNaN(T_stat[i])){T_stat[i] = 0.0;}
            }
            i = 0;
            while(i < ROH_haplotypes.size())
            {
                if(ROH_haplotypes[i] == regions[loopregions].getHaplotype_R())
                {
                    regions[loopregions].Update_LSM(LSM[i]); regions[loopregions].Update_Tstat(T_stat[i]);
                    regions[loopregions].Update_Effect(effect_beta[i]);  break;
                }
                if(ROH_haplotypes[i] != regions[loopregions].getHaplotype_R()){i++;}
            }
            //cout << "Worked" << endl;
            //cout << regions[loopregions].getChr_R() << " " << regions[loopregions].getStPos_R() << " " << regions[loopregions].getEnPos_R() << " ";
            //cout<< regions[loopregions].getStartIndex_R() << " " << regions[loopregions].getEndIndex_R() << " " << regions[loopregions].getHaplotype_R() << " ";
            //cout << regions[loopregions].getRawPheno_R() << " " << regions[loopregions].getLSM_R() << " " << regions[loopregions].gettval() << endl;
            vector < int > linker_roh_category;
            for(int i = 1; i < (ROH_haplotypes.size()+1); i++){linker_roh_category.push_back(i);}
            //for(int i = 0; i < ROH_haplotypes.size(); i++)
            //{
            //  cout << linker_roh_category[i] << " " << ROH_haplotypes[i] << " " << LSM[i] << " " << T_stat[i] << endl;
            //}
            ROWS = ROH_haplotypes.size(); i = 0;
            while(i < ROWS)
            {
                if(unfav_direc == "low")
                {
                    while(1)
                    {
                        if(T_stat[i] > (-1 * one_sided_t))
                        {
                            linker_roh_category.erase(linker_roh_category.begin()+i); ROH_haplotypes.erase(ROH_haplotypes.begin()+i);
                            LSM.erase(LSM.begin()+i); effect_beta.erase(effect_beta.begin()+i); T_stat.erase(T_stat.begin()+i); ROWS = ROWS -1; break;
                        }
                        if(T_stat[i] <= (-1* one_sided_t)){i++; break;}
                    }
                }
                if(unfav_direc == "high")
                {
                    while(1)
                    {
                        if(T_stat[i] < one_sided_t)
                        {
                            linker_roh_category.erase(linker_roh_category.begin()+i); ROH_haplotypes.erase(ROH_haplotypes.begin()+i);
                            LSM.erase(LSM.begin()+i); effect_beta.erase(effect_beta.begin()+i); T_stat.erase(T_stat.begin()+i); ROWS = ROWS -1; break;
                        }
                        if(T_stat[i] >= one_sided_t){i++; break;}
                    }
                }
            }
            //for(int i = 0; i < ROH_haplotypes.size(); i++)
            // {
            //     cout << linker_roh_category[i] << " " << ROH_haplotypes[i] << " " << LSM[i] << " " << T_stat[i] << endl;
            //}
            if(ROH_haplotypes.size() > 0)
            {
                for(int i = 0; i < ROH_haplotypes.size(); i++)
                {
                    if(unfav_direc == "low")
                    {
                        if(ROH_haplotypes[i] != regions[loopregions].getHaplotype_R() && mean_ROH[linker_roh_category[i]] < phenotype_cutoff )
                        {
                            ROH_haplotypes_region[loopregions].push_back(ROH_haplotypes[i]);
                            raw_pheno_region[loopregions].push_back(mean_ROH[linker_roh_category[i]]);
                            lsm_region[loopregions].push_back(LSM[i]); beta_roh_effect_region[loopregions].push_back(effect_beta[i]);
                            t_stat_region[loopregions].push_back(T_stat[i]);
                        }
                    }
                    if(unfav_direc == "high")
                    {
                        if(ROH_haplotypes[i] != regions[loopregions].getHaplotype_R() && mean_ROH[linker_roh_category[i]] > phenotype_cutoff )
                        {
                            ROH_haplotypes_region[loopregions].push_back(ROH_haplotypes[i]);
                            raw_pheno_region[loopregions].push_back(mean_ROH[linker_roh_category[i]]);
                            lsm_region[loopregions].push_back(LSM[i]); beta_roh_effect_region[loopregions].push_back(effect_beta[i]);
                            t_stat_region[loopregions].push_back(T_stat[i]);
                        }
                    }
                }
            }
        }
        if(solutionsneg999 == upddim_lhs)       /* If get nan value then will just give it a tstat of 0.0 and skip over */
        {
            for(int i = 0; i < ROH_haplotypes.size(); i++)
            {
                regions[loopregions].Update_LSM(0.0); regions[loopregions].Update_Tstat(0.0); regions[loopregions].Update_Effect(0.0);
            }
        }
        if(loopregions % 10 == 0){logfile << "                    - " << loopregions << endl;}
    }
    /* add to regions and then delete 2-d vectors */
    for(int i = 0; i < raw_pheno_region.size(); i++)
    {
        if(raw_pheno_region[i].size() > 0)
        {
            for(int j = 0; j < raw_pheno_region[i].size(); j++)
            {
                int chr = regions[i].getChr_R(); int strpos = regions[i].getStPos_R(); int endpos = regions[i].getEnPos_R();
                int strind = regions[i].getStartIndex_R(); int endind = regions[i].getEndIndex_R();
                Unfavorable_Regions region_temp(chr,strpos,endpos,strind,endind, ROH_haplotypes_region[i][j],raw_pheno_region[i][j],beta_roh_effect_region[i][j],lsm_region[i][j],t_stat_region[i][j]);
                regions.push_back(region_temp);
            }
        }
    }
    /* delete 2-D vectors */
    for(int i = 0; i < raw_pheno_region.size(); i++)
    {
        ROH_haplotypes_region[i].clear(); raw_pheno_region[i].clear(); lsm_region[i].clear(); t_stat_region[i].clear();
    }
    ROH_haplotypes_region.clear(); raw_pheno_region.clear(); lsm_region.clear(); t_stat_region.clear();
    //for(int i = 0; i < regions.size(); i++)
    //{
    //    cout << regions[i].getChr_R() << " " << regions[i].getStPos_R() << " " << regions[i].getEnPos_R() << " ";
    //    cout << regions[i].getStartIndex_R() << " " << regions[i].getEndIndex_R() << " " << regions[i].getHaplotype_R() << " ";
    //    cout << regions[i].getRawPheno_R()<< " " <<regions[i].getEffect() << " " << regions[i].getLSM_R() << " " << regions[i].gettval() << endl;
    //}
    //cout << endl << endl;
    int ROWS = regions.size(); int i = 0;
    if(unfav_direc == "low")
    {
        while(i < ROWS)
        {
            while(1)
            {
                if(regions[i].gettval() > (-1 * one_sided_t))
                {
                    regions.erase(regions.begin()+i); ROWS = ROWS -1; break;        /* Reduce size of population so i stays the same */
                }
                if(regions[i].gettval() <= (-1* one_sided_t)){i++; break;}
            }
        }
    }
    if(unfav_direc == "high")
    {
        while(i < ROWS)
        {
            while(1)
            {
                if(regions[i].gettval() < one_sided_t)
                {
                    regions.erase(regions.begin()+i); ROWS = ROWS -1; break;        /* Reduce size of population so i stays the same */
                }
                if(regions[i].gettval() >= one_sided_t){i++; break;}
            }
        }
    }
}
///////////////////////////
//// Step 3 of Algorithm //
///////////////////////////
void Step3(vector < Unfavorable_Regions > &regions,vector <double> const &pheno,vector <string> const &genotype,vector < int > const &phenogenorownumber,vector < string > const &id)
{
    //for(int i = 0; i < regions.size(); i++)
    //{
    //    cout << regions[i].getChr_R() << " " << regions[i].getStPos_R() << " " << regions[i].getEnPos_R() << " ";
    //    cout << regions[i].getStartIndex_R() << " " << regions[i].getEndIndex_R() << " " << regions[i].getHaplotype_R() << " ";
    //    cout << regions[i].getRawPheno_R()<< " " <<regions[i].getEffect()<< " " <<regions[i].getLSM_R()<< " " <<regions[i].gettval() << endl;
    //}
    //cout << endl << endl;
    sort(regions.begin(), regions.end(), sortByStart);
    //for(int i = 0; i < regions.size(); i++)
    //{
    //    cout << regions[i].getChr_R() << " " << regions[i].getStPos_R() << " " << regions[i].getEnPos_R() << " ";
    //    cout << regions[i].getStartIndex_R() << " " << regions[i].getEndIndex_R() << " " << regions[i].getHaplotype_R() << " ";
    //   cout << regions[i].getRawPheno_R() << " " << regions[i].getEffect() << " " << regions[i].getLSM_R() << " " << regions[i].gettval() << endl;
    //}
    //cout << endl << endl;
    int location = 0;
    while(location < regions.size())
    {
        int location2 = location + 1;
        while(location2 < regions.size())
        {
            //cout << regions[location].getStartIndex_R() << " " << regions[location].getEndIndex_R() << " ";
            //cout << regions[location].getHaplotype_R() << endl;
            //cout << regions[location2].getStartIndex_R() << " " << regions[location2].getEndIndex_R() << " ";
            //cout << regions[location2].getHaplotype_R() << endl;
            /* find all individuals that fall into roh haplotype location */
            vector < string > individual1;
            vector < string > individual2;
            for(int i = 0; i < pheno.size(); i++)
            {
                int length_i = regions[location].getEndIndex_R() - regions[location].getStartIndex_R();                   /* Get length of haplotype i */
                int length_i_1 = regions[location2].getEndIndex_R() - regions[location2].getStartIndex_R();               /* Get length of haplotype i-1 */
                // Location i
                string temp = genotype[phenogenorownumber[i]].substr(regions[location].getStartIndex_R(),length_i);                           /* Grab substring */
                if(temp == regions[location].getHaplotype_R()){individual1.push_back(id[i]);}
                // Location i -1
                temp = genotype[phenogenorownumber[i]].substr(regions[location2].getStartIndex_R(),length_i_1);                               /* Grab substring */
                if(temp == regions[location2].getHaplotype_R()){individual2.push_back(id[i]);}
            }
            //for(int i = 0; i < individual1.size(); i++){cout << individual1[i] << " ";}
            //cout << endl << endl;
            //for(int i = 0; i < individual2.size(); i++){cout << individual2[i] << " ";}
            //cout << endl << endl;
            int totalmatched = 0;
            sort(individual1.begin(),individual1.end());
            individual1.erase(unique(individual1.begin(),individual1.end()),individual1.end());
            sort(individual2.begin(),individual2.end());
            individual2.erase(unique(individual2.begin(),individual2.end()),individual2.end());
            //for(int i = 0; i < individual1.size(); i++){cout << individual1[i] << " ";}
            //cout << endl << endl;
            //for(int i = 0; i < individual2.size(); i++){cout << individual2[i] << " ";}
            //cout << endl << endl;
            //cout << individual1.size() << " " << individual2.size() << endl;
            /* count number matched and reference is individual 2 */
            if(individual1.size() > individual2.size())
            {
                for(int i = 0; i < individual2.size(); i++)
                {
                    for(int j = 0; j < individual1.size(); j++)
                    {
                        if(individual2[i] == individual1[j]){totalmatched += 1;}
                    }
                }
                if(totalmatched == individual2.size()){regions.erase(regions.begin()+(location2));}
                if(totalmatched != individual2.size()){location2 += 1;}
                //cout << regions.size() << endl << endl;
            }
            /* count number matched and reference is individual 2 */
            if(individual1.size() <= individual2.size())
            {
                for(int i = 0; i < individual1.size(); i++)
                {
                    for(int j = 0; j < individual2.size(); j++)
                    {
                        if(individual1[i] == individual2[j]){totalmatched += 1;}
                    }
                }
                if(totalmatched == individual2.size()){regions.erase(regions.begin()+(location));}
                if(totalmatched != individual2.size()){location2 += 1;}
                //cout << regions.size() << endl << endl;
            }
        }
        location++;
    }
    //for(int i = 0; i < regions.size(); i++)
    //{
    //    cout << regions[i].getChr_R() << " " << regions[i].getStPos_R() << " " << regions[i].getEnPos_R() << " ";
    //    cout << regions[i].getStartIndex_R() << " " << regions[i].getEndIndex_R() << " " << regions[i].getHaplotype_R() << " ";
    //    cout << regions[i].getRawPheno_R() << " " << regions[i].getEffect() << " " << regions[i].getLSM_R() << " " << regions[i].gettval() << endl;
    //}
    //cout << endl << endl;
}
///////////////////////////////////////
// Output Regions that Passed Step l //
///////////////////////////////////////
void OutputStage2(string Stage2loc,vector < Unfavorable_Regions > &regions,int chromo)
{
    if(chromo == 0)
    {
        std::ofstream output6(Stage2loc, std::ios_base::app | std::ios_base::out);
        output6 << "Chromosome StartPos EndPos StartIndex EndIndex Genotype PhenoMean BetaEffect LSM T-Stat" << endl;
    }
    for(int i = 0; i < regions.size(); i++)
    {
        std::ofstream output6(Stage2loc, std::ios_base::app | std::ios_base::out);
        output6 << regions[i].getChr_R() << " " << regions[i].getStPos_R() << " " << regions[i].getEnPos_R() << " ";
        output6 << (regions[i].getStartIndex_R()+1) << " " << regions[i].getEndIndex_R() << " " << regions[i].getHaplotype_R() << " ";
        output6 << regions[i].getRawPheno_R() << " " << regions[i].getEffect() << " " << regions[i].getLSM_R() << " " << regions[i].gettval() << endl;
    }
}


