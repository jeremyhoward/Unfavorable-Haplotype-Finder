#include <iostream>
#include <algorithm>
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <omp.h>
#include <fstream>
#include <ctime>
#include <string>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <sstream>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Cholesky>
#include <Eigen/QR>
#include <mkl.h>
#include <random>

#include "HaplofinderClasses.h"

using namespace std;

//////////////////////////////////
/// Haplotype Finder Functions ///
//////////////////////////////////
void ReadMapFile_Index(string mapfile, vector < int > &chr, vector < int > &position, vector < int > &index, vector < CHR_Index> &chr_index,ostream& logfile);
void ReadPhenoFile_Index(string phenofile, vector<string> &id, vector<double> &pheno, vector<int> &phenogenorownumber, vector<vector<string>> &FIXED_CLASS, vector<vector<string >> &uniqueclass,vector<vector<double>> &FIXED_COV, vector<int> const &fixed_class_col, vector<int> const &fixed_cov_col, int id_column, int phenocolumn,ostream& logfile);
void ReadGenoFile_Index(string genofile, vector<string> &genotype, vector<string> &genotypeID, vector<int> &phenogenorownumber, vector<string> const &id, ostream& logfile);
void uniquephenotypeanimals(vector < string > &uniqueID, vector < string > const &id);
void GenerateAinv(string pedigreefile,vector < string > const &uniqueID,vector < string > const &id, double* Relationshipinv_mkl);
void GenerateFullRankXtX(double* X_Dense, vector <int> &keepremove, vector <double> &MeanPerCovClass,  vector<double> const &pheno, vector<vector<string>> const &FIXED_CLASS, vector<vector<string >> const &uniqueclass,vector<vector<double>> const &FIXED_COV,ostream& logfile);
void GenerateLHSRed(vector <int> const &X_i, vector <int> const &X_j, vector <double> const &X_A, int dimension, double* Relationshipinv_mkl, vector <int> &ZW_i, vector <int> &ZW_j, vector <double> &ZW_A, vector <string> id, vector <string> uniqueID,vector <int> &LHSred_i, vector <int> &LHSred_j, vector <double> &LHSred_A, int dim_lhs, vector <double> const &lambda);
void updateLHSinv(vector <int> const &X_i, vector <int> const &X_j, vector <double> const &X_A, int dimension, vector <int> &ZW_i, vector <int> &ZW_j, vector <double> &ZW_A,vector <int> &LHSred_i, vector <int> &LHSred_j, vector <double> &LHSred_A,vector <string> uniqueID, vector <int> const &sub_genotype,float * LHSinvupdated,int dim_lhs, int upddim_lhs, float * solutions, vector < double > const &pheno);
void estimateROHeffect(float * LHSinvupdated,float * solutions,int upddim_lhs,vector < string > const &factor_red, vector < int > const &zero_columns_red, vector <double> &LSM, vector <double> &T_stat, double resvar,vector<vector<string>> &FIXED_CLASS, vector<vector<string >> &uniqueclass,vector<vector<double>> &FIXED_COV,vector <double> const &MeanPerCovClass);
void doublecheckasreml(vector <CHR_Index> chr_index,int min_Phenotypes,vector <int> const &width,vector <string> const &genotype,vector < int > const &phenogenorownumber,vector <double> const &pheno,int dim_lhs,vector <int> const &X_i, vector <int> const &X_j, vector <double> const &X_A, int dimension, vector <int> &ZW_i, vector <int> &ZW_j, vector <double> &ZW_A,vector <string> const &id,vector <string> uniqueID,vector <int> &LHSred_i, vector <int> &LHSred_j, vector <double> &LHSred_A,vector < string > const &factor_red, vector < int > const &zero_columns_red,double res_var,vector<vector<string>> &FIXED_CLASS, vector<vector<string >> &uniqueclass,vector<vector<double>> &FIXED_COV,vector <double> const &MeanPerCovClass,string unfav_direc);
double phenocutoff(vector <CHR_Index> chr_index,int null_samples,int min_Phenotypes,vector <int> const &width,vector <string> const &genotype,vector < int > const &phenogenorownumber,vector <double> const &pheno,int dim_lhs,vector <int> const &X_i, vector <int> const &X_j, vector <double> const &X_A, int dimension, vector <int> &ZW_i, vector <int> &ZW_j, vector <double> &ZW_A,vector <string> uniqueID,vector <int> &LHSred_i, vector <int> &LHSred_j, vector <double> &LHSred_A,vector < string > const &factor_red, vector < int > const &zero_columns_red,double res_var,vector<vector<string>> &FIXED_CLASS, vector<vector<string >> &uniqueclass,vector<vector<double>> &FIXED_COV,vector <double> const &MeanPerCovClass,string unfav_direc,ostream& logfile);
void Step1(vector < Unfavorable_Regions_sub > regions_sub, double phenotype_cutoff, string unfav_direc, int chromo, vector <CHR_Index> chr_index,int min_Phenotypes,vector <int> const &width,vector <string> const &genotype,vector < int > const &phenogenorownumber,vector <double> const &pheno, vector <string> const &id,vector<int> const &chr, vector<int> const &position, vector<int> const &index,vector < Unfavorable_Regions > &regions);
void Step2(vector < Unfavorable_Regions > &regions,int min_Phenotypes,vector <string> const &genotype,vector < int > const &phenogenorownumber,vector <double> const &pheno,int dim_lhs,vector <int> const &X_i, vector <int> const &X_j, vector <double> const &X_A, int dimension, vector <int> &ZW_i, vector <int> &ZW_j, vector <double> &ZW_A,vector <string> uniqueID,vector <int> &LHSred_i, vector <int> &LHSred_j, vector <double> &LHSred_A,vector < string > const &factor_red, vector < int > const &zero_columns_red,double res_var,vector<vector<string>> &FIXED_CLASS, vector<vector<string >> &uniqueclass,vector<vector<double>> &FIXED_COV,vector <double> const &MeanPerCovClass,string unfav_direc,double one_sided_t,double phenotype_cutoff,ostream& logfile);
void Step3(vector < Unfavorable_Regions > &regions,vector <double> const &pheno,vector <string> const &genotype,vector < int > const &phenogenorownumber,vector < string > const &id);
void OutputStage1(string Stage1loc,vector < Unfavorable_Regions > &regions);
void OutputStage2(string Stage2loc,vector < Unfavorable_Regions > &regions,int chromo);




using namespace std;

int main(int argc, char* argv[])
{
    using Eigen::MatrixXd; using Eigen::SparseMatrix; using Eigen::VectorXd;
    time_t full_begin_time = time(0);
    cout<<"\n#############################################################\n";
    cout<<"###################################################   #######\n";
    cout<<"###############################################   /~|   #####\n";
    cout<<"############################################   _- `~~~', ####\n";
    cout<<"##########################################  _-~       )  ####\n";
    cout<<"#######################################  _-~          |  ####\n";
    cout<<"####################################  _-~            ;  #####\n";
    cout<<"##########################  __---___-~              |   #####\n";
    cout<<"#######################   _~   ,,                  ;  `,,  ##\n";
    cout<<"#####################  _-~    ;'                  |  ,'  ; ##\n";
    cout<<"###################  _~      '                    `~'   ; ###\n";
    cout<<"############   __---;                                 ,' ####\n";
    cout<<"########   __~~  ___                                ,' ######\n";
    cout<<"#####  _-~~   -~~ _          N                    ,' ########\n";
    cout<<"##### `-_         _           C                  ; ##########\n";
    cout<<"#######  ~~----~~~   ;         S                ; ###########\n";
    cout<<"#########  /          ;         U               ; ###########\n";
    cout<<"#######  /             ;                      ; #############\n";
    cout<<"#####  /                `                    ; ##############\n";
    cout<<"###  /                                      ; ###############\n";
    cout<<"#                                            ################\n";
    cout<<"-------------------------------------------------------------\n";
    cout<<"- Unfavorable ROH Haplo_Finder                              -\n";
    cout<<"- Author: J. Howard                                         -\n";
    cout<<"- Institution: NCSU                                         -\n";
    cout<<"- Date: 8/12/2015                                           -\n";
    cout<<"-------------------------------------------------------------\n";
    /* Figure out where you are currently at then just append to string */
    char * cwd;
    cwd = (char*) malloc( FILENAME_MAX * sizeof(char) );
    getcwd(cwd,FILENAME_MAX);
    string path(cwd);
    if(argc != 2){cout << "Program ended due to a parameter file not given!" << endl; exit (EXIT_FAILURE);}
    string paramterfile = argv[1];
    /* create path to ouput results */
    string logfileloc = path + "/LogFile";
    fstream checklog; checklog.open(logfileloc, std::fstream::out | std::fstream::trunc); checklog.close();         /* Remove previous file */
    std::ofstream logfile(logfileloc, std::ios_base::out | std::ios_base::out);               /* open log file to output verbage throughout code */
    /********************************************************************************************************************************************/
    /********************************************************************************************************************************************/
    /********************************************************************************************************************************************/
    /********************************************************************************************************************************************/
    /***                                                    Parameters to Read In                                                             ***/
    /********************************************************************************************************************************************/
    /********************************************************************************************************************************************/
    /********************************************************************************************************************************************/
    /********************************************************************************************************************************************/
    /* Paremeteters from the file */
    string phenofile, genofile, mapfile, pedigreefile, perm_effect, determine_cutoff, unfav_direc,subtract_mean, doublecheck;
    int id_column, phenocolumn, null_samples, threads;
    double res_var, add_var, perm_var, phenotype_cutoff, minimum_freq, one_sided_t;
    vector < int > fixed_class_col; vector < int > fixed_cov_col; vector < int > width;
    logfile << "==================================\n";
    logfile << "== Read in Parameters from file ==\n";
    logfile << "==================================\n";
    logfile << "Name of parameter file was: '" << paramterfile << "'"<< endl;
    logfile << "Parameters Specified in Paramter File: " << endl;
    /* read parameters file and generate correct variables */
    vector <string> parm;
    string parline;
    ifstream parfile;
    parfile.open(paramterfile);
    if(parfile.fail()){cout << "Parameter file not found. Check log file." << endl; exit (EXIT_FAILURE);}
    while (getline(parfile,parline)){parm.push_back(parline);} /* Stores in vector and each new line push back to next space */
    int search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("THREADS:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(": ", 0); parm[search].erase(0, pos + 1);
            parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end());
            threads = atoi(parm[search].c_str());
            logfile << "    - Number of threads used:\t\t\t\t\t\t\t\t\t" << "'" << threads << "'." << endl; break;
        }
        search++; if(search >= parm.size()){threads = 1; logfile << "    - Number of threads used:\t\t\t\t\t\t\t\t\t" << "'1' (Default)." << endl; break;}
    }
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("MAP_FILE:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(": ", 0); parm[search].erase(0, pos + 1);
            parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end()); mapfile = parm[search]; break;
        }
        search++; if(search >= parm.size()){cout << "Couldn't find 'MAP_FILE:' variable in parameter file!" << endl; exit (EXIT_FAILURE);}
    }
    logfile << "    - Map File (Chromosome & Position):\t\t\t\t\t\t\t\t" << "'" << mapfile << "'." << endl;
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("PHENO_FILE:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(": ", 0); parm[search].erase(0, pos + 1);
            parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end()); phenofile = parm[search]; break;
        }
        search++; if(search >= parm.size()){cout << "Couldn't find 'PHENO_FILE:' variable in parameter file!" << endl; exit (EXIT_FAILURE);}
    }
    logfile << "    - Phenotype File (Model Effect & Phenotype):\t\t\t\t\t\t" << "'" << phenofile << "'." << endl;
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("GENO_FILE:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(": ", 0); parm[search].erase(0, pos + 1);
            parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end()); genofile = parm[search]; break;
        }
        search++;
        if(search >= parm.size()){cout << "Couldn't find 'GENO_FILE:' variable in parameter file!" << endl; exit (EXIT_FAILURE);}
    }
    logfile << "    - Genotype File (Genotype):\t\t\t\t\t\t\t\t\t" << "'" << genofile << "'." << endl;
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("PEDIGREE:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(": ", 0); parm[search].erase(0, pos + 1);
            parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end()); pedigreefile = parm[search]; break;
        }
        search++;
        if(search >= parm.size()){cout << "Couldn't find 'PEDIGREE:' variable in parameter file!" << endl; exit (EXIT_FAILURE);}
    }
    logfile << "    - Pedigree File (Animal Sire Dam):\t\t\t\t\t\t\t\t" << "'" << pedigreefile << "'." << endl;
    logfile << "    - Columns Used in Model:" << endl;
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("ID:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(": ", 0); parm[search].erase(0, pos + 1);
            parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end());
            id_column = atoi(parm[search].c_str()); break;
        }
        search++;
        if(search >= parm.size()){cout << "Couldn't find 'ID:' variable in parameter file!" << endl; exit (EXIT_FAILURE);}
    }
    logfile << "        - Column that refers to Individual's ID:\t\t\t\t\t\t" << "'" << id_column << "'." << endl;
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("CLASS:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(": ", 0); parm[search].erase(0, pos + 1);
            pos = parm[search].find(",",0);
            /* if only have one then pos will be same as std::string::npos and then just break */
            if(pos == std::string::npos)
            {
                parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end());
                fixed_class_col.push_back(stoi(parm[search])); break;
            }
            string quit = "NO";             /* Don't know how many their are, but seperated by "," */
            while(quit != "YES")
            {
                size_t pos = parm[search].find(",",0);
                if(pos > 0)                                                     /* hasn't reached last one yet */
                {
                    fixed_class_col.push_back(stoi(parm[search].substr(0,pos))); parm[search].erase(0, pos + 1);
                }
                if(pos == std::string::npos){quit = "YES";}                     /* has reached last one so now kill while loop */
            }
            break;
        }
        search++;
        if(search >= parm.size()){logfile << "        - No Cross-Classified Fixed Effects." << endl; break;}
    }
    for(int i = 0; i < fixed_class_col.size(); i++)
    {
        logfile << "        - Column that refers to Individuals Cross-Classified Fixed Effect " << i + 1;
        logfile << ":\t\t\t" << "'" << fixed_class_col[i] << "'." << endl;
    }
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("COV:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(": ", 0); parm[search].erase(0, pos + 1);
            pos = parm[search].find(",",0);
            /* if only have one then pos will be same as std::string::npos and then just break */
            if(pos == std::string::npos)
            {
                parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end());
                fixed_cov_col.push_back(stoi(parm[search])); break;
            }
            string quit = "NO";             /* Don't know how many their are, but seperated by "," */
            while(quit != "YES")
            {
                size_t pos = parm[search].find(",",0);
                if(pos > 0)                                                     /* hasn't reached last one yet */
                {
                    fixed_cov_col.push_back(stoi(parm[search].substr(0,pos))); parm[search].erase(0, pos + 1);
                }
                if(pos == std::string::npos){quit = "YES";}                     /* has reached last one so now kill while loop */
            }
            break;
        }
        search++;
        if(search >= parm.size()){logfile << "        - No Covariate Fixed Effects." << endl; break;}
    }
    for(int i = 0; i < fixed_cov_col.size(); i++)
    {
        logfile << "        - Column that refers to Individuals Covariate Fixed Effect " << i + 1;
        logfile << ":\t\t\t\t" << "'" << fixed_cov_col[i] << "'." << endl;
    }
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("PHENOTYPE:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(": ", 0); parm[search].erase(0, pos + 1);
            parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end());
            phenocolumn = atoi(parm[search].c_str()); break;
        }
        search++;
        if(search >= parm.size()){cout << "Couldn't find 'PHENOTYPE:' variable in parameter file!" << endl; exit (EXIT_FAILURE);}
    }
    logfile << "        - Column that refers to Individual's Phenotype:\t\t\t\t\t\t" << "'" << phenocolumn << "'." << endl;
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("VAR_RESIDUAL:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(": ", 0); parm[search].erase(0, pos + 1);
            parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end());
            res_var = atof(parm[search].c_str()); break;
        }
        search++;
        if(search >= parm.size()){cout << "Couldn't find 'VAR_RESIDUAL:' variable in parameter file!" << endl; exit (EXIT_FAILURE);}
    }
    logfile << "    - Variance components for a model excluding ROH effects:" << endl;
    logfile << "        - Residual Variance Set at:\t\t\t\t\t\t\t\t" << "'" << res_var << "'." << endl;
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("VAR_ANIMAL:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(": ", 0); parm[search].erase(0, pos + 1);
            parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end());
            add_var = atof(parm[search].c_str()); break;
        }
        search++;
        if(search >= parm.size()){cout << "Couldn't find 'VAR_ANIMAL:' variable in parameter file!" << endl; exit (EXIT_FAILURE);}
    }
    logfile << "        - Animal Variance Set at:\t\t\t\t\t\t\t\t" << "'" << add_var << "'." << endl;
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("VAR_PERMANENT:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(": ", 0); parm[search].erase(0, pos + 1);
            parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end());
            perm_var = atof(parm[search].c_str());
            logfile << "        - Permanent Animal Variance Set at:\t\t\t\t\t\t\t" << "'" << perm_var << "'." << endl; break;
        }
        search++;
        if(search >= parm.size())
        {
            perm_var = 0.0;
            logfile << "        - Permanent Animal Variance Set at:\t\t\t\t\t\t\t" << "'" << perm_var << "'(Default)." << endl; break;
        }
        if(search >= parm.size()){cout << "Couldn't find 'VAR_ANIMAL:' variable in parameter file!" << endl; exit (EXIT_FAILURE);}
    }
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("WINDOW:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(":", 0); parm[search].erase(0, pos + 2);
            int timescycled = 0;
            for(int i = 0; i < 30; i++)
            {
                size_t posinner = parm[search].find(",",0);
                if(posinner > 0)                                                                         /* hasn't reached last one yet */
                {
                    string temp = parm[search].substr(0,posinner);
                    width.push_back(atoi(temp.c_str())); parm[search].erase(0, posinner + 1); timescycled++;
                }
                if(posinner == std::string::npos){i = 30;}
            }
            logfile << "    - Window sizes:\t\t\t\t\t\t\t\t\t\t" << "'";
            for(int i = 0; i < width.size(); i++)
            {
                if(i == 0){logfile << width[i];}
                if(i > 0){logfile << "," << width[i];}
            }
            logfile << "'" << endl; break;
        }
        search++;
        if(search >= parm.size())
        {
            int starting = 50;
            for(int i = 0; i < 8; i++)
            {
                width.push_back(starting); starting = starting -5;
            }
            logfile << "    - Window sizes:\t\t\t\t\t\t\t\t\t\t" << "'";
            for(int i = 0; i < width.size(); i++)
            {
                if(i == 0){logfile << width[i];}
                if(i > 0){logfile << "," << width[i];}
            }
            logfile << "'(Default)" << endl; break;
        }
    }
    logfile << "    - How to determine phenotype cutoff for windows to investigate:" << endl;
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("CUTOFF:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(": ", 0); parm[search].erase(0, pos + 2);
            pos = parm[search].find(" ",0); determine_cutoff = parm[search].substr(0,pos); parm[search].erase(0, pos + 1);
            if(determine_cutoff == "data")
            {
                null_samples = atoi(parm[search].c_str());
                logfile << "        - Method to determine cutoff:\t\t\t\t\t\t\t\t" << "'" << determine_cutoff << "'." << endl;
                logfile << "        - Number of samples to grab:\t\t\t\t\t\t\t\t" << "'" << null_samples << "'." << endl;
                break;
            }
            if(determine_cutoff == "value")
            {
                phenotype_cutoff = atof(parm[search].c_str());
                logfile << "        - Method to determine cutoff:\t\t\t\t\t\t\t\t" << "'" << determine_cutoff << "'." << endl;
                logfile << "        - Phenotype Cutoff:\t\t\t\t\t\t\t\t\t" << "'" << phenotype_cutoff  << "'." << endl;
                break;
            }
        }
        search++;
        if(search >= parm.size()){cout << "Couldn't find 'CUTOFF:' variable in parameter file!" << endl; exit (EXIT_FAILURE);}
    }
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("UNFAV_DIRECTION:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(": ", 0); parm[search].erase(0, pos + 1);
            parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end());
            unfav_direc = parm[search];
            logfile << "    - Direction of unfavorable phenotype:\t\t\t\t\t\t\t" << "'" << unfav_direc << "'." << endl;
            break;
        }
        search++;
        if(search >= parm.size()){cout << "Couldn't find 'UNFAV_DIRECTION:' variable in parameter file!" << endl; exit (EXIT_FAILURE);}
    }
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("SUBTRACT_MEAN:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(": ", 0); parm[search].erase(0, pos + 1);
            parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end());
            subtract_mean = parm[search];
            logfile << "    - Subtract of mean phenotype:\t\t\t\t\t\t\t\t" << "'" << subtract_mean << "'." << endl;
            break;
        }
        search++;
        if(search >= parm.size())
        {
            subtract_mean = "no";
            logfile << "    - Subtract of mean phenotype:\t\t\t\t\t\t\t\t" << "'" << subtract_mean << "'(Default)." << endl; break;
        }
    }
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("MINIMUM_FREQ:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(": ", 0); parm[search].erase(0, pos + 1);
            parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end());
            minimum_freq = atof(parm[search].c_str());
            logfile << "    - Minimum ROH Haplotype Frequency:\t\t\t\t\t\t\t\t" << "'" << minimum_freq << "'." << endl;
            break;
        }
        search++;
        if(search >= parm.size())
        {
            minimum_freq = 0.0075;
            logfile << "    - Minimum ROH Haplotype Frequency:\t\t\t\t\t\t\t\t" << "'" << minimum_freq << "' (Default)." << endl; break;
        }
    }
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("ONE_SIDED_T_CUTOFF:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(": ", 0); parm[search].erase(0, pos + 1);
            parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end());
            one_sided_t = atof(parm[search].c_str());
            logfile << "    - One sided T-value cutoff to declare significance:\t\t\t\t\t\t" << "'" << one_sided_t  << "'." << endl;
            break;
        }
        search++;
        if(search >= parm.size())
        {
            one_sided_t = 2.326;
            logfile << "    - One sided T-value cutoff to declare significance:\t\t\t\t\t\t" << "'" << one_sided_t;
            logfile << "' (Default)." << endl; break;
        }
    }
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("ASREML_CHECK:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(": ", 0); parm[search].erase(0, pos + 1);
            parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end());
            doublecheck = parm[search].c_str();
            logfile << "    - Double Check the Program with external solver:\t\t\t\t\t\t" << "'" << doublecheck  << "'." << endl;
            break;
        }
        search++;
        if(search >= parm.size())
        {
            doublecheck = "no";
            logfile << "    - Double Check the Program with external solver:\t\t\t\t\t\t" << "'" << doublecheck  << ".";
            logfile << "' (Default)." << endl; break;
        }
    }
    if(perm_var == 0.0){perm_effect = "no";}
    if(perm_var > 0.0){perm_effect = "yes";}
    logfile << "Finished Reading in Parameter File" << endl;
    /********************************************************************************************************************************************/
    /********************************************************************************************************************************************/
    /********************************************************************************************************************************************/
    /********************************************************************************************************************************************/
    /***                                             Important vectors that are utilized                                                      ***/
    /********************************************************************************************************************************************/
    /********************************************************************************************************************************************/
    /********************************************************************************************************************************************/
    /********************************************************************************************************************************************/
    /* Vectors Used Throughout Program */
    vector < int > chr;                                 /* stores chromosome in vector */
    vector < int > position;                            /* stores position */
    vector < int > index;                               /* used to grab chromosome */
    vector < CHR_Index > chr_index;                     /* Class to store chromosomal information */
    vector < string > id;                               /* ID */
    vector < double > pheno;                            /* Phenotype */
    vector < int > phenogenorownumber;                  /* match to genotype row */
    /* depending on how many fixed effect parameters their are and at what point need to make different sized matrices */
    vector < vector < string > > FIXED_CLASS;           /* Stores Classification Fixed Effects */
    vector < vector < string > > uniqueclass;           /* Number of levels within each classification variable */
    vector < vector < double > > FIXED_COV;             /* Stores Covariate Fixed Effects */
    vector < string > genotype;                         /* Genotype String */
    vector < string > genotypeID;                       /* ID pertaining to Genotype String */
    vector < string > uniqueID;                         /* Sets the rows and cols up for ZtZ */
    vector <int> X_i; vector <int> X_j; vector <double> X_A;                    /* Used to store X in sparse ija format */
    vector <int> ZW_i; vector <int> ZW_j; vector <double> ZW_A;                 /* Used to store ZW in sparse ija format */
    vector <int> LHSred_i; vector <int> LHSred_j; vector <double> LHSred_A;     /* Used to store LHS in sparse symmetric ija format */
    /********************************************************************************************************************************************/
    /***                                             Read in files to fill vectors                                                            ***/
    /********************************************************************************************************************************************/
    /* set number of threads used */
    omp_set_num_threads(threads); mkl_set_num_threads_local(threads);
    /* Read in correct files */
    ReadMapFile_Index(mapfile,chr,position,index,chr_index,logfile);
    ReadPhenoFile_Index(phenofile,id,pheno,phenogenorownumber,FIXED_CLASS,uniqueclass,FIXED_COV,fixed_class_col,fixed_cov_col,id_column,phenocolumn,logfile);
    int totalclasslevels = 0;
    for(int i = 0; i < uniqueclass.size(); i++){totalclasslevels += (uniqueclass[i].size());}
    if(subtract_mean == "yes")                          /* if subtract mean do it now */
    {
        double mean_phenotype = 0.0;
        for(int i = 0; i < pheno.size(); i++){mean_phenotype += pheno[i]; phenogenorownumber.push_back(-5);}
        mean_phenotype = mean_phenotype / double(pheno.size());               /* estimate mean */
        for(int i = 0; i < pheno.size(); i++){pheno[i] = pheno[i] - mean_phenotype;}
    }
    ReadGenoFile_Index(genofile,genotype,genotypeID,phenogenorownumber,id,logfile);
    cout << " - Data Read In." << endl;
    /********************************************************************************************************************************************/
    /***                                           Set up Static Matrices for Algorithm                                                       ***/
    /********************************************************************************************************************************************/
    logfile << "\n==================================================================================\n";
    logfile << "==\tSet Up Static Matrices and Generate Inverse of LHS for Reduced Model \t==\n";
    logfile << "==================================================================================\n";
    /* First LHSinv and RHS portion that will be used repeatedly; */
    time_t fulla_begin_time = time(0);
    /**************************************************/
    /*** Read in pedigree and construct A then Ainv ***/
    /**************************************************/
    time_t fullped_begin_time = time(0);
    uniquephenotypeanimals(uniqueID,id);                /* Figure out unique ids for dim of A matrix and ZtZ or WtW */
    double* Relationshipinv_mkl = new double[uniqueID.size()*uniqueID.size()];
    for(int i = 0; i < (uniqueID.size()*uniqueID.size()); i++){Relationshipinv_mkl[i] = 0.0;}
    GenerateAinv(pedigreefile,uniqueID,id,Relationshipinv_mkl);        /* Generate A inverse */
    time_t fullped_end_time = time(0);
    logfile << "    - Generated Ainv: " << uniqueID.size() << " " << uniqueID.size() << " (Took: ";
    logfile << difftime(fullped_end_time,fullped_begin_time) << " seconds)" << endl;
    /********************************************************************************/
    /***     Generate XtX that is off full rank to Figure out Dimension of LHS    ***/
    /********************************************************************************/
    time_t fulllhs_begin_time = time(0);
    double* X_Dense = new double[pheno.size()*(1 + totalclasslevels + FIXED_COV[0].size())];
    for(int i = 0; i < (pheno.size()*(1 + totalclasslevels + FIXED_COV[0].size())); i++){X_Dense[i] = 0.0;}
    vector < int > keepremove ((1 + totalclasslevels + FIXED_COV[0].size()),0);
    vector < double > MeanPerCovClass(FIXED_COV[0].size(),0.0);
    GenerateFullRankXtX(X_Dense,keepremove,MeanPerCovClass,pheno,FIXED_CLASS,uniqueclass,FIXED_COV,logfile);
    int dimension = 0; int checkcolumn = 0;
    for(int i = 0; i < keepremove.size(); i++){dimension += keepremove[i];}
    for(int i = 0; i < pheno.size(); i++)
    {
        X_i.push_back(X_j.size()); int newcolumn = 0;
        for(int j = 0; j < keepremove.size(); j++)
        {
            //cout <<X_Dense[(i*keepremove.size())+j] << " ";
            if(keepremove[j] == 1)
            {
                if(X_Dense[(i*keepremove.size())+j] != 0){X_j.push_back(newcolumn); X_A.push_back(X_Dense[(i*keepremove.size())+j]);}
                newcolumn++;
            }
        }
    }
    delete [] X_Dense;
    /********************************************************************************/
    /***                     Generate LHS based on reduced Model                  ***/
    /********************************************************************************/
    /* Now know the dimension of LHS so fill LHS with appropriate columns */
    int dim_lhs; vector < double > lambda(2,0.0);
    if(perm_effect == "yes"){dim_lhs = dimension+uniqueID.size()+uniqueID.size(); lambda[0] = res_var / add_var; lambda[1] = res_var / perm_var;}
    if(perm_effect == "no"){dim_lhs = dimension+uniqueID.size(); lambda[0] = res_var / add_var;}
    GenerateLHSRed(X_i,X_j,X_A,dimension,Relationshipinv_mkl,ZW_i,ZW_j,ZW_A,id,uniqueID,LHSred_i,LHSred_j,LHSred_A,dim_lhs,lambda);
    time_t fulllhs_end_time = time(0);
    logfile<<"    - Generated LHS reduced model: "<<dim_lhs<<"-"<<dim_lhs<<" (Took: "<<difftime(fulllhs_end_time,fulllhs_begin_time)<<" seconds)"<<endl;
    delete [] Relationshipinv_mkl;
    /********************************************************************************/
    /***                  Generate vector add zero for contrasts                  ***/
    /********************************************************************************/
    vector < string > factor_red; vector < int > zero_columns_red; int i_r = 0; int i_f = 0;
    factor_red.push_back("int"); zero_columns_red.push_back(1); i_f++;
    for(int j = 0; j < FIXED_CLASS[0].size(); j++)      /* loop through fixed effect that are consistent across models */
    {
        /* first figure out number of ones zeroed out */
        int startat = i_f;
        for(int checknum = startat; checknum < (startat + uniqueclass[j].size()); checknum++)
        {
            zero_columns_red.push_back(keepremove[checknum]);
            stringstream ss; ss << (j + 1); string str = ss.str();
            factor_red.push_back("Fixed_Class" + str); i_f++;
        }
    }
    for(int j = 0; j < FIXED_COV[0].size(); j++)        /* loop through fixed covariate effects */
    {
        zero_columns_red.push_back(1); stringstream ss; ss << (j + 1); string str = ss.str(); factor_red.push_back("Cov_Class" + str);
    }
    for(int i = 0; i < uniqueID.size(); i++){zero_columns_red.push_back(1); factor_red.push_back("Random");}
    if(lambda[1] > 0.0){for(int i = 0; i < uniqueID.size(); i++){zero_columns_red.push_back(1); factor_red.push_back("Random");}}
    //cout << zero_columns_red.size() << " " << factor_red.size() << endl;
    int min_Phenotypes = minimum_freq * id.size() + 0.5;  /* Determines Minimum Number and rounds up correctly */
    /* If someone is using this with a small dataset make sure it is at least 2 in order to not create issues */
    if(min_Phenotypes < 2){min_Phenotypes = 2;}
    cout << " - Static Matrices Created." << endl;
    if(doublecheck == "yes")
    {
        logfile << endl << " Running model once and outputing least square means and t-stats to double check!!" << endl;
        doublecheckasreml(chr_index,min_Phenotypes,width,genotype,phenogenorownumber,pheno,dim_lhs,X_i,X_j,X_A,dimension,ZW_i,ZW_j,ZW_A,id,uniqueID,LHSred_i,LHSred_j,LHSred_A,factor_red,zero_columns_red,res_var,FIXED_CLASS,uniqueclass,FIXED_COV,MeanPerCovClass,unfav_direc);
    }
    if(determine_cutoff == "data")
    {
        logfile << "\n==================================\n";
        logfile << "==   Generate Phenotype Cutoff  ==\n";
        logfile << "==================================\n";
        logfile << "    - Randomly sample haplotype regions to generate phenotype cutoff! " << endl;
        phenotype_cutoff = phenocutoff(chr_index,null_samples,min_Phenotypes,width,genotype,phenogenorownumber,pheno,dim_lhs,X_i,X_j,X_A,dimension,ZW_i,ZW_j,ZW_A,uniqueID,LHSred_i,LHSred_j,LHSred_A,factor_red,zero_columns_red,res_var,FIXED_CLASS,uniqueclass,FIXED_COV,MeanPerCovClass,unfav_direc,logfile);
    }
    logfile << "        - Minimum phenotype cutoff: " << phenotype_cutoff << endl;
    cout << " - Phenotypic Cutoff Generated: (" << phenotype_cutoff << ")                                 \r" << std::flush << endl;
    logfile << "\n==================================================\n";
    logfile << "==\tStart to Loop Through Phenotypes    \t==\n";
    logfile << "==\tand Identify Unfavorable Haplotypes \t==\n";
    logfile << "==================================================" << endl;
    string Stage1loc = path + "/Stage1_Regions"; string Stage2loc = path + "/Stage2_Regions";
    fstream checkstg1; checkstg1.open(Stage1loc, std::fstream::out | std::fstream::trunc); checkstg1.close();      /* Remove previous file */
    fstream checkstg2; checkstg2.open(Stage2loc, std::fstream::out | std::fstream::trunc); checkstg2.close();         /* Remove previous file */
    logfile << "    - Start to loop across all chromosomes: " << endl;
    cout << " - Start to loop across all chromosomes: " << endl;
    time_t loop_begin_time = time(0);
    for(int chromo = 0; chromo < chr_index.size(); chromo++)
    {
        logfile << "        - Starting Chromosome: " << chromo + 1 << ":" << endl;
        time_t chr_begin_time = time(0);
        vector < Unfavorable_Regions > regions;                         /* vector of objects to store everything about unfavorable region */
        vector < Unfavorable_Regions_sub > regions_sub;                 /* vector of objects to store everything about unfavorable region */
        time_t begin_time = time(0);
        Step1(regions_sub,phenotype_cutoff,unfav_direc,chromo,chr_index,min_Phenotypes,width,genotype,phenogenorownumber,pheno,id,chr,position,index,regions);
        OutputStage1(Stage1loc,regions);
        time_t end_time = time(0);
        logfile << "            - Number of haplotypes after step 1: "<<regions.size()<<" (Took: "<<difftime(end_time,begin_time)<<" seconds)."<<endl;
        begin_time = time(0);
        Step2(regions,min_Phenotypes,genotype,phenogenorownumber,pheno,dim_lhs,X_i,X_j,X_A,dimension,ZW_i,ZW_j,ZW_A,uniqueID,LHSred_i,LHSred_j,LHSred_A,factor_red,zero_columns_red,res_var,FIXED_CLASS,uniqueclass,FIXED_COV,MeanPerCovClass,unfav_direc,one_sided_t,phenotype_cutoff,logfile);
        end_time = time(0);
        logfile << "            - Number of haplotypes after step 2: "<<regions.size()<<" (Took: "<<difftime(end_time,begin_time)<<" seconds)."<<endl;
        Step3(regions,pheno,genotype,phenogenorownumber,id);
        logfile << "            - Number of haplotypes after step 3: " << regions.size() << endl;
        OutputStage2(Stage2loc,regions,chromo);
        time_t chr_end_time = time(0);
        logfile << "        - Finished Chromosome: " << chromo + 1 << " (Took: " << difftime(chr_end_time,chr_begin_time) << " seconds)" << endl;
        cout << "   - Finished Chromosome: " << chromo + 1 << " (Took: " << difftime(chr_end_time,chr_begin_time) << " seconds)" << endl;
    }
    time_t loop_end_time = time(0);
    cout << " - Finished looping across all chromosomes: " << endl;
    logfile << "    - Finished looping across all chromosomes (Took: " << difftime(loop_end_time,loop_begin_time) << " seconds)" << endl;
    time_t full_end_time = time(0);
    logfile << "    - Finished Program (Took: " << difftime(full_end_time,full_begin_time) << " seconds)" << endl;
}

