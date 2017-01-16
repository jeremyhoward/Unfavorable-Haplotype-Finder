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
// Class to store index start and stop site for each chromosome //
class CHR_Index
{
private:
    int Chromosome;             /* Chromsome */
    int StartIndex;             /* Start Index */
    int EndIndex;               /* End Index */
    int Num_SNP;
public:
    CHR_Index();
    CHR_Index(int chr = 0, int stind = 0, int enind = 0, int numsnp = 0);
    ~CHR_Index();
    int getChr(){return Chromosome;}
    int getStInd(){return StartIndex;}
    int getEnInd(){return EndIndex;}
    int getNumSnp(){return Num_SNP;}
};
// Constructors ROH_Index
CHR_Index::CHR_Index(){Chromosome = 0; StartIndex = 0; EndIndex = 0; Num_SNP = 0;}
CHR_Index::CHR_Index(int chr, int stind, int enind, int numsnp){Chromosome = chr; StartIndex = stind; EndIndex = enind; Num_SNP = numsnp;}
// Destructors
CHR_Index::~CHR_Index(){}
// Class to store regions that passed Stage 1-2 //
class Unfavorable_Regions
{
private:
    int Chromosome_R;               /* Chromosome it is in */
    int StartPos_R;                 /* Start Position in base pairs */
    int EndPos_R;                   /* End Position in base pairs */
    int StartIndex_R;               /* Index it starts easier to grab later */
    int EndIndex_R;                 /* Index in ends easier to grab later */
    std::string Haplotype_R;        /* Unfavorable Haplotype */
    double Raw_Phenotype;           /* Uncorrected Phenotype */
    double effect;                  /* actual effect from model */
    double LS_Mean;                 /* Corrected Phenotype */
    double t_value;                 /* T-Statistisc */
public:
    Unfavorable_Regions();
    Unfavorable_Regions(int chr = 0, int st_pos = 0,int en_pos = 0, int st_ind = 0, int en_ind = 0, std::string hap = "", double pheno = 0.0, double eff = 0.0, double lsm = 0.0, double tval = 0.0);
    ~Unfavorable_Regions();
    int getChr_R(){return Chromosome_R;}
    int getStPos_R(){return StartPos_R;}
    int getEnPos_R(){return EndPos_R;}
    int getStartIndex_R(){return StartIndex_R;}
    int getEndIndex_R(){return EndIndex_R;}
    std::string getHaplotype_R(){return Haplotype_R;}
    double getRawPheno_R(){return Raw_Phenotype;}
    double getEffect(){return effect;}
    double getLSM_R(){return LS_Mean;}
    double gettval(){return t_value;}
    void Update_Haplotype(std::string temp);
    void Update_RawPheno(double temp);
    void Update_Effect(double temp);
    void Update_LSM(double temp);
    void Update_Tstat(double temp);
    friend bool sortByStart(const Unfavorable_Regions&, const Unfavorable_Regions&);
};
// Constructors Unfavorable Haplotypes
Unfavorable_Regions::Unfavorable_Regions()
{
    Chromosome_R = 0; StartPos_R = 0; EndPos_R = 0; StartIndex_R = 0; EndIndex_R = 0; Haplotype_R = "0"; Raw_Phenotype= 0.0; effect = 0.0; LS_Mean = 0; t_value = 0.0;
}
Unfavorable_Regions::Unfavorable_Regions(int chr, int st_pos, int en_pos, int st_ind, int en_ind, std::string hap, double pheno, double eff, double lsm, double tval)
{
    Chromosome_R = chr; StartPos_R = st_pos; EndPos_R = en_pos; StartIndex_R = st_ind; EndIndex_R = en_ind; Haplotype_R = hap; Raw_Phenotype = pheno; effect = eff; LS_Mean = lsm; t_value = tval;
}
// Destructors
Unfavorable_Regions::~Unfavorable_Regions(){};
void Unfavorable_Regions::Update_Haplotype(std::string temp){Haplotype_R = temp;}
void Unfavorable_Regions::Update_RawPheno(double temp){Raw_Phenotype = temp;}
void Unfavorable_Regions::Update_Effect(double temp){effect = temp;}
void Unfavorable_Regions::Update_LSM(double temp){LS_Mean = temp;}
void Unfavorable_Regions::Update_Tstat(double temp){t_value = temp;}
// Sort objects
bool sortByStart(const Unfavorable_Regions &lhs, const Unfavorable_Regions &rhs) {return lhs.StartPos_R < rhs.StartPos_R;}


// Class to store regions within each chromosome/
class Unfavorable_Regions_sub
{
private:
    int StartIndex_s;               /* start of haplotype */
    int EndIndex_s;                 /* end of haplotype */
    std::string Haplotype_s;        /* haplotype that is the worst */
    int Number_s;                   /* number of animals that have the worst haplotype */
    double Phenotype_s;             /* Mean phenotype of worst haplotype */
    std::string Animal_ID_s;        /* string of animals IDs; */
public:
    Unfavorable_Regions_sub();
    Unfavorable_Regions_sub(int st = 0, int end = 0, std::string hap_s = "", int num = 0, double pheno_s = 0.0, std::string animal_s = "");
    ~Unfavorable_Regions_sub();
    int getStartIndex_s(){return StartIndex_s;}
    int getEndIndex_s(){return EndIndex_s;}
    std::string getHaplotype_s(){return Haplotype_s;}
    int getNumber_s(){return Number_s;}
    double getPhenotype_s(){return Phenotype_s;}
    std::string getAnimal_ID_s(){return Animal_ID_s;}
    /* can do in parallel put first need to initialize things and then update */
    void Update_substart(int temp);
    void Update_subend(int temp);
    void Update_subHaplotype(std::string temp);
    void Update_subNumber(int temp);
    void Update_subPhenotype(double temp);
    void Update_subAnimal_IDs(std::string temp);
    // Sort Phenotype function
    friend bool sortByPheno(const Unfavorable_Regions_sub&, const Unfavorable_Regions_sub&);
};
void Unfavorable_Regions_sub::Update_substart(int temp){StartIndex_s = temp;}
void Unfavorable_Regions_sub::Update_subend(int temp){EndIndex_s = temp;}
void Unfavorable_Regions_sub::Update_subHaplotype(std::string temp){Haplotype_s = temp;}
void Unfavorable_Regions_sub::Update_subNumber(int temp){Number_s = temp;}
void Unfavorable_Regions_sub::Update_subPhenotype(double temp){Phenotype_s = temp;}
void Unfavorable_Regions_sub::Update_subAnimal_IDs(std::string temp){Animal_ID_s = temp;}

// Constructors Unfavorable Haplotypes_sub
Unfavorable_Regions_sub::Unfavorable_Regions_sub()
{
    StartIndex_s = 0; EndIndex_s = 0; Haplotype_s = ""; Number_s = 0; Phenotype_s = 0.0; Animal_ID_s = "";
}
Unfavorable_Regions_sub::Unfavorable_Regions_sub(int st, int end, std::string hap_s, int num, double pheno_s, std::string animal_s)
{
    StartIndex_s = st; EndIndex_s = end; Haplotype_s = hap_s; Number_s = num; Phenotype_s = pheno_s; Animal_ID_s = animal_s;
}
// Destructors
Unfavorable_Regions_sub::~Unfavorable_Regions_sub(){};
// Sort objects
bool sortByPheno(const Unfavorable_Regions_sub &lhs, const Unfavorable_Regions_sub &rhs) {return lhs.Phenotype_s < rhs.Phenotype_s;}


using namespace std;
int main()
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
    /* create path to ouput results */
    string logfileloc = path + "/LogFile";
    fstream checklog; checklog.open(logfileloc, std::fstream::out | std::fstream::trunc); checklog.close();         /* Remove previous file */
    std::ofstream logfile(logfileloc, std::ios_base::out);               /* open log file to output verbage throughout code */
    /* Read in Parameter File */
    string paramterfile;                                        /* Name of parameter file */
    cout << "What is the file name of the parameter file: ";
    cin >> paramterfile;
    cout << endl;
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
            threads = atoi(parm[search].c_str()); break;
        }
        search++;
        if(search >= parm.size()){cout << "Couldn't find 'THREADS:' variable in parameter file!" << endl; exit (EXIT_FAILURE);}
    }
    logfile << "    - Number of threads used:\t\t\t\t\t\t\t\t\t" << "'" << threads << "'." << endl;
    search = 0;
    while(1)
    {
        size_t fnd = parm[search].find("MAP_FILE:");
        if(fnd!=std::string::npos)
        {
            size_t pos = parm[search].find(": ", 0); parm[search].erase(0, pos + 1);
            parm[search].erase(remove(parm[search].begin(), parm[search].end(), ' '), parm[search].end()); mapfile = parm[search]; break;
        }
        search++;
        if(search >= parm.size()){cout << "Couldn't find 'MAP_FILE:' variable in parameter file!" << endl; exit (EXIT_FAILURE);}
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
        search++;
        if(search >= parm.size()){cout << "Couldn't find 'PHENO_FILE:' variable in parameter file!" << endl; exit (EXIT_FAILURE);}
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
    logfile << "\n==================================\n";
    logfile << "==\tReading in Map file \t==\n";
    logfile << "==================================\n";
    /* set number of threads used */
    omp_set_num_threads(threads);
    mkl_set_num_threads_local(threads);
    /* Read in file */
    vector <string> numbers;
    string line;
    ifstream infile;
    infile.open(mapfile.c_str());
    if(infile.fail()){cout << "Error Opening Map File \n"; exit (EXIT_FAILURE);}
    while (getline(infile,line)){numbers.push_back(line);}      /* Stores in vector and each new line push back to next space */
    int snp = numbers.size();                                   /* Determine number of SNP */
    logfile << "   - Total number of SNP markers " << snp << endl;
    vector < int > chr(snp,0);                                 /* stores chromosome in vector */
    vector < int > position(snp,0);                            /* stores position */
    vector < int > index(snp,0);                               /* used to grab chromosome */
    for(int i = 0; i < numbers.size(); i++)
    {
        /*  Grab chromosome number */
        string temp = numbers[i]; size_t pos = temp.find(" ", 0); string tempa = temp.substr(0,pos); chr[i] = atoi(tempa.c_str()); temp.erase(0, pos+1);
        position[i] = atoi(temp.c_str());                       /* Grab Position */
        index[i] = i;
    }
    numbers.clear();                                            /* clear vector that holds each row */
    vector < int > change_locations;                            /* used to track at which point the chromosome switches */
    change_locations.push_back(0);                              /* first SNP is in chromosome 1 */
    vector<CHR_Index> chr_index;                                /* Class to store chromosomal information */
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
    logfile << "   - Index locations for each SNP start - stop!!" << endl;
    for(int i = 0; i < chr_index.size(); i++)
    {
        logfile << "   - Chromosome " << chr_index[i].getChr() << ": " << chr_index[i].getStInd() << " - " << chr_index[i].getEnInd();
        logfile << "; Number of SNP: " << chr_index[i].getNumSnp() << endl;
    }
    logfile << endl;
    logfile << "==================================\n";
    logfile << "==\tReading in Pheno file\t==\n";
    logfile << "==================================\n";
    /* Import file and put each row into a vector */
    ifstream infile1;
    infile1.open(phenofile.c_str());
    if(infile1.fail()){cout << "Error Opening Phenotype File \n"; exit (EXIT_FAILURE);}
    while (getline(infile1,line)){numbers.push_back(line);}                             /* Stores in vector and each new line push back to next space */
    int rows = numbers.size();                                                          /* Determine number of SNP */
    logfile << "    - Total Number of observations in datafile is " << rows << endl;
    vector < string > id(numbers.size(),"");                                            /* ID */
    vector < double > pheno (numbers.size(),0.0);                                       /* Phenotype */
    vector < int > number(numbers.size(),1);                                            /* Number of phenotypes for unique id */
    vector < int > phenogenorownumber(numbers.size(),-5);                               /* match to genotype row */
    /* depending on how many fixed effect parameters their are and at what point need to make different sized matrices */
    vector < vector < string > > FIXED_CLASS;
    vector < vector < double > > FIXED_COV;
    for(int i = 0; i < numbers.size(); i++)
    {
        vector < string > temp;
        for(int j = 0; j < fixed_class_col.size(); j++){temp.push_back("");}
        FIXED_CLASS.push_back(temp);
    }
    /* calculate mean phenotype */
    double mean_phenotype = 0.0;
    for(int i = 0; i < numbers.size(); i++)
    {
        vector < double > temp;
        for(int j = 0; j < fixed_cov_col.size(); j++){temp.push_back(0.0);}
        FIXED_COV.push_back(temp);
    }
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
            logfile << "Number of columns doesn't match up with way parameter file was specified.";
            logfile << " Check row number " << i + 1 << ". Program Ended." << endl; exit (EXIT_FAILURE);
        }
        if(col_variables.size() == (1 + fixed_class_col.size() + fixed_cov_col.size() + 1))
       {
            id[i] = col_variables[id_column-1];                             /* Save ID */
            pheno[i] = atof(col_variables[phenocolumn-1].c_str());          /* Save Phenotype */
            mean_phenotype += pheno[i];
            /* Save cross-classified variables */
            for(int j = 0; j < fixed_class_col.size(); j++){FIXED_CLASS[i][j] = col_variables[fixed_class_col[j]-1];}
            for(int j = 0; j < fixed_cov_col.size(); j++){FIXED_COV[i][j] = atof((col_variables[fixed_cov_col[j]-1]).c_str());}
        }
        if(i == 0)
        {
            logfile << "    - This is what first line got partitioned into:" << endl;
            logfile << "       - ID: " << id[i] << endl;
            logfile << "       - Phenotype: " << pheno[i]  << endl;
            for(int j = 0; j < fixed_class_col.size(); j++)
            {
                logfile << "       - Classified Variable " << j + 1 << ": " << FIXED_CLASS[i][j]  << endl;
            }
            for(int j = 0; j < fixed_cov_col.size(); j++)
            {
                logfile << "       - Covariate Variable " << j + 1 << ": " << FIXED_COV[i][j] << endl;
            }
            
        }
    }
    mean_phenotype = mean_phenotype / double(numbers.size());               /* estimate mean */
    numbers.clear();            /* Clear to free up memory */
    /* if subtract mean do it now */
    if(subtract_mean == "yes"){for(int i = 0; i < pheno.size(); i++){pheno[i] = pheno[i] - mean_phenotype;}}
    logfile << "==================================\n";
    logfile << "==\tReading in Geno file \t==\n";
    logfile << "==================================\n";
    /* Import file and put each row into a vector */
    ifstream infile2;
    infile2.open(genofile.c_str());
    if(infile2.fail()){cout << "Error Opening Genotype File \n"; exit (EXIT_FAILURE);}
    while (getline(infile2,line)){numbers.push_back(line);}                             /* Stores in vector and each new line push back to next space */
    int genorows = numbers.size();
    logfile << "    - Total Number of lines in genotype file is " << genorows << endl;
    vector < string > genotype(genorows,"");
    vector < string > genotypeID(genorows,"");
    /* Put each column into a vector */
    for(int i = 0; i < genorows; i++)
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
        if(col_variables.size() != 2)
        {
            logfile << "Incorrect number of columns in genofile.";
            logfile << " Check row number " << i + 1 << ". Program Ended." << endl; exit (EXIT_FAILURE);
        }
        if(col_variables.size() == 2)
        {
            genotypeID[i] = col_variables[0];
            genotype[i] = col_variables[1];
        }
    }
    /* now go back and find ID in phenotype file and match it up with genotype file */
    for(int i = 0; i < rows; i++)
    {
        int currentgenoline = 0; string kill = "NO";
        while(kill == "NO")
        {
            if(genotypeID[currentgenoline] != id[i]){currentgenoline++;}
            if(genotypeID[currentgenoline] == id[i]){phenogenorownumber[i] = currentgenoline; kill = "YES";}
            if(currentgenoline > genorows){logfile << endl << " Problem matching geno and pheno ID's; Check record " << i + 1 << endl; exit (EXIT_FAILURE);}
        }
    }
    /* Double check to make sure all match */
    int numbermismatch = 0;
    for(int i = 0; i < 100; i++)
    {
        if(id[i] != genotypeID[phenogenorownumber[i]]){numbermismatch += 1;}
    }
    logfile << "    - Number of mismatched ID lines: " << numbermismatch << " !!" << endl;
    if(numbermismatch != 0){logfile << endl << " - Mismatched ID lines. " << endl; exit (EXIT_FAILURE);}
    /* In order to make computation easier sum phenotypes and track number */
    vector < string > id_sum;                                          /* animal id */
    vector < double > pheno_sum;                                       /* Phenotype sum for an animal */
    vector < int > numb_sum;                                           /* Number of phenotypes for a given animal */
    vector < int > genotyperow_sum;                                 /* Genotype String location */
    /* copy first then remove constant genotypes after */
    for(int i = 0; i < pheno.size(); i++)
    {
        id_sum.push_back(id[i]);
        pheno_sum.push_back(pheno[i]);
        numb_sum.push_back(1);
        genotyperow_sum.push_back(phenogenorownumber[i]);
    }
    /* convert to sum of phenotype to remove constant genotypes in order to save space*/
    int ROWS = pheno_sum.size();                                                        /* Current Size of summary statistics */
    int i = 0;                                                                          /* Start at first row and look forward */
    while(i < ROWS)
    {
        int j = i + 1;
        while(j < ROWS)
        {
            if(id_sum[i] == id_sum[j])                                                      /* if have same then combine */
            {
                numb_sum[i] += 1; pheno_sum[i] += pheno_sum[j];
                id_sum.erase(id_sum.begin()+j); pheno_sum.erase(pheno_sum.begin()+j);
                genotyperow_sum.erase(genotyperow_sum.begin()+j); numb_sum.erase(numb_sum.begin()+j);
                ROWS = ROWS -1;                                                             /* Reduce size of population so i stays the same */
            }
            if(id_sum[i] != id_sum[j]){j++;}                                                /* not the same ID so move to next row */
        }
        i++;
    }
    rows = id_sum.size();                                                               /* Determine number of unique ID's */
    logfile << "    - Total number of unique animals in phenotype file " << pheno_sum.size() << endl;
    if(pheno_sum.size() != genotype.size())
    {
        logfile << "    - Have extra genotypes in genotype file!! If this is a lot larger than number of unique animals " << endl;
        logfile << "    - then it may take up less memory if the genotype file was trimmed to only contain animals you need!" << endl;
    }
    logfile << endl;
    logfile << "==================================\n";
    logfile << "==\tSet Up Static Matrices \t==\n";
    logfile << "==================================\n";
    /* First need to set up LHS and RHS portion that will be used repeatedly; First need to figure out what an appropriate cutoff is for phenotypes */
    /* This part is important, which greatly reduces computational time by only selecting a set of haplotypes that are likely to be significantly */
    /* from a non-ROH haplotype */
    time_t fulla_begin_time = time(0);
    /**************************************************/
    /*** Read in pedigree and construct A then Ainv ***/
    /**************************************************/
    /* Read in pedigree file and create Ainv for animals that have a phenotype can be numeric or character,but parents come before progeny */
    time_t fullped_begin_time = time(0);
    vector < string > animal; vector < string > sire; vector < string > dam;
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
    /* Construct Full A Matrix then grab only animals that have phenotypes then invert with cholesky decomposition */
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
    logfile << "    - Generate Relationship Matrix " << endl;
    logfile << "        - Full A Size: " << FullRelationship.rows() << " " << FullRelationship.cols() << endl;
    //Renumber ID to match up with pedigree
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
    vector < int > uniqueID;
    for(int i = 0; i < id.size(); i++){uniqueID.push_back(renum_id[i]);}
    ROWS = pheno.size();
    i = 0;                                                                          /* Start at first row and look forward */
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
    int relsize = uniqueID.size();
    MatrixXd SubsetPedigreeRelationship(relsize,relsize);
    /* get ids for all animals that go in subset relationship */
    for(int i = 0; i < relsize; i++)
    {
        for(int j = 0; j < relsize; j++)
        {
            SubsetPedigreeRelationship(i,j) = FullRelationship((uniqueID[i] - 1),(uniqueID[j] - 1));
        }
    }
    logfile << "        - Reduced A Size: " << SubsetPedigreeRelationship.rows() << " " << SubsetPedigreeRelationship.cols() << endl;
    FullRelationship.resize(0,0);
    /*******************************************/
    /*** MKL's Cholesky Decomposition of A   ***/
    /*******************************************/
    // Set up variables to use for functions //
    unsigned long i_pa = 0, j_pa = 0;
    unsigned long na = SubsetPedigreeRelationship.cols();
    long long int infoa = 0;
    const long long int int_na =(int)na;
    char lowera ='L';
    double* Relationshipinv_mkl=new double[na*na];
    /* Copy it to a 2-dim array that is dynamically stored that all the computations will be on */
    #pragma omp parallel for private(j_pa)
    for(i_pa = 0; i_pa < na; i_pa++)
    {
        for(j_pa=0; j_pa < na; j_pa++){Relationshipinv_mkl[i_pa*na + j_pa]=SubsetPedigreeRelationship(i_pa,j_pa);}
    }
    dpotrf(&lowera, &int_na, Relationshipinv_mkl, &int_na, &infoa);             /* Calculate upper triangular L matrix */
    dpotri(&lowera, &int_na, Relationshipinv_mkl, &int_na, &infoa);             /* Calculate inverse of lower triangular matrix result is the inverse */
    /* Copy upper triangler part to lower traingular part and then you have the inverse ! */
    MatrixXd Ainv(na,na);
    int number_nonzero = 0;
    #pragma omp parallel for private(j_pa)
    for(j_pa = 0; j_pa < na; j_pa++)
    {
        for(i_pa = 0; i_pa <= j_pa; i_pa++)
        {
            Ainv(i_pa,j_pa) = Ainv(j_pa,i_pa) = Relationshipinv_mkl[i_pa*na+j_pa];
            if(Ainv(i_pa,j_pa) != 0){number_nonzero += 1;}
        }
    }
    // free memory
    delete[] Relationshipinv_mkl;
    time_t fullped_end_time = time(0);
    logfile << "    - Ainv: " << Ainv.rows()<< " " <<Ainv.cols() << " (Took: ";
    logfile << difftime(fullped_end_time,fullped_begin_time) << " seconds)" << endl;
    logfile << "        - Number of non-zero cells: " << number_nonzero << endl;
    SubsetPedigreeRelationship.resize(0,0);
    /**********************************************************************/
    /*** Now Begin to set up LHS and RHS matrices that can be reused    ***/
    /**********************************************************************/
    /* Generate sub x-matrix of every fixed effect except for the haplotype effect */
    /* For a given class figure out how man unique class variables their are */
    vector < vector < string > > uniqueclass;
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
    logfile << "    - Number of levels for each class fixed effect: " << endl;
    int totalclasslevels = 0;
    for(int i = 0; i < uniqueclass.size(); i++)
    {
        logfile << "        - Fixed Class Level " << i + 1 << " levels: " << uniqueclass[i].size() << "." << endl;
        totalclasslevels += (uniqueclass[i].size());
    }
    logfile << "    - Setting up submatrices: " << endl;
    /* Set up Xsub which is all the fixed class and each covariate */
    MatrixXd y(pheno.size(),1);
    MatrixXd X_dense(pheno.size(),(1 + totalclasslevels + FIXED_COV[0].size()));
    for(int i = 0; i < pheno.size(); i++)
    {
        for(int j = 0; j < (1 + totalclasslevels + FIXED_COV[0].size()); j++){X_dense(i,j) = 0;}
    }
    /* For each covariate class need to determine mean to use to set up L matrix */
    vector < double > MeanPerCovClass(FIXED_COV[0].size(),0.0);
    for(int i = 0; i < pheno.size(); i++)
    {
        X_dense(i,0) = 1; /* intercept */
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
            /* need to figure out where it should be taking into account intercept and previous class effects; first class effect is zeroed out */
            col = columnstart + col;
            X_dense(i,col) = 1;
            columnstart += uniqueclass[j].size();
        }
        /* Loop across covariate classes and add number to matrix */
        for(int j = 0; j < FIXED_COV[0].size(); j++)
        {
            X_dense(i,columnstart) = FIXED_COV[i][j]; MeanPerCovClass[j] += FIXED_COV[i][j]; columnstart += 1;
        }
        y(i,0) = pheno[i];
    }
    logfile << "        - X:\t" << X_dense.rows() << " " << X_dense.cols() << endl;
    /* Initially zero out first level of each effect */
    vector < int > keepremove (X_dense.cols(),0); int keepremoveindex = 1;
    keepremove[0] = 1;  /* keep intercept */
    for(int i = 0; i < uniqueclass.size(); i++)
    {
        for(int j = 0; j < uniqueclass[i].size(); j++)
        {
            if(j > 0){keepremove[keepremoveindex] = 1;}     /* Zero out first level */
            keepremoveindex++;
        }
    }
    for(int i = 0; i < FIXED_COV[0].size(); i++){keepremove[keepremoveindex] = 1; keepremoveindex++;}
    int dimension = 0;
    for(int i = 0; i < keepremove.size(); i++){dimension += keepremove[i];}
    /* First check if full rank by remove out first level of effect */
    MatrixXd check(pheno.size(),dimension);
    int checkcolumn = 0;
    for(int i = 0; i < keepremove.size(); i++)
    {
        if(keepremove[i] == 1){check.col(checkcolumn) = X_dense.col(i); checkcolumn++;}
    }
    Eigen::FullPivLU <MatrixXd> lu(check);
    int rank = lu.rank();
    if(rank != dimension)
    {
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
    }
    dimension = 0;
    
    
    
//    vector < int > columnsums(keepremove.size(),0);
//    vector < double > phenosums(keepremove.size(),0.0);
//    for(int i = 0; i < keepremove.size(); i++)
//    {
//        for(int j = 0; j < pheno.size(); j++)
//        {
//            if(X_dense(j,i) == 1){columnsums[i] += 1; phenosums[i] += y(j,0); }
//        }
//    }
//    for(int i = 0; i < keepremove.size(); i++){cout << i + 1 << " " << columnsums[i] << " " << phenosums[i] << endl;}
    
    
    
    for(int i = 0; i < keepremove.size(); i++){dimension += keepremove[i];}
    MatrixXd X_densefullrank(pheno.size(),dimension); check.resize(0,0);
    checkcolumn = 0;
    for(int i = 0; i < keepremove.size(); i++)
    {
        if(keepremove[i] == 1){X_densefullrank.col(checkcolumn) = X_dense.col(i); checkcolumn++;}
    }
    logfile << "        - X (full rank):\t" << X_densefullrank.rows() << " " << X_densefullrank.cols() << endl;
    
//    vector < int > columnsumsa(X_densefullrank.cols(),0);
//    vector < double > phenosumsa(X_densefullrank.cols(),0.0);
//    for(int i = 0; i < X_densefullrank.cols(); i++)
//    {
//        for(int j = 0; j < pheno.size(); j++)
//        {
//            if(X_densefullrank(j,i) == 1){columnsumsa[i] += 1; phenosumsa[i] += y(j,0); }
//        }
//    }
//    for(int i = 0; i < X_densefullrank.cols(); i++){cout << i + 1 << " " << columnsumsa[i] << " " << phenosumsa[i] << endl;}
//
//    exit (EXIT_FAILURE);
    
    Eigen::FullPivLU <MatrixXd> lua(X_densefullrank);
    rank = lua.rank();
    for(int j = 0; j < FIXED_COV[0].size(); j++){MeanPerCovClass[j]  = MeanPerCovClass[j] / pheno.size();}
    SparseMatrix < double > X_sub(X_densefullrank.rows(),X_densefullrank.cols());
    /* fill sparse matrix */
    for(int i = 0; i < X_densefullrank.rows(); i++)
    {
        for(int j = 0; j < X_densefullrank.cols(); j++)
        {
            if(X_densefullrank(i,j) != 0){X_sub.insert(i,j) = X_densefullrank(i,j);}
        }
    }
    X_densefullrank.resize(0,0);  X_dense.resize(0,0);
    SparseMatrix<double> X_subtX_suba(X_sub.cols(),X_sub.cols());
    X_subtX_suba = X_sub.transpose() * X_sub;
    logfile << "        - XtXsub:\t" << X_subtX_suba.rows() << " " << X_subtX_suba.cols() << endl;
//    
//    MatrixXd Check(X_subtX_suba.rows(),X_subtX_suba.rows()); Check = X_subtX_suba;
//    
//    for(int i = 0; i < 15; i++)
//    {
//        for(int j = 0; j < 15; j++)
//        {
//            cout << Check(i,j) << " ";
//        }
//        cout << endl;
//    }
    /* Declare scalar vector for alpha */
    VectorXd lambda(2);
    if(perm_effect == "yes"){lambda(0) = res_var / add_var; lambda(1) = res_var / perm_var;}
    if(perm_effect == "no"){lambda(0) = res_var / add_var; lambda(1) = res_var / perm_var;}
    /* At this time Z and W are the same; easier to fill the matrix in triplet form i, j, value for ZtZ instead of doing multiplication */
    /* Know that ZtZ and WtW are just a diagonal matrix of the number of observations */
    SparseMatrix<double> Z(pheno.size(),uniqueID.size());
    vector < double > diag(uniqueID.size(), 0);
    for(int i = 0; i < pheno.size(); i++)
    {
        int col = -5; int j = 0;
        while(j < uniqueID.size())
        {
            if(renum_id[i] == uniqueID[j]){Z.insert(i,j) = 1.0; diag[j] += 1; break;}
            if(renum_id[i] != uniqueID[j]){j++;}
            if(j == uniqueID.size()){cout << "Can't Find" << endl; exit (EXIT_FAILURE);}
        }
    }
    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletList;                     /* Used for additive effect part */
    tripletList.reserve(number_nonzero);
    std::vector<T> tripletLista;
    tripletLista.reserve(Ainv.rows());              /* Used for permanent environment part */
    for(int i = 0; i < Ainv.rows(); i++)
    {
        for(int j = 0; j < Ainv.cols(); j++)
        {
            if(Ainv(i,j) * lambda(0) != 0)
            {
                if(i != j)
                {
                    double temp = (Ainv(i,j) * lambda(0)); /* off-diagonal won't be a function of diagonals */
                    tripletList.push_back(T(i,j,temp));
                }
                if(i == j)
                {
                    double temp = diag[i] + (Ainv(i,j) * lambda(0)); /* diagonal will be a function of diagonals */
                    tripletList.push_back(T(i,j,temp));
                    if(perm_effect == "yes")
                    {
                        temp = diag[i] + (1 * lambda(1));
                        tripletLista.push_back(T(i,j,temp));
                    }
                }
            }
        }
    }
    SparseMatrix<double> ZtZa(Z.cols(),Z.cols());
    ZtZa.setFromTriplets(tripletList.begin(), tripletList.end());
    logfile << "        - ZtZ:\t\t" << ZtZa.rows() << " " << ZtZa.cols() << endl;
    SparseMatrix<double> WtWa(Z.cols(),Z.cols());
    if(perm_effect == "yes")
    {
        WtWa.setFromTriplets(tripletLista.begin(), tripletLista.end());
        logfile << "        - WtW:\t\t" << WtWa.rows() << " " << WtWa.cols() << endl;
    }
    diag.clear(); tripletList.clear(); tripletLista.clear();
    SparseMatrix <double> X_subtZa(X_sub.cols(),Z.cols()); X_subtZa = X_sub.transpose() * Z;        /* XsubtZ */
    logfile << "        - XsubtZ:\t" << X_subtZa.rows() << " " << X_subtZa.cols() << endl;
    SparseMatrix <double> X_subtWa(X_sub.cols(),Z.cols());
    if(perm_effect == "yes")
    {
        X_subtWa = X_sub.transpose() * Z;
        logfile << "        - XsubtW:\t" << X_subtWa.rows() << " " << X_subtWa.cols() << endl;
    }                                     /* XtW */
    SparseMatrix <double> ZtX_suba(Z.cols(),X_sub.cols()); ZtX_suba = Z.transpose() * X_sub;        /* Z'X */
    logfile << "        - ZtXsub:\t" << ZtX_suba.rows() << " " << ZtX_suba.cols() << endl;
    
    SparseMatrix <double> ZtWa(Z.cols(),Z.cols());
    if(perm_effect == "yes")
    {
        ZtWa = Z.transpose() * Z;                        /* Z'W */
        logfile << "        - ZtW:\t\t" << ZtWa.rows() << " " << ZtWa.cols() << endl;
    }
    SparseMatrix <double> WtX_suba(Z.cols(),X_sub.cols()); SparseMatrix <double> WtZa(Z.cols(),Z.cols());
    if(perm_effect == "yes")
    {
        WtX_suba = Z.transpose() * X_sub;        /* W'X */
        WtZa = Z.transpose() * Z;                        /* W'X */
        logfile << "        - WtXsub:\t" << WtX_suba.rows() << " " << WtX_suba.cols() << endl;
        logfile << "        - WtZ:\t\t" << WtZa.rows() << " " << WtZa.cols() << endl;
    }
    logfile << endl;
    /* Generate Zty and if needed Wty */
    MatrixXd X_subty(X_sub.cols(),1);
    MatrixXd Zty(Z.cols(),1);
    MatrixXd Wty(Z.cols(),1);
    if(perm_effect == "yes"){X_subty = X_sub.transpose() * y; Zty = Z.transpose() * y; Wty = Z.transpose() * y;}
    if(perm_effect == "no"){X_subty = X_sub.transpose() * y; Zty = Z.transpose() * y;}
    
//    for(int i = 0; i < X_sub.cols(); i++){cout << i + 1 << " " << X_subty(i) << endl;}
//    exit (EXIT_FAILURE);
//    
    int min_Phenotypes = minimum_freq * id.size() + 0.5;                                /* Determines Minimum Number and rounds up correctly */
    if(determine_cutoff == "data")
    {
        logfile << "==================================\n";
        logfile << "==   Generate Phenotype Cutoff  ==\n";
        logfile << "==================================\n";
        logfile << "    - Randomly sample haplotype regions to generate phenotype cutoff! " << endl;
        vector < vector < double > > sample_raw_pheno;
        vector < vector < double > > sample_raw_adj_pheno;
        vector < vector < double > > sample_t_value;
        //make it so it is number of rows based on null_samples
        for(int i = 0; i < null_samples; i++)
        {
            vector < double > temp;
            sample_raw_pheno.push_back(temp);
            sample_raw_adj_pheno.push_back(temp);
            sample_t_value.push_back(temp);
        }
        mt19937 gen(1337);
        #pragma omp parallel for
        for(int sample = 0; sample < null_samples; sample++)
        {
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
            //cout << chromo << " " << startsnp << " " << width[width_index] << endl;
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
            for(int i = 0; i < pheno.size(); i++)
            {
                mean_ROH[sub_genotype[i]] += pheno[i];
                number_ROH[sub_genotype[i]] += number[i];
            }
            for(int i = 0; i < mean_ROH.size(); i++){mean_ROH[i] = mean_ROH[i] / number_ROH[i]; category_ROH[i] = i;}
            for(int i = 1; i < mean_ROH.size(); i++){sample_raw_pheno[sample].push_back(mean_ROH[i]);}
            //for(int i = 0; i < mean_ROH.size(); i++)
            //{
            //    if(i == 0){cout << category_ROH[i] << "\t" << mean_ROH[i] << "\t" << number_ROH[i] << "\t" << "Non_ROH" << endl;}
            //    if(i >= 1){cout << category_ROH[i] << "\t" << mean_ROH[i] << "\t" << number_ROH[i] << "\t" << ROH_haplotypes[i-1] << endl;}
            //}
            //cout << endl;
            fstream checktestdf; checktestdf.open("TestDFa.txt", std::fstream::out | std::fstream::trunc); checktestdf.close();
            std::ofstream output5("TestDFa.txt", std::ios_base::app | std::ios_base::out);
            for(int i = 0; i < pheno.size(); i++)
            {
                output5 << id[i] << " ";
                for(int j = 0; j < FIXED_CLASS[0].size(); j++){output5 << FIXED_CLASS[i][j] << " ";}
                for(int j = 0; j < FIXED_COV[0].size(); j++){output5 << FIXED_COV[i][j]<< " ";}
                output5 << sub_genotype[i] << " " << pheno[i] << endl;
            }
            SparseMatrix <double> X_hap(pheno.size(),ROH_haplotypes.size());
            /* Set up X_hap and non_roh is zeroed out */
            for(int i = 0; i < pheno.size(); i++){if(sub_genotype[i] > 0){X_hap.insert(i,(sub_genotype[i]-1)) = 1;}}
            /* LHS */
            /* Row 1 */
            SparseMatrix<double> X_subtX_hapa(X_sub.cols(),X_hap.cols()); X_subtX_hapa = X_sub.transpose() * X_hap;
            /* Row2 */
            SparseMatrix<double> X_haptX_suba(X_hap.cols(),X_sub.cols()); X_haptX_suba = X_hap.transpose() * X_sub;
            SparseMatrix<double> XtX_hapa(X_hap.cols(),X_hap.cols()); XtX_hapa = X_hap.transpose() * X_hap;
            SparseMatrix<double> X_haptZa(X_hap.cols(),Z.cols()); X_haptZa = X_hap.transpose() * Z;
            SparseMatrix<double> X_haptWa(X_hap.cols(),Z.cols());
            if(perm_effect == "yes"){X_haptWa = X_hap.transpose() * Z;}
            /* Row3 */
            SparseMatrix <double> ZtXhapa(Z.cols(),X_hap.cols()); ZtXhapa = Z.transpose() * X_hap;
            /* Row4 */
            SparseMatrix <double> WtXhapa(Z.cols(),X_hap.cols());
            if(perm_effect == "yes"){WtXhapa = Z.transpose() * X_hap;}
            /* Can't combine sparse matrices together so convert to dense combine then convert LHS back to sparse */
            int rows_lhs;
            if(perm_effect == "yes"){rows_lhs = X_subtX_suba.rows()+X_haptX_suba.rows()+ZtX_suba.rows()+WtX_suba.rows();}
            if(perm_effect == "no"){rows_lhs = X_subtX_suba.rows()+X_haptX_suba.rows()+ZtX_suba.rows();}
            MatrixXd LHS(rows_lhs,rows_lhs);
            if(perm_effect == "yes")
            {
                /* Row 1 */
                MatrixXd X_subtX_sub(X_subtX_suba.rows(),X_subtX_suba.cols()); X_subtX_sub = X_subtX_suba;
                MatrixXd X_subtX_hap(X_subtX_hapa.rows(),X_subtX_hapa.cols()); X_subtX_hap = X_subtX_hapa;
                MatrixXd X_subtZ(X_subtZa.rows(),X_subtZa.cols()); X_subtZ=X_subtZa;
                MatrixXd X_subtW(X_subtWa.rows(),X_subtWa.cols()); X_subtW=X_subtWa;
                /* Row 2 */
                MatrixXd X_haptX_sub(X_haptX_suba.rows(),X_haptX_suba.cols()); X_haptX_sub = X_haptX_suba;
                MatrixXd XtX_hap(XtX_hapa.rows(),XtX_hapa.cols()); XtX_hap = XtX_hapa;
                MatrixXd X_haptZ(X_haptZa.rows(),X_haptZa.cols()); X_haptZ = X_haptZa;
                MatrixXd X_haptW(X_haptWa.rows(),X_haptWa.cols()); X_haptW = X_haptWa;
                /* Row 3 */
                MatrixXd ZtX_sub(ZtX_suba.rows(),ZtX_suba.cols()); ZtX_sub=ZtX_suba;
                MatrixXd ZtXhap(ZtXhapa.rows(),ZtXhapa.cols()); ZtXhap=ZtXhapa;
                MatrixXd ZtZ(ZtZa.rows(),ZtZa.cols()); ZtZ=ZtZa;
                MatrixXd ZtW(ZtWa.rows(),ZtWa.cols()); ZtW=ZtWa;
                /* Row 4 */
                MatrixXd WtX_sub(WtX_suba.rows(),WtX_suba.cols()); WtX_sub=WtX_suba;
                MatrixXd WtXhap(WtXhapa.rows(),WtXhapa.cols()); WtXhap=WtXhapa;
                MatrixXd WtZ(WtZa.rows(),WtZa.cols()); WtZ=WtZa;
                MatrixXd WtW(WtWa.rows(),WtWa.cols()); WtW=WtWa;
                /* Make LHS */
                LHS << X_subtX_sub,X_subtX_hap,X_subtZ,X_subtW,
                X_haptX_sub,XtX_hap,X_haptZ,X_haptW,
                ZtX_sub,ZtXhap,ZtZ,ZtW,
                WtX_sub,WtXhap,WtZ,WtW;
                /* remove all to save on storage most will be sparse */
                X_subtX_sub.resize(0,0); X_subtX_hap.resize(0,0); X_subtZ.resize(0,0); X_subtW.resize(0,0);
                X_haptX_sub.resize(0,0); XtX_hap.resize(0,0); X_haptZ.resize(0,0); X_haptW.resize(0,0);
                ZtX_sub.resize(0,0); ZtXhap.resize(0,0); ZtZ.resize(0,0); ZtW.resize(0,0);
                WtX_sub.resize(0,0); WtXhap.resize(0,0); WtZ.resize(0,0); WtW.resize(0,0);
            }
            if(perm_effect == "no")
            {
                /* Row 1 */
                MatrixXd X_subtX_sub(X_subtX_suba.rows(),X_subtX_suba.cols()); X_subtX_sub = X_subtX_suba;
                MatrixXd X_subtX_hap(X_subtX_hapa.rows(),X_subtX_hapa.cols()); X_subtX_hap = X_subtX_hapa;
                MatrixXd X_subtZ(X_subtZa.rows(),X_subtZa.cols()); X_subtZ=X_subtZa;
                /* Row 2 */
                MatrixXd X_haptX_sub(X_haptX_suba.rows(),X_haptX_suba.cols()); X_haptX_sub = X_haptX_suba;
                MatrixXd XtX_hap(XtX_hapa.rows(),XtX_hapa.cols()); XtX_hap = XtX_hapa;
                MatrixXd X_haptZ(X_haptZa.rows(),X_haptZa.cols()); X_haptZ = X_haptZa;
                /* Row 3 */
                MatrixXd ZtX_sub(ZtX_suba.rows(),ZtX_suba.cols()); ZtX_sub=ZtX_suba;
                MatrixXd ZtXhap(ZtXhapa.rows(),ZtXhapa.cols()); ZtXhap=ZtXhapa;
                MatrixXd ZtZ(ZtZa.rows(),ZtZa.cols()); ZtZ=ZtZa;
                /* Make LHS */
                LHS << X_subtX_sub,X_subtX_hap,X_subtZ,
                X_haptX_sub,XtX_hap,X_haptZ,
                ZtX_sub,ZtXhap,ZtZ;
                /* remove all to save on storage most will be sparse */
                X_subtX_sub.resize(0,0); X_subtX_hap.resize(0,0); X_subtZ.resize(0,0);
                X_haptX_sub.resize(0,0); XtX_hap.resize(0,0); X_haptZ.resize(0,0);
                ZtX_sub.resize(0,0); ZtXhap.resize(0,0); ZtZ.resize(0,0);
            }
            /*******/
            /* RHS */
            /*******/
            MatrixXd RHS(rows_lhs,1);
            if(perm_effect == "yes")
            {
                MatrixXd X_hapty(X_hap.cols(),1);
                X_hapty = X_hap.transpose() * y;
                RHS << X_subty,
                X_hapty,
                Zty,
                Wty;
                /* remove all */
                X_hapty.resize(0,0);
            }
            if(perm_effect == "no")
            {
                MatrixXd X_hapty(X_hap.cols(),1);
                X_hapty = X_hap.transpose() * y;
                RHS << X_subty,
                X_hapty,
                Zty;
                /* remove all */
                X_hapty.resize(0,0);
            }
            /*******************************************/
            /*** MKL's Cholesky Decomposition of LHS ***/
            /*******************************************/
            MatrixXd LHSinv(LHS.rows(),LHS.cols());
            int N = (int)LHS.cols();
            unsigned long i_p = 0, j_p = 0;
            unsigned long n = LHS.cols();
            long long int info = 0;
            const long long int int_n =(int)n;
            char lower='L';
            float* Vi_mkl=new float[n*n];
            /* Copy it to a 2-dim array that is dynamically stored that all the computations will be on */
            #pragma omp parallel for private(j_p)
            for(i_p=0; i_p<n; i_p++)
            {
                for(j_p=0; j_p<n; j_p++)
                {
                    Vi_mkl[i_p*n+j_p]=LHS(i_p,j_p);
                }
            }
            spotrf(&lower, &int_n, Vi_mkl, &int_n, &info);          /* Calculate upper triangular L matrix */
            spotri(&lower, &int_n, Vi_mkl, &int_n, &info);          /* Calculate inverse of lower triangular matrix result is the inverse */
            /* Copy upper triangler part to lower traingular part and then you have the inverse ! */
            #pragma omp parallel for private(j_p)
            for(j_p=0; j_p<n; j_p++)
            {
                for(i_p=0; i_p<=j_p; i_p++)
                {
                    LHSinv(i_p,j_p) = LHSinv(j_p,i_p) = Vi_mkl[i_p*n+j_p];
                }
            }
            // free memory
            delete[] Vi_mkl;
            /***********************/
            /*** Get Beta and SE ***/
            /***********************/
            MatrixXd b(LHS.cols(),1);
            MatrixXd b_se(LHS.cols(),1);
            b = LHSinv * RHS;
            /* Scale LHS*/
            for(int i = 0; i < LHSinv.rows(); i++)
            {
                for(int j = 0; j < LHSinv.cols(); j++){LHSinv(i,j) = LHSinv(i,j) * res_var;}
            }
            /********************************************************/
            /* Make it so beta has solutions for zero'd out effects */
            /********************************************************/
            int numberzeroed = 0;
            for(int i = 0; i < keepremove.size(); i++){if(keepremove[i] == 0){numberzeroed++;}}
            /* need to add in zero'd out columns to LHSinv and b used for contrasts*/
            /* full matrix is size LHS + number of zeroed out plus one for haplotype */
            MatrixXd b_full((LHS.cols() + numberzeroed + 1),1);
            MatrixXd LHSinv_full((LHS.cols() + numberzeroed + 1),(LHS.cols() + numberzeroed + 1));
            for(int i = 0; i < LHSinv_full.rows(); i++)
            {
                b_full(i,0) = 0;
                for(int j = 0; j < LHSinv_full.cols(); j++){LHSinv_full(i,j) = 0;}
            }
            /* Create vector in order to set up L more effectively */
            vector < string > factor(b_full.rows(),"Random"); /* Initialize to Random all fixed will be changed */
            /* which columns to add zeros too */
            vector < int > zero_columns;
            /* intercept */
            int i_r = 0;        /* which row are you at in reduced b and LHSinv */
            int i_f = 0;        /* which row are you at in reduced b and LHSinv */
            /* first one is intercept */
            b_full(i_f,0) = b(i_r,0); factor[i_f ] = "int"; i_f++; i_r++;
            /* now loop through fixed effect that are consistent across models */
            for(int j = 0; j < FIXED_CLASS[0].size(); j++)
            {
                /* first figure out number of ones zeroed out */
                int startat = i_f;
                for(int checknum = startat; checknum < (startat + uniqueclass[j].size()); checknum++)
                {
                    if(keepremove[checknum] == 0)
                    {
                        b_full(i_f,0) = 0; zero_columns.push_back(checknum);
                    }
                    if(keepremove[checknum] != 0)
                    {
                        b_full(i_f) = b(i_r,0); i_r++;
                    }
                    stringstream ss; ss << (j + 1); string str = ss.str();
                    factor[i_f] = "Fixed_Class" + str; i_f++;
                }
            }
            /* now loop through fixed covariate effects just copy */
            for(int j = 0; j < FIXED_COV[0].size(); j++)
            {
                b_full(i_f) = b(i_r,0);
                stringstream ss; ss << (j + 1); string str = ss.str();
                factor[i_f] = "Cov_Class" + str; i_f++; i_r++;
            }
            /* now loop through haplotype effects just copy */
            b_full(i_f,0) = 0; factor[i_f] = "haplotype"; zero_columns.push_back(i_f); i_f++;
            for(int j = 0; j < ROH_haplotypes.size(); j++)
            {
                b_full(i_f) = b(i_r,0); factor[i_f] = "haplotype"; i_f++; i_r++;
            }
            if(perm_effect == "yes"){b_full.block(i_f,0,(2*ZtZa.cols()),1) = b.block(i_r,0,(2*ZtZa.cols()),1);}
            if(perm_effect == "no"){b_full.block(i_f,0,(ZtZa.cols()),1) = b.block(i_r,0,(ZtZa.cols()),1);}
            //cout << zero_columns.size() << endl;
            b.resize(0,0);
            //for(int i = 0; i < zero_columns.size(); i++){cout << zero_columns[i] << " " << b_full(zero_columns[i],0) << endl;}
            //for(int i = 0; i < i_f+5; i++){cout << b_full(i,0) << " " << factor[i] << endl;}
            /********************************************************/
            /* Make it so LHSinv has solutions for zero'd out effects */
            /********************************************************/
            int where_at_in_reduced_i = 0; int where_at_in_zerocolumns_i = 0;
            for(int i = 0; i < LHSinv_full.rows(); i++)
            {
                if(i != zero_columns[where_at_in_zerocolumns_i])
                {
                    int where_at_in_reduced_j = 0; int where_at_in_zerocolumns_j = 0;
                    for(int j = 0; j < LHSinv_full.cols(); j++)
                    {
                        if(j != zero_columns[where_at_in_zerocolumns_j])
                        {
                            LHSinv_full(i,j) = LHSinv(where_at_in_reduced_i,where_at_in_reduced_j);
                            where_at_in_reduced_j++;
                        }
                        if(j == zero_columns[where_at_in_zerocolumns_j]){where_at_in_zerocolumns_j++;}
                    }
                    where_at_in_reduced_i++;
                }
                if(i == zero_columns[where_at_in_zerocolumns_i]){where_at_in_zerocolumns_i++;}
            }
            LHSinv.resize(0,0);
            
            /* First get least square mean for each ROH haplotypes */
            for(int j = 0; j < (ROH_haplotypes.size()+1); j++)
            {
                MatrixXd Lvec(1,b_full.rows());
                int current_haplotype = 0;
                for(int i = 0; i < b_full.rows(); i++)
                {
                    if(factor[i] == "int"){Lvec(0,i) = 1;}
                    for(int k = 0; k < FIXED_CLASS[0].size(); k++)
                    {
                        stringstream ss; ss << (k + 1); string str = ss.str();
                        string lookup = "Fixed_Class" + str;
                        if(factor[i] == lookup){Lvec(0,i) = 1 / double(uniqueclass[k].size());}
                    }
                    for(int k = 0; k < FIXED_COV[0].size(); k++)
                    {
                        stringstream ss; ss << (k + 1); string str = ss.str();
                        string lookup = "Cov_Class" + str;
                        if(factor[i] == lookup){Lvec(0,i) = MeanPerCovClass[k];}

                    }
                    if(factor[i] == "haplotype")
                    {
                        while(1)
                        {
                            if(current_haplotype == j){Lvec(0,i) = 1; current_haplotype++; break;}
                            if(current_haplotype != j){Lvec(0,i) = 0; current_haplotype++; break;}
                        }
                    }
                    if(factor[i] == "Random"){Lvec(0,i) = 0;}
                }
                if(j > 0){double temp = (Lvec * b_full).value(); sample_raw_adj_pheno[sample].push_back(temp);}
            }
            for(int j = 1; j < (ROH_haplotypes.size()+1); j++)
            {
                MatrixXd Lvec(2,b_full.rows());
                int current_haplotype = 1; int baseline = 0;
                for(int i = 0; i < b_full.rows(); i++)
                {
                    if(factor[i] == "int"){Lvec(0,i) = 1;Lvec(1,i) = Lvec(0,i) ;}
                    for(int k = 0; k < FIXED_CLASS[0].size(); k++)
                    {
                        stringstream ss; ss << (k + 1); string str = ss.str();
                        string lookup = "Fixed_Class" + str;
                        if(factor[i] == lookup)
                        {
                            Lvec(0,i) = 1 / double(uniqueclass[k].size()); Lvec(1,i) = Lvec(0,i);
                        }
                    }
                    for(int k = 0; k < FIXED_COV[0].size(); k++)
                    {
                        stringstream ss; ss << (k + 1); string str = ss.str();
                        string lookup = "Cov_Class" + str;
                        if(factor[i] == lookup)
                        {
                            Lvec(0,i) = MeanPerCovClass[k]; Lvec(1,i) = Lvec(0,i);
                        }
                    }
                    if(factor[i] == "haplotype" && baseline > 0)
                    {
                        while(1)
                        {
                            if(current_haplotype == j){Lvec(0,i) = 0; Lvec(1,i) = -1.0; current_haplotype++; break;}
                            if(current_haplotype != j){Lvec(0,i) = 0; Lvec(1,i) = 0; current_haplotype++; break;}
                        }
                    }
                    if(factor[i] == "haplotype" && baseline == 0){Lvec(0,i) = 1; Lvec(1,i) = 0.0; baseline++;}
                    if(factor[i] == "Random"){Lvec(0,i) = 0; Lvec(1,i) = 0;}
                }
                // SE is var(a) + var(b) - 2*cov(a,b)
                MatrixXd SE_Matrix(Lvec.rows(),Lvec.rows());
                SE_Matrix = (Lvec * LHSinv_full * Lvec.transpose());
                double SE = SE_Matrix(0,0) + SE_Matrix(1,1) - (2 * SE_Matrix(0,1));
                MatrixXd Means_Matrix (Lvec.rows(),1);
                Means_Matrix = (Lvec * b_full);
                double LSM_Diff = Means_Matrix(0,0) - Means_Matrix(1,0);
                double temp = LSM_Diff / double(sqrt(SE));
                sample_t_value[sample].push_back(temp);
            }
            //for(int i = 0; i < sample_t_value[sample].size(); i++)
            //{
            //    cout << sample_raw_pheno[sample][i] << " " << sample_raw_adj_pheno[sample][i] << " " << sample_t_value[sample][i] << endl;
            //}
            if(sample % 10 == 0){logfile << "      - " << sample << endl;}
        }
        logfile << endl << "    - Done Sampling Now Determine Cutoff." << endl;
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
        /* Find mean */
        double sum = 0;
        for(int i = 0; i < pheno_given_percentile.size(); i++){sum += pheno_given_percentile[i];}
        phenotype_cutoff = sum / pheno_given_percentile.size();
        logfile << "        - Minimum phenotype cutoff: " << phenotype_cutoff << endl;
        pheno_given_percentile.clear();
    }
    logfile << "==================================================\n";
    logfile << "==\tStart to Loop Through Phenotypes    \t==\n";
    logfile << "==\tand Identify Unfavorable Haplotypes \t==\n";
    logfile << "==================================================\n";
    logfile << endl;
    string Stage1loc = path + "/Stage1_Regions";
    string Stage2loc = path + "/Stage2_Regions";
    fstream checkstg1; checkstg1.open(Stage1loc, std::fstream::out | std::fstream::trunc); checkstg1.close();      /* Remove previous file */
    fstream checkstg2; checkstg2.open(Stage2loc, std::fstream::out | std::fstream::trunc); checkstg2.close();         /* Remove previous file */
    logfile << "    - Start to loop across all chromosomes: " << endl;
    time_t loop_begin_time = time(0);
    for(int chromo = 0; chromo < chr_index.size(); chromo++)
    {
        logfile << "        - Starting Chromosome: " << chromo + 1 << ":" << endl;
        time_t chr_begin_time = time(0);
        vector < Unfavorable_Regions > regions;                         /* vector of objects to store everything about unfavorable region */
        vector < Unfavorable_Regions_sub > regions_sub;                 /* vector of objects to store everything about unfavorable region */
        time_t begin_time = time(0);
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
                    vector < string > ROH_haplotypes;                                               /* Tabulates all unique ROH haplotypes */
                    vector < int > ROH_ID;                                                          /* Numeric ID for unique haplotype */
                    vector < int > Haplo_number;                                                    /* Number of phenotypes that have haplotype */
                    vector < int > sub_genotype;
                    for(int i = 0; i < pheno_sum.size(); i++){sub_genotype.push_back(0);}
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
                    for(int i = 0; i < pheno_sum.size(); i++)
                    {
                        string tempa = genotype[genotyperow_sum[i]].substr(chr_index[chromo].getStInd(),chr_index[chromo].getNumSnp());
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
                    //for(int i = 0; i < ROH_haplotypes.size(); i++)
                    //{
                    //   cout << ROH_haplotypes[i] << "\t" << ROH_ID[i] << "\t" << Haplo_number[i] << endl;
                    //}
                    //cout << endl;
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
                    for(int i = 0; i < pheno_sum.size(); i++)
                    {
                        mean_ROH[sub_genotype[i]] += pheno_sum[i];
                        number_ROH[sub_genotype[i]] += numb_sum[i];
                    }
                    for(int i = 0; i < mean_ROH.size(); i++){mean_ROH[i] = mean_ROH[i] / number_ROH[i]; category_ROH[i] = i;}
                    //for(int i = 0; i < mean_ROH.size(); i++)
                    //{
                    //    cout << category_ROH[i] << "\t" << mean_ROH[i] << "\t" << number_ROH[i] << "\t" << haplo_string[i] << endl;
                    //}
                    //cout << endl;
                    /* if below minimum phenotype set mean at non_roh and then sort; this makes it so don't have to loop through phenotypes again */
                    for(int i = 1; i < mean_ROH.size(); i++)
                    {
                        if(number_ROH[i] < min_Phenotypes){mean_ROH[i] = mean_ROH[0]; haplo_string[i] = "Non_ROH";}
                    }
                    //for(int i = 0; i < mean_ROH.size(); i++)
                    //{
                    //    cout << category_ROH[i] << "\t" << mean_ROH[i] << "\t" << number_ROH[i] << "\t" << haplo_string[i] << endl;
                    //}
                    //cout << endl;
                    /* Mean for each ROH Category Tabulated; Now sort and grab lowest one then save */
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
                    //cout << endl;
                    /* only save if below minimum phenotype threshold */
                    if(unfav_direc == "low")
                    {
                        if(mean_ROH[0] < phenotype_cutoff && haplo_string[0] != "Non_ROH")
                        {
                            string stringedid = "";
                            for(int i = 0; i < id_sum.size(); i++)
                            {
                                string tempa = genotype[genotyperow_sum[i]].substr(chr_index[chromo].getStInd(),chr_index[chromo].getNumSnp());
                                string temp = tempa.substr(scan,width[increment]);                         /* Grab substring */
                                if(temp == ROH_haplotypes[(category_ROH[0]-1)]){stringedid = stringedid + "_" + id_sum[i];}
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
                            for(int i = 0; i < id_sum.size(); i++)
                            {
                                string tempa = genotype[genotyperow_sum[i]].substr(chr_index[chromo].getStInd(),chr_index[chromo].getNumSnp());
                                string temp = tempa.substr(scan,width[increment]);                         /* Grab substring */
                                if(temp == ROH_haplotypes[(category_ROH[0]-1)]){stringedid = stringedid + "_" + id_sum[i];}
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
                    //if(scan >= 122 && increment == 0)
                    //{
                    //    cout << scan << " " << haplotype_worst[0] << " " << start_pos[0] << " " << end_pos[0] << endl; exit (EXIT_FAILURE);
                    //}
                    //if(scan < 150 && start_pos.size() > 0){cout << scan << " " << haplotype_worst[0] << " " << start_pos[0] << " " << end_pos[0] << endl;}
                }
            }
            /* remove ones that were below threshold which are ones that are still -5 */
            i = 0;
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
            ROWS = regions_sub.size();                           /* Current Size of summary statistics */
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
        time_t end_time = time(0);
        logfile << "            - Number of initial haplotypes prior to condensing steps: " << regions_sub.size() << " (Took: ";
        logfile << difftime(end_time,begin_time) << " seconds)." << endl;
        begin_time = time(0);
        //cout << regions_sub.size() << endl;
        //for(int i = 0; i < regions_sub.size(); i++)
        //{
        //    cout << regions_sub[i].getStartIndex_s() << " " << regions_sub[i].getEndIndex_s() << " " << regions_sub[i].getHaplotype_s() << " ";
        //    cout << regions_sub[i].getNumber_s() << " " << regions_sub[i].getPhenotype_s() << endl;
        //}
        //cout << endl;
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
        ROWS = regions_sub.size();                                              /* Current Size of summary statistics */
        i = 1;                                                                  /* Start at one because always look back at previous one */
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
        end_time = time(0);
        logfile << "            - Number of haplotypes after step 1 condensing: " << regions_sub.size() << " (Took: ";
        logfile << difftime(end_time,begin_time) << " seconds)." << endl;
        //cout << regions_sub.size() << endl;
        //for(int i = 0; i < regions_sub.size(); i++)
        //for(int i = 0; i < 20; i++)
        //{
        //    cout << regions_sub[i].getStartIndex_s() << " " << regions_sub[i].getEndIndex_s() << " " << regions_sub[i].getHaplotype_s() << " ";
        //    cout << regions_sub[i].getNumber_s() << " " << regions_sub[i].getPhenotype_s() << endl;
        //}
        //cout << endl;
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
        for(int i = 0; i < regions.size(); i++)
        {
            std::ofstream output5(Stage1loc, std::ios_base::app | std::ios_base::out);
            output5 << regions[i].getChr_R() << " " << regions[i].getStPos_R() << " " << regions[i].getEnPos_R() << " ";
            output5 << (regions[i].getStartIndex_R()+1) << " " << regions[i].getEndIndex_R() << " " << regions[i].getHaplotype_R() << " ";
            output5 << regions[i].getRawPheno_R() << endl;
        }
        if(doublecheck == "yes")
        {
            logfile << " Running model once and outputing least square means and t-stats to double check!!" << endl;
            vector < int > sub_genotype(pheno.size(),0);                                    /* haplotype for each individual */
            vector < string > ROH_haplotypes;                                               /* Tabulates all unique ROH haplotypes */
            vector < int > ROH_ID;                                                          /* Numeric ID for unique haplotype */
            vector < int > Haplo_number;                                                    /* Number of phenotypes that have haplotype */
            vector < double > effect_beta;
            for(int i = 0; i < pheno.size(); i++)
            {
                int length = regions[0].getEndIndex_R() - regions[0].getStartIndex_R();             /* Get length of haplotype */
                string temp = genotype[phenogenorownumber[i]].substr(regions[0].getStartIndex_R(),length);    /* Grab substring */
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
                            if(temp.compare(ROH_haplotypes[h]) == 0){sub_genotype[i] = h + 1; Haplo_number[h] = Haplo_number[h] + 1; stop = "KILL";}
                            if(temp.compare(ROH_haplotypes[h]) != 0){h++;}           /* Not the same */
                            if(h == ROH_haplotypes.size())                          /* If number not match = size of hapLibary then add */
                            {
                                ROH_haplotypes.push_back(temp);Haplo_number.push_back(1);ROH_ID.push_back((h+1));
                                sub_genotype[i] = ROH_haplotypes.size(); stop = "KILL";
                            }
                        }
                    }
                    if(ROH_haplotypes.size() == 0)                                  /* Haplotype library will be empty for first individual */
                    {
                        ROH_haplotypes.push_back(temp); Haplo_number.push_back(1); ROH_ID.push_back((1)); sub_genotype[i] = ROH_haplotypes.size();
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
            for(int i = 0; i < pheno.size(); i++){mean_ROH[sub_genotype[i]] += pheno[i]; number_ROH[sub_genotype[i]] += number[i];}
            for(int i = 0; i < mean_ROH.size(); i++){mean_ROH[i] = mean_ROH[i] / number_ROH[i]; category_ROH[i] = i;}
            fstream checktestdf; checktestdf.open("TestDFa.txt", std::fstream::out | std::fstream::trunc); checktestdf.close();
            std::ofstream output5("TestDFa.txt", std::ios_base::app | std::ios_base::out);
            for(int i = 0; i < pheno.size(); i++)
            {
                output5 << id[i] << " ";
                for(int j = 0; j < FIXED_CLASS[0].size(); j++){output5 << FIXED_CLASS[i][j] << " ";}
                for(int j = 0; j < FIXED_COV[0].size(); j++){output5 << FIXED_COV[i][j]<< " ";}
                output5 << sub_genotype[i] << " " << pheno[i] << endl;
            }
            /************************************************/
            /*** Now Begin to set up LHS and RHS matrices ***/
            /************************************************/
            SparseMatrix <double> X_hap(pheno.size(),ROH_haplotypes.size());
            /* Set up X_hap and non_roh is zeroed out */
            for(int i = 0; i < pheno.size(); i++){if(sub_genotype[i] > 0){X_hap.insert(i,(sub_genotype[i]-1)) = 1;}}
            /* LHS */
            SparseMatrix<double> X_subtX_hapa(X_sub.cols(),X_hap.cols()); X_subtX_hapa = X_sub.transpose() * X_hap;
            SparseMatrix<double> X_haptX_suba(X_hap.cols(),X_sub.cols()); X_haptX_suba = X_hap.transpose() * X_sub;
            SparseMatrix<double> XtX_hapa(X_hap.cols(),X_hap.cols()); XtX_hapa = X_hap.transpose() * X_hap;
            SparseMatrix<double> X_haptZa(X_hap.cols(),Z.cols()); X_haptZa = X_hap.transpose() * Z;
            SparseMatrix<double> X_haptWa(X_hap.cols(),Z.cols());
            if(perm_effect == "yes"){X_haptWa = X_hap.transpose() * Z;}
            SparseMatrix <double> ZtXhapa(Z.cols(),X_hap.cols()); ZtXhapa = Z.transpose() * X_hap;
            SparseMatrix <double> WtXhapa(Z.cols(),X_hap.cols());
            if(perm_effect == "yes"){WtXhapa = Z.transpose() * X_hap;}
            /* Can't combine sparse matrices together so convert to dense combine then convert LHS back to sparse */
            int rows_lhs;
            if(perm_effect == "yes"){rows_lhs = X_subtX_suba.rows()+X_haptX_suba.rows()+ZtX_suba.rows()+WtX_suba.rows();}
            if(perm_effect == "no"){rows_lhs = X_subtX_suba.rows()+X_haptX_suba.rows()+ZtX_suba.rows();}
            MatrixXd LHS(rows_lhs,rows_lhs);
            if(perm_effect == "yes")
            {
                MatrixXd X_subtX_sub(X_subtX_suba.rows(),X_subtX_suba.cols()); X_subtX_sub = X_subtX_suba;
                MatrixXd X_subtX_hap(X_subtX_hapa.rows(),X_subtX_hapa.cols()); X_subtX_hap = X_subtX_hapa;
                MatrixXd X_subtZ(X_subtZa.rows(),X_subtZa.cols()); X_subtZ=X_subtZa;
                MatrixXd X_subtW(X_subtWa.rows(),X_subtWa.cols()); X_subtW=X_subtWa;
                MatrixXd X_haptX_sub(X_haptX_suba.rows(),X_haptX_suba.cols()); X_haptX_sub = X_haptX_suba;
                MatrixXd XtX_hap(XtX_hapa.rows(),XtX_hapa.cols()); XtX_hap = XtX_hapa;
                MatrixXd X_haptZ(X_haptZa.rows(),X_haptZa.cols()); X_haptZ = X_haptZa;
                MatrixXd X_haptW(X_haptWa.rows(),X_haptWa.cols()); X_haptW = X_haptWa;
                MatrixXd ZtX_sub(ZtX_suba.rows(),ZtX_suba.cols()); ZtX_sub=ZtX_suba;
                MatrixXd ZtXhap(ZtXhapa.rows(),ZtXhapa.cols()); ZtXhap=ZtXhapa;
                MatrixXd ZtZ(ZtZa.rows(),ZtZa.cols()); ZtZ=ZtZa;
                MatrixXd ZtW(ZtWa.rows(),ZtWa.cols()); ZtW=ZtWa;
                MatrixXd WtX_sub(WtX_suba.rows(),WtX_suba.cols()); WtX_sub=WtX_suba;
                MatrixXd WtXhap(WtXhapa.rows(),WtXhapa.cols()); WtXhap=WtXhapa;
                MatrixXd WtZ(WtZa.rows(),WtZa.cols()); WtZ=WtZa;
                MatrixXd WtW(WtWa.rows(),WtWa.cols()); WtW=WtWa;
                /* Make LHS */
                LHS << X_subtX_sub,X_subtX_hap,X_subtZ,X_subtW,
                X_haptX_sub,XtX_hap,X_haptZ,X_haptW,
                ZtX_sub,ZtXhap,ZtZ,ZtW,
                WtX_sub,WtXhap,WtZ,WtW;
                /* remove all to save on storage most will be sparse */
                X_subtX_sub.resize(0,0); X_subtX_hap.resize(0,0); X_subtZ.resize(0,0); X_subtW.resize(0,0);
                X_haptX_sub.resize(0,0); XtX_hap.resize(0,0); X_haptZ.resize(0,0); X_haptW.resize(0,0);
                ZtX_sub.resize(0,0); ZtXhap.resize(0,0); ZtZ.resize(0,0); ZtW.resize(0,0);
                WtX_sub.resize(0,0); WtXhap.resize(0,0); WtZ.resize(0,0); WtW.resize(0,0);
            }
            if(perm_effect == "no")
            {
                MatrixXd X_subtX_sub(X_subtX_suba.rows(),X_subtX_suba.cols()); X_subtX_sub = X_subtX_suba;
                MatrixXd X_subtX_hap(X_subtX_hapa.rows(),X_subtX_hapa.cols()); X_subtX_hap = X_subtX_hapa;
                MatrixXd X_subtZ(X_subtZa.rows(),X_subtZa.cols()); X_subtZ=X_subtZa;
                MatrixXd X_haptX_sub(X_haptX_suba.rows(),X_haptX_suba.cols()); X_haptX_sub = X_haptX_suba;
                MatrixXd XtX_hap(XtX_hapa.rows(),XtX_hapa.cols()); XtX_hap = XtX_hapa;
                MatrixXd X_haptZ(X_haptZa.rows(),X_haptZa.cols()); X_haptZ = X_haptZa;
                MatrixXd ZtX_sub(ZtX_suba.rows(),ZtX_suba.cols()); ZtX_sub=ZtX_suba;
                MatrixXd ZtXhap(ZtXhapa.rows(),ZtXhapa.cols()); ZtXhap=ZtXhapa;
                MatrixXd ZtZ(ZtZa.rows(),ZtZa.cols()); ZtZ=ZtZa;
                /* Make LHS */
                LHS << X_subtX_sub,X_subtX_hap,X_subtZ,
                X_haptX_sub,XtX_hap,X_haptZ,
                ZtX_sub,ZtXhap,ZtZ;
                /* remove all to save on storage most will be sparse */
                X_subtX_sub.resize(0,0); X_subtX_hap.resize(0,0); X_subtZ.resize(0,0);
                X_haptX_sub.resize(0,0); XtX_hap.resize(0,0); X_haptZ.resize(0,0);
                ZtX_sub.resize(0,0); ZtXhap.resize(0,0); ZtZ.resize(0,0);
            }
            //logfile << "                - (LHS: " << LHS.rows() << "-" << LHS.cols() << ")"<< endl;
            /*******/
            /* RHS */
            /*******/
            MatrixXd RHS(rows_lhs,1);
            if(perm_effect == "yes")
            {
                MatrixXd X_hapty(X_hap.cols(),1);
                X_hapty = X_hap.transpose() * y;
                RHS << X_subty,
                X_hapty,
                Zty,
                Wty;
                /* remove all */
                X_hapty.resize(0,0);
            }
            if(perm_effect == "no")
            {
                MatrixXd X_hapty(X_hap.cols(),1);
                X_hapty = X_hap.transpose() * y;
                RHS << X_subty,
                X_hapty,
                Zty;
                /* remove all */
                X_hapty.resize(0,0);
            }
            //logfile << "                - (RHS: " << RHS.rows() << "-" << RHS.cols() << ")" << endl;
            /*******************************************/
            /*** MKL's Cholesky Decomposition of LHS ***/
            /*******************************************/
            MatrixXd LHSinv(LHS.rows(),LHS.cols());
            int N = (int)LHS.cols();
            unsigned long i_p = 0, j_p = 0;
            unsigned long n = LHS.cols();
            long long int info = 0;
            const long long int int_n =(int)n;
            char lower='L';
            double* Vi_mkl=new double[n*n];
            /* Copy it to a 2-dim array that is dynamically stored that all the computations will be on */
            #pragma omp parallel for private(j_p)
            for(i_p=0; i_p<n; i_p++){for(j_p=0; j_p<n; j_p++){Vi_mkl[i_p*n+j_p]=LHS(i_p,j_p);}}
            dpotrf(&lower, &int_n, Vi_mkl, &int_n, &info);          /* Calculate upper triangular L matrix */
            dpotri(&lower, &int_n, Vi_mkl, &int_n, &info);          /* Calculate inverse of lower triangular matrix result is the inverse */
            /* Copy upper triangler part to lower traingular part and then you have the inverse ! */
            #pragma omp parallel for private(j_p)
            for(j_p=0; j_p<n; j_p++){for(i_p=0; i_p<=j_p; i_p++){LHSinv(i_p,j_p) = LHSinv(j_p,i_p) = Vi_mkl[i_p*n+j_p];}}
            // free memory
            delete[] Vi_mkl;
            /***********************/
            /*** Get Beta and SE ***/
            /***********************/
            MatrixXd b(LHS.cols(),1);
            MatrixXd b_se(LHS.cols(),1);
            b = LHSinv * RHS;
            /* Scale LHS*/
            #pragma omp parallel for private(j_p)
            for(j_p=0; j_p<n; j_p++)
            {
                for(i_p=0; i_p<=j_p; i_p++)
                {
                    LHSinv(i_p,j_p) = LHSinv(j_p,i_p) = LHSinv(j_p,i_p) * res_var;
                }
            }
            //fstream checktest; checktest.open("CheckLHSa", std::fstream::out | std::fstream::trunc); checktest.close();
            //std::ofstream output8("CheckLHSa", std::ios_base::app | std::ios_base::out);
            //for(int i = 0; i < LHS.rows(); i++)
            //{
            //    for(int j = 0; j < LHS.cols(); j++)
            //    {
            //        if(j != LHS.cols() -1){output8 << LHSinv(i,j) << " ";}
            //        if(j == LHS.cols() -1){output8 << LHSinv(i,j);}
            //    }
            //    output8 << endl;
            //}
            //fstream checktesta; checktesta.open("CheckRHSa", std::fstream::out | std::fstream::trunc); checktesta.close();
            //std::ofstream output9("CheckRHSa", std::ios_base::app | std::ios_base::out);
            //for(int i = 0; i < RHS.rows(); i++)
            //{
            //    output9 << RHS(i,0) << endl;
            //}
            //for(int i = 0; i < 100; i++)
            //{
            //    cout << b(i,0) << endl;
            //}
            //cout << endl;
            /********************************************************/
            /* Make it so beta has solutions for zero'd out effects */
            /********************************************************/
            int numberzeroed = 0;
            for(int i = 0; i < keepremove.size(); i++){if(keepremove[i] == 0){numberzeroed++;}}
            /* need to add in zero'd out columns to LHSinv and b used for contrasts*/
            /* full matrix is size LHS + number of zeroed out plus one for haplotype */
            MatrixXd b_full((LHS.cols() + numberzeroed + 1),1);
            MatrixXd LHSinv_full((LHS.cols() + numberzeroed + 1),(LHS.cols() + numberzeroed + 1));
            for(int i = 0; i < LHSinv_full.rows(); i++)
            {
                b_full(i,0) = 0;
                for(int j = 0; j < LHSinv_full.cols(); j++){LHSinv_full(i,j) = 0;}
            }
            /* Create vector in order to set up L more effectively */
            vector < string > factor(b_full.rows(),"Random"); /* Initialize to Random all fixed will be changed */
            /* which columns to add zeros too */
            vector < int > zero_columns;
            /* intercept */
            int i_r = 0;        /* which row are you at in reduced b and LHSinv */
            int i_f = 0;        /* which row are you at in reduced b and LHSinv */
            /* first one is intercept */
            b_full(i_f,0) = b(i_r,0); factor[i_f ] = "int"; i_f++; i_r++;
            /* now loop through fixed effect that are consistent across models */
            for(int j = 0; j < FIXED_CLASS[0].size(); j++)
            {
                /* first figure out number of ones zeroed out */
                int startat = i_f;
                for(int checknum = startat; checknum < (startat + uniqueclass[j].size()); checknum++)
                {
                    if(keepremove[checknum] == 0)
                    {
                        b_full(i_f,0) = 0; zero_columns.push_back(checknum);
                    }
                    if(keepremove[checknum] != 0)
                    {
                        b_full(i_f) = b(i_r,0); i_r++;
                    }
                    stringstream ss; ss << (j + 1); string str = ss.str();
                    factor[i_f] = "Fixed_Class" + str; i_f++;
                }
            }
            /* now loop through fixed covariate effects just copy */
            for(int j = 0; j < FIXED_COV[0].size(); j++)
            {
                b_full(i_f) = b(i_r,0);
                stringstream ss; ss << (j + 1); string str = ss.str();
                factor[i_f] = "Cov_Class" + str; i_f++; i_r++;
            }
            /* now loop through haplotype effects just copy */
            b_full(i_f,0) = 0; factor[i_f] = "haplotype"; zero_columns.push_back(i_f); i_f++;
            for(int j = 0; j < ROH_haplotypes.size(); j++)
            {
                b_full(i_f) = b(i_r,0); factor[i_f] = "haplotype"; i_f++; i_r++;
            }
            if(perm_effect == "yes"){b_full.block(i_f,0,(2*ZtZa.cols()),1) = b.block(i_r,0,(2*ZtZa.cols()),1);}
            if(perm_effect == "no"){b_full.block(i_f,0,(ZtZa.cols()),1) = b.block(i_r,0,(ZtZa.cols()),1);}
            //cout << zero_columns.size() << endl;
            b.resize(0,0);
            //for(int i = 0; i < zero_columns.size(); i++){cout << zero_columns[i] << " " << b_full(zero_columns[i],0) << endl;}
            //for(int i = 0; i < i_f+5; i++){cout << b_full(i,0) << " " << factor[i] << endl;}
            /********************************************************/
            /* Make it so LHSinv has solutions for zero'd out effects */
            /********************************************************/
            int where_at_in_reduced_i = 0; int where_at_in_zerocolumns_i = 0;
            for(int i = 0; i < LHSinv_full.rows(); i++)
            {
                if(i != zero_columns[where_at_in_zerocolumns_i])
                {
                    int where_at_in_reduced_j = 0; int where_at_in_zerocolumns_j = 0;
                    for(int j = 0; j < LHSinv_full.cols(); j++)
                    {
                        if(j != zero_columns[where_at_in_zerocolumns_j])
                        {
                            LHSinv_full(i,j) = LHSinv(where_at_in_reduced_i,where_at_in_reduced_j);
                            where_at_in_reduced_j++;
                        }
                        if(j == zero_columns[where_at_in_zerocolumns_j]){where_at_in_zerocolumns_j++;}
                    }
                    where_at_in_reduced_i++;
                }
                if(i == zero_columns[where_at_in_zerocolumns_i]){where_at_in_zerocolumns_i++;}
            }
            LHSinv.resize(0,0);
            /* Fill beta estimate of roh effect */
            int current_haplotype = 0;
            for(int i = 0; i < b_full.rows(); i++)
            {
                if(factor[i] == "haplotype")
                {
                    while(1)
                    {
                        if(current_haplotype == 0){current_haplotype++; break;}
                        if(current_haplotype > 0){effect_beta.push_back(b_full(i,0)); current_haplotype++; break;}
                    }
                }
            }
            vector < double > LSM(ROH_haplotypes.size(),0);
            vector < double > T_stat(ROH_haplotypes.size(),0);
            /* First get least square mean for each ROH haplotypes */
            #pragma omp parallel for
            for(int j = 0; j < (ROH_haplotypes.size()+1); j++)
            {
                MatrixXd Lvec(1,b_full.rows());
                int current_haplotype = 0;
                for(int i = 0; i < b_full.rows(); i++)
                {
                    if(factor[i] == "int"){Lvec(0,i) = 1;}
                    for(int k = 0; k < FIXED_CLASS[0].size(); k++)
                    {
                        stringstream ss; ss << (k + 1); string str = ss.str();
                        string lookup = "Fixed_Class" + str;
                        if(factor[i] == lookup){Lvec(0,i) = 1 / double(uniqueclass[k].size());}
                    }
                    for(int k = 0; k < FIXED_COV[0].size(); k++)
                    {
                        stringstream ss; ss << (k + 1); string str = ss.str();
                        string lookup = "Cov_Class" + str;
                        if(factor[i] == lookup){Lvec(0,i) = MeanPerCovClass[k];}
                        
                    }
                    if(factor[i] == "haplotype")
                    {
                        while(1)
                        {
                            if(current_haplotype == j){Lvec(0,i) = 1; current_haplotype++; break;}
                            if(current_haplotype != j){Lvec(0,i) = 0; current_haplotype++; break;}
                        }
                    }
                    if(factor[i] == "Random"){Lvec(0,i) = 0;}
                }
                if(j > 0){double temp = (Lvec * b_full).value(); LSM[j-1] = temp;}
            }
            #pragma omp parallel for
            for(int j = 1; j < (ROH_haplotypes.size()+1); j++)
            {
                MatrixXd Lvec(2,b_full.rows());
                int current_haplotype = 1; int baseline = 0;
                for(int i = 0; i < b_full.rows(); i++)
                {
                    if(factor[i] == "int"){Lvec(0,i) = 1;Lvec(1,i) = Lvec(0,i) ;}
                    for(int k = 0; k < FIXED_CLASS[0].size(); k++)
                    {
                        stringstream ss; ss << (k + 1); string str = ss.str();
                        string lookup = "Fixed_Class" + str;
                        if(factor[i] == lookup)
                        {
                            Lvec(0,i) = 1 / double(uniqueclass[k].size()); Lvec(1,i) = Lvec(0,i);
                        }
                    }
                    for(int k = 0; k < FIXED_COV[0].size(); k++)
                    {
                        stringstream ss; ss << (k + 1); string str = ss.str();
                        string lookup = "Cov_Class" + str;
                        if(factor[i] == lookup)
                        {
                            Lvec(0,i) = MeanPerCovClass[k]; Lvec(1,i) = Lvec(0,i);
                        }
                    }
                    if(factor[i] == "haplotype" && baseline > 0)
                    {
                        while(1)
                        {
                            if(current_haplotype == j){Lvec(0,i) = 0; Lvec(1,i) = -1.0; current_haplotype++; break;}
                            if(current_haplotype != j){Lvec(0,i) = 0; Lvec(1,i) = 0; current_haplotype++; break;}
                        }
                    }
                    if(factor[i] == "haplotype" && baseline == 0){Lvec(0,i) = 1; Lvec(1,i) = 0.0; baseline++;}
                    if(factor[i] == "Random"){Lvec(0,i) = 0; Lvec(1,i) = 0;}
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
            logfile << endl << endl;
            logfile << "Wanted to double check with external program: " << endl;
            logfile << "Location and summary statistics about haplotype: " << endl;
            logfile << "    - Chromosome: " << regions[0].getChr_R() << endl;
            logfile << "    - Start Position Mb: " << regions[0].getStPos_R() << endl;
            logfile << "    - End Position Mb: " << regions[0].getEnPos_R() << endl;
            logfile << "    - Unfavorable Haplotype: " << regions[0].getHaplotype_R() << endl;
            logfile << "    - Phenotypic Effect of Unfavorable Haplotype: " << regions[0].getRawPheno_R() << endl;
            logfile << "Output of non-ROH and each unique ROH haplotype from complete model: " << endl;
            logfile << " ROH ID -- Least Square Mean -- Beta Estimate -- T_stat of difference between non-ROH and haplotype " << endl;
            for(int i = 0; i < ROH_haplotypes.size(); i++)
            {
                logfile << i + 1 << " " << LSM[i] << " " << effect_beta[i] << " " << T_stat[i] << endl;
            }
            logfile << endl;
            exit (EXIT_FAILURE);
        }
        logfile << "            - Begin step 2 with full model: " << endl;
        begin_time = time(0);
        /* Run this step in parralel and in order to remove clashing create a 2-d matrix for windows that contain multiple significant haplotypes */
        /* and then add to regions object at the end. only need to make 4 new ones because others will be the same for a given region. */
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
        #pragma omp parallel for
        for(int loopregions = 0; loopregions < regions.size(); loopregions++)
        {
            time_t fullb_begin_time = time(0);
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
            for(int i = 0; i < pheno.size(); i++){mean_ROH[sub_genotype[i]] += pheno[i]; number_ROH[sub_genotype[i]] += number[i];}
            for(int i = 0; i < mean_ROH.size(); i++){mean_ROH[i] = mean_ROH[i] / number_ROH[i]; category_ROH[i] = i;}
            //for(int i = 0; i < mean_ROH.size(); i++)
            //{
            //    if(i == 0){cout << category_ROH[i] << "\t" << mean_ROH[i] << "\t" << number_ROH[i] << "\t" << "Non_ROH" << endl;}
            //    if(i >= 1){cout << category_ROH[i] << "\t" << mean_ROH[i] << "\t" << number_ROH[i] << "\t" << ROH_haplotypes[i-1] << endl;}
            //}
            //cout << endl;
            fstream checktestdf; checktestdf.open("TestDFa.txt", std::fstream::out | std::fstream::trunc); checktestdf.close();
            std::ofstream output5("TestDFa.txt", std::ios_base::app | std::ios_base::out);
            for(int i = 0; i < pheno.size(); i++)
            {
                output5 << id[i] << " ";
                for(int j = 0; j < FIXED_CLASS[0].size(); j++){output5 << FIXED_CLASS[i][j] << " ";}
                for(int j = 0; j < FIXED_COV[0].size(); j++){output5 << FIXED_COV[i][j]<< " ";}
                output5 << sub_genotype[i] << " " << pheno[i] << endl;
            }
            /************************************************/
            /*** Now Begin to set up LHS and RHS matrices ***/
            /************************************************/
            SparseMatrix <double> X_hap(pheno.size(),ROH_haplotypes.size());
            /* Set up X_hap and non_roh is zeroed out */
            for(int i = 0; i < pheno.size(); i++){if(sub_genotype[i] > 0){X_hap.insert(i,(sub_genotype[i]-1)) = 1;}}
            /* LHS */
            /* Row 1 */
            SparseMatrix<double> X_subtX_hapa(X_sub.cols(),X_hap.cols()); X_subtX_hapa = X_sub.transpose() * X_hap;
            /* Row2 */
            SparseMatrix<double> X_haptX_suba(X_hap.cols(),X_sub.cols()); X_haptX_suba = X_hap.transpose() * X_sub;
            SparseMatrix<double> XtX_hapa(X_hap.cols(),X_hap.cols()); XtX_hapa = X_hap.transpose() * X_hap;
            SparseMatrix<double> X_haptZa(X_hap.cols(),Z.cols()); X_haptZa = X_hap.transpose() * Z;
            SparseMatrix<double> X_haptWa(X_hap.cols(),Z.cols());
            if(perm_effect == "yes"){X_haptWa = X_hap.transpose() * Z;}
            /* Row3 */
            SparseMatrix <double> ZtXhapa(Z.cols(),X_hap.cols()); ZtXhapa = Z.transpose() * X_hap;
            /* Row4 */
            SparseMatrix <double> WtXhapa(Z.cols(),X_hap.cols());
            if(perm_effect == "yes"){WtXhapa = Z.transpose() * X_hap;}
            /* Can't combine sparse matrices together so convert to dense combine then convert LHS back to sparse */
            int rows_lhs;
            if(perm_effect == "yes"){rows_lhs = X_subtX_suba.rows()+X_haptX_suba.rows()+ZtX_suba.rows()+WtX_suba.rows();}
            if(perm_effect == "no"){rows_lhs = X_subtX_suba.rows()+X_haptX_suba.rows()+ZtX_suba.rows();}
            MatrixXd LHS(rows_lhs,rows_lhs);
            if(perm_effect == "yes")
            {
                /* Row 1 */
                MatrixXd X_subtX_sub(X_subtX_suba.rows(),X_subtX_suba.cols()); X_subtX_sub = X_subtX_suba;
                MatrixXd X_subtX_hap(X_subtX_hapa.rows(),X_subtX_hapa.cols()); X_subtX_hap = X_subtX_hapa;
                MatrixXd X_subtZ(X_subtZa.rows(),X_subtZa.cols()); X_subtZ=X_subtZa;
                MatrixXd X_subtW(X_subtWa.rows(),X_subtWa.cols()); X_subtW=X_subtWa;
                /* Row 2 */
                MatrixXd X_haptX_sub(X_haptX_suba.rows(),X_haptX_suba.cols()); X_haptX_sub = X_haptX_suba;
                MatrixXd XtX_hap(XtX_hapa.rows(),XtX_hapa.cols()); XtX_hap = XtX_hapa;
                MatrixXd X_haptZ(X_haptZa.rows(),X_haptZa.cols()); X_haptZ = X_haptZa;
                MatrixXd X_haptW(X_haptWa.rows(),X_haptWa.cols()); X_haptW = X_haptWa;
                /* Row 3 */
                MatrixXd ZtX_sub(ZtX_suba.rows(),ZtX_suba.cols()); ZtX_sub=ZtX_suba;
                MatrixXd ZtXhap(ZtXhapa.rows(),ZtXhapa.cols()); ZtXhap=ZtXhapa;
                MatrixXd ZtZ(ZtZa.rows(),ZtZa.cols()); ZtZ=ZtZa;
                MatrixXd ZtW(ZtWa.rows(),ZtWa.cols()); ZtW=ZtWa;
                /* Row 4 */
                MatrixXd WtX_sub(WtX_suba.rows(),WtX_suba.cols()); WtX_sub=WtX_suba;
                MatrixXd WtXhap(WtXhapa.rows(),WtXhapa.cols()); WtXhap=WtXhapa;
                MatrixXd WtZ(WtZa.rows(),WtZa.cols()); WtZ=WtZa;
                MatrixXd WtW(WtWa.rows(),WtWa.cols()); WtW=WtWa;
                /* Make LHS */
                LHS << X_subtX_sub,X_subtX_hap,X_subtZ,X_subtW,
                X_haptX_sub,XtX_hap,X_haptZ,X_haptW,
                ZtX_sub,ZtXhap,ZtZ,ZtW,
                WtX_sub,WtXhap,WtZ,WtW;
                /* remove all to save on storage most will be sparse */
                X_subtX_sub.resize(0,0); X_subtX_hap.resize(0,0); X_subtZ.resize(0,0); X_subtW.resize(0,0);
                X_haptX_sub.resize(0,0); XtX_hap.resize(0,0); X_haptZ.resize(0,0); X_haptW.resize(0,0);
                ZtX_sub.resize(0,0); ZtXhap.resize(0,0); ZtZ.resize(0,0); ZtW.resize(0,0);
                WtX_sub.resize(0,0); WtXhap.resize(0,0); WtZ.resize(0,0); WtW.resize(0,0);
            }
            if(perm_effect == "no")
            {
                /* Row 1 */
                MatrixXd X_subtX_sub(X_subtX_suba.rows(),X_subtX_suba.cols()); X_subtX_sub = X_subtX_suba;
                MatrixXd X_subtX_hap(X_subtX_hapa.rows(),X_subtX_hapa.cols()); X_subtX_hap = X_subtX_hapa;
                MatrixXd X_subtZ(X_subtZa.rows(),X_subtZa.cols()); X_subtZ=X_subtZa;
                /* Row 2 */
                MatrixXd X_haptX_sub(X_haptX_suba.rows(),X_haptX_suba.cols()); X_haptX_sub = X_haptX_suba;
                MatrixXd XtX_hap(XtX_hapa.rows(),XtX_hapa.cols()); XtX_hap = XtX_hapa;
                MatrixXd X_haptZ(X_haptZa.rows(),X_haptZa.cols()); X_haptZ = X_haptZa;
                /* Row 3 */
                MatrixXd ZtX_sub(ZtX_suba.rows(),ZtX_suba.cols()); ZtX_sub=ZtX_suba;
                MatrixXd ZtXhap(ZtXhapa.rows(),ZtXhapa.cols()); ZtXhap=ZtXhapa;
                MatrixXd ZtZ(ZtZa.rows(),ZtZa.cols()); ZtZ=ZtZa;
                /* Make LHS */
                LHS << X_subtX_sub,X_subtX_hap,X_subtZ,
                X_haptX_sub,XtX_hap,X_haptZ,
                ZtX_sub,ZtXhap,ZtZ;
                /* remove all to save on storage most will be sparse */
                X_subtX_sub.resize(0,0); X_subtX_hap.resize(0,0); X_subtZ.resize(0,0);
                X_haptX_sub.resize(0,0); XtX_hap.resize(0,0); X_haptZ.resize(0,0);
                ZtX_sub.resize(0,0); ZtXhap.resize(0,0); ZtZ.resize(0,0);
            }
            //logfile << "                - (LHS: " << LHS.rows() << "-" << LHS.cols() << ")"<< endl;
            /*******/
            /* RHS */
            /*******/
            MatrixXd RHS(rows_lhs,1);
            MatrixXd X_hapty(X_hap.cols(),1);
            X_hapty = X_hap.transpose() * y;
            if(perm_effect == "yes")
            {
                RHS << X_subty,
                X_hapty,
                Zty,
                Wty;
            }
            if(perm_effect == "no")
            {
                RHS << X_subty,
                X_hapty,
                Zty;
            }
            X_hapty.resize(0,0); /* clear X_hapty size */
            //logfile << "                - (RHS: " << RHS.rows() << "-" << RHS.cols() << ")" << endl;
            /*******************************************/
            /*** MKL's Cholesky Decomposition of LHS ***/
            /*******************************************/
            MatrixXd LHSinv(LHS.rows(),LHS.cols());
            int N = (int)LHS.cols();
            unsigned long i_p = 0, j_p = 0;
            unsigned long n = LHS.cols();
            long long int info = 0;
            const long long int int_n =(int)n;
            char lower='L';
            float* Vi_mkl=new float[n*n];
            /* Copy it to a 2-dim array that is dynamically stored that all the computations will be on */
            #pragma omp parallel for private(j_p)
            for(i_p=0; i_p<n; i_p++)
            {
                for(j_p=0; j_p<n; j_p++)
                {
                    Vi_mkl[i_p*n+j_p]=LHS(i_p,j_p);
                }
            }
            //dpotrf(&lower, &int_n, Vi_mkl, &int_n, &info);          /* Calculate upper triangular L matrix */
            //dpotri(&lower, &int_n, Vi_mkl, &int_n, &info);          /* Calculate inverse of lower triangular matrix result is the inverse */
            spotrf(&lower, &int_n, Vi_mkl, &int_n, &info);          /* Calculate upper triangular L matrix */
            spotri(&lower, &int_n, Vi_mkl, &int_n, &info);          /* Calculate inverse of lower triangular matrix result is the inverse */
            /* Copy upper triangler part to lower traingular part and then you have the inverse ! */
            #pragma omp parallel for private(j_p)
            for(j_p=0; j_p<n; j_p++)
            {
                for(i_p=0; i_p<=j_p; i_p++){LHSinv(i_p,j_p) = LHSinv(j_p,i_p) = Vi_mkl[i_p*n+j_p];}
            }
            // free memory
            delete[] Vi_mkl;
            /***********************/
            /*** Get Beta and SE ***/
            /***********************/
            MatrixXd b(LHS.cols(),1);
            MatrixXd b_se(LHS.cols(),1);
            b = LHSinv * RHS;
            /* Scale LHS*/
            #pragma omp parallel for private(j_p)
            for(j_p=0; j_p<n; j_p++)
            {
                for(i_p=0; i_p<=j_p; i_p++){LHSinv(i_p,j_p) = LHSinv(j_p,i_p) = LHSinv(j_p,i_p) * res_var;}
            }
            //fstream checktest; checktest.open("CheckLHSa", std::fstream::out | std::fstream::trunc); checktest.close();
            //std::ofstream output8("CheckLHSa", std::ios_base::app | std::ios_base::out);
            //for(int i = 0; i < LHS.rows(); i++)
            //{
            //    for(int j = 0; j < LHS.cols(); j++)
            //    {
            //        if(j != LHS.cols() -1){output8 << LHS(i,j) << " ";}
            //        if(j == LHS.cols() -1){output8 << LHS(i,j);}
            //    }
            //    output8 << endl;
            //}
            //fstream checktesta; checktesta.open("CheckRHSa", std::fstream::out | std::fstream::trunc); checktesta.close();
            //std::ofstream output9("CheckRHSa", std::ios_base::app | std::ios_base::out);
            //for(int i = 0; i < RHS.rows(); i++)
            //{
            //    output9 << RHS(i,0) << endl;
            //}
            //for(int i = 0; i < 20; i++)
            //{
            //    cout << b(i,0) << " ";
            //}
            //cout << endl;
            /********************************************************/
            /* Make it so beta has solutions for zero'd out effects */
            /********************************************************/
            int numberzeroed = 0;
            for(int i = 0; i < keepremove.size(); i++){if(keepremove[i] == 0){numberzeroed++;}}
            /* need to add in zero'd out columns to LHSinv and b used for contrasts*/
            /* full matrix is size LHS + number of zeroed out plus one for haplotype */
            MatrixXd b_full((LHS.cols() + numberzeroed + 1),1);
            MatrixXd LHSinv_full((LHS.cols() + numberzeroed + 1),(LHS.cols() + numberzeroed + 1));
            for(int i = 0; i < LHSinv_full.rows(); i++)
            {
                b_full(i,0) = 0;
                for(int j = 0; j < LHSinv_full.cols(); j++){LHSinv_full(i,j) = 0;}
            }
            /* Create vector in order to set up L more effectively */
            vector < string > factor(b_full.rows(),"Random"); /* Initialize to Random all fixed will be changed */
            /* which columns to add zeros too */
            vector < int > zero_columns;
            /* intercept */
            int i_r = 0;        /* which row are you at in reduced b and LHSinv */
            int i_f = 0;        /* which row are you at in reduced b and LHSinv */
            /* first one is intercept */
            b_full(i_f,0) = b(i_r,0); factor[i_f ] = "int"; i_f++; i_r++;
            /* now loop through fixed effect that are consistent across models */
            for(int j = 0; j < FIXED_CLASS[0].size(); j++)
            {
                /* first figure out number of ones zeroed out */
                int startat = i_f;
                for(int checknum = startat; checknum < (startat + uniqueclass[j].size()); checknum++)
                {
                    if(keepremove[checknum] == 0)
                    {
                        b_full(i_f,0) = 0; zero_columns.push_back(checknum);
                    }
                    if(keepremove[checknum] != 0)
                    {
                        b_full(i_f) = b(i_r,0); i_r++;
                    }
                    stringstream ss; ss << (j + 1); string str = ss.str();
                    factor[i_f] = "Fixed_Class" + str; i_f++;
                }
            }
            /* now loop through fixed covariate effects just copy */
            for(int j = 0; j < FIXED_COV[0].size(); j++)
            {
                b_full(i_f) = b(i_r,0);
                stringstream ss; ss << (j + 1); string str = ss.str();
                factor[i_f] = "Cov_Class" + str; i_f++; i_r++;
            }
            /* now loop through haplotype effects just copy */
            b_full(i_f,0) = 0; factor[i_f] = "haplotype"; zero_columns.push_back(i_f); i_f++;
            for(int j = 0; j < ROH_haplotypes.size(); j++)
            {
                b_full(i_f) = b(i_r,0); factor[i_f] = "haplotype"; i_f++; i_r++;
            }
            if(perm_effect == "yes"){b_full.block(i_f,0,(2*ZtZa.cols()),1) = b.block(i_r,0,(2*ZtZa.cols()),1);}
            if(perm_effect == "no"){b_full.block(i_f,0,(ZtZa.cols()),1) = b.block(i_r,0,(ZtZa.cols()),1);}
            //cout << zero_columns.size() << endl;
            b.resize(0,0);
            //for(int i = 0; i < zero_columns.size(); i++){cout << zero_columns[i] << " " << b_full(zero_columns[i],0) << endl;}
            //for(int i = 0; i < i_f+5; i++){cout << b_full(i,0) << " " << factor[i] << endl;}
            /********************************************************/
            /* Make it so LHSinv has solutions for zero'd out effects */
            /********************************************************/
            int where_at_in_reduced_i = 0; int where_at_in_zerocolumns_i = 0;
            for(int i = 0; i < LHSinv_full.rows(); i++)
            {
                if(i != zero_columns[where_at_in_zerocolumns_i])
                {
                    int where_at_in_reduced_j = 0; int where_at_in_zerocolumns_j = 0;
                    for(int j = 0; j < LHSinv_full.cols(); j++)
                    {
                        if(j != zero_columns[where_at_in_zerocolumns_j])
                        {
                            LHSinv_full(i,j) = LHSinv(where_at_in_reduced_i,where_at_in_reduced_j);
                            where_at_in_reduced_j++;
                        }
                        if(j == zero_columns[where_at_in_zerocolumns_j]){where_at_in_zerocolumns_j++;}
                    }
                    where_at_in_reduced_i++;
                }
                if(i == zero_columns[where_at_in_zerocolumns_i]){where_at_in_zerocolumns_i++;}
            }
            LHSinv.resize(0,0);
            /* Fill beta estimate of roh effect */
            int current_haplotype = 0;
            for(int i = 0; i < b_full.rows(); i++)
            {
                if(factor[i] == "haplotype")
                {
                    while(1)
                    {
                        if(current_haplotype == 0){current_haplotype++; break;}
                        if(current_haplotype > 0){effect_beta.push_back(b_full(i,0)); current_haplotype++; break;}
                    }
                }
            }
            vector < double > LSM(ROH_haplotypes.size(),0);
            vector < double > T_stat(ROH_haplotypes.size(),0);
            /* First get least square mean for each ROH haplotypes */
            for(int j = 0; j < (ROH_haplotypes.size()+1); j++)
            {
                MatrixXd Lvec(1,b_full.rows());
                int current_haplotype = 0;
                for(int i = 0; i < b_full.rows(); i++)
                {
                    if(factor[i] == "int"){Lvec(0,i) = 1;}
                    for(int k = 0; k < FIXED_CLASS[0].size(); k++)
                    {
                        stringstream ss; ss << (k + 1); string str = ss.str();
                        string lookup = "Fixed_Class" + str;
                        if(factor[i] == lookup){Lvec(0,i) = 1 / double(uniqueclass[k].size());}
                    }
                    for(int k = 0; k < FIXED_COV[0].size(); k++)
                    {
                        stringstream ss; ss << (k + 1); string str = ss.str();
                        string lookup = "Cov_Class" + str;
                        if(factor[i] == lookup){Lvec(0,i) = MeanPerCovClass[k];}
                        
                    }
                    if(factor[i] == "haplotype")
                    {
                        while(1)
                        {
                            if(current_haplotype == j){Lvec(0,i) = 1; current_haplotype++; break;}
                            if(current_haplotype != j){Lvec(0,i) = 0; current_haplotype++; break;}
                        }
                    }
                    if(factor[i] == "Random"){Lvec(0,i) = 0;}
                }
                if(j > 0){double temp = (Lvec * b_full).value(); LSM[j-1] = temp;}
            }
            for(int j = 1; j < (ROH_haplotypes.size()+1); j++)
            {
                MatrixXd Lvec(2,b_full.rows());
                int current_haplotype = 1; int baseline = 0;
                for(int i = 0; i < b_full.rows(); i++)
                {
                    if(factor[i] == "int"){Lvec(0,i) = 1;Lvec(1,i) = Lvec(0,i) ;}
                    for(int k = 0; k < FIXED_CLASS[0].size(); k++)
                    {
                        stringstream ss; ss << (k + 1); string str = ss.str();
                        string lookup = "Fixed_Class" + str;
                        if(factor[i] == lookup)
                        {
                            Lvec(0,i) = 1 / double(uniqueclass[k].size()); Lvec(1,i) = Lvec(0,i);
                        }
                    }
                    for(int k = 0; k < FIXED_COV[0].size(); k++)
                    {
                        stringstream ss; ss << (k + 1); string str = ss.str();
                        string lookup = "Cov_Class" + str;
                        if(factor[i] == lookup)
                        {
                            Lvec(0,i) = MeanPerCovClass[k]; Lvec(1,i) = Lvec(0,i);
                        }
                    }
                    if(factor[i] == "haplotype" && baseline > 0)
                    {
                        while(1)
                        {
                            if(current_haplotype == j){Lvec(0,i) = 0; Lvec(1,i) = -1.0; current_haplotype++; break;}
                            if(current_haplotype != j){Lvec(0,i) = 0; Lvec(1,i) = 0; current_haplotype++; break;}
                        }
                    }
                    if(factor[i] == "haplotype" && baseline == 0){Lvec(0,i) = 1; Lvec(1,i) = 0.0; baseline++;}
                    if(factor[i] == "Random"){Lvec(0,i) = 0; Lvec(1,i) = 0;}
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
            //for(int i = 0; i < ROH_haplotypes.size(); i++)
            //{
            //    cout << regions[loopregions].getHaplotype_R() << " " << ROH_haplotypes[i] << " " << T_stat[i] << " " << effect_beta[i] << endl;
            //}
            //cout << endl;
            /* Find one that matches up with region ROH haplotype */
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
            //{
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
            time_t fullb_end_time = time(0);
            if(loopregions == 0){logfile << "               - Takes " << difftime(fullb_end_time,fullb_begin_time) << " seconds per window." << endl;}
            if(loopregions % 5 == 0){logfile << "               - " << loopregions << endl;}
        }
        end_time = time(0);
        logfile << "            - Finished step 2 with full model (Took: ";
        logfile << difftime(end_time,begin_time) << " seconds)." << endl;
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
        logfile << "            - Number of regions prior to removing ones with t-value below |2.326|: " << regions.size() << endl;
        //for(int i = 0; i < regions.size(); i++)
        //{
        //    cout << regions[i].getChr_R() << " " << regions[i].getStPos_R() << " " << regions[i].getEnPos_R() << " ";
        //    cout << regions[i].getStartIndex_R() << " " << regions[i].getEndIndex_R() << " " << regions[i].getHaplotype_R() << " ";
        //    cout << regions[i].getRawPheno_R()<< " " <<regions[i].getEffect() << " " << regions[i].getLSM_R() << " " << regions[i].gettval() << endl;
        //}
        //cout << endl << endl;
        ROWS = regions.size(); i = 0;
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
        logfile << "            - Number of regions after removing ones with t-value below |2.326|: " << regions.size() << endl;
        logfile << "            - Begin the final stage to remove nested haplotypes (minimizes double counting)." << endl;
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
        logfile << "            - Number of regions after final stage of editing: " << regions.size() << endl;
        //for(int i = 0; i < regions.size(); i++)
        //{
        //    cout << regions[i].getChr_R() << " " << regions[i].getStPos_R() << " " << regions[i].getEnPos_R() << " ";
        //    cout << regions[i].getStartIndex_R() << " " << regions[i].getEndIndex_R() << " " << regions[i].getHaplotype_R() << " ";
        //    cout << regions[i].getRawPheno_R() << " " << regions[i].getEffect() << " " << regions[i].getLSM_R() << " " << regions[i].gettval() << endl;
        //}
        //cout << endl << endl;
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
        time_t chr_end_time = time(0);
        logfile << "        - Finished Chromosome: " << chromo + 1 << " (Took: " << difftime(chr_end_time,chr_begin_time) << " seconds)" << endl;
    }
    time_t loop_end_time = time(0);
    logfile << "    - Finished looping across all chromosomes (Took: " << difftime(loop_end_time,loop_begin_time) << " seconds)" << endl;
    time_t full_end_time = time(0);
    logfile << "    - Finished Program (Took: " << difftime(full_end_time,full_begin_time) << " seconds)" << endl;
}
