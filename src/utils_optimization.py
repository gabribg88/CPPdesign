import os
import math
import time
import itertools
import random
import numpy as np
import pandas as pd
from functools import reduce
from collections import defaultdict
from peptides import Peptide
from Bio import Align
from Bio.Align import substitution_matrices
from sklearn.neighbors import LocalOutlierFactor


def load_dataset():
    key = ['ID', 'Sequence', 'CPP', 'Dataset']

    features = ['physicochemical',
                'AAC',
                'CKSAAP type 1',
                'TPC type 1',
                'DPC type 1',
                'DDE',
                'GAAC',
                'CKSAAGP type 1',
                'GDPC type 1',
                'GTPC type 1',
                'Moran',
                'Geary',
                'NMBroto',
                'CTDC',
                'CTDT',
                'CTDD',
                'CTriad',
                'KSCTriad',
                'SOCNumber',
                'QSOrder',
                'PAAC',
                'APAAC',
                'ASDC',
                'AC',
                'CC',
                'ACC',
                'EAAC',
                'EGAAC',
                'AAIndex',
                'BLOSUM62',
                'ZScale'
               ]
    dfList = []
    features_dict = {}

    for feature in features:
        tmp = pd.read_pickle(os.path.join('..', 'features', 'comb_' + feature.split(' ')[0] + '.pickle'))
        features_dict[feature] = list(set(tmp.columns.tolist()) - set(key))
        dfList.append(tmp)

    comb = reduce(lambda df1, df2: pd.merge(df1, df2, on=['ID', 'Sequence', 'CPP', 'Dataset']), dfList)

    train = comb[comb.Dataset=='train'].copy()
    test  = comb[comb.Dataset=='test'].copy()

    feature_names = sorted(list(itertools.chain.from_iterable(list(features_dict.values()))))
    return comb, feature_names

class Featurizer():
    def __init__(self, data_folder:str):
        self.data_folder = data_folder
        self.feature_names = ['CTDC_charge.G1',
                             'CTDD_hydrophobicity_ARGP820101.1.residue50',
                             'SOCNumber_gGrantham.lag1',
                             'DDE_LA',
                             'Geary_ANDN920101.lag2',
                             'DDE_SP',
                             'NetC',
                             'ASDC_VG',
                             'CTDC_solventaccess.G3',
                             'APAAC_Pc1.V']
        
    def Count(self, seq1, seq2):
        sum = 0
        for aa in seq1:
            sum = sum + seq2.count(aa)
        return sum

    def Count1(self, aaSet, sequence):
        number = 0
        for aa in sequence:
            if aa in aaSet:
                number = number + 1
        cutoffNums = [1, math.floor(0.25 * number), math.floor(0.50 * number), math.floor(0.75 * number), number]
        cutoffNums = [i if i >= 1 else 1 for i in cutoffNums]

        code = []
        for cutoff in cutoffNums:
            myCount = 0
            for i in range(len(sequence)):
                if sequence[i] in aaSet:
                    myCount += 1
                    if myCount == cutoff:
                        code.append((i + 1) / len(sequence) * 100)
                        break
            if myCount == 0:
                code.append(0)
        return code
    
    def compute_CTDC_charge_G1(self, sequence):
        return self.Count('KR', sequence) / len(sequence)
    
    def compute_CTDD_hydrophobicity_ARGP820101_1_residue50(self, sequence):
        return self.Count1('QSTNGDE', sequence)[2]
    
    def compute_SOCNumber_gGrantham_lag1(self, sequence):
        n=1
        dataFile1 = os.path.join(self.data_folder, 'Grantham.txt')
        AA1 = 'ARNDCQEGHILKMFPSTWYV'
        DictAA1 = {}
        for i in range(len(AA1)):
            DictAA1[AA1[i]] = i
        with open(dataFile1) as f:
            records = f.readlines()[1:]
        AADistance1 = []
        for i in records:
            array = i.rstrip().split()[1:] if i.rstrip() != '' else None
            AADistance1.append(array)
        AADistance1 = np.array([float(AADistance1[i][j]) for i in range(len(AADistance1)) for j in range(len(AADistance1[i]))]).reshape((20, 20))
        return sum([AADistance1[DictAA1[sequence[j]]][DictAA1[sequence[j + n]]] ** 2 for j in range(len(sequence) - n)]) / (len(sequence) - n)
    
    def compute_DDE_LA(self, sequence):
        AA = 'ACDEFGHIKLMNPQRSTVWY'
        myCodons = {'A': 4, 'C': 2, 'D': 2, 'E': 2, 'F': 2, 'G': 4, 'H': 2, 'I': 3, 'K': 2, 'L': 6,
                    'M': 1, 'N': 2, 'P': 4, 'Q': 2, 'R': 6, 'S': 6, 'T': 4, 'V': 4, 'W': 1, 'Y': 2
                    }
        N_LA = 0
        for i in range(len(sequence)-1):
            if sequence[i:i+2] == 'LA':
                N_LA += 1
        Dc = N_LA / (len(sequence) - 1)
        Tm = (myCodons['L'] / 61) * (myCodons['A'] / 61)
        Tv = (Tm * (1 - Tm)) / (len(sequence) - 1)
        return (Dc - Tm) / math.sqrt(Tv)
    
    def compute_Geary_ANDN920101_lag2(self, sequence):
        props = 'ANDN920101'.split(';')
        nlag = 2
        fileAAidx = os.path.join(self.data_folder, 'AAidx.txt')
        AA = 'ARNDCQEGHILKMFPSTWYV'
        with open(fileAAidx) as f:
            records = f.readlines()[1:]
        myDict = {}
        for i in records:
            array = i.rstrip().split('\t')
            myDict[array[0]] = array[1:]
        AAidx = []
        AAidxName = []
        for i in props:
            if i in myDict:
                AAidx.append(myDict[i])
                AAidxName.append(i)
            else:
                print('"' + i + '" properties not exist.')
        AAidx1 = np.array([float(j) for i in AAidx for j in i])
        AAidx = AAidx1.reshape((len(AAidx), 20))
        propMean = np.mean(AAidx, axis=1)
        propStd = np.std(AAidx, axis=1)
        for i in range(len(AAidx)):
            for j in range(len(AAidx[i])):
                AAidx[i][j] = (AAidx[i][j] - propMean[i]) / propStd[i]
        index = {}
        for i in range(len(AA)):
            index[AA[i]] = i

        N = len(sequence)
        for prop in range(len(props)):
            xmean = sum([AAidx[prop][index[aa]] for aa in sequence]) / N
            for n in range(1, nlag + 1):
                if len(sequence) > nlag:
                    # if key is '-', then the value is 0
                    rn = (N - 1) / (2 * (N - n)) * ((sum(
                        [(AAidx[prop][index.get(sequence[j], 0)] - AAidx[prop][index.get(sequence[j + n], 0)]) ** 2
                        for
                        j in range(len(sequence) - n)])) / (sum(
                        [(AAidx[prop][index.get(sequence[j], 0)] - xmean) ** 2 for j in range(len(sequence))])))
                else:
                    rn = 'NA'
        return rn
    
    def compute_DDE_SP(self, sequence):
        AA = 'ACDEFGHIKLMNPQRSTVWY'
        myCodons = {'A': 4, 'C': 2, 'D': 2, 'E': 2, 'F': 2, 'G': 4, 'H': 2, 'I': 3, 'K': 2, 'L': 6,
                    'M': 1, 'N': 2, 'P': 4, 'Q': 2, 'R': 6, 'S': 6, 'T': 4, 'V': 4, 'W': 1, 'Y': 2
                    }
        N_SP = 0
        for i in range(len(sequence)-1):
            if sequence[i:i+2] == 'SP':
                N_SP += 1
        Dc = N_SP / (len(sequence) - 1)
        Tm = (myCodons['S'] / 61) * (myCodons['P'] / 61)
        Tv = (Tm * (1 - Tm)) / (len(sequence) - 1)
        return (Dc - Tm) / math.sqrt(Tv)
    
    def compute_NetC(self, sequence):
        return Peptide(sequence).charge(pH=7.4)
    
    def compute_ASDC_VG(self, sequence):
        AA = 'ACDEFGHIKLMNPQRSTVWY'
        VG_counter = 0
        sum = 0
        for j in range(len(sequence)):
            for k in range(j + 1, len(sequence)):
                if sequence[j] in AA and sequence[k] in AA:
                    if sequence[j] + sequence[k] == 'VG':
                        VG_counter += 1
                sum += 1
        return VG_counter / sum
    
    def compute_CTDC_solventaccess_G3(self, sequence):
        return 1 - (self.Count('ALFCGIVW', sequence) / len(sequence)) - (self.Count('RKQEND', sequence) / len(sequence))
    
    def compute_APAAC_Pc1_V(self, sequence):
        lambdaValue = 1
        w = 0.05
        dataFile = os.path.join(self.data_folder, 'PAAC.txt')
        with open(dataFile) as f:
            records = f.readlines()
        AA = ''.join(records[0].rstrip().split()[1:])
        AADict = {}
        for i in range(len(AA)):
            AADict[AA[i]] = i
        AAProperty = []
        AAPropertyNames = []
        for i in range(1, len(records) - 1):
            array = records[i].rstrip().split() if records[i].rstrip() != '' else None
            AAProperty.append([float(j) for j in array[1:]])
            AAPropertyNames.append(array[0])
        AAProperty1 = []
        for i in AAProperty:
            meanI = sum(i) / 20
            fenmu = math.sqrt(sum([(j - meanI) ** 2 for j in i]) / 20)
            AAProperty1.append([(j - meanI) / fenmu for j in i])
        theta = []
        for n in range(1, lambdaValue + 1):
            for j in range(len(AAProperty1)):
                theta.append(
                    sum([AAProperty1[j][AADict[sequence[k]]] * AAProperty1[j][AADict[sequence[k + n]]] for k in
                        range(len(sequence) - n)]) / (len(sequence) - n))
        return sequence.count('V') / (1 + w * sum(theta))
        
    def compute_features(self, sequence):
        features = pd.DataFrame(data=0.0, index=range(1), columns=self.feature_names)
        features['CTDC_charge.G1'] = self.compute_CTDC_charge_G1(sequence)
        features['CTDD_hydrophobicity_ARGP820101.1.residue50'] = self.compute_CTDD_hydrophobicity_ARGP820101_1_residue50(sequence)
        features['SOCNumber_gGrantham.lag1'] = self.compute_SOCNumber_gGrantham_lag1(sequence)
        features['DDE_LA'] = self.compute_DDE_LA(sequence)
        features['Geary_ANDN920101.lag2'] = self.compute_Geary_ANDN920101_lag2(sequence)
        features['DDE_SP'] = self.compute_DDE_SP(sequence)
        features['NetC'] = self.compute_NetC(sequence)
        features['ASDC_VG'] = self.compute_ASDC_VG(sequence)
        features['CTDC_solventaccess.G3'] = self.compute_CTDC_solventaccess_G3(sequence)
        features['APAAC_Pc1.V'] = self.compute_APAAC_Pc1_V(sequence)
        return features
    
    
# Function to predict penetration
def pred_penetration(features_query, models, best_iteration, models_number):
    proba_p = 0.0
    for model in models:
        proba_p += model.predict(features_query, num_iteration=best_iteration) / models_number
    return proba_p[0]

# Function to calculate anomaly score
def anomaly_score(features_query, clf_anomaly):
    pos_score = -clf_anomaly.score_samples(features_query)
    return pos_score

# Function to compute distance
def compute_distance(aligner, TARGET, query):
    weigth = aligner.align(TARGET, TARGET)
    alignments = aligner.align(TARGET, query)
    distance = 1 - alignments.score / weigth.score
    return distance

# Function to compute fitness score
def my_fitness(query, params_fit):
    models = params_fit['model']
    featurizer = params_fit['featurizer']
    best_iteration = params_fit['best_iteration']
    n_models = params_fit['n_models']
    
    features_query = featurizer.compute_features(query)
    proba_p = pred_penetration(features_query, models['models'].boosters, best_iteration, n_models)

    clf_anomaly = params_fit['clf_anomaly']
    pos_score = anomaly_score(features_query, clf_anomaly)[0]

    TARGET = params_fit['target_ligand']
    aligner = params_fit['aligner']
    distance = compute_distance(aligner, TARGET, query)

    num_diff_residues = sum(1 for q, t in zip(query, TARGET) if q != t)

    a = 1
    b = 1
    x_s = 1.5
    anom_scor = np.max((b*(pos_score**2-x_s**2), 0))
    fitness = distance  + 1*np.max( ( a*(1 - proba_p)**2-a*(0.2)**2, 0)) + 1*anom_scor
    return fitness,num_diff_residues

# Class defining an Individual
class Individual():
    def __init__(self, chromosome, params_fit={}):
        self.chromosome = chromosome
        self.params_fit = params_fit
        self.fitness, self.num_diff_residues = self.comp_fitness(params_fit)

    # Compute fitness according to my_fitness function
    def comp_fitness(self, params_fit):
        my_fun = params_fit['obj']
        fitness, num_diff_residues = my_fun(''.join(self.chromosome), params_fit)
        return fitness, num_diff_residues

    # Update the chromosome of the individual
    def update_chromosome(self, chromosome):
        self.chromosome = chromosome

    # Method to create random genes
    @staticmethod
    def mutated_genes(genes):
        gene = random.choice(genes)
        return gene

        # Method to create chromosome or string of genes

    @staticmethod
    def create_gnome(len_gnome, genes, target, max_diff_pct):
        num_diff = int(len_gnome * max_diff_pct / 100)  # calculate the number of different letters allowed
        gnome = list(target)  # start from the target sequence
        for _ in range(num_diff):
            idx = random.randint(0, len_gnome - 1)  # select a random position
            new_gene = Individual.mutated_genes(genes)  # generate a new gene
            while new_gene == gnome[idx]:  # ensure the new gene is different from the existing one
                new_gene = Individual.mutated_genes(genes)
            gnome[idx] = new_gene  # replace the gene at the selected position
        return gnome
        # Perform mating and produce new offspring
    def mate(self, par2, genes):
            child_chromosome = []
            for gp1, gp2 in zip(self.chromosome, par2.chromosome):
                prob = random.random()

                if prob < 0.45:
                    child_chromosome.append(gp1)
                elif prob < 0.90:
                    child_chromosome.append(gp2)
                else:
                    child_chromosome.append(Individual.mutated_genes(genes))
            return Individual(child_chromosome, self.params_fit)