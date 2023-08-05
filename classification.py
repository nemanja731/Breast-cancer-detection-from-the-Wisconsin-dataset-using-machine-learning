import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

#ucitavanje baze
def load_data():
    dataset = pd.read_csv('data.csv')
    #izbacivanje kolona koje nisu od znacaja (kolona 'id' i kolona 'Unnamed: 32')
    dataset = dataset.drop(columns=['id', 'Unnamed: 32'])
    dataset['diagnosis'] = dataset['diagnosis'].replace('B', 0).replace('M', 1)
    #print('Velicina baze: ', dataset.shape)
    #print('Prikaz prvih 10 redova baze:\n', dataset.head(10))
    return dataset

#provera da li ima nedostajucih vrednosti
def missing_values(dataset):
    missing_values_per_column = dataset.isna().sum()
    print("\nNedostajuce vrednosti po kolonama:\n", missing_values_per_column)
    missing_values = missing_values_per_column.sum()
    print("\nBroj nedostajucih vrednosti:", missing_values)

def class_distribution(dataset):
    numMalignant = sum(dataset['diagnosis'] == 'M')
    numBenign = sum(dataset['diagnosis'] == 'B')
    print('Broj uzoraka sa malignim oboljenjem: ', numMalignant)
    print('Broj uzoraka sa belignim oboljenjem: ', numBenign)
    
    #prikaz raspodele klasa
    plt.bar(0, numBenign, color ='lightskyblue', width=0.5, label ='Benigna')
    plt.bar(1, numMalignant, color ='mediumaquamarine', width=0.5, label ='Maligna')
    plt.axis([-0.5, 1.5, 0, 400])
    plt.xticks([0, 1])
    plt.xlabel('Klase')
    plt.ylabel('Broj uzoraka')
    plt.title('Prikaz raspodele klasa', weight='bold')
    plt.legend()
    plt.show()

def show_correlations(mean, se, worst):
    data = [mean, se, worst]
    colors = ["lightskyblue", "mediumaquamarine", "peachpuff"]
    titles = ["srednje vrednosti", "srednje-kvadratne greske", "najvece greske"]
    file_names = ["mean_correlation_map.png", "se_correlation_map.png", "worst_correlation_map.png"]
    for i in range (len(data)):
        correlation = data[i].drop(columns=['diagnosis']).corrwith(data[i].diagnosis)
        correlation.plot(kind='bar', grid=False, color=colors[i])
        plt.title("Korelacija atributa " + titles[i] + " sa dijagnozom", weight='bold')
        plt.savefig(file_names[i])
        plt.show()

def show_color_map(dataset, mean, se, worst):
    data = [dataset, mean, se, worst]
    file_names = ["correlation.png", "mean_correlation.png", "se_correlation.png", "worst_correlation.png"]
    for i in range (len(data)):
        corr = data[i].corr()
        plt.figure(figsize = (20, 20))
        sns.heatmap(corr, cmap = 'coolwarm', annot = True)
        plt.savefig(file_names[i])
        plt.show()

def correlation(dataset):
    #kreiranje korelacija
    mean = dataset[dataset.columns[:11]]
    se = dataset.drop(dataset.columns[1:11], axis=1)
    se = se.drop(se.columns[11:], axis=1)
    worst = dataset.drop(dataset.columns[1:21], axis=1)

    # #prikaz korelacija za dijagnozom
    # show_correlations(mean, se, worst)
    # # prikaz korelacija sa dijagnozom kolor mapom
    # show_color_map(dataset, mean, se, worst)

    corr = dataset.corr()
    print(corr[abs(corr['diagnosis']) > 0.5])
    cc = corr[abs(corr['diagnosis']) > 0.5].index
    print('- Broj prediktora kojima je koeficijent korelacije veci od 0.5 = ', len(cc))
    print('--------------------------------------------------')
    print('- Prediktori najkorelisaniji sa izlazom: \n ', cc)

    acc = dataset[dataset.columns[:]].corr()['diagnosis']
    print('Svi prediktori sa odgovarajucim koeficijentom korelacije sa izlazom: \n\n',acc)

    dataset = dataset[['diagnosis', 'radius_mean', 'area_mean',
       'compactness_mean', 'concavity_mean', 'concave points_mean',
       'perimeter_worst', 'area_worst', 'compactness_worst',
       'concavity_worst', 'concave points_worst','texture_mean', 
         'smoothness_mean', 'symmetry_mean','area_se','fractal_dimension_se',
         'texture_worst', 'smoothness_worst', 'symmetry_worst',
       'fractal_dimension_worst']]

    acc = dataset[dataset.columns[:]].corr()['diagnosis']
    print('Izdvojeni prediktori sa odgovarajucim koeficijentom korelacije sa izlazom: \n\n',acc)

if __name__ == '__main__':
    dataset = load_data()
    #missing_values(dataset)
    #class_distribution(dataset)
    correlation(dataset)
