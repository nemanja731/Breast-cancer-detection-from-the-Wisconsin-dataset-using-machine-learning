import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
import xgboost
from sklearn import tree, metrics
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#ucitavanje baze
def load_data():
    dataset = pd.read_csv('data.csv')
    #izbacivanje kolona koje nisu od znacaja (kolona 'id' i kolona 'Unnamed: 32')
    dataset = dataset.drop(columns=['id', 'Unnamed: 32'])
    dataset['diagnosis'] = dataset['diagnosis'].replace('B', 0).replace('M', 1)
    print('Velicina baze: ', dataset.shape)
    print('Prikaz poslednjih 10 redova baze:\n', dataset.tail(10))
    return dataset

#provera da li ima nedostajucih vrednosti
def missing_values(dataset):
    missing_values_per_column = dataset.isna().sum()
    print("\nNedostajuce vrednosti po kolonama:\n", missing_values_per_column)
    missing_values = missing_values_per_column.sum()
    print("\nBroj nedostajucih vrednosti:", missing_values)

def class_distribution(dataset):
    numBenign = sum(dataset['diagnosis'] == 0)
    numMalignant = sum(dataset['diagnosis'] == 1)
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
        seaborn.heatmap(corr, cmap = 'coolwarm', annot = True)
        plt.savefig(file_names[i])
        plt.show()

def correlation(dataset):
    #kreiranje korelacija
    mean = dataset[dataset.columns[:11]]
    se = dataset.drop(dataset.columns[1:11], axis=1)
    se = se.drop(se.columns[11:], axis=1)
    worst = dataset.drop(dataset.columns[1:21], axis=1)
    #prikaz korelacija za dijagnozom
    show_correlations(mean, se, worst)
    # prikaz korelacija sa dijagnozom kolor mapom
    show_color_map(dataset, mean, se, worst)

    corr = dataset.corr()
    dataset = dataset[['diagnosis', 'radius_mean', 'area_mean',
       'compactness_mean', 'concavity_mean', 'concave points_mean',
       'perimeter_worst', 'area_worst', 'compactness_worst',
       'concavity_worst', 'concave points_worst','texture_mean', 
         'smoothness_mean', 'symmetry_mean','area_se','fractal_dimension_se',
         'texture_worst', 'smoothness_worst', 'symmetry_worst',
       'fractal_dimension_worst']]
    new_corr = dataset.corr()

    print(corr[abs(corr['diagnosis']) > 0.5])
    cc = corr[abs(corr['diagnosis']) > 0.5].index
    print('- Broj prediktora kojima je koeficijent korelacije veci od 0.5 = ', len(cc))
    print('--------------------------------------------------')
    print('- Prediktori najkorelisaniji sa izlazom: \n ', cc)

    print('Svi prediktori sa odgovarajucim koeficijentom korelacije sa izlazom: \n\n', corr['diagnosis'])
    print('Izdvojeni prediktori sa odgovarajucim koeficijentom korelacije sa izlazom: \n\n', new_corr['diagnosis'])

    #return dataset

def positive_negative_hist(dataset):
    fig, ax = plt.subplots(ncols = 5, nrows = 6, figsize = (30, 35))
    index = 0
    ax = ax.flatten()
    for col, value in dataset.iloc[:, 1:].items():
        val0 = value.loc[dataset['diagnosis'] == 0]
        val1 = value.loc[dataset['diagnosis'] == 1]
        seaborn.histplot(val0, ax=ax[index], color="lightskyblue", label="100% Equities", kde=True, stat="density", linewidth=0)
        seaborn.histplot(val1, ax=ax[index], color="peachpuff", label="100% Equities", kde=True, stat="density", linewidth=0)
        index += 1
    plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)
    plt.savefig('positive_negative_hist.png')
    plt.show()

def final_atributes(dataset):
    #odabir konacnih obelezja za koriscenje pri obucavanju i testiranju
    dataset = dataset[['diagnosis', 'radius_mean', 'area_mean',
        'compactness_mean', 'concavity_mean', 'concave points_mean',
        'perimeter_worst', 'area_worst', 'compactness_worst',
        'concavity_worst', 'concave points_worst','texture_mean', 
        'smoothness_mean', 'symmetry_mean','area_se','fractal_dimension_se',
        'texture_worst', 'smoothness_worst', 'symmetry_worst',
        'fractal_dimension_worst']]
    return dataset

def divide_train_test(dataset):
    X = dataset.drop(columns = ['diagnosis'])
    y = dataset['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, stratify=y, shuffle=True, random_state = 0)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test, y_train, y_test

def plot_confusion_matrix(prediction_lr, y_test, name):
    confusion_matrix = metrics.confusion_matrix(y_test, prediction_lr)
    cm_display = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
    cm_display.plot()
    plt.show()
    print('Tacnost modela ' + name + ' iznosi: {:.3f}%'.format(accuracy_score(y_test,prediction_lr)*100))
    print('Balansirana tacnost modela ' + name + ' iznosi: {:.3f}%'.format(f1_score(y_test,prediction_lr)*100))

def classify(classifier, name, X_train, X_test, y_train, y_test):
    classifier.fit(X_train, y_train)
    prediction = classifier.predict(X_test)
    print(classification_report(y_test, prediction, target_names = ['Maligna masa: 0','Benigna masa: 1']))
    plot_confusion_matrix(prediction, y_test, name)

def decision_tree(X_train, X_test, y_train, y_test, dataset):
    classifier = DecisionTreeClassifier()
    classify(classifier, "stabla odlucivanja", X_train, X_test, y_train, y_test) 
    classifier.fit(X_train, y_train)
    #prikaz stabla
    feature_names =[l for l in list(dataset.columns)[1:]]
    class_names = ['Benign', 'Malignant']
    fig = plt.figure(figsize=(30, 20))
    _ = tree.plot_tree(classifier, feature_names = feature_names, class_names=class_names, filled=True, fontsize=8)
    plt.savefig('tree.png')
    plt.show() 

def logistic_regression(X_train, X_test, y_train, y_test):
    classifier = LogisticRegression()
    classify(classifier, "logisticke regresije", X_train, X_test, y_train, y_test)

def random_forest(X_train, X_test, y_train, y_test):
    classifier = RandomForestClassifier()
    classify(classifier, "slucajne sume", X_train, X_test, y_train, y_test)

def XGBoost(X_train, X_test, y_train, y_test):
    classifier = xgboost.XGBClassifier(n_estimators=350, subsample=0.8, max_depth=7, eval_metric = 'logloss', use_label_encoder = False)
    classify(classifier, "XGBoost", X_train, X_test, y_train, y_test)

def AdaBoost(X_train, X_test, y_train, y_test):
    classifier = AdaBoostClassifier(n_estimators=50, learning_rate=1)
    classify(classifier, "AdaBoost", X_train, X_test, y_train, y_test)

def naive_bayes(X_train, X_test, y_train, y_test):
    classifier = GaussianNB()
    classify(classifier, "Naivnog Bajesa", X_train, X_test, y_train, y_test)

if __name__ == '__main__':
    #analiziranje klasa i atributa
    dataset = load_data()
    missing_values(dataset)
    class_distribution(dataset)
    correlation(dataset)
    positive_negative_hist(dataset)

    #izbor atributa
    dataset = final_atributes(dataset)

    #podela skupa podataka
    X_train, X_test, y_train, y_test = divide_train_test(dataset)
    
    #treniranje i klasifikacija razlicitim metodama
    decision_tree(X_train, X_test, y_train, y_test, dataset)
    logistic_regression(X_train, X_test, y_train, y_test)
    random_forest(X_train, X_test, y_train, y_test)
    XGBoost(X_train, X_test, y_train, y_test)
    AdaBoost(X_train, X_test, y_train, y_test)
    naive_bayes(X_train, X_test, y_train, y_test)

