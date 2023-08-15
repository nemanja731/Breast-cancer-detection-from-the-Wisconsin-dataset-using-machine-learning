import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
from xgboost import XGBClassifier
from sklearn import tree, metrics
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def analyze_base():
    dataset = load_data()
    missing_values(dataset)
    #class_distribution(dataset)
    correlation(dataset)
    #positive_negative_hist(dataset)
    return dataset

#ucitavanje baze
def load_data():
    dataset = pd.read_csv('data.csv')
    #izbacivanje kolona koje nisu od znacaja (kolona 'id' i kolona 'Unnamed: 32')
    dataset = dataset.drop(columns=['id', 'Unnamed: 32'])
    print('\nVelicina baze: ', dataset.shape)
    dataset['diagnosis'] = dataset['diagnosis'].replace('B', 0).replace('M', 1)
    print('\n\nPrikaz poslednjih 10 redova baze:\n\n', dataset.tail(10))
    return dataset

#provera da li ima nedostajucih vrednosti
def missing_values(dataset):
    missing_values_per_column = dataset.isna().sum()
    print("\nNedostajuce vrednosti po kolonama:\n\n", missing_values_per_column)
    missing_values = missing_values_per_column.sum()
    print("\nBroj nedostajucih vrednosti:", missing_values)

def class_distribution(dataset):
    numBenign = sum(dataset['diagnosis'] == 0)
    numMalignant = sum(dataset['diagnosis'] == 1)
    print('Broj uzoraka sa benignim oboljenjem: ', numBenign)
    print('Broj uzoraka sa malignim oboljenjem: ', numMalignant, '\n')

    #prikaz raspodele klasa
    plt.bar(0, numBenign, color ='lightskyblue', width=0.5, label ='Benigna')
    plt.bar(1, numMalignant, color ='mediumaquamarine', width=0.5, label ='Maligna')
    plt.axis([-0.5, 1.5, 0, 400])
    plt.xticks([0, 1])
    plt.xlabel('Klase')
    plt.ylabel('Broj uzoraka')
    plt.title('Prikaz raspodele klasa', weight='bold')
    plt.legend()
    plt.savefig('./images/class_distribution.png')
    plt.show()

def correlation_with_diagnosis(mean, se, worst, titles):
    data = [mean, se, worst]
    colors = ["lightskyblue", "mediumaquamarine", "peachpuff"]
    file_names = ["mean_corr_class.png", "se_corr_class.png", "worst_corr_class.png"]
    for i in range (len(data)):
        correlation = data[i].drop(columns=['diagnosis']).corrwith(data[i].diagnosis)
        correlation.plot(kind='bar', grid=False, color=colors[i], figsize = (20, 20))
        plt.title("Korelacija atributa " + titles[i] + " sa dijagnozom", weight='bold')
        plt.ylabel('Nivo korelisanosti')
        plt.savefig("./images/" + file_names[i])
        plt.show()

def mutual_correlation(dataset, mean, se, worst, titles):
    data = [dataset, mean, se, worst]
    file_names = ["cm.png", "mean_cm.png", "se_cm.png", "worst_cm.png"]
    titles.insert(0, "svih atributa")
    for i in range (len(data)):
        corr = data[i].corr()
        plt.figure(figsize = (20, 20))
        seaborn.heatmap(corr, cmap = 'coolwarm', annot = True)
        plt.title("Kolormapa " + titles[i], weight='bold')
        plt.savefig("./images/" + file_names[i])
        plt.show()

def correlation(dataset):
    #kreiranje korelacija
    mean = dataset[dataset.columns[:11]]
    se = dataset.drop(dataset.columns[1:11], axis=1)
    se = se.drop(se.columns[11:], axis=1)
    worst = dataset.drop(dataset.columns[1:21], axis=1)

    titles = ["srednje vrednosti", "srednje-kvadratne greske", "najvece greske"]
    #prikaz korelacija za dijagnozom
    #correlation_with_diagnosis(mean, se, worst, titles)
    #prikaz korelacija sa dijagnozom kolor mapom
    #mutual_correlation(dataset, mean, se, worst, titles)

    corr = dataset.corr()
    print(corr[abs(corr['diagnosis']) > 0.3])
    cc = corr[abs(corr['diagnosis']) > 0.3].index
    print('\nBroj prediktora kojima je koeficijent korelacije veci od 0.3 = ', len(cc))
    print('\nAtributi najkorelisaniji sa izlazom: \n ', cc)

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
    plt.savefig('./images/positive_negative_histogram.png')
    plt.title("Prikaz pozitivnih i negativnih primeraka za sve atribute", weight="bold")
    plt.show()

def final_attributes(dataset):
    #odabir konacnih obelezja za koriscenje pri obucavanju i testiranju
    dataset = dataset[['diagnosis', 'radius_mean', 'compactness_mean',
        'concavity_mean', 'concave points_mean', 'area_worst',
        'compactness_worst', 'concavity_worst', 'concave points_worst',
        'texture_mean', 'smoothness_mean', 'symmetry_mean','area_se',
        'texture_worst', 'smoothness_worst', 'symmetry_worst', 'fractal_dimension_worst']]
    new_corr = dataset.corr()
    print('\nIzdvojeni prediktori sa odgovarajucim koeficijentom korelacije sa izlazom: \n\n', new_corr['diagnosis'])
    return dataset

def divide_dataset(dataset):
    X = dataset.drop(columns = ['diagnosis'])
    y = dataset['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test, y_train, y_test

def train_and_clasify(dataset, X_train, X_test, y_train, y_test):
    decision_tree(X_train, X_test, y_train, y_test, dataset)
    logistic_regression(X_train, X_test, y_train, y_test)
    random_forest(X_train, X_test, y_train, y_test)
    XGBoost(X_train, X_test, y_train, y_test)
    AdaBoost(X_train, X_test, y_train, y_test)
    naive_bayes(X_train, X_test, y_train, y_test)

def plot_confusion_matrix(prediction_lr, y_test, name, filename):
    confusion_matrix = metrics.confusion_matrix(y_test, prediction_lr)
    cm_display = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
    cm_display.plot()
    plt.savefig('./images/conf_mat_' + filename + '.png')
    plt.title("Konfuziona matrica - " + name, weight="bold")
    plt.xlabel("Predikcija")
    plt.ylabel("Stvarna vrednost")
    plt.show()
    print('Tacnost modela ' + name + ' iznosi: {:.3f}%'.format(accuracy_score(y_test,prediction_lr)*100))
    print('Balansirana tacnost modela ' + name + ' iznosi: {:.3f}%'.format(f1_score(y_test,prediction_lr)*100))

def classify(classifier, name, filename, X_train, X_test, y_train, y_test):
    classifier.fit(X_train, y_train)
    prediction = classifier.predict(X_test)
    print(classification_report(y_test, prediction, target_names = ['Maligna masa: 0','Benigna masa: 1']))
    plot_confusion_matrix(prediction, y_test, name, filename)

def decision_tree(X_train, X_test, y_train, y_test, dataset):
    print("\n-----------------------\n")
    print("Stablo odlucivanja:\n")
    classifier = DecisionTreeClassifier()
    classify(classifier, "stabla odlucivanja", "td", X_train, X_test, y_train, y_test) 
    classifier.fit(X_train, y_train)
    #prikaz stabla
    feature_names =[l for l in list(dataset.columns)[1:]]
    class_names = ['Benign', 'Malignant']
    fig = plt.figure(figsize=(30, 20))
    _ = tree.plot_tree(classifier, feature_names = feature_names, class_names=class_names, filled=True, fontsize=8)
    plt.title("Prikaz stabla odlucivanja", weight="bold")
    plt.savefig('./images/tree.png')
    plt.show() 

def logistic_regression(X_train, X_test, y_train, y_test):
    print("\n-----------------------\n")
    print("Logisticka regresija:\n")
    classifier = LogisticRegression()
    classify(classifier, "logisticke regresije", "lr", X_train, X_test, y_train, y_test)

def random_forest(X_train, X_test, y_train, y_test):
    print("\n-----------------------\n")
    print("Slucajna suma:\n")
    classifier = RandomForestClassifier()
    classify(classifier, "slucajne sume", "rf", X_train, X_test, y_train, y_test)

def XGBoost(X_train, X_test, y_train, y_test):
    print("\n-----------------------\n")
    print("XGBoost:\n")
    classifier = XGBClassifier(n_estimators=350, subsample=0.8, max_depth=7, eval_metric = 'logloss')
    classify(classifier, "XGBoost", "xgb", X_train, X_test, y_train, y_test)

def AdaBoost(X_train, X_test, y_train, y_test):
    print("\n-----------------------\n")
    print("AdaBoost:\n")
    classifier = AdaBoostClassifier(n_estimators=50, learning_rate=1)
    classify(classifier, "AdaBoost", "adb", X_train, X_test, y_train, y_test)

def naive_bayes(X_train, X_test, y_train, y_test):
    print("\n-----------------------\n")
    print("Naivni Bajes:\n")
    classifier = GaussianNB()
    classify(classifier, "Naivnog Bajesa", "nb", X_train, X_test, y_train, y_test)
    print("\n-----------------------\n")

if __name__ == '__main__':
    #analiziranje klasa i atributa
    dataset = analyze_base()
    #izbor atributa
    dataset = final_attributes(dataset)
    #podela skupa podataka
    X_train, X_test, y_train, y_test = divide_dataset(dataset)
    #treniranje i klasifikacija razlicitim metodama
    train_and_clasify(dataset, X_train, X_test, y_train, y_test)

#X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, stratify=y, shuffle=True, random_state = 0)
    