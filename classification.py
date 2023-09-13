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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, RFE, mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from skfeature.function.similarity_based import fisher_score
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    recall_score,
    classification_report,
    accuracy_score,
    precision_score,
    f1_score,
)

# analiza baze
def analyze_base():
    dataset = load_data()
    # missing_values(dataset)
    # class_distribution(dataset)
    # correlation(dataset)
    # plot_class_histogram_for_every_attribute(dataset)
    return dataset


# ucitavanje baze
def load_data():
    dataset = pd.read_csv("data.csv")
    # izbacivanje kolona koje nisu od znacaja (kolona 'id' i kolona 'Unnamed: 32')
    dataset = dataset.drop(columns=["id", "Unnamed: 32"])
    # print("\nVelicina baze: ", dataset.shape)
    dataset["diagnosis"] = dataset["diagnosis"].replace("B", 0).replace("M", 1)
    # print("\n\nPrikaz poslednjih 10 redova baze:\n\n", dataset.tail(10))
    return dataset


# provera da li ima nedostajucih vrednosti
def missing_values(dataset):
    missing_values_per_column = dataset.isna().sum()
    print("\nNedostajuce vrednosti po kolonama:\n\n", missing_values_per_column)
    missing_values = missing_values_per_column.sum()
    print("\nUkupan broj nedostajucih vrednosti:", missing_values)


# distribucija klase
def class_distribution(dataset):
    numBenign = sum(dataset["diagnosis"] == 0)
    numMalignant = sum(dataset["diagnosis"] == 1)
    print("Broj uzoraka sa benignim oboljenjem: ", numBenign)
    print("Broj uzoraka sa malignim oboljenjem: ", numMalignant, "\n")

    # prikaz raspodele klasa
    plt.bar(0, numBenign, color="lightskyblue", width=0.5, label="Benigna")
    plt.bar(1, numMalignant, color="mediumaquamarine", width=0.5, label="Maligna")
    plt.axis([-0.5, 1.5, 0, 400])
    plt.xticks([0, 1])
    plt.xlabel("Klase")
    plt.ylabel("Broj uzoraka")
    plt.title("Prikaz raspodele klasa", weight="bold")
    plt.legend()
    plt.savefig("./images/class_distribution.png")
    plt.show()


# kreiranje korelacija
def correlation(dataset):
    mean = dataset[dataset.columns[:11]]
    se = dataset.drop(dataset.columns[1:11], axis=1)
    se = se.drop(se.columns[11:], axis=1)
    worst = dataset.drop(dataset.columns[1:21], axis=1)

    titles = ["srednje vrednosti", "srednje-kvadratne greske", "najvece greske"]
    # prikaz korelacija za dijagnozom
    correlation_with_diagnosis(mean, se, worst, titles)
    # prikaz korelacija sa dijagnozom kolor mapom
    mutual_correlation(dataset, mean, se, worst, titles)

    # corr = dataset.corr()
    # print(corr[abs(corr["diagnosis"]) > 0.5])
    # cc = corr[abs(corr["diagnosis"]) > 0.5].index
    # print("\nBroj prediktora kojima je koeficijent korelacije veci od 0.5 = ", len(cc))
    # print("\nAtributi najkorelisaniji sa izlazom: \n ", cc)


# korelacija atributa iz svake grupe sa klasom
def correlation_with_diagnosis(mean, se, worst, titles):
    data = [mean, se, worst]
    colors = ["lightskyblue", "mediumaquamarine", "peachpuff"]
    file_names = ["mean_corr_class.png", "se_corr_class.png", "worst_corr_class.png"]
    for i in range(len(data)):
        correlation = data[i].drop(columns=["diagnosis"]).corrwith(data[i].diagnosis)
        correlation.plot(kind="bar", grid=False, color=colors[i], figsize=(20, 20))
        plt.title("Korelacija atributa " + titles[i] + " sa dijagnozom", weight="bold")
        plt.ylabel("Nivo korelisanosti")
        plt.savefig("./images/" + file_names[i])
        plt.show()


# korelacija u vidu kolormape
def mutual_correlation(dataset, mean, se, worst, titles):
    data = [dataset, mean, se, worst]
    file_names = ["cm.png", "mean_cm.png", "se_cm.png", "worst_cm.png"]
    titles.insert(0, "svih atributa")
    for i in range(len(data)):
        corr = data[i].corr()
        plt.figure(figsize=(20, 20))
        seaborn.heatmap(corr, cmap="coolwarm", annot=True)
        plt.title("Kolormapa " + titles[i], weight="bold")
        plt.savefig("./images/" + file_names[i])
        plt.show()


# prikaz histograma raspodele klase u zavisnosti od atributa
def plot_class_histogram_for_every_attribute(dataset):
    fig, ax = plt.subplots(ncols=5, nrows=6, figsize=(30, 35))
    index = 0
    ax = ax.flatten()
    for col, value in dataset.iloc[:, 1:].items():
        val0 = value.loc[dataset["diagnosis"] == 0]
        val1 = value.loc[dataset["diagnosis"] == 1]
        seaborn.histplot(
            val0,
            ax=ax[index],
            color="lightskyblue",
            label="100% Equities",
            kde=True,
            stat="density",
            linewidth=0,
        )
        seaborn.histplot(
            val1,
            ax=ax[index],
            color="peachpuff",
            label="100% Equities",
            kde=True,
            stat="density",
            linewidth=0,
        )
        index += 1

    plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)
    plt.savefig("./images/plot_class_histogram_for_every_attributeogram.png")
    plt.title("Prikaz pozitivnih i negativnih primeraka za sve atribute", weight="bold")
    plt.show()


# fisher score za odredjivanje atributa
def fisher_score_(X, y):
    ranks = fisher_score.fisher_score(X.to_numpy(), y.to_numpy())
    feat_importances = pd.Series(ranks, X.columns[0 : len(X.columns)])
    feat_importances.plot(kind="barh", color="teal")
    plt.savefig("./images/fisher_score.png")
    plt.title("Fisher score date klase")
    plt.xlabel("Vrednost")
    plt.ylabel("Atribut")
    plt.show()


# information gain za odredjivanje atributa
def information_gain(k, X, y):
    ranks = mutual_info_classif(X, y)
    feat_importances = pd.Series(ranks, X.columns[0 : len(X.columns)])
    feat_importances.plot(kind="barh", color="teal")
    plt.savefig("./images/information_gain.png")
    plt.title("Information Gain date klase")
    plt.xlabel("Vrednost")
    plt.ylabel("Atribut")
    plt.show()
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    print("\nInformation Gain (15 najboljih atributa):")
    print(selected_features)


# rekurzivna eliminacija atributa za odredjivanje atributa
def rfe_(k, X, y):
    model = DecisionTreeClassifier()
    rfe = RFE(estimator=model, n_features_to_select=k)
    rfe.fit_transform(X, y)
    selected_features = X.columns[rfe.support_]
    print("\nRFE  (15 najboljih atributa):")
    print(selected_features)
    print("\n")


# odabir konacnih obelezja za koriscenje pri obucavanju i testiranju
def choose_attributes(dataset):
    X_old = dataset.drop(columns=["diagnosis"])
    y_old = dataset["diagnosis"]
    k = 15
    # fisher_score_(X_old, y_old)
    # information_gain(k, X_old, y_old)
    # rfe_(k, X_old, y_old)

    # dataset = dataset[
    #     [
    #         "diagnosis",
    #         "area_se",
    #         "radius_worst",
    #         "concave points_worst",
    #         "concavity_mean",
    #         "symmetry_worst",
    #         "texture_worst",
    #     ]
    # ]

    new_corr = dataset.corr()
    # print(
    #     "\nIzdvojeni prediktori sa odgovarajucim koeficijentom korelacije sa izlazom: \n\n",
    #     new_corr["diagnosis"],
    # )
    return dataset


# treniranje i klasifikacija
def train_and_classify(dataset, classifier, name, filename):
    X = dataset.drop(columns=["diagnosis"])
    y = dataset["diagnosis"]
    _, num_attributes = dataset.shape
    # 1/5 = 20% test, 4/5 = 80% training
    num_splits = 5
    folds = StratifiedKFold(n_splits=num_splits)
    scores_accuracy = []
    scores_precision = []
    scores_f1_score = []
    scores_recall = []
    i = 1

    for train_index, test_index in folds.split(X, y):
        prediction, y_test = train_and_classify_one_model(
            classifier, train_index, test_index, X, y
        )
        # make_report(classifier, prediction, y_test, name, filename, i)
        scores_accuracy.append(accuracy_score(prediction, y_test))
        scores_precision.append(precision_score(prediction, y_test))
        scores_f1_score.append(f1_score(prediction, y_test))
        scores_recall.append(recall_score(prediction, y_test))
        i += 1

    reportForAllModels(
        name,
        num_attributes - 1,
        num_splits,
        scores_accuracy,
        scores_precision,
        scores_recall,
        scores_f1_score,
    )

# treniranje i klasifikacija za 1 iteraciju 1 modela
def train_and_classify_one_model(classifier, train_index, test_index, X, y):
    X_train, X_test, y_train, y_test = (
        X.iloc[train_index, :],
        X.iloc[test_index, :],
        y.iloc[train_index],
        y.iloc[test_index],
    )
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    classifier.fit(X_train, y_train)
    prediction = classifier.predict(X_test)
    return prediction, y_test


# pravljenje izvestaja za 1 iteraciju 1 modela
def make_report(classifier, prediction, y_test, name, filename, i):
    if (i != 1):
        print("\n-----------------------\n")
    print(str(i) + ". Klasifikator:\n")
    print(
        classification_report(
            y_test, prediction, target_names=["Maligna masa: 0", "Benigna masa: 1"]
        )
    )
    print(
        "Tacnost modela iznosi: {:.2f}%".format(
            accuracy_score(y_test, prediction) * 100
        )
    )
    print(
        "Preciznost modela iznosi: {:.2f}%".format(
            precision_score(y_test, prediction) * 100
        )
    )
    print(
        "Recall modela iznosi: {:.2f}%".format(recall_score(y_test, prediction) * 100)
    )
    print(
        "Balansirana tacnost iznosi: {:.2f}%".format(f1_score(y_test, prediction) * 100)
    )
    plot_confusion_matrix(y_test, prediction, i, name, filename)
    if (filename == "dt"):
        plot_tree(classifier, i)


# pravljenje konacnog izvestaja za 1 model
def reportForAllModels(
    name,
    num_attributes,
    num_splits,
    scores_accuracy,
    scores_precision,
    scores_recall,
    scores_f1_score,
):
    print(
        "\nIzabrano je {} atributa pri analizi, a rezultati su sledeci:\n".format(
            num_attributes
        )
    )
    print(
        "Usrednjena tacnost za "
        + str(num_splits)
        + " modela za " + name +  " iznosi: {:.2f}%".format(100 * np.average(scores_accuracy))
    )
    print(
        "Usrednjena preciznost za "
        + str(num_splits)
        + " modela  " + name +  " iznosi: {:.2f}%".format(100 * np.average(scores_precision))
    )
    print(
        "Usrednjen recall za "
        + str(num_splits)
        + " modela  " + name +  " iznosi: {:.2f}%".format(100 * np.average(scores_recall))
    )
    print(
        "Usrednjen f1 score za "
        + str(num_splits)
        + " modela  " + name +  " iznosi: {:.2f}%".format(100 * np.average(scores_f1_score))
    )


# prikaz konfuzione matrice
def plot_confusion_matrix(y_test, prediction, i, name, filename):
    confusion_matrix = metrics.confusion_matrix(y_test, prediction)
    cm_display = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
    cm_display.plot()
    plt.savefig('./images/conf_mat_' + filename + str(i) + '.png')
    plt.title("Konfuziona matrica - " + name + " " + str(i), weight="bold")
    plt.xlabel("Predikcija")
    plt.ylabel("Stvarna vrednost")
    plt.show()
    print('Tacnost modela ' + name + ' iznosi: {:.3f}%'.format(accuracy_score(y_test,prediction)*100))
    print('Balansirana tacnost modela ' + name + ' iznosi: {:.3f}%'.format(f1_score(y_test,prediction)*100))


# prikaz stabla
def plot_tree(classifier, i):
    feature_names = [l for l in list(dataset.columns)[1:]]
    class_names = ["Benigni", "Maligni"]
    fig = plt.figure(figsize=(30, 20))
    _ = tree.plot_tree(
        classifier,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        fontsize=8,
    )
    plt.title("Prikaz " + str(i) + ". stabla odlucivanja", weight="bold")
    plt.savefig("./images/tree" + str(i) + ".png")
    plt.show()


def decision_tree(dataset):
    print("\n-----------------------\n")
    print("Stablo odlucivanja:\n")
    classifier = DecisionTreeClassifier()
    train_and_classify(dataset, classifier, "Stablo odlucivanja", "dt") 


def random_forest(dataset):
    print("\n-----------------------\n")
    print("Slucajne sume:\n")
    classifier = RandomForestClassifier(n_estimators=50)
    train_and_classify(dataset, classifier, "Slucajne sume", "rf")


def XGBoost(dataset):
    print("\n-----------------------\n")
    print("XGBoost:\n")
    classifier = XGBClassifier(n_estimators=350, subsample=0.8, max_depth=7, eval_metric = 'logloss')
    train_and_classify(dataset, classifier, "XGBoost", "xgb")


def AdaBoost(dataset):
    print("\n-----------------------\n")
    print("AdaBoost:\n")
    classifier = AdaBoostClassifier(n_estimators=50, learning_rate=1)
    train_and_classify(dataset, classifier, "AdaBoost", "adb")


def logistic_regression(dataset):
    print("\n-----------------------\n")
    print("Logisticka regresija:\n")
    classifier = LogisticRegression()
    train_and_classify(dataset, classifier, "Logisticka regresija", "lr")


def naive_bayes(dataset):
    print("\n-----------------------\n")
    print("Naivni Bajes:\n")
    classifier = GaussianNB()
    train_and_classify(dataset, classifier, "Naivnog Bajesa", "nb")
    print("\n-----------------------\n")

# poziva treniranje i klasifikaciju za svaki model
def classify(dataset):
    dataset = dataset[
        [
            "diagnosis",
            "area_se",
            "radius_worst",
            "concave points_worst",
            "concavity_mean",
            "symmetry_worst",
            "texture_worst",
        ]
    ]
    decision_tree(dataset)
    random_forest(dataset)
    XGBoost(dataset)
    AdaBoost(dataset)
    logistic_regression(dataset)
    naive_bayes(dataset)

if __name__ == "__main__":
    # analiziranje klasa i atributa
    dataset = analyze_base()
    # izbor atributa
    dataset = choose_attributes(dataset)
    # podela skupa podataka, treniranje i klasifikacija razlicitim modelima
    classify(dataset)

    # dataset = dataset[
    #     [
    #         "diagnosis",
    #         "compactness_mean",
    #         "area_mean",
    #         "concavity_mean",
    #         "fractal_dimension_se",
    #     ]
    # ]

    # dataset = dataset[
    #     [
    #         "diagnosis",
    #         "radius_mean",
    #         "perimeter_mean",
    #         "area_mean",
    #         "compactness_mean",
    #         "concavity_mean",
    #         "concave points_mean",
    #         "radius_se",
    #         "perimeter_se",
    #         "area_se",
    #         "radius_worst",
    #         "perimeter_worst",
    #         "area_worst",
    #         "compactness_worst",
    #         "concavity_worst",
    #         "concave points_worst",
    #     ]
    # ]
    # dataset = dataset[
    #     [
    #         "diagnosis",
    #         "perimeter_mean",
    #         "radius_mean",
    #         "compactness_mean",
    #         "concave points_mean",  # 1
    #         "radius_se",
    #         "perimeter_se",
    #         "radius_worst",  # 2
    #         "perimeter_worst",  # 3
    #         "compactness_worst",
    #         "concave points_worst",  # 5
    #         "compactness_se",
    #         "concave points_se",
    #         "texture_worst",
    #         "area_worst",  # 4
    #     ]
    # ]

    # dataset = dataset[["diagnosis", "concavity_mean", "concavity_se", "area_mean"]]

    # dataset = dataset[
    #     [
    #         "diagnosis",
    #         "radius_mean",
    #         "compactness_mean",
    #         "concavity_mean",
    #         "concave points_mean",
    #         "area_worst",
    #         "compactness_worst",
    #         "concavity_worst",
    #         "concave points_worst",
    #         "texture_mean",
    #         "smoothness_mean",
    #         "symmetry_mean",
    #         "area_se",
    #         "texture_worst",
    #         "smoothness_worst",
    #         "symmetry_worst",
    #         "fractal_dimension_worst",
    #     ]
    # ]

    # dataset = dataset[
    #     [
    #         "diagnosis",
    #         "radius_mean",
    #         "perimeter_mean",
    #         "area_mean",
    #         "compactness_mean",
    #         "concave points_mean",
    #     ]
    # ]