import pandas as pd

#ucitavanje baze
def load_data():
    dataset = pd.read_csv('data.csv')
    #izbacivanje kolona koje nisu od znacaja (kolona 'id' i kolona 'Unnamed: 32')
    dataset = dataset.drop(columns=['id', 'Unnamed: 32'])
    print('Velicina baze: ', dataset.shape)
    print('Prikaz prvih 10 redova baze:\n', dataset.head(10))

if __name__ == '__main__':
    load_data()