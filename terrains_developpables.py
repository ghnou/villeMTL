__author__ = 'pougomg'
import pandas as pd
import xlrd

def get_lands_statistics(my_book):

    secteur = {'Jaune': 'Secteur 1', 'Vert': 'Secteur 2', 'Bleu pâle': 'Secteur 3', 'Bleu': 'Secteur 4',
               'Mauve': 'Secteur 5', 'Rouge': 'Secteur 6', 'Noir': 'Secteur 7'}
    my_book.rename(columns={'couleur': 'Secteur'}, inplace=True)
    my_book.loc[:, 'Secteur'] = my_book.loc[:, 'Secteur'].map(secteur)
    my_book.set_index('id_uev', inplace=True)
    print(my_book.info())
    stat_terrain = my_book[['Secteur', 'SuperficieTerrain']].groupby('Secteur').describe()
    print(stat_terrain.reset_index())
    val_terrain = my_book[['Secteur', 'Valeur terrain totale PROVISOIRE']].groupby('Secteur').describe()

    with pd.ExcelWriter('terrains developpables description jan-30.xlsx') as writer:
        stat_terrain.to_excel(writer, sheet_name='Superficie Terrain')
        val_terrain.to_excel(writer, sheet_name='Valeur Terrain')




if __name__ == '__main__':
    myBook = pd.read_excel('terrains developpables.xlsx', sheet_name='Terrains développables', index_col=0)
    get_lands_statistics(myBook)
