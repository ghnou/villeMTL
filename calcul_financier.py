from calcul_de_couts import Calcul_prix_terrain
import pandas as pd
import numpy as np
import xlrd
from import_parameter import __FILES_NAME__,__BATIMENT__, __SECTEUR__, __UNITE_TYPE__, get_nombre_unite, get_house_price

__author__ = 'pougomg'

def calcul_detail_financier(batim, secteur, ensemble, quality, periode ,myBook):


    # prix_terrain = Calcul_prix_terrain(batim, secteur, ensemble, myBook)
    # print(prix_terrain)
    cout_total_construction = 0

    tab_financial = pd.DataFrame(list(range(1,121)), columns=['mois'])
    tab_financial['achat terrain'] = 1
    tab_financial['debut des ventes'] = 0
    tab_financial['debut des ventes'].loc[tab_financial['mois'] >
                                          myBook.sheet_by_name('Financement').
                                              cell(21, 2 + __BATIMENT__.index(batim)).value] = 1
    nu = get_nombre_unite(batim, secteur, myBook)
    ntu = 0
    pm = get_house_price(batim, secteur, myBook)
    print(nu)

    for value in __UNITE_TYPE__:

        tab_financial[ value +" vendues"] = 0
        tab_financial[ value +" revenus"] = 0

    for value in range(len(__UNITE_TYPE__)):
        ntu += nu[value][1]
        index = list(tab_financial[tab_financial['debut des ventes'] == 1 ].iloc[0:nu[value][3]].index)
        tab_financial[__UNITE_TYPE__[value] +" vendues"].loc[index] = nu[value][2]
        tab_financial[__UNITE_TYPE__[value] +" revenus"].loc[index] = nu[value][2] * pm[value][1]


    # tab_financial['total unite vendues'] = tab_financial.to_sql()
    unite = (" vendues,".join(__UNITE_TYPE__) + " vendues").split(',')
    tab_financial['total unite vendues'] = tab_financial[unite].sum(axis = 1)
    tab_financial['cumul'] = tab_financial['total unite vendues'].cumsum()

    unite = (" revenus,".join(__UNITE_TYPE__) + " revenus").split(',')
    tab_financial['total revenus'] = tab_financial[unite].sum(axis = 1)

    tab_financial['45%'] = 0
    tab_financial['50%'] = 0
    tab_financial['livraison'] = 0
    tab_financial['45%'].loc[tab_financial['cumul']/ntu >
                                          myBook.sheet_by_name('Financement').
                                              cell(15, 2 + __BATIMENT__.index(batim)).value] = 1

    tab_financial['50%'].loc[tab_financial['cumul']/ntu >
                                          myBook.sheet_by_name('Financement').
                                              cell(16, 2 + __BATIMENT__.index(batim)).value] = 1

    tab_financial['45% cum']  = tab_financial['45%'].cumsum()
    tab_financial['livraison'].loc[tab_financial['45% cum'] >
                                          myBook.sheet_by_name('Financement').
                                              cell(24, 2 + __BATIMENT__.index(batim)).value] = 1


    tab_financial['calcul-eq prev']  = 0
    tab_financial['calcul-eq prev'].loc[tab_financial['livraison'] == 0] = tab_financial['total revenus'] * myBook.sheet_by_name('Financement').cell(7, 2 + __BATIMENT__.index(batim)).value

    # tab_financial['calcul-eq prev sum'] = tab_financial['calcul-eq prev'].cumsum()
    # tab_financial['equite atteinte'] = 0
    # tab_financial['equite atteinte'].loc[tab_financial['calcul-eq prev sum']>
    #                                       myBook.sheet_by_name('Financement').
    #                                           cell(15, 2 + __BATIMENT__.index(batim)).value] = 1

    print(tab_financial[['mois', '45%', '50%', 'calcul-eq prev']] )


if __name__ == '__main__':

    myBook = xlrd.open_workbook(__FILES_NAME__)
    calcul_detail_financier(__BATIMENT__[7], __SECTEUR__[4],0,"Base",120, myBook)