from calcul_de_couts import Calcul_prix_terrain
import pandas as pd
import numpy as np
import xlrd
from import_parameter import __FILES_NAME__,__BATIMENT__, __SECTEUR__, __UNITE_TYPE__, get_nombre_unite, get_house_price

__author__ = 'pougomg'

def calcul_detail_financier(batim, secteur, ensemble, quality, periode ,myBook):


    prix_terrain = Calcul_prix_terrain(batim, secteur, ensemble, myBook)[0]

    print(prix_terrain, myBook.sheet_by_name('Financement').cell(6, 2 + __BATIMENT__.index(batim)).value)
    credit_terrain = prix_terrain *(1 - myBook.sheet_by_name('Financement').cell(6, 2 + __BATIMENT__.index(batim)).value)
    cout_total_construction = 50431666

    tab_financial = pd.DataFrame(list(range(1, 121)), columns=['mois'])
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

        tab_financial[value +" vendues"] = 0
        tab_financial[value +" revenus"] = 0

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

    tab_financial['45 % des unités en prévente - Obtention du permis de construction et  Début de la construction'] = 0
    tab_financial['50 % des unités vendues'] = 0
    tab_financial['Livraison de l immeuble'] = 0
    tab_financial['45 % des unités en prévente - Obtention du permis de construction et  Début de la construction'].loc[tab_financial['cumul']/ntu >
                                          myBook.sheet_by_name('Financement').
                                              cell(15, 2 + __BATIMENT__.index(batim)).value] = 1

    tab_financial['50 % des unités vendues'].loc[tab_financial['cumul']/ntu >
                                          myBook.sheet_by_name('Financement').
                                              cell(16, 2 + __BATIMENT__.index(batim)).value] = 1

    tab_financial['45% cum']  = tab_financial['45 % des unités en prévente - Obtention du permis de construction et  Début de la construction'].cumsum()
    tab_financial['Livraison de l immeuble'].loc[tab_financial['45% cum'] >
                                          myBook.sheet_by_name('Financement').
                                              cell(24, 2 + __BATIMENT__.index(batim)).value] = 1


    tab_financial['Pour calcul - Équité - dépôts de prévente']  = 0
    tab_financial['Pour calcul - Équité - dépôts de prévente'].loc[tab_financial['Livraison de l immeuble'] == 0] = tab_financial['total revenus'] * myBook.sheet_by_name('Financement').cell(7, 2 + __BATIMENT__.index(batim)).value
    tab_financial['Pour calcul - Équité - dépôts de prévente total'] = tab_financial['Pour calcul - Équité - dépôts de prévente'].cumsum()

    tab_financial['solde pret'] = 0
    tab_financial['solde pret'].loc[tab_financial['50 % des unités vendues'] == 0] = credit_terrain
    TI = ((((1 + (myBook.sheet_by_name('Financement').cell(13, 2 + __BATIMENT__.index(batim)).value / 2)) ** 2) ** (1 / 12)) - 1)
    tab_financial['interet'] = tab_financial['solde pret'].astype(float) * TI
    tab_financial['taxes fonciere'] = 17.92

    financement_projet = prix_terrain + tab_financial['interet'].sum() + tab_financial['taxes fonciere'].sum() + cout_total_construction
    print(financement_projet)

    eq_t = prix_terrain - credit_terrain
    eq_pro = financement_projet*(1 - myBook.sheet_by_name('Financement').
                                              cell(3, 2 + __BATIMENT__.index(batim)).value)
    eq_av = financement_projet * myBook.sheet_by_name('Financement').cell(5, 2 + __BATIMENT__.index(batim)).value
    eq_prev = financement_projet * myBook.sheet_by_name('Financement').cell(4, 2 + __BATIMENT__.index(batim)).value

    eq_prev_ter = credit_terrain
    print(eq_pro)

    tab_financial['Équité - dépôts de prévente'] = tab_financial['Pour calcul - Équité - dépôts de prévente']
    tab_financial['Équité - dépôts de prévente'].loc[tab_financial['Pour calcul - Équité - dépôts de prévente total'] > eq_prev] = 0


    ##########Revenus#########

    tab_financial['depot prevente qui ne rentrent pas dans l equite'] = tab_financial['Pour calcul - Équité - dépôts de prévente'] - tab_financial['Équité - dépôts de prévente']

    tab_financial['Pour calcul - Revenus - reste des préventes'] = tab_financial['total revenus'] - tab_financial['depot prevente qui ne rentrent pas dans l equite'] - tab_financial['Équité - dépôts de prévente']

    tab_financial['t'] = tab_financial['Pour calcul - Revenus - reste des préventes'].cumsum()
    tab_financial['x'] = tab_financial['Livraison de l immeuble'].cumsum()
    tab_financial['Revenus - reste des préventes'] = tab_financial['t']*tab_financial['x']
    tab_financial['Revenus - reste des préventes'].loc[tab_financial['x'] > 1] = 0
    tab_financial['Ventes après livraison'] = tab_financial['total revenus'] * tab_financial['Livraison de l immeuble']
    # tab_financial['calcul-eq prev sum'] = tab_financial['calcul-eq prev'].cumsum()
    # tab_financial['equite atteinte'] = 0
    # tab_financial['equite atteinte'].loc[tab_financial['calcul-eq prev sum']>
    #                                       myBook.sheet_by_name('Financement').
    #                                           cell(15, 2 + __BATIMENT__.index(batim)).value] = 1

    # print(tab_financial[['mois', '45%', '50%', 'interet']] )


if __name__ == '__main__':

    myBook = xlrd.open_workbook(__FILES_NAME__)
    calcul_detail_financier(__BATIMENT__[7], __SECTEUR__[4],0,"Base",120, myBook)