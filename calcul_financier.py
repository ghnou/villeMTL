from calcul_de_couts import Calcul_prix_terrain
import pandas as pd
import numpy as np
import xlrd
from import_parameter import __FILES_NAME__,__BATIMENT__, __SECTEUR__, __UNITE_TYPE__, get_nombre_unite, get_house_price

__author__ = 'pougomg'


def calcul_detail_financier(batim, secteur, ensemble, quality, periode ,myBook):

    
    prix_terrain = Calcul_prix_terrain(batim, secteur, ensemble, myBook)[0]
    credit_terrain = prix_terrain *(1 - myBook.sheet_by_name('Financement').cell(6, 2 + __BATIMENT__.index(batim)).value)
    cout_total_construction = 50431666

    tab_financial = pd.DataFrame(list(range(1, 121)), columns=['0'])
    tab_financial['1'] = 1
    tab_financial['2'] = 0
    tab_financial['2'].loc[tab_financial['2'] >
                                          myBook.sheet_by_name('Financement').
                                              cell(21, 2 + __BATIMENT__.index(batim)).value] = 1
    
    #nb unite batiment
    nu = get_nombre_unite(batim, secteur, myBook)
    ntu = 0
    pm = get_house_price(batim, secteur, myBook)
    print(nu)

    for value in range(len(__UNITE_TYPE__)):

        tab_financial[str(value + 25) ] = 0
        tab_financial[str(value + 35)] = 0

    for value in range(len(__UNITE_TYPE__)):
        ntu += nu[value][1]
        index = list(tab_financial[tab_financial['2'] == 1 ].iloc[0:nu[value][3]].index)
        tab_financial[str(value + 25)].loc[index] = nu[value][2]
        tab_financial[str(value + 35)].loc[index] = nu[value][2] * pm[value][1]


    unite = [str(value + 25) for value in range(len(__UNITE_TYPE__))]
    tab_financial['33'] = tab_financial[unite].sum(axis = 1)
    tab_financial['34'] = tab_financial['33'].cumsum()

    unite = [str(value + 25) for value in range(len(__UNITE_TYPE__))]
    tab_financial['42'] = tab_financial[unite].sum(axis = 1)

    tab_financial['3'] = 0
    tab_financial['4'] = 0
    tab_financial['7'] = 0
    tab_financial['3'].loc[tab_financial['34']/ntu >
                                          myBook.sheet_by_name('Financement').
                                              cell(15, 2 + __BATIMENT__.index(batim)).value] = 1

    tab_financial['4'].loc[tab_financial['34']/ntu >
                                          myBook.sheet_by_name('Financement').
                                              cell(16, 2 + __BATIMENT__.index(batim)).value] = 1

    tab_financial['45% cum']  = tab_financial['3'].cumsum()
    tab_financial['7'].loc[tab_financial['45% cum'] >
                                          myBook.sheet_by_name('Financement').
                                              cell(24, 2 + __BATIMENT__.index(batim)).value] = 1


    tab_financial['12']  = 0
    tab_financial['12'].loc[tab_financial['7'] == 0] = tab_financial['42'] * myBook.sheet_by_name('Financement').cell(7, 2 + __BATIMENT__.index(batim)).value
    tab_financial['14'] = tab_financial['12'].cumsum()
    
    eq_t = prix_terrain - credit_terrain
    tab_financial['10'] = eq_t
    tab_financial['19'] = 0
    tab_financial['19'].loc[tab_financial['4'] == 0] = credit_terrain
    TI = ((((1 + (myBook.sheet_by_name('Financement').cell(13, 2 + __BATIMENT__.index(batim)).value / 2)) ** 2) ** (1 / 12)) - 1)
    tab_financial['20'] = tab_financial['19'].astype(float) * TI
    tab_financial['59'] = 17.92

    financement_projet = prix_terrain + tab_financial['20'].sum() + tab_financial['59'].sum() + cout_total_construction
    print(financement_projet)

    
    eq_pro = financement_projet*(1 - myBook.sheet_by_name('Financement').
                                              cell(3, 2 + __BATIMENT__.index(batim)).value)
    eq_av = financement_projet * myBook.sheet_by_name('Financement').cell(5, 2 + __BATIMENT__.index(batim)).value
    eq_prev = financement_projet * myBook.sheet_by_name('Financement').cell(4, 2 + __BATIMENT__.index(batim)).value

    tab_financial['13'] = tab_financial['12']
    tab_financial['13'].loc[tab_financial['14'] > eq_prev] = 0

    eq_prev_ter = eq_t + tab_financial['13'].sum()
    eq_rest = (eq_pro - eq_prev_ter) if (eq_pro - eq_prev_ter)> 0 else 0
    print(eq_pro)




    ##########Revenus#########

    tab_financial['15'] = tab_financial['12'] - tab_financial['13']

    tab_financial['16'] = tab_financial['42'] - tab_financial['15'] - tab_financial['13']

    tab_financial['t'] = tab_financial['16'].cumsum()
    tab_financial['x'] = tab_financial['7'].cumsum()
    tab_financial['17'] = tab_financial['t']*tab_financial['x']
    tab_financial['17'].loc[tab_financial['x'] > 1] = 0
    tab_financial['18'] = tab_financial['42'] * tab_financial['7']
    
    tab_financial['6'] = tab_financial['4'] + tab_financial['5']
    tab_financial['6'].loc[tab_financial['6'] < 2 ] = 0
    tab_financial['6'].loc[tab_financial['6'] == 2 ] = 1
    
    tab_financial['6c'] = tab_financial['6'].cumsum()
    tab_financial['13c'] = tab_financial['13'].cumsum()
    tab_financial['15c'] = tab_financial['15'].cumsum()
    tab_financial['6c'].loc[tab_financial['6c'] > 1 ] = 0
    tab_financial['21'] = tab_financial['6c']*(financement_projet -tab_financial['13c']-tab_financial['15c'] - tab_financial['17'] - tab_financial['10'] - eq_rest)
    
    columns = list(tab_financial.columns)
    
    entete_to_return = [str(value) for value in range(61) if str(value) in columns]
    
    pd.to_csv()
    
    
    
    # tab_financial['calcul-eq prev sum'] = tab_financial['calcul-eq prev'].cumsum()
    # tab_financial['equite atteinte'] = 0
    # tab_financial['equite atteinte'].loc[tab_financial['calcul-eq prev sum']>
    #                                       myBook.sheet_by_name('Financement').
    #                                           cell(15, 2 + __BATIMENT__.index(batim)).value] = 1

    # print(tab_financial[['mois', '45%', '50%', 'interet']] )


if __name__ == '__main__':

    myBook = xlrd.open_workbook(__FILES_NAME__)
    calcul_detail_financier(__BATIMENT__[7], __SECTEUR__[4],0,"Base",120, myBook)