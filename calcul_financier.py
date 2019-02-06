from calcul_de_couts import  calcul_cout_batiment
import pandas as pd
import numpy as np
import xlrd
from import_parameter import get_building_cost_parameter
from lexique import __FINANCE_FILES_NAME__,__BATIMENT__, __SECTEUR__, __UNITE_TYPE__, __COUTS_FILES_NAME__,\
    __FINANCE_PARAM_SHEET__, __ECOULEMENT_SHEET__
from obtention_intrant import get_cb1_characteristics

__author__ = 'pougomg'

def debut_des_ventes(group, d, fonction_ecoulement):

    """
        This function is used to calculate the beginning of the sales of the building.
        :param
        group (pd.DataFrame): Dataframe of month timeline
         d (dict): Duree moyenne entre l'achat de terrain et le debut de la prevente

        :return
        pd.Serie: Serie contenant le timeline de debut de prevente
    """
    fonction_ecoulement = fonction_ecoulement[[group.name]].reset_index(drop=True)
    f3mo = fonction_ecoulement.loc[0, group.name]
    f3mo_after = fonction_ecoulement.loc[1, group.name]
    group.where(group > d[group.name], 0, inplace=True)
    group.where(group == 0, 1, inplace=True)

    t = group.reset_index(drop=True)
    x1 = (f3mo/3).round(2) * (np.heaviside(t.index - d[group.name] + 1, 0) - np.heaviside(t.index - d[group.name] - 2, 0))
    x2 = (f3mo_after/3).round(2) * (np.heaviside(t.index - d[group.name] - 2, 0))
    # print((x1 + x2))
    group['ecoulement'] = x1 + x2
    t = group[['ecoulement']].cumsum()
    t.loc[t.loc[:, 'ecoulement'] > 1, 'ecoulement'] = 0
    group['sum'] = t
    group.loc[group.loc[:, 'sum'] == 0, 'ecoulement'] = 0
    group.loc[group['sum'].idxmax() + 1 , 'ecoulement'] = 1 - group.loc[:, 'sum'].max()

    return group[['1', 'ecoulement']]

def calcul_ecoulement_et_vente(group, nombre_total_unite):

    print(group.name)
    print(nombre_total_unite)

    t = nombre_total_unite[nombre_total_unite['category'].isin(__UNITE_TYPE__)][[group.name, 'category', 'value']].set_index('category').transpose()

    print(t)
    return
    result = pd.DataFrame([t.values[0] for i in range(group.shape[0])], columns=t.columns).reset_index(drop=True)

    print(result.mul(group['ecoulement'].values, axis=0))

    print(t.columns)





def calcul_detail_financier(cost_table, financials_param, secteur, batiment, my_book, timeline) -> pd.DataFrame:

    """""
    This function is used to compute the cost of a builiding given a specific sector.

    :param
    cost_table (pd.DataFrame):  The table containing all the building useful information necessary to compute the financials.

    secteur (str): sector of the building

    batiment (range): range containing the building we want to compute the costs. eg: ['B1', 'B7']

    :return

    financials_result (pd.Dataframe)

    """""
    fsh = my_book.sheet_by_name(__FINANCE_PARAM_SHEET__)
    financials_result = []
    for bat in batiment:
         financials_result.append(pd.DataFrame([[bat, i + 1, 1] for i in range(timeline)], columns=['batiment', '1', '2']))
    financials_result = pd.concat(financials_result,
                                  ignore_index=True)
    # Debut des ventes
    d_ = dict()
    pos = [21, 2]
    for _ in range(len(__BATIMENT__)):
        d_[__BATIMENT__[_]] = fsh.cell(pos[0], pos[1] + _).value

    # Fonction d'ecoulement
    esh = my_book.sheet_by_name(__ECOULEMENT_SHEET__)
    tab = []

    pos_first_3mo = [4, 2]

    for line in range(len(__SECTEUR__)):
        _ = [__SECTEUR__[line], '3mo']
        for col in range(len(__BATIMENT__)):
            _.append(esh.cell(line + pos_first_3mo[0], col + pos_first_3mo[1]).value)
        tab.append(_)

    pos_next_3mo = [15, 2]

    for line in range(len(__SECTEUR__)):
        _ = [__SECTEUR__[line], 'n3mo']
        for col in range(len(__BATIMENT__)):
            _.append(esh.cell(line + pos_next_3mo[0], col + pos_next_3mo[1]).value)
        tab.append(_)

    fonction_ecoulement = pd.DataFrame(tab, columns=['sector', 'value'] + __BATIMENT__)
    fonction_ecoulement = fonction_ecoulement[fonction_ecoulement['sector'] == secteur]

    financials_result[['3', 'ecoulement']] = financials_result[['1']].groupby(financials_result['batiment']).apply(debut_des_ventes, d_, fonction_ecoulement)

    # Ventes (Ecoulement et revenus bruts)

    financials_result[['ecoulement']].groupby(financials_result['batiment']).apply(calcul_ecoulement_et_vente,
                                                                                   cost_table[(cost_table['value'] == 'ntu') |
                                                                                   (cost_table['value'] == 'price')])




    return
    print(fonction_ecoulement[batiment].groupby(fonction_ecoulement['sector']).apply(ventes_ecoulement))
    print(financials_result.head(20))
    print(cost_table[cost_table['value'] == 'ntu'].groupby('sector').apply(ventes_ecoulement, fonction_ecoulement))
    # print(financials_result)



def calcul_detail_financier_(batim, secteur, ensemble, quality, periode ,myBook):

    
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

    myBook = xlrd.open_workbook(__COUTS_FILES_NAME__)
    intrant_param = get_cb1_characteristics(myBook)
    cost_param = get_building_cost_parameter(myBook)

    intrant_param = intrant_param[['sector', 'category', 'value', 'B1', 'B3', 'B8']]
    intrant_param = intrant_param[(intrant_param['sector'] == 'Secteur 7') & (intrant_param['value'] != 'pptu')]
    cost_param = cost_param[['sector', 'category', 'value', 'B1', 'B3', 'B8']]
    cost_param = cost_param[cost_param['sector'] == 'Secteur 7']
    cost_table = calcul_cout_batiment(intrant_param, cost_param, 'Secteur 7', ['B1', 'B3', 'B8'], myBook)

    myBook = xlrd.open_workbook(__FINANCE_FILES_NAME__)
    calcul_detail_financier(cost_table, None, 'Secteur 7', ['B1', 'B3', 'B8'], myBook, 120)
