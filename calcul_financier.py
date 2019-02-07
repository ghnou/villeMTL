import numpy as np
import pandas as pd
import xlrd

from calcul_de_couts import calcul_cout_batiment
from import_parameter import get_building_cost_parameter
from lexique import __FINANCE_FILES_NAME__, __BATIMENT__, __SECTEUR__, __UNITE_TYPE__, __COUTS_FILES_NAME__, \
    __FINANCE_PARAM_SHEET__, __ECOULEMENT_SHEET__
from obtention_intrant import get_cb1_characteristics, get_cb4_characteristics

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

    group['3'] = group['1'].where(group['1'].astype(int) >= d[group.name], 0)
    group['3'] = group['3'].where(group['3'] == 0, 1)

    t = group.reset_index(drop=True)
    x1 = (f3mo / 3).round(2) * (
                np.heaviside(t.index - d[group.name] + 2, 0) - np.heaviside(t.index - d[group.name] - 1, 0))
    x2 = (f3mo_after / 3).round(2) * (np.heaviside(t.index - d[group.name] - 1, 0))
    group['ecoulement'] = x1 + x2
    t = group[['ecoulement']].cumsum()
    t.loc[t.loc[:, 'ecoulement'] > 1, 'ecoulement'] = 0
    group['sum'] = t
    group.loc[group.loc[:, 'sum'] == 0, 'ecoulement'] = 0
    group.loc[group['sum'].idxmax() + 1, 'ecoulement'] = 1 - group.loc[:, 'sum'].max()

    return group[['3', 'ecoulement']]


def calcul_ecoulement_et_vente(group, nombre_total_unite):
    ntu = nombre_total_unite[(nombre_total_unite['category'].isin(__UNITE_TYPE__)) &
                             (nombre_total_unite['value'] == 'ntu')][[group.name, 'category']].set_index(
        'category').transpose()

    price = nombre_total_unite[(nombre_total_unite['category'].isin(__UNITE_TYPE__)) &
                               (nombre_total_unite['value'] == 'price')][[group.name, 'category']].set_index(
        'category').transpose()
    result = pd.DataFrame([ntu.values[0] for i in range(group.shape[0])], columns=ntu.columns).reset_index(drop=True)

    ecoulement_name = dict()
    for i in range(len(__UNITE_TYPE__)):
        ecoulement_name[__UNITE_TYPE__[i]] = str(37 + i)
    result = result.mul(group['ecoulement'].values, axis=0)
    price = price[ntu.columns]
    mul = result.mul(price.values[0], axis=1)

    result.rename(columns=ecoulement_name, inplace=True)

    result['45'] = result.sum(axis=1)
    result['46'] = result['45'].cumsum()

    ecoulement_name = dict()
    for i in range(len(__UNITE_TYPE__)):
        ecoulement_name[__UNITE_TYPE__[i]] = str(47 + i)
    mul.rename(columns=ecoulement_name, inplace=True)
    mul['54'] = mul.sum(axis=1)
    result = pd.concat([result, mul], axis=1)

    return result


def mid_prcent_unite_vendu(group, d, ntu):
    batim = group.name

    nu = ntu.reset_index().loc[0, batim]
    group['6'] = group['46'].where(group['46'] >= nu, 0)
    group['6'] = group['6'].where(group['6'] == 0, 1)

    group['4'] = group['46'] / nu
    group['4'] = group['4'].where(group['4'] >= d[batim], 0)
    group['4'] = group['4'].where(group['4'] == 0, 1)

    return group[['4', '6']]


def liv_immeuble(group, d):
    batim = group.name
    group = group.cumsum()
    group.where(group > d[batim], 0, inplace=True)
    group.where(group == 0, 1, inplace=True)
    return group


def depot_prevente(group, d):
    group['7'] = ((1 - group['5']) * group['54']).mul(d[group.name])
    group['s7'] = group['7'].cumsum()
    group['s4'] = group['4'].cumsum()
    group['s4'] = group[['s4']].where(group['s4'] == 1, 0)
    group.loc[group[['s4']].idxmax(), '4'] = 0
    group['8'] = group['s4'] * group['s7'] + group['4'] * group['7']
    group['11'] = ((1 - group['5']) * group['54']).mul(1 - d[group.name])
    group['s5'] = group['5'].cumsum()
    group['s5'] = group[['s5']].where(group['s5'] == 1, 0)
    group['12'] = group['s5'] * (group['11'].cumsum())
    group['13'] = group['5'] * group['54']

    return group[['7', '8', '11', '12', '13']]


def sortie_de_fond(group, data, d):
    terrain = data.loc[data['value'] == 'financement terrain', group.name].values[0]
    soft_cost = data.loc[data['value'] == 'soft cost', group.name].values[0]
    hard_cost = data.loc[data['value'] == 'construction cost', group.name].values[0]

    group['15'] = 0
    group['26'] = 0
    group.loc[group.head(1).index, '15'] = -1 * terrain
    group.loc[group.head(1).index, '26'] = d[group.name] * terrain

    group['s5'] = 1 + group['5'].cumsum()
    group['s5'] = group[['s5']].where(group['s5'] == 1, 0)
    group['soft_per'] = group['s5'] * group['1']
    n_soft = group['soft_per'].max()

    group['16'] = -(1 - group['5']) * (soft_cost / n_soft)

    group['17'] = -(group['4'] - group['5']) * (hard_cost / (group['4'].sum() - group['5'].sum()))

    group['18'] = group['15'] + group['16'] + group['17'] + group['26']

    group['s18'] = group['18'].cumsum()
    group['21'] = -1 * group['s18'] - 0.25 * (terrain + soft_cost + hard_cost)
    group['21'] = group[['21']].where(group['21'] > 0, 0)
    group['21'] = group[['21']].where(group['21'] == 0, 1)
    group['19'] = (1 - group['21']) * group['18']
    # group['20'] = (1 - group['19']) * group['18']
    group['20'] = group['18'].where(group['19'] != 0, group['18'])
    group['20'] = group['20'].where(group['19'] == 0, 0)
    group['60'] = group['4'] + group['21']
    group['61'] = group[['60']].where(group['60'] == 2, 0)
    group['61'] = group[['61']].where(group['61'] == 0, 1)

    return group[['15', '16', '17', '18', '19', '20', '21', '26', '60', '61']]


def financement_terrain(group, d):
    # print(group)
    group['s60'] = group[['60']].where(group['60'] >= 2, 1)
    group['s60'] = group[['s60']].where(group['s60'] == 1, 0)

    rate = (1 + d[group.name] / 2) ** (1 / 6) - 1
    rate_tab = [(1 + rate) ** (i) for i in range(group.shape[0])]
    group['rate'] = rate_tab

    group['t'] = group['26'].max()
    group['27'] = group['t'] * group['rate'] * group['s60']
    group['29'] = group['27'] * rate

    index = group[group['s60'] == 0].head(1).index
    group.loc[index.values[0], 's60'] = 1
    group['x'] = group['t'] * group['rate'] * group['s60']
    group['28'] = group['x'] * rate

    group['s61'] = group['61'].cumsum()
    group['s61'] = group[['s61']].where(group['s61'] == 1, 0)

    group['22'] = group['s61'] * group['27']

    group['30'] = group['s61'] * (group['27'] - (group['18'].sum() - group['18'].cumsum() + group['18']))

    return group[['22', '27', '28', '29', '30']]


def other(group, data):
    group['s4'] = group['4'].cumsum()
    group['s4'] = group[['s4']].where(group['s4'] == 1, 0)
    group['s7'] = group['7'].cumsum()
    group['y'] = data[data['value'] == 'max_equite'][group.name].values[0]
    group['x'] = group['s4'] * group['s7']
    group['z'] = group['s4'] * group['y']
    group['9'] = group[['x', 'y']].min(axis=1)
    group['10'] = group['8'] - group['9']
    group['14'] = group['10'] + group['12'] + group['13']
    return group[['9', '10', '14']]


def projet_interest(group, d):
    r = [[0, 0, 0, 0, 0]]
    timeline = group['61'].cumsum().values
    p = group['14'].values
    af = group['30'].values
    rate = (1 + d[group.name] / 2) ** (1 / 6) - 1
    index = group.index
    for time in range(1, len(timeline)):

        if timeline[time] == 0:
            v31 = 0
            v23 = 0
        elif timeline[time] == 1:
            v31 = af[time] - sum(p[:time + 1])
            v23 = -1 * sum(p[:time + 1])
        else:
            v31 = r[time - 1][0] + r[time - 1][2] - p[time]

            if v31 > 0:
                v23 = -1 * p[time]
            else:
                v23 = 0
        v24 = 0
        if r[time - 1][0] > 0 and v31 < 0:
            v24 = -1 * r[time - 1]

        v32 = 0 if v31 < 0 else v31
        v33 = v32 * rate
        r.append([v31, v32, v33, v23, v24])
    v34 = []

    for v in range(1, len(timeline)):
        if r[v][2] == 0:
            v34.append(0)
        else:
            v34.append(r[v - 1][2])
    v34.append(0)
    value = pd.DataFrame(r, columns=['31', '32', '33', '23', '24'])
    value['34'] = v34
    value.set_index(index)
    group = pd.concat([group, value], axis=1)
    group['65'] = group['5'] + group['6']
    group['65'] = group['65'].where(group['65'] == 2, 0)
    group['65'] = group['65'].where(group['65'] == 0, 1)

    group['35'] = group['32'].where(group['32'] > 0, 0)
    group['35'] = group['35'] * group['65'] * -1
    group['25'] = group['24']
    group.loc[group['24'] == 0, '25'] = group['23']

    return group[['23', '24', '25', '31', '32', '33', '34', '35']]


def cashflow(group):
    group['36'] = group.sum(axis=1)
    return group[['36']]


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
    summary = pd.DataFrame(None, columns=cost_table.columns)
    ntu = cost_table[(cost_table['value'] == 'ntu')]
    cost = cost_table[(cost_table['category'] == 'partial')]
    cost_total = cost_table[(cost_table['category'] == 'total')]
    summary = pd.concat([summary, ntu, cost, cost_total])

    c = cost_table[(cost_table['value'] == 'ntu') & (cost_table['category'] == 'ALL')]

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

    result = financials_result[['ecoulement']].groupby(financials_result['batiment']).apply(calcul_ecoulement_et_vente,
                                                                                            cost_table[(cost_table['value'] == 'ntu') |
                                                                                   (cost_table['value'] == 'price')])

    result = result.reset_index(drop=True)
    financials_result = pd.concat([financials_result, result], axis=1)

    # 50% des unites construites
    d_ = dict()
    pos = [16, 2]
    for _ in range(len(__BATIMENT__)):
        d_[__BATIMENT__[_]] = fsh.cell(pos[0], pos[1] + _).value

    c = cost_table[(cost_table['value'] == 'ntu') & (cost_table['category'] == 'ALL')]
    financials_result[['4', '6']] = financials_result[['46']].groupby(financials_result['batiment']).apply(
        mid_prcent_unite_vendu, d_, c)

    # livraison de l'immeuble
    d_ = dict()
    pos = [24, 2]
    for _ in range(len(__BATIMENT__)):
        d_[__BATIMENT__[_]] = fsh.cell(pos[0], pos[1] + _).value

    financials_result['5'] = financials_result[['4']].groupby(financials_result['batiment']).apply(liv_immeuble, d_)

    # ------> ENTREES DE FOND

    # depot de prevente pour calcul
    d_ = dict()
    pos = [7, 2]
    for _ in range(len(__BATIMENT__)):
        d_[__BATIMENT__[_]] = fsh.cell(pos[0], pos[1] + _).value

    # Depot de prevente total - pour calcul
    tab = financials_result[['4', '5', '54']].groupby(financials_result['batiment'])
    financials_result[['7', '8', '11', '12', '13']] = tab.apply(depot_prevente, d_)

    # --------- SORTIE DE FONDS
    d_ = dict()
    pos = [6, 2]
    for _ in range(len(__BATIMENT__)):
        d_[__BATIMENT__[_]] = fsh.cell(pos[0], pos[1] + _).value

    cost = cost_table[cost_table['category'] == 'partial']
    tab = financials_result[['1', '4', '5']].groupby(financials_result['batiment'])
    financials_result[['15', '16', '17', '18', '19', '20', '21', '26', '60', '61']] = tab.apply(sortie_de_fond, cost,
                                                                                                d_)

    # ------------ FINANCEMENT

    # Financement terrain
    d_ = dict()
    pos = [13, 2]
    for _ in range(len(__BATIMENT__)):
        d_[__BATIMENT__[_]] = fsh.cell(pos[0], pos[1] + _).value

    tab = financials_result[['18', '26', '60', '61']].groupby(financials_result['batiment'])
    financials_result[['22', '27', '28', '29', '30']] = tab.apply(financement_terrain, d_)

    fin_proj = financials_result[['28']].groupby(financials_result['batiment']).sum().reset_index()
    fin_proj = fin_proj.set_index('batiment').transpose()
    c = summary[summary['category'] == 'partial'][batiment].sum()
    fin_proj = fin_proj.reset_index(drop=True) + c.values

    d_ = dict()
    pos = [5, 2]
    for _ in range(len(__BATIMENT__)):
        d_[__BATIMENT__[_]] = fsh.cell(pos[0], pos[1] + _).value

    _ = []
    for b in batiment:
        _.append(d_[b])
    max_equite = fin_proj * _

    max_equite['sector'] = secteur
    max_equite['category'] = 'total'
    max_equite['value'] = 'max_equite'
    max_equite = max_equite[cost_table.columns]

    fin_proj['sector'] = secteur
    fin_proj['category'] = 'total'
    fin_proj['value'] = 'financement projet (ne compte pas interet projet'
    fin_proj = fin_proj[cost_table.columns]
    summary = pd.concat([summary, fin_proj, max_equite], ignore_index=True)

    tab = financials_result[['4', '7', '8', '12', '13']].groupby(financials_result['batiment'])
    financials_result[['9', '10', '14']] = tab.apply(other, summary)

    d_ = dict()
    pos = [14, 2]
    for _ in range(len(__BATIMENT__)):
        d_[__BATIMENT__[_]] = fsh.cell(pos[0], pos[1] + _).value

    tab = financials_result[['5', '6', '14', '30', '61']].groupby(financials_result['batiment'])
    financials_result[['23', '24', '25', '31', '32', '33', '34', '35']] = tab.apply(projet_interest, d_)

    tab = financials_result[['9', '14', '15', '16', '17', '22', '25', '26', '30', '35']].groupby(
        financials_result['batiment'])
    financials_result['36'] = tab.apply(cashflow)

    # KEY STATISTICS ------------------------------------------------------------------------->

    print(summary)
    inter_terr = financials_result[['29']].groupby(financials_result['batiment']).sum().reset_index()
    inter_terr = inter_terr.set_index('batiment').transpose()
    inter_terr['category'] = 'total'
    inter_terr['value'] = 'total interet terrrain'
    inter_terr['sector'] = secteur
    inter_terr = inter_terr[cost_table.columns]

    inter_proj = financials_result[['34']].groupby(financials_result['batiment']).sum().reset_index()
    inter_proj = inter_proj.set_index('batiment').transpose()
    inter_proj['category'] = 'total'
    inter_proj['value'] = 'total interet projet'
    inter_proj['sector'] = secteur
    inter_proj = inter_proj[cost_table.columns]

    x = financials_result[['33']].groupby(financials_result['batiment']).sum().reset_index()
    y = financials_result[['30']].groupby(financials_result['batiment']).sum().reset_index()

    print("Internal rate of return {0:.2f}%".format(100 * np.irr(financials_result['36'])))

    summary = pd.concat([summary, inter_terr, inter_proj], ignore_index=True)
    # print(cost_param)

    # print(financials_result[['1', '4', '21', '5', '27', '28', '30']].head(50))
    return




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

    intrant_param = get_cb4_characteristics(intrant_param, "Secteur 7", 5389.0, None, None, None, None)

    intrant_param = intrant_param[['sector', 'category', 'value', 'B8']]
    intrant_param = intrant_param[(intrant_param['sector'] == 'Secteur 7')]

    cost_param = cost_param[['sector', 'category', 'value', 'B8']]
    cost_param = cost_param[cost_param['sector'] == 'Secteur 7']
    cost_table = calcul_cout_batiment(intrant_param, cost_param, 'Secteur 7', ['B8'], myBook)

    myBook = xlrd.open_workbook(__FINANCE_FILES_NAME__)
    calcul_detail_financier(cost_table, None, 'Secteur 7', ['B8'], myBook, 120)
