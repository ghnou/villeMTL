import numpy as np
import pandas as pd
import xlrd

from calcul_de_couts import calculate_cost
from lexique import __FILES_NAME__, __BATIMENT__, __SECTEUR__, __UNITE_TYPE__
from obtention_intrant import get_all_informations

__author__ = 'pougomg'


def debut_des_ventes(data, d, fonction_ecoulement):

    """
        This function is used to calculate the beginning of the sales of the building.
        :param
        group (pd.DataFrame): Dataframe of month timeline
         d (dict): Duree moyenne entre l'achat de terrain et le debut de la prevente

        :return
        pd.Serie: Serie contenant le timeline de debut de prevente
    """

    group = data.copy()
    name = data.name
    value = d[name].values[0]
    f3mo = fonction_ecoulement.loc[fonction_ecoulement['value'] == 'ecob3mo', name].values[0]
    f3mo_after = fonction_ecoulement.loc[fonction_ecoulement['value'] == 'ecoa3mo', name].values[0]


    group['3'] = group['1'].where(group['1'].astype(int) >= value, 0)
    group['3'] = group['3'].where(group['3'] == 0, 1)

    t = group.reset_index(drop=True)
    x1 = (f3mo / 3)* ( np.heaviside(t.index - value + 2, 0) - np.heaviside(t.index - value - 1, 0))
    x2 = (f3mo_after / 3) * (np.heaviside(t.index - value - 1, 0))

    group['ecoulement'] = x1 + x2

    return group[['3', 'ecoulement']]


def calculate_vente_ecoulement(t, ecoulement, studios, cc1, cc2, cc3, penth, cc2f, cc3f):

    global residuel, ntu, i, unit_sold_f3mo, us, nu, after3mo

    value = ecoulement * ntu.values
    value = value + residuel
    result = value.astype(float).round(0)

    if i>0:
        residuel = value - result
        unit_sold_f3mo = unit_sold_f3mo + result
        v = nu.values - us - result
        _ = pd.DataFrame(v[0], columns=['value'], index=__UNITE_TYPE__)
        _['result'] = result[0]
        _.loc[:,'result'] = _['result'].where(_['value']>=0,_['value'] + _['result'])
        result = np.array([_['result']])
        us = us + result

    i += 1
    if i == after3mo + 4:
        ntu = ntu - unit_sold_f3mo

    return pd.Series(result[0])


def calcul_ecoulement_et_vente(data, nombre_total_unite):

    global residuel, ntu, i, unit_sold_f3mo, nu, us, after3mo
    i = 0
    after3mo = data.reset_index()
    after3mo = after3mo[after3mo['4'] == 1].head(1).index[0]
    name = data.name
    group = data.copy()
    ntu = nombre_total_unite[(nombre_total_unite['category'].isin(__UNITE_TYPE__)) &
                             (nombre_total_unite['value'] == 'ntu')][[name, 'category']].set_index(
        'category').transpose()
    ntu = ntu.astype(float).round(0)
    ntu = ntu[__UNITE_TYPE__]
    nu = ntu
    result = pd.concat([ntu] * group.shape[0], ignore_index=True)
    result.loc[:, __UNITE_TYPE__] = 0

    # Price infos
    price = nombre_total_unite[(nombre_total_unite['category'].isin(__UNITE_TYPE__)) &
                               (nombre_total_unite['value'] == 'price')][[name, 'category']].set_index(
        'category').transpose()
    price = pd.concat([price] * group.shape[0], ignore_index=True)

    si = nombre_total_unite[nombre_total_unite['value'] == 'si'][name].values[0]

    stat = nombre_total_unite[nombre_total_unite['value'] == 'stat'][name].values[0]

    parc = nombre_total_unite[nombre_total_unite['value'] == 'frais_parc'][name].values[0]
    result.set_index(group.index, inplace=True)
    group = pd.concat([group, result], axis=1)

    # Ecoulement de ventes
    residuel = np.zeros(len(__UNITE_TYPE__))
    unit_sold_f3mo = np.zeros(len(__UNITE_TYPE__))
    us = np.zeros(len(__UNITE_TYPE__))

    # result = group.loc[group['3'].cumsum() > 0]
    result = group.apply(lambda row: calculate_vente_ecoulement(*row[['3', 'ecoulement'] + __UNITE_TYPE__]),
                         axis=1)
    result.columns = __UNITE_TYPE__
    # result = result.astype(int)

    price = price[__UNITE_TYPE__]
    mul = result.mul(price.values[0], axis=1)

    result['42'] = result.sum(axis=1)
    result['43'] = result['42'].cumsum()

    ecoulement_name = dict()
    for i in range(len(__UNITE_TYPE__)):
        ecoulement_name[__UNITE_TYPE__[i]] = str(34 + i)

    result.rename(columns=ecoulement_name, inplace=True)

    ecoulement_name = dict()
    for i in range(len(__UNITE_TYPE__)):
        ecoulement_name[__UNITE_TYPE__[i]] = str(44 + i)

    mul.rename(columns=ecoulement_name, inplace=True)
    mul['51'] = mul.sum(axis=1)

    result = pd.concat([result, mul], axis=1)
    result['52'] = result['42'] * si * stat
    result['14'] = -1 * parc
    result.loc[group['3'].cumsum()!=1, '14'] = 0

    return result


def mid_prcent_unite_vendu(data, d, ntu):

    batim = data.name
    group = data.copy()

    nu = int(ntu.reset_index().loc[0, batim])

    value = d[batim].values[0]
    group['9'] = group['43'].where(group['43'] >= nu, 0)
    group['9'] = group['9'].where(group['9'] == 0, 1)

    group['4'] = group['43'] / nu
    group['4'] = group['4'].where(group['4'] >= value, 0)
    group['4'] = group['4'].where(group['4'] == 0, 1)

    return group[['4', '9']]


def liv_immeuble(data, d):

    batim = data.name
    group = data.copy()

    value = d[batim].values[0]
    group = group.cumsum()
    group.where(group >= value, 0, inplace=True)
    group.where(group == 0, 1, inplace=True)

    return group


def get_25(data, d):
    batim = data.name
    group = data.copy()
    value = d[batim].values[0]
    group['x'] = -1 * group.sum(axis=1)
    group['7'] = 0
    group.loc[group['x'].cumsum() > 0.25 * value, '7'] = 1

    return group[['7']]

def depot_prevente(data, d):

    group = data.copy()
    name = data.name
    value = d[name].values[0]

    group['21'] = ((1 - group['8']) * group['51']).mul(value)
    group['23'] = ((1 - group['8']) * group['51']).mul(1 - value)

    group['x'] = 1
    group.loc[group['5'].cumsum() != 1, 'x'] = 0
    group['22'] = (group['x'].shift(-1) * group['21'].cumsum()).shift(1) + group['21'] * group['5']

    group['x'] = 1
    group.loc[group['8'].cumsum() != 1, 'x'] = 0
    group['24'] = group['x'] * group['23'].cumsum()
    group['25'] = group['8'] * group['51']
    group['26'] = group['x'] * group['52'].cumsum()


    return group[['21', '22', '23', '24', '25', '26']]


def sortie_de_fond(data, cost, d):
    "5 = 8, 4 = 5"
    group = data.copy()
    name = data.name
    value = d[name].values[0]

    terrain = cost.loc[cost['value'] == 'financement terrain', name].values[0]
    soft_cost = cost.loc[cost['value'] == 'soft cost', name].values[0]
    hard_cost = cost.loc[cost['value'] == 'hard cost', name].values[0]

    group['10'] = 0
    group['27'] = 0
    group.loc[group.head(1).index, '10'] = -1 * terrain
    group.loc[group.head(1).index, '27'] = value * terrain
    group.loc[:, '19'] = group['27']

    group['s8'] = 1 + group['8'].cumsum()
    group['s8'] = group[['s8']].where(group['s8'] == 1, 0)
    group['soft_per'] = group['s8'] * group['1']
    n_soft = group['soft_per'].max()

    group['11'] = -(1 - group['8']) * (soft_cost / n_soft)

    group['12'] = -(group['5'] - group['8']) * (hard_cost / (group['5'].sum() - group['8'].sum()))

    # group['18'] = group['15'] + group['16'] + group['17'] + group['26']
    #
    # group['s18'] = group['18'].cumsum()
    # group['21'] = -1 * group['s18'] - 0.25 * (terrain + soft_cost + hard_cost)
    # group['21'] = group[['21']].where(group['21'] > 0, 0)
    # group['21'] = group[['21']].where(group['21'] == 0, 1)
    #
    # group['21s'] = group['21'].shift(1).fillna(0)
    # group['19'] = (1 - group['21s']) * group['18']
    # # group['20'] = (1 - group['19']) * group['18']
    # group['20'] = group['18'].where(group['19'] != 0, group['18'])
    # group['20'] = group['20'].where(group['19'] == 0, 0)
    # group['60'] = group['4'] + group['21']
    # group['61'] = group[['60']].where(group['60'] == 2, 0)
    # group['61'] = group[['61']].where(group['61'] == 0, 1)

    return group[['10', '11', '12', '19', '27']]


def financement_terrain(data, d):

    group = data.copy()
    name = data.name

    group['s60'] = group[['60']].where(group['60'] >= 2, 1)
    group['s60'] = group[['s60']].where(group['s60'] == 1, 0)

    rate = (1 + d[name] / 2) ** (1 / 6) - 1
    rate_tab = [(1 + rate) ** (i) for i in range(group.shape[0])]
    group['rate'] = rate_tab

    group['t'] = group['26'].max()
    group['27'] = group['t'] * group['rate'] * group['s60']
    group['29'] = group['27'] * rate

    index = group[group['s60'] == 0].head(1).index
    group.loc[index.values[0], 's60'] = 1
    group.loc[index.values[0], '27']  = group.loc[index.values[0], 't']* group.loc[index.values[0], 'rate']
    group['x'] = group['t'] * group['rate'] * group['s60']
    group['28'] = group['x'] * rate

    group['s61'] = group['61'].cumsum()
    group['s61'] = group[['s61']].where(group['s61'] == 1, 0)

    group['22'] = -1 * group['s61'] * group['27']

    group['30'] = group['s61'] * (group['27'] - (group['18'].sum() - group['18'].cumsum()))

    return group[['22', '27', '28', '29', '30']]


def other(data, summary):

    group = data.copy()
    name = data.name
    group['s4'] = group['4'].cumsum()
    group['s4'] = group[['s4']].where(group['s4'] == 1, 0)
    group['s7'] = group['7'].cumsum()
    group['y'] = summary[summary['value'] == 'Maximum equite fin - preventes'][name].values[0]
    group['x'] = group['s4'] * group['s7']
    group['z'] = group['s4'] * group['y']
    group['9'] = group[['x', 'y']].min(axis=1)
    group['10'] = group['8'] - group['9']
    group['14'] = group['10'] + group['12'] + group['13']

    return group[['9', '10', '14']]


def projet_interest(data, d):

    group = data.copy()
    name = data.name

    r = [[0, 0, 0, 0, 0]]
    timeline = group['61'].cumsum().values
    p = group['14'].values
    af = group['30'].values
    rate = (1 + d[name] / 2) ** (1 / 6) - 1
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
            v24 = -1 * r[time - 1][0]

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
    value.set_index(index, inplace=True)
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


def calculate_irr(group):

    return  100 * np.irr(group.fillna(0).values)


def calcul_detail_financier(cost_table, secteur, batiment, my_book, timeline) -> pd.DataFrame:

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

    financials_result = []
    for bat in batiment:
        x = pd.DataFrame([[bat, i + 1, 1] for i in range(timeline)], columns=['batiment', '1', '2'])
        x['3'] = [0, 0] + [1 for i in range(timeline - 2)]
        financials_result.append(x)
    financials_result = pd.concat(financials_result,
                                  ignore_index=True)



    ###################################################################################################################
    #
    # Time line and sales Zone.
    #
    ###################################################################################################################

    # Debut des ventes
    data = cost_table[cost_table['value'] == 'dm_ach_prev']
    fonction_ecoulement = cost_table[cost_table['value'].isin(['ecob3mo', 'ecoa3mo'])]


    # Fonction d'ecoulement
    t = financials_result[['1']].groupby(financials_result['batiment'])
    financials_result[['4', 'ecoulement']] = t.apply(debut_des_ventes, data, fonction_ecoulement).reset_index(level=0, drop=True)


    # Ventes (Ecoulement et revenus bruts)
    result = financials_result[['3', '4', 'ecoulement']].groupby(financials_result['batiment'])
    result = result.apply(calcul_ecoulement_et_vente, cost_table[cost_table['value'].isin(['ntu', 'price', 'si', 'stat', 'frais_parc'])])

    result = result.reset_index(drop=True)
    financials_result = pd.concat([financials_result, result], axis=1)



    # 50% des unites construites

    data = cost_table[cost_table['value'] == 'nv_min_prev_av_deb']
    c = cost_table[(cost_table['value'] == 'ntu') & (cost_table['category'] == 'ALL')]
    t = financials_result[['43']].groupby(financials_result['batiment'])
    financials_result[['5', '9']] = t.apply(mid_prcent_unite_vendu, data, c).reset_index(level=0, drop=True)


    # livraison de l'immeuble
    data = cost_table[cost_table['value'] == 'dur_moy_const']

    t = financials_result[['5']].groupby(financials_result['batiment'])
    financials_result['8'] = t.apply(liv_immeuble, data).reset_index(level=0, drop=True)


     ###################################################################################################################
    #
    # Sortie de fond Zone.
    #
    ###################################################################################################################

    data = cost_table[cost_table['value'] == 'eq_terr']


    cost = cost_table[cost_table['category'] == 'partial']
    tab = financials_result[['1', '5', '8']].groupby(financials_result['batiment']).apply(sortie_de_fond, cost, data)
    financials_result[['10', '11', '12', '19', '27']] = tab.reset_index(level=0, drop=True)


    ###################################################################################################################


    ###################################################################################################################
    #
    # Entree de fond Zone.
    #
    ###################################################################################################################

    # depot de prevente pour calcul
    data = cost_table[cost_table['value'] == 'pp_prev']

    # Depot de prevente total - pour calcul
    tab = financials_result[['5', '8', '51', '52']].groupby(financials_result['batiment'])
    financials_result[['21', '22', '23', '24', '25', '26']] = tab.apply(depot_prevente, data).reset_index(level=0, drop=True)

    ###################################################################################################################
    #
    # 25% equite atteinte.
    #
    ###################################################################################################################

    data = cost_table[cost_table['value'] == 'cout total du projet']
    print(data)
    tab = financials_result[['10', '11', '12', '14', '27']].groupby(financials_result['batiment'])
    financials_result[['6']] = tab.apply(get_25, data).reset_index(level=0, drop=True)
    # financials_result.to_excel('x.xlsx')


    return
    #
    # Financement.
    #
    ###################################################################################################################

    # Financement terrain
    d_ = dict()
    pos = [13, 2]
    for _ in range(len(__BATIMENT__)):
        d_[__BATIMENT__[_]] = fsh.cell(pos[0], pos[1] + _).value

    tab = financials_result[['18', '26', '60', '61']].groupby(financials_result['batiment'])
    financials_result[['22', '27', '28', '29', '30']] = tab.apply(financement_terrain, d_).reset_index(level=0, drop=True)

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
    max_equite['value'] = 'Maximum equite fin - preventes'
    max_equite = max_equite[cost_table.columns]

    fin_proj['sector'] = secteur
    fin_proj['category'] = 'total'
    fin_proj['value'] = 'financement projet (ne compte pas interet du financement projet)'
    fin_proj = fin_proj[cost_table.columns]
    summary = pd.concat([summary, fin_proj, max_equite], ignore_index=True)

    tab = financials_result[['4', '7', '8', '12', '13']].groupby(financials_result['batiment'])
    financials_result[['9', '10', '14']] = tab.apply(other, summary).reset_index(level=0, drop=True)


    d_ = dict()
    pos = [14, 2]
    for _ in range(len(__BATIMENT__)):
        d_[__BATIMENT__[_]] = fsh.cell(pos[0], pos[1] + _).value

    tab = financials_result[['5', '6', '14', '30', '61']].groupby(financials_result['batiment']).apply(projet_interest,
                                                                                                       d_)
    financials_result[['23', '24', '25', '31', '32', '33', '34', '35']] = tab.reset_index(level=0, drop=True)

    financials_result['36'] = financials_result[['9', '14', '15', '16', '17', '22', '25', '26', '30', '35']].sum(axis=1)


    ###################################################################################################################
    #
    # Key Statistics.
    #
    ###################################################################################################################

    # total interet terrain
    inter_terr = financials_result[['29']].groupby(financials_result['batiment']).sum().reset_index()
    inter_terr = inter_terr.set_index('batiment').transpose()
    inter_terr['category'] = 'total'
    inter_terr['value'] = 'total interet terrain'
    inter_terr['sector'] = secteur
    inter_terr = inter_terr[cost_table.columns]

    # total interet projet
    inter_proj = financials_result[['34']].groupby(financials_result['batiment']).sum().reset_index()
    inter_proj = inter_proj.set_index('batiment').transpose()
    inter_proj['category'] = 'total'
    inter_proj['value'] = 'total interet projet'
    inter_proj['sector'] = secteur
    inter_proj = inter_proj[cost_table.columns]

    # total cout avec interet
    total_cout_interet = summary[summary['value'] == 'cout total du projet'][batiment].reset_index()
    total_cout_interet = total_cout_interet + inter_terr[batiment].reset_index(drop=True) + inter_proj[batiment].reset_index(drop=True)
    total_cout_interet['category'] = 'total'
    total_cout_interet['value'] = 'total cout avec interet'
    total_cout_interet['sector'] = secteur
    total_cout_interet = total_cout_interet[cost_table.columns]

    # Financement projet (avec interets)
    fin_proj_av_i = financials_result[['30', '33']].sum(axis=1).groupby(financials_result['batiment']).sum().reset_index()
    fin_proj_av_i = fin_proj_av_i.set_index('batiment').transpose()
    fin_proj_av_i['sector'] = secteur
    fin_proj_av_i['category'] = 'total'
    fin_proj_av_i['value'] = 'Financement projet (avec interets)'
    fin_proj_av_i = fin_proj_av_i[cost_table.columns]

    # Equite dans le projet
    eq_dans_proj = financials_result[['19']].groupby(financials_result['batiment']).sum().reset_index()
    eq_dans_proj =-1 * eq_dans_proj.set_index('batiment').transpose()
    eq_dans_proj['sector'] = secteur
    eq_dans_proj['category'] = 'total'
    eq_dans_proj['value'] = 'Equite dans le projet'
    eq_dans_proj = eq_dans_proj[cost_table.columns]

    # Revenus totaux
    rev_totaux = financials_result[['54']].groupby(financials_result['batiment']).sum().reset_index()
    rev_totaux = rev_totaux.set_index('batiment').transpose()
    rev_totaux['sector'] = secteur
    rev_totaux['category'] = 'total'
    rev_totaux['value'] = 'revenus totaux'
    rev_totaux = rev_totaux[cost_table.columns]

    # Profit net
    prof_net = rev_totaux[batiment].reset_index(drop=True) - total_cout_interet[batiment].reset_index(drop=True)
    prof_net['sector'] = secteur
    prof_net['category'] = 'total'
    prof_net['value'] = 'profit net'
    prof_net = prof_net[cost_table.columns]



    # Somme cash flow
    csh_flow = financials_result[['36']].groupby(financials_result['batiment']).sum().reset_index()
    csh_flow = csh_flow.set_index('batiment').transpose()
    csh_flow['category'] = 'total'
    csh_flow['value'] = 'somme cash flow'
    csh_flow['sector'] = secteur
    csh_flow = csh_flow[cost_table.columns]

    # TRI
    tri = financials_result['36'].groupby(financials_result['batiment']).apply(calculate_irr)
    tri = tri.to_frame().transpose()
    tri['category'] = 'total'
    tri['value'] = 'TRI'
    tri['sector'] = secteur
    tri = tri[cost_table.columns]

    summary = pd.concat([summary, inter_terr, inter_proj, total_cout_interet, fin_proj_av_i, eq_dans_proj, rev_totaux,
                         prof_net, csh_flow, tri],
                        ignore_index=True)

    financials_result.to_excel('out.xlsx')
    print(summary[['category', 'value'] + batiment])
    summary.to_excel('summary.xlsx')
    return

def calculate_financial(type, secteur, batiment, params, timeline=120, *args):

    cost_table = calculate_cost(type, secteur, batiment, params, *args)
    calcul_detail_financier(cost_table, secteur, batiment, None, timeline)


if __name__ == '__main__':

    myBook = xlrd.open_workbook(__FILES_NAME__)
    x = get_all_informations(myBook)
    args = dict()
    supter = [50000]
    densite = [10]
    calculate_financial('CA3', [__SECTEUR__[-1]], __BATIMENT__[-1: ], x, 120, args)
