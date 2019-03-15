import numpy as np
import pandas as pd
import xlrd

import time

from calcul_de_couts import calculate_cost, calcul_cout_batiment
from lexique import __FILES_NAME__, __BATIMENT__, __SECTEUR__, __UNITE_TYPE__
from unicodedata import normalize
from obtention_intrant import get_all_informations, get_intrants, get_cb3_characteristic, get_ca_characteristic

__author__ = 'pougomg'


#######################################################################################################################
#
# Importants functions
#
#######################################################################################################################

def format_for_output(data):

    sector = data.name
    group = data.copy()
    group.loc[group['value'] == 'ntu', 'value'] = group['category']
    group.drop(['category', 'type', 'sector'], axis=1, inplace=True)
    group = group.set_index('value').transpose().reset_index()
    group.rename(columns={'index': 'batiment', 'ALL': 'Nombre unites'}, inplace=True)
    group['sector'] = sector

    return group


def debut_des_ventes(data, dm_1, dm_prev, parc_fee):

    """
        This function is used to calculate the beginning of the sales of the building.
        :param
        group (pd.DataFrame): Dataframe of month timeline
         d (dict): Duree moyenne entre l'achat de terrain et le debut de la prevente

        :return
        pd.Serie: Serie contenant le timeline de debut de prevente
    """
    group = data.copy()
    batiment = data.name[1]
    sector = data.name[0]

    value = dm_1[batiment].values[0]
    group['3'] = group['1'].where(group['1'].astype(int) > value, 0)
    group['3'] = group['3'].where(group['3'] == 0, 1)
    group['4'] = group['3']

    value = dm_prev[batiment].values[0]
    group['5'] = group['1'].where(group['1'].astype(int) > value, 0)
    group['5'] = group['5'].where(group['5'] == 0, 1)

    parc = parc_fee[(parc_fee['value'] == 'frais_parc') &
                             (parc_fee['sector'] == sector)][[batiment, 'category']].set_index('category').transpose()

    rem = parc_fee[(parc_fee['value'] == 'rem') &
                             (parc_fee['sector'] == sector)][[batiment, 'category']].set_index('category').transpose()

    contrib_terr_hs = parc_fee[(parc_fee['value'] == 'contrib_terr_hs') &
                   (parc_fee['sector'] == sector)][[batiment, 'category']].set_index('category').transpose()
    contrib_fin = parc_fee[(parc_fee['value'] == 'contrib_fin') &
                               (parc_fee['sector'] == sector)][[batiment, 'category']].set_index('category').transpose()

    rem = rem['unique'].values[0]
    parc = parc['unique'].values[0]
    contrib_terr_hs = contrib_terr_hs['unique'].values[0]
    contrib_fin = contrib_fin['unique'].values[0]

    group['17'] = 0
    group['18'] = 0
    group['16'] = 0
    group.loc[:, '17'] = group['17'].where(group['3'].cumsum() != 1, -parc)
    group.loc[:, '18'] = group['18'].where(group['3'].cumsum() != 1, -rem)
    group.loc[:, '16'] = group['18'].where(group['3'].cumsum() != 1, -contrib_terr_hs -contrib_fin)
    return group[['3', '4', '5', '16', '17', '18']]


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

    # global residuel, ntu, i, unit_sold_f3mo, nu, us, after3mo
    # i = 0
    # after3mo = data.reset_index()
    # after3mo = after3mo[after3mo['4'] == 1].head(1).index[0]
    # name = data.name
    # group = data.copy()
    # ntu = nombre_total_unite[(nombre_total_unite['category'].isin(__UNITE_TYPE__)) &
    #                          (nombre_total_unite['value'] == 'ntu')][[name, 'category']].set_index(
    #     'category').transpose()
    # ntu = ntu.astype(float).round(0)
    # ntu = ntu[__UNITE_TYPE__]
    # nu = ntu
    # result = pd.concat([ntu] * group.shape[0], ignore_index=True)
    # result.loc[:, __UNITE_TYPE__] = 0
    #
    # # Price infos
    # price = nombre_total_unite[(nombre_total_unite['category'].isin(__UNITE_TYPE__)) &
    #                            (nombre_total_unite['value'] == 'price')][[name, 'category']].set_index(
    #     'category').transpose()
    # price = pd.concat([price] * group.shape[0], ignore_index=True)
    #
    # si = nombre_total_unite[nombre_total_unite['value'] == 'si'][name].values[0]
    #
    # stat = nombre_total_unite[nombre_total_unite['value'] == 'stat'][name].values[0]
    #
    # result.set_index(group.index, inplace=True)
    # group = pd.concat([group, result], axis=1)
    #
    # # Ecoulement de ventes
    # residuel = np.zeros(len(__UNITE_TYPE__))
    # unit_sold_f3mo = np.zeros(len(__UNITE_TYPE__))
    # us = np.zeros(len(__UNITE_TYPE__))
    #
    # # result = group.loc[group['3'].cumsum() > 0]
    # result = group.apply(lambda row: calculate_vente_ecoulement(*row[['3', 'ecoulement'] + __UNITE_TYPE__]),
    #                      axis=1)
    # result.columns = __UNITE_TYPE__
    # result = result.astype(int)

    ############################################################################################################
    #
    # Revenus
    #
    ############################################################################################################

    sector, batiment = data.name
    group = data.copy()
    global start_pos, ecob3mo, ecoa3mo, residu

    ntu = nombre_total_unite[(nombre_total_unite['category'].isin(__UNITE_TYPE__)) &
                             (nombre_total_unite['value'] == 'ntu') &
                             (nombre_total_unite['sector'] == sector)][[batiment, 'category']].set_index('category').transpose()
    ecob3mo = nombre_total_unite[(nombre_total_unite['category'].isin(__UNITE_TYPE__)) &
                             (nombre_total_unite['value'] == 'ecob3mo')
                              ][[batiment, 'category']].set_index('category').transpose()
    ecoa3mo = nombre_total_unite[(nombre_total_unite['category'].isin(__UNITE_TYPE__)) &
                             (nombre_total_unite['value'] == 'ecoa3mo')
                              ][[batiment, 'category']].set_index('category').transpose()

    residu = [[0, 0, 0, 0, 0, 0, 0]]
    start_pos = 0

    def fonction_ecoulement(timeline, studio, cc_1, cc_2, cc_3, penth, cc_2_fam, cc_3_fam):

        global start_pos, ecob3mo, ecoa3mo, residu

        if timeline == 1:
            start_pos += 1
            if start_pos < 4:
                value = ecob3mo.values.astype(int)
                residu = residu + ecob3mo.values - value
                value = value + residu.astype(int)
                residu = residu - residu.astype(int)
                return pd.Series(value[0])
            else:
                value = ecoa3mo.values.astype(int)
                residu = residu + ecoa3mo.values - value
                value = value + residu.astype(int)
                residu = residu - residu.astype(int)
                return pd.Series(value[0])
        else:
            return pd.Series(np.array([0, 0, 0, 0, 0, 0, 0]))

    di = dict()

    for x in __UNITE_TYPE__:
        di[x] = 0
    group = group.assign(**di)

    group.loc[group.index[1]:, __UNITE_TYPE__] = group.loc[group.index[1]:, ['4'] + __UNITE_TYPE__].apply(
        lambda row: fonction_ecoulement(*row[['4'] + __UNITE_TYPE__]),
        axis=1).values

    group = group[__UNITE_TYPE__].where(group[__UNITE_TYPE__].cumsum() <= ntu[__UNITE_TYPE__].values,
                                        group[__UNITE_TYPE__] - group[__UNITE_TYPE__].cumsum() + ntu[__UNITE_TYPE__].values[0])
    result = group[__UNITE_TYPE__].where(group[__UNITE_TYPE__] > 0, 0)

    price = nombre_total_unite[(nombre_total_unite['category'].isin(__UNITE_TYPE__)) &
                             (nombre_total_unite['value'] == 'price') &
                             (nombre_total_unite['sector'] == sector)][[batiment, 'category']].set_index('category').transpose()
    price = price[__UNITE_TYPE__]
    mul = result[__UNITE_TYPE__].mul(price.values[0], axis=1)

    result['48'] = result.sum(axis=1)
    result['49'] = result['48'].cumsum()

    ecoulement_name = dict()
    for i in range(len(__UNITE_TYPE__)):
        ecoulement_name[__UNITE_TYPE__[i]] = str(40 + i)

    result.rename(columns=ecoulement_name, inplace=True)

    ecoulement_name = dict()
    for i in range(len(__UNITE_TYPE__)):
        ecoulement_name[__UNITE_TYPE__[i]] = str(50 + i)

    mul.rename(columns=ecoulement_name, inplace=True)
    mul['58'] = mul.sum(axis=1)

    si = nombre_total_unite[(nombre_total_unite['value'] == 'si') &
                             (nombre_total_unite['sector'] == sector)][[batiment, 'category']].set_index('category').transpose()

    stat = nombre_total_unite[(nombre_total_unite['value'] == 'stat') &
                             (nombre_total_unite['sector'] == sector)][[batiment, 'category']].set_index('category').transpose()

    result = pd.concat([result, mul], axis=1)
    result['59'] = result['48'] * si.values[0] * stat.values[0]

    return result


def mid_prcent_unite_vendu(data, d, ntu):

    batim = data.name[1]
    sector = data.name[0]
    group = data.copy()
    nu = int(ntu[ntu['sector'] == sector].reset_index().loc[0, batim])
    value = d[batim].values[0]

    group['11'] = group['49'].where(group['49'] >= nu, 0)
    group['11'] = group['11'].where(group['11'] == 0, 1)

    group['6'] = group['49'] / nu
    group['6'] = group['6'].where(group['6'] >= value, 0)
    group['6'] = group['6'].where(group['6'] == 0, 1)

    group.loc[:, '8'] = group['6'] + group['5']

    group['8'] = group['8'].where(group['8']>1, 0)
    group['8'] = group['8'].where(group['8']==0, 1)


    return group[['6', '8', '11']]


def liv_immeuble(data, d):

    batim = data.name[1]
    group = data.copy()

    value = d[batim].values[0]
    group.loc[:, '8'] = group['8'].cumsum()
    group.loc[:, '10'] = group['8'].where(group['8'] > value, 0)
    group.loc[:, '10'] = group['10'].where(group['10'] == 0, 1)
    group['10'] = group['10'].where(group['10'] == 0, 1)
    group['60'] = 0
    group.loc[group['10'].cumsum() == 1, '60'] = group['1']
    group['61'] = group['60']

    return group[['10', '60', '61']]


def get_25(data, d):
    batim = data.name
    group = data.copy()
    value = d[batim].values[0]
    group['x'] = -1 * group.sum(axis=1)
    group['7'] = 0
    group.loc[group['x'].cumsum() > 0.25 * value, '7'] = 1

    return group[['7']]


def financement_commencer(data):

    group = data.copy()
    group['7'] = group.sum(axis=1)
    group.loc[group['7'] < 2, '7'] = 0
    group.loc[group['7'] == 2, '7'] = 1

    return group[['7']]


def sold_fin_terr(data, d):

    batiment = data.name[1]
    group = data.copy()
    rate = d[batiment].values[0] / 12
    global cum, interet

    def calculate(time_line, sorties, entree):
        global interet, cum
        cum = (sorties - entree + cum + interet) * (1 - time_line)
        interet = rate * cum

        return cum

    group['70'] = group['68'] - group['69']
    cum = group['70'].head(1).values[0]
    interet = cum * rate
    group.loc[group.index[1]:, '70'] = group.loc[group.index[1]:].apply(lambda row: calculate(*row[['10', '68', '69']]),
                                                                        axis=1)
    group['71'] = group['70'] * rate

    nb_month = group['60'].loc[group['60'].idxmax()]

    group['15'] = -1 * (1 - group['10']) * (group['71'].sum()/(nb_month - 1))

    return group[['15', '70', '71']]


def pret_projet(data, d):

    name = data.name
    group = data.copy()
    rate = (1 + d[name].values[0] / 2) ** (1 / 6) - 1

    group['rate'] = 0
    group.loc[group['7'].cumsum()>0, 'rate'] = rate

    # group['rate'] = (1 + group['rate']).cumprod()
    # group.loc[group['7'].cumsum() == 0, 'rate'] = 0

    group['x'] = group[['11', '12', '15', '14', '22']].sum(axis=1)
    group['x'] = group['7'] * group['x']
    group['y'] = (group[['24', '25', '26']].sum(axis=1)) * group['7']

    value = group[['x', 'y', 'rate']].values
    # print(group[['x', 'y', 'rate']].head(50))

    tab = [[0, 0, 0]]

    for line in range(1, value.shape[0]):

        interet = (value[line -1][2]) * tab[line -1][1]

        pret = value[line][0] - interet

        if value[line][0] >= 0:
            pret = 0
        else:
            pret = - pret

        cum = tab[line -1][1] + pret - value[line][1]

        cum = 0 if cum < 0 else cum
        interet = 0 if cum == 0 else interet

        tab.append([pret, cum, interet])
    x = pd.DataFrame(tab, columns=['30', '31', '32'])
    x.index = group.index

    return x[['30', '31', '32']]

    # print(group[['x', 'rate']].head(50))


def depot_prevente(data, d):

    group = data.copy()
    batiment = data.name[1]
    value = d[batiment].values[0]

    group['26'] = ((1 - group['10']) * group['58']).mul(value)
    group['28'] = ((1 - group['10']) * group['58']).mul(1 - value)

    group['x'] = 1
    group['27'] = group['6'] * (group['6'].where(group['6'].cumsum() == 1, 0) * group['26'].cumsum() +
                                group['6'].where(group['6'].cumsum() > 1, 0) * group['26'])
    group['29'] = 0
    group.loc[group['10'].cumsum() == 1, '29'] = group['28'].cumsum()

    group['30'] = 0
    group.loc[group['10'] == 1, '30'] = group['58']
    group['31'] = group['10'] * (group['10'].where(group['10'].cumsum() == 1, 0) * group['59'].cumsum() +
                                 group['10'].where(group['10'].cumsum() > 1,  0) * group['59'])
    group['32'] = group[['27', '29', '30', '31']].sum(axis=1)

    group['67'] = 0
    group.loc[group['6'].cumsum() > 1, '67'] = group['26']

    group['68'] = -1 * (1 - group['10']) * ((1 - group['7']) * group['12'] + group['7'] * group[['12', '13', '14']].sum(axis=1))
    group['69'] = group[['67', '24']].sum(axis=1)

    return group[['26', '27', '28', '29', '30', '31', '32', '67', '68', '69']]


def sortie_de_fond(data, cost, d):

    group = data.copy()
    batiment = data.name[1]
    sector = data.name[0]
    cost = cost[cost['sector'] == sector]
    terrain = cost.loc[cost['value'] == 'financement terrain',
                       batiment].values[0]
    soft_cost = cost.loc[cost['value'] == 'soft cost',
                         batiment].values[0]
    hard_cost = cost.loc[cost['value'] == 'hard cost',
                         batiment].values[0]
    nb_month = group['60'].loc[group['60'].idxmax()]
    nb_month_hard = d.loc[d['value'] == 'dur_moy_const', batiment].values[0]

    group['12'] = 0
    group.loc[group.head(1).index, '12'] = -1 * terrain

    eq_ter = d.loc[d['value'] == 'eq_terr', batiment].values[0]
    group.loc[:, '24'] = -1 * group['12'] * eq_ter
    group.loc[:, '33'] = group['24']

    group['13'] = -1 * (1 - group['10']) * soft_cost/(nb_month - 1)
    group['14'] = -1 * group['8'] * (1 - group['10']) * hard_cost/nb_month_hard
    group['23'] = -1 * group[['12', '13', '14', '16', '17', '18', '24']].sum(axis=1)

    cout_projet_est = terrain + soft_cost + hard_cost
    group.loc[:, '7'] = 0
    group.loc[group['23'].cumsum() >= 0.25 * cout_projet_est, '7'] = 1

    group.loc[:, '9'] = group['5'] * group['6'] * group['7']
    group['34'] = 0
    group.loc[group['9'] == 0, '34'] = group['33'].cumsum()

    group['19'] = 0
    group.loc[group['9'].cumsum() == 1, '19'] = - terrain * eq_ter

    return group[['7', '9', '12', '13', '14', '19', '23', '24', '33', '34']]


def financement_projet(data, d):

    group = data.copy()
    batiment = data.name[1]
    sector = data.name[0]
    global cum

    def calculate(remb, prev, liv, stat, pret):
        global cum
        cum = 0 if (remb < 0) or cum + pret - prev - liv - stat < 0 else cum + pret - prev - liv - stat
        return cum

    rate = (1 + d[batiment].values[0]/ 2) ** (1 / 6) - 1

    group['36'] = -1 * group['9'] * group[['13', '14', '15', '16', '17', '19', '27']].sum(axis=1)
    group.loc[group['36'] <= 0, '36'] = 0
    # group.loc[group['36'] > 0, '36'] = - group['36']
    group['x'] = group['36'].shift(1).fillna(0)
    group['x'] = group['x'].where(group['x'] > 0, 0)
    group['x'] = group['x'].where(group['x'] != 0, 1)

    group['22'] = -1 * (1 - group['10']) * (group['11'].where(group['11'].cumsum() == 1, 0)) * group['x'] * group['36'] * rate

    group['37'] = group['36'] - group[['29', '30', '31']].sum(axis=1)
    cum = group['37'].head(1).values[0]

    group.loc[group.index[1]:, '37'] = group.loc[group.index[1]:].apply(
        lambda row: calculate(*row[['22', '29', '30', '31', '36']]), axis=1)

    group.loc[:, 'x'] = group['37'].shift(1).fillna(0)
    group.loc[:, 'x'] = group['37'] - group['x']
    group['20'] = 0
    group.loc[group['37'] > 0, '20'] = group['x'].where(group['x'] < 0, 0) - group['36'].where(group['x'] < 0, 0)

    group['21'] = - group['37'].shift(1).fillna(0)
    group.loc[group['37'] != 0, '21'] = 0
    group.loc[group['21'] > 0, '21'] = 0
    group['25'] = group['36']

    return group[['20', '21', '22', '25', '36', '37']]


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


def remb_terr(data):

    group = data.copy()
    name = data.name

    group['28'] = group['28'].shift(1).fillna(0)
    group['15'] = 0
    group.loc[group['7'].cumsum() == 1, '15'] = -1 * group['28']

    return group[['15']]


def remb_proj(data):

    group = data.copy()
    group['x'] = group['31'].shift(1).fillna(0)
    group.loc[:, 'x'] = group['31'] - group['x']
    group['y'] = group['x'] - group['30']
    group['16'] = 0

    group.loc[group[(group['31'] > 0) & (group['x'] < 0)].index, '16'] =group['y']
    group['17'] = 0

    group['x'] = group['31'].shift(1).fillna(0)
    group.loc[group[(group['31'] ==0) & (group['x'] >0)].index, '17'] = -group['x']

    return group[['16', '17']]


def cashflow(group):

    group['36'] = group.sum(axis=1)

    return group[['36']]


def calculate_irr(group):

    return 100 * np.irr(group['39'].fillna(0).values)


def calcul_detail_financier(secteur, batiment,  timeline, cost_table, finance_params) -> pd.DataFrame:

    """""
    This function is used to compute the cost of a builiding given a specific sector.

    :param
    cost_table (pd.DataFrame):  The table containing all the building useful information necessary to compute the financials.

    secteur (list): sector of the building

    batiment (range): range containing the building we want to compute the costs. eg: ['B1', 'B7']
    
    timeline: duration max of the project
    
    finance_params: financials intrants

    :return

    financials_result (pd.Dataframe)

    """""
    summary = pd.DataFrame(None, columns=cost_table.columns)
    ntu = cost_table[(cost_table['value'] == 'ntu')]
    supbtu = cost_table[(cost_table['value'] == 'supbtu')]
    sup_bru_one_floor = cost_table[(cost_table['value'] == 'sup_bru_one_floor')]
    cost = cost_table[(cost_table['category'] == 'partial')]
    contrib_terr_ss = cost_table[(cost_table['value'] == 'contrib_terr_ss')]

    cost_total = cost_table[(cost_table['category'] == 'total')]
    summary = pd.concat([summary, ntu, supbtu, sup_bru_one_floor, cost, cost_total, contrib_terr_ss])
    summary = summary.groupby(['sector']).apply(format_for_output).reset_index(drop=True)

    c = cost_table[(cost_table['value'] == 'ntu') & (cost_table['category'] == 'ALL')]
    go = cost_table[(cost_table['value'] == 'go_no_go') & (cost_table['category'] == 'ALL')]

    financials_result = []

    for sector in secteur:

        c = go[batiment].loc[go['sector'] == sector]
        c = c.iloc[:, c.gt(0).any().values]

        for bat in c.columns:
            x = pd.DataFrame([[sector, bat, i + 1, 1] for i in range(timeline)], columns=['sector', 'batiment', '1', '2'])
            financials_result.append(x)

    financials_result = pd.concat(financials_result, ignore_index=True)

    ###################################################################################################################
    #
    # Time line and sales Zone.
    #
    ###################################################################################################################

    # Debut des ventes
    dm_1 = finance_params[finance_params['value'] == 'dm_1']
    dm_prev = finance_params[finance_params['value'] == 'dm_prev']
    t = financials_result[['sector', 'batiment', '1']].groupby(['sector', 'batiment'])
    financials_result[['3', '4', '5', '16', '17', '18']] = t.apply(debut_des_ventes, dm_1, dm_prev,
                                                 cost_table[cost_table['value'].isin(['rem', 'frais_parc', 'contrib_terr_hs', 'contrib_fin'])]).reset_index(drop=True)

    # Ventes (Ecoulement et revenus bruts)
    result = financials_result[['sector', 'batiment', '3', '4']].groupby(['sector', 'batiment'])
    data = finance_params[finance_params['value'].isin(['ecob3mo', 'ecoa3mo'])]
    data = data[cost_table.columns]
    params = cost_table[cost_table['value'].isin(['ntu', 'ecob3mo', 'ecoa3mo', 'price', 'si', 'stat'])]
    params = pd.concat([params, data], ignore_index=True)

    result = result.apply(calcul_ecoulement_et_vente, params)
    result = result.reset_index(drop=True)
    financials_result = pd.concat([financials_result, result], axis=1)
    financials_result.to_excel('t.xlsx')

    # 50% des unites construites
    data = finance_params[finance_params['value'] == 'nv_min_prev_av_deb']
    c = cost_table[(cost_table['value'] == 'ntu') & (cost_table['category'] == 'ALL')]
    t = financials_result[['sector', 'batiment', '5', '49']].groupby(['sector', 'batiment'])
    financials_result[['6', '8', '11']] = t.apply(mid_prcent_unite_vendu, data, c).reset_index(drop=True)

    # livraison de l'immeuble
    data = finance_params[finance_params['value'] == 'dur_moy_const']

    t = financials_result[['sector', 'batiment', '1', '8']].groupby(['sector', 'batiment'])
    financials_result[['10', '60', '61']] = t.apply(liv_immeuble, data).reset_index(drop=True)

     ###################################################################################################################
    #
    # Sortie de fond Zone.
    #
    ###################################################################################################################

    data = finance_params[finance_params['value'].isin(['eq_terr', 'dur_moy_const'])]
    cost = cost_table[(cost_table['category'] == 'partial')]

    tab = financials_result[['sector', 'batiment', '5', '6', '8', '10', '16', '17', '18', '60']].groupby(['sector', 'batiment'])
    financials_result[['7', '9', '12', '13', '14', '19', '23', '24', '33', '34']] = tab.apply(sortie_de_fond, cost, data).reset_index(drop=True)


    ###################################################################################################################
    #
    # Entree de fond Zone.
    #
    ###################################################################################################################

    # depot de prevente pour calcul
    data = finance_params[finance_params['value'] == 'pp_prev']

    # Depot de prevente total - pour calcul
    tab = financials_result[['sector', 'batiment', '6', '7', '10', '12', '13', '14', '24', '58', '59']].groupby(
        ['sector', 'batiment'])
    x = ['26', '27', '28', '29', '30', '31', '32', '67', '68', '69']
    financials_result[x] = tab.apply(depot_prevente, data).reset_index(drop=True)

    ###################################################################################################################
    #
    # Interets.
    #
    ###################################################################################################################

    # Calcul interet simplifié
    data = finance_params[finance_params['value'] == 'interet_terrain']

    tab = financials_result[['sector', 'batiment', '10', '60', '68', '69']].groupby(
        ['sector', 'batiment'])
    financials_result[['15', '70', '71']] = tab.apply(sold_fin_terr, data).reset_index(drop=True)

    ###################################################################################################################
    #
    # Pret projet.
    #
    ###################################################################################################################
    data = finance_params[finance_params['value'] == 'interet_projet']

    x = ['sector', 'batiment', '9', '10', '11', '13', '14', '15', '16', '17', '19', '27', '29', '30', '31']
    tab = financials_result[x].groupby(['sector', 'batiment'])
    financials_result[['20', '21', '22', '25', '36', '37']] = tab.apply(financement_projet, data).reset_index(drop=True)

    ###################################################################################################################
    #
    # CashFlow
    #
    ###################################################################################################################

    entete_for_cashflow = ['12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '24', '25', '27', '29', '30', '31']

    financials_result['39'] = financials_result[entete_for_cashflow].sum(axis=1)

    ###################################################################################################################
    #
    # Key Statistics.
    #
    ###################################################################################################################

    # total interet terrain
    inter_terr = financials_result[['sector', 'batiment', '71']].groupby(['sector', 'batiment']).sum().reset_index()
    inter_terr = inter_terr.rename(columns={'71': 'total interet terrain'})
    summary = pd.merge(summary, inter_terr, on=['sector', 'batiment'])

    # total revenu
    rev = financials_result[['sector', 'batiment', '58', '59']]
    rev.loc[:, 'x'] = financials_result[['58', '59']].sum(axis=1)
    rev = rev[['sector', 'batiment', 'x']].groupby(['sector', 'batiment']).sum().reset_index()
    rev = rev.rename(columns={'x': 'revenus totaux'})
    summary = pd.merge(summary, rev, on=['sector', 'batiment'])

    # TRI
    tri = financials_result[['sector', 'batiment', '39']].groupby(['sector', 'batiment']).apply(calculate_irr)
    tri = np.round(100 * ((1 + tri/100)**12 - 1), 2)
    tri = tri.reset_index().rename(columns={0: 'TRI'})
    summary = pd.merge(summary, tri, on=['sector', 'batiment'])

    # Cout du projet
    summary.loc[:, 'cout du projet'] = summary[["cout total du projet", 'total interet terrain']].sum(axis=1)

    # marge beneficiare
    summary.loc[:, 'marge beneficiaire'] = 100 * (summary['revenus totaux'] / summary["cout du projet"] - 1)

    inter_terr = financials_result[['sector', 'batiment', '58']].groupby(['sector', 'batiment']).sum().reset_index()
    # inter_terr = inter_terr.rename(columns={'58': 'total interet terrain'})
    summary = pd.merge(summary, inter_terr, on=['sector', 'batiment'])
    # print(summary)
    return summary


def calculate_financial(type, secteur, batiment, params, timeline, cost, finance_params, *args):

    cost_table = calculate_cost(type, secteur, batiment, params, cost, *args)
    cost_table.to_excel('cost.xlsx')
    return calcul_detail_financier(secteur, batiment, timeline, cost_table, finance_params)


if __name__ == '__main__':

    myBook = xlrd.open_workbook(__FILES_NAME__)
    x = get_all_informations(myBook)
    cost_params = x[(x['type'].isin(['pcost'])) & (x['sector'] == 'Secteur 1')]

    args = dict()
    # supter = [50000]
    # densite = [10]
    finance_params = x[(x['type'].isin(['financial'])) & (x['sector'] == 'Secteur 1')]
    #
    # # tab = []
    # print(__BATIMENT__[5:6])
    result = calculate_financial('CA3', __SECTEUR__, __BATIMENT__, x, 120, cost_params, finance_params, args)
    # r = result.loc[result[result['value'] == 'marge beneficiaire'].index[0], __BATIMENT__]
    # best_batiment = r.astype(float).idxmax(skipna=True)
    # result = result[result['value'].isin(['ntu', 'TRI', 'marge beneficiaire'])]
    # result = result[['category', 'value', best_batiment]]
    # result.loc[result['value'] == 'ntu', 'value'] = result['category']
    # result = result[['value', best_batiment]]
    # result.set_index('value', inplace=True)

    def get_summary_value(group):

        data = group.copy()

        id_batiment = data.loc[:, 'ID'].values[0]
        sup_ter = data.loc[:, 'sup_ter'].values[0]
        denm_p = data.loc[:, 'denm_p'].values[0]
        vat = data.loc[:, 'vat'].values[0]
        sector = data.loc[:, 'sector'].values[0]

        args = dict()
        args['sup_ter'] = [[sup_ter]]
        args['denm_p'] = [[denm_p]]
        args['vat'] = [[vat]]
        params = x[x['sector'] == sector]
        params.loc[:, 'sector'] = id_batiment

        result = get_cb3_characteristic([id_batiment], __BATIMENT__, params, args)

        return result

    def add_cost_params(group, terr):

        id_batiment = group.name
        data = group.copy()

        sector = terr.loc[terr['ID'] == id_batiment, 'sector'].values[0]
        params = cost[cost['sector'] == sector]
        params.loc[:, 'sector'] = id_batiment

        return pd.concat([data, params[data.columns]], ignore_index=True)

    #
    #
    #
    # couleur_secteur = {}
    # couleur = ['Jaune', 'Vert', 'Bleu pâle', 'Bleu', 'Mauve', 'Rouge', 'Noir']
    #
    # for pos in range(len( __SECTEUR__)):
    #     couleur_secteur[couleur[pos]] = __SECTEUR__[pos]
    #
    # terrain_dev = pd.read_excel(__FILES_NAME__, sheet_name='terrains')
    #
    # header_dict = {'SuperficieTerrain_Pi2': 'sup_ter', 'COS max formule': 'denm_p', 'couleur': 'sector',
    #                'Valeur terrain p2 PROVISOIRE': 'vat'}
    # terrain_dev.rename(columns = header_dict, inplace=True)
    #
    # terrain_dev = terrain_dev[['ID', 'sup_ter', 'denm_p', 'sector', 'vat']]
    # # terrain_dev = terrain_dev[terrain_dev['sup_ter'] >= 1000]
    #
    # terrain_dev.loc[:, 'sector'] = terrain_dev['sector'].replace(couleur_secteur)
    #
    # start = time.time()
    # terr = terrain_dev.drop_duplicates(['sup_ter', 'denm_p', 'sector', 'vat']).reset_index(drop=True).tail(250)
    # cb3 = terr.groupby('ID').apply(get_summary_value).reset_index(drop=True)
    # ca3 = get_ca_characteristic(cb3['sector'].unique(), __BATIMENT__, cb3)
    #
    # # Add cost intrants.
    # cost = calcul_cout_batiment(cb3['sector'].unique(), __BATIMENT__, ca3, cost_params)
    # end = time.time()
    #
    # print(end - start)


