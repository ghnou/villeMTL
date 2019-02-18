__author__ = 'pougomg'

import numpy as np
import pandas as pd
import xlrd

from import_parameter import get_building_cost_parameter
from lexique import __UNITE_TYPE__, __QUALITE_BATIMENT__, __COUTS_FILES_NAME__, __BATIMENT__, __SECTEUR__
from obtention_intrant import get_cb1_characteristics, get_cb4_characteristics


def ajouter_caraterisque_par_type_unite(sh, tab, name, pos, unique):

    for unite in __UNITE_TYPE__:
        _ = [unite, 'ALL', name]
        line = pos[0] if unique else __UNITE_TYPE__.index(unite) + pos[0]
        for batiment in range(len(__BATIMENT__)):
            value = sh.cell(line, pos[1] + batiment).value
            value = 0 if value == "" else value
            _.append(value)
        tab.append(_)
    return tab


def get_qu(group, dict_of_qu):
    tab = [group[batiment].map(dict_of_qu).values.tolist() for batiment in __BATIMENT__]
    tab = np.array(tab).transpose()
    tab = pd.DataFrame(tab, columns=__BATIMENT__)
    group = group.reset_index()

    return tab


def apply_mutation_function(x) -> float:

    """"
    This Function is used to compute the frais de mutation.
    :param x: Price of the land
    :return frais de mutation.

     """
    if x >= 1007000:
        return (x - 1007000) * 0.025 + 16111.5
    elif x >= 503500:
        return (x - 503500) * 0.02 + 6041.5
    elif x >= 251800:
        return (x - 251800) * 0.015 + 2266
    elif x >= 50400:
        return (x - 50400) * 0.01 + 252
    else:
        return x * 0.005


def calcul_cout_batiment(table_of_intrant, cost_param, secteur, batiment, myBook) -> pd.DataFrame:

    """""
    This function is used to compute the cost of a builiding given a specific sector.

    :param
    table_of_intrant (pd.DataFrame):  The table containing all the building useful information necessary to compute the costs.

    cost_param (pd.DataFrame): A dataframe containing all the cost units params.

    secteur (str): sector of the building

    batiment (range): range containing the building we want to compute the costs. eg: ['B1', 'B7']
    
    sup_tot_hs
    sup_ss
    suptu
    ntu
    supbtu
    cir
    pisc
    cub
    sup_com
    decont
    sup_parc
    cont_soc
    vat
    sup_ter
    price
    

    :return

    cout_result (pd.Dataframe)

    """""

    # Coquille
    tc = cost_param[(cost_param['value'] == 'tcq') & (cost_param['category'] == 'ALL')].reset_index(drop=True)
    supths = table_of_intrant[(table_of_intrant['value'] == 'sup_tot_hs') & (table_of_intrant['category'] == 'ALL')][
        batiment].reset_index(drop=True)
    result = tc[batiment] * supths
    result[['sector', 'value']] = tc[['sector', 'value']]
    result['category'] = 'unique'
    result = result[cost_param.columns]
    cout_result = result

    # Sous Sol
    tss = cost_param[(cost_param['value'] == 'tss') & (cost_param['category'] == 'ALL')].reset_index(drop=True)
    sup_ss = table_of_intrant[(table_of_intrant['value'] == 'sup_ss') & (table_of_intrant['category'] == 'ALL')][
        batiment].reset_index(drop=True)
    result = tss[batiment] * sup_ss
    result[['sector', 'value']] = tss[['sector', 'value']]
    result['category'] = 'unique'
    result = result[cost_param.columns]
    cout_result = pd.concat([cout_result, result],
                            ignore_index=True)

    # Travaux Finition unite de marche
    suptu = table_of_intrant[
        (table_of_intrant['value'] == 'suptu') & (table_of_intrant['category'].isin(__UNITE_TYPE__[0:5]))].reset_index(
        drop=True)
    suptu = suptu[batiment].groupby(suptu['sector']).sum().reset_index(drop=True)
    tfu = cost_param[(cost_param['value'] == 'tfum') & (cost_param['category'] == 'ALL')].reset_index(drop=True)
    result = tfu[batiment] * suptu
    result[['sector', 'value']] = tfu[['sector', 'value']]
    result['category'] = 'unique'
    result = result[cost_param.columns]
    cout_result = pd.concat([cout_result, result],
                            ignore_index=True)

    # Allocation pour cuisine, salle de bain unite de marche
    ntu = table_of_intrant[
        (table_of_intrant['value'] == 'ntu') & (table_of_intrant['category'].isin(__UNITE_TYPE__[0:5]))].reset_index(
        drop=True)
    ntu = ntu[batiment].groupby(ntu['sector']).sum().reset_index(drop=True)
    val = cost_param[(cost_param['value'] == 'all_cuis') & (cost_param['category'] == 'ALL')].reset_index(drop=True)
    result = ntu[batiment] * val

    result['sector'] = val['sector']
    result['value'] = 'cuisum'
    result['category'] = 'unique'
    result = result[cost_param.columns]
    cout_result = pd.concat([cout_result, result],
                            ignore_index=True)
    val = cost_param[(cost_param['value'] == 'all_sdb') & (cost_param['category'] == 'ALL')].reset_index(drop=True)
    result = ntu[batiment] * val
    result['sector'] = val['sector']
    result['value'] = 'saldbum'
    result['category'] = 'unique'
    result = result[cost_param.columns]
    cout_result = pd.concat([cout_result, result],
                            ignore_index=True)

    # Travaux Finition unite abordables
    suptu = table_of_intrant[
        (table_of_intrant['value'] == 'suptu') & (table_of_intrant['category'].isin(__UNITE_TYPE__[5:]))].reset_index(
        drop=True)
    suptu = suptu[batiment].groupby(suptu['sector']).sum().reset_index(drop=True)
    tfu = cost_param[(cost_param['value'] == 'tfum') & (cost_param['category'] == 'ALL')].reset_index(drop=True)
    result = tfu[batiment] * suptu
    result['sector'] = tfu['sector']
    result['value'] = 'tfuf'
    result['category'] = 'unique'
    result = result[cost_param.columns]
    cout_result = pd.concat([cout_result, result],
                            ignore_index=True)

    # Allocation pour cuisine, salle de bain unite de marche
    ntu = table_of_intrant[
        (table_of_intrant['value'] == 'ntu') & (table_of_intrant['category'].isin(__UNITE_TYPE__[5:]))].reset_index(
        drop=True)
    ntu = ntu[batiment].groupby(ntu['sector']).sum().reset_index(drop=True)
    val = cost_param[(cost_param['value'] == 'all_cuis') & (cost_param['category'] == 'ALL')].reset_index(drop=True)
    result = ntu[batiment] * val

    result['sector'] = val['sector']
    result['value'] = 'cuisuf'
    result['category'] = 'unique'
    result = result[cost_param.columns]
    cout_result = pd.concat([cout_result, result],
                            ignore_index=True)
    val = cost_param[(cost_param['value'] == 'all_sdb') & (cost_param['category'] == 'ALL')].reset_index(drop=True)
    result = ntu[batiment] * val
    result['sector'] = val['sector']
    result['value'] = 'saldbuf'
    result['category'] = 'unique'
    result = result[cost_param.columns]
    cout_result = pd.concat([cout_result, result],
                            ignore_index=True)

    # --> Total travaux unite de marche

    totbum = cout_result[(cout_result['value'] == 'tfum') | (cout_result['value'] == 'cuisum')
                         | (cout_result['value'] == 'saldbum')]
    totbum = totbum[batiment].groupby(totbum['sector']).sum()

    # --> Total travaux unite fam

    totbuf = cout_result[(cout_result['value'] == 'tfuf') | (cout_result['value'] == 'cuisuf')
                         | (cout_result['value'] == 'saldbuf')]
    totbuf = totbuf[batiment].groupby(totbuf['sector']).sum()

    # Couts des finitions
    sh = myBook.sheet_by_name('PCOUTS')
    qu_pos = [74, 4]
    dict_cost_finitions = dict()
    for i in range(3):
        dict_cost_finitions[__QUALITE_BATIMENT__[i]] = sh.cell(qu_pos[0] + i, qu_pos[1]).value

    sh = myBook.sheet_by_name('Scénarios')

    # --> Unites de marches
    qum = []
    qum_pos = [77, 2]
    for sect in range(len(__SECTEUR__)):
        _ = []
        for bat in range(len(__BATIMENT__)):
           _.append(sh.cell(sect + qum_pos[0], bat + qum_pos[1]).value)
        qum.append(_)
    qum = pd.DataFrame(qum, columns=__BATIMENT__)
    qum['value'] = 'qum'
    qum['sector'] = __SECTEUR__

    qum = qum[qum['sector'] == secteur]
    qum = qum.groupby('value').apply(get_qu, dict_cost_finitions).reset_index(drop=True)
    qum = qum[batiment]

    totbum = totbum.reset_index(drop=True) * qum
    totbum['category'] = 'unique'
    totbum['sector'] = secteur
    totbum['value'] = 'cfum'
    totbum = totbum[cost_param.columns]
    cout_result = pd.concat([cout_result, totbum],
                            ignore_index=True)
    # --> Unites familiale
    quf = []
    quf_pos = [86, 2]
    for sect in range(len(__SECTEUR__)):
        _ = []
        for bat in range(len(__BATIMENT__)):
           _.append(sh.cell(sect + quf_pos[0], bat + quf_pos[1]).value)
        quf.append(_)
    quf = pd.DataFrame(quf, columns=__BATIMENT__)
    quf['value'] = 'quf'
    quf['sector'] = __SECTEUR__

    quf = quf[quf['sector'] == secteur]
    quf = quf.groupby('value').apply(get_qu, dict_cost_finitions).reset_index(drop=True)
    quf = quf[batiment]

    totbuf = totbuf.reset_index(drop=True) * quf
    totbuf['category'] = 'unique'
    totbuf['sector'] = secteur
    totbuf['value'] = 'cfuf'
    totbuf = totbuf[cost_param.columns]
    cout_result = pd.concat([cout_result, totbuf],
                            ignore_index=True)

    # Travaux finitions aires communes
    aire_commune = table_of_intrant[((table_of_intrant['value'] == 'supbtu')
                              |(table_of_intrant['value'] == 'cir'))
                              & (table_of_intrant['category'] == 'ALL')][
        ['sector'] + batiment].reset_index(drop=True)

    aire_commune = aire_commune.groupby('sector').prod().reset_index(drop=True)
    tvfac = cost_param[(cost_param['value'] == 'tvfac') & (cost_param['category'] == 'ALL')][batiment].reset_index(
        drop=True)
    tvfac = tvfac * aire_commune
    tvfac['category'] = 'unique'
    tvfac['sector'] = secteur
    tvfac['value'] = 'tvfac'
    tvfac = tvfac[cost_param.columns]

    cout_result = pd.concat([cout_result, tvfac],
                            ignore_index=True)

    # ascenceurs
    asc = cost_param[(cost_param['value'] == 'asc') & (cost_param['category'] == 'ALL')]
    asc.loc[:, 'category'] = 'unique'
    cout_result = pd.concat([cout_result, asc],
                            ignore_index=True)

    # Cout additionnel Piscine
    pisc = table_of_intrant[(table_of_intrant['value'] == 'pisc')
                              & (table_of_intrant['category'] == 'ALL')][
        batiment].reset_index(drop=True)
    pisc.replace({'Non': 0, 'Oui': 1}, inplace=True)

    c_ad_pisc = cost_param[(cost_param['value'] == 'c_ad_pisc') & (cost_param['category'] == 'ALL')][
        batiment].reset_index(drop=True)
    c_ad_pisc = c_ad_pisc * pisc
    c_ad_pisc['category'] = 'unique'
    c_ad_pisc['sector'] = secteur
    c_ad_pisc['value'] = 'c_ad_pisc'
    c_ad_pisc = c_ad_pisc[cost_param.columns]
    cout_result = pd.concat([cout_result, c_ad_pisc],
                            ignore_index=True)

    # Cout additionnel chalet urbain
    cub = table_of_intrant[(table_of_intrant['value'] == 'cub')
                              & (table_of_intrant['category'] == 'ALL')][
        batiment].reset_index(drop=True)
    cub.replace({'Non': 0, 'Oui': 1}, inplace=True)

    c_ad_cu = cost_param[(cost_param['value'] == 'c_ad_cu') & (cost_param['category'] == 'ALL')][batiment].reset_index(
        drop=True)
    c_ad_cu = c_ad_cu * cub
    c_ad_cu['category'] = 'unique'
    c_ad_cu['sector'] = secteur
    c_ad_cu['value'] = 'c_ad_cu'
    c_ad_cu = c_ad_cu[cost_param.columns]
    cout_result = pd.concat([cout_result, c_ad_cu],
                            ignore_index=True)

    # cout additionnel espace commmerciaux
    sup_com = table_of_intrant[(table_of_intrant['value'] == 'sup_com')
                              & (table_of_intrant['category'] == 'ALL')][
        batiment].reset_index(drop=True)
    cir = table_of_intrant[(table_of_intrant['value'] == 'cir')
                              & (table_of_intrant['category'] == 'ALL')][
        batiment].reset_index(drop=True)
    c_ad_com = cost_param[(cost_param['value'] == 'c_ad_com') & (cost_param['category'] == 'ALL')][
        batiment].reset_index(drop=True)

    c_ad_com = c_ad_com * sup_com * (1 - cir)
    c_ad_com['category'] = 'unique'
    c_ad_com['sector'] = secteur
    c_ad_com['value'] = 'c_ad_com'
    c_ad_com = c_ad_com[cost_param.columns]
    cout_result = pd.concat([cout_result, c_ad_com],
                            ignore_index=True)

    # Imprevus sur travaux
    su = cout_result[
        (cout_result['value'].isin(['tcq', 'tss', 'cfum', 'cfuf', 'tvfac', 'asc', 'c_ad_pisc', 'c_ad_cu', 'c_ad_com']))
        & (cout_result['category'] == 'unique')][['sector'] + batiment].reset_index(drop=True)

    su = su.groupby('sector').sum().reset_index(drop=True)
    it = cost_param[(cost_param['value'] == 'it') & (cost_param['category'] == 'ALL')][batiment].reset_index(drop=True)
    it = su /(1 - it) - su
    it['category'] = 'unique'
    it['sector'] = secteur
    it['value'] = 'it'
    it = it[cost_param.columns]
    cout_result = pd.concat([cout_result, it],
                            ignore_index=True)

    # Sous total couts de construction
    cct = su + it
    cct['category'] = 'partial'
    cct['sector'] = secteur
    cct['value'] = 'construction cost'
    cct = cct[cost_param.columns]
    cout_result = pd.concat([cout_result, cct],
                            ignore_index=True)

    ### --> SOFT COST

    # Arpenteur-geometre
    apt_geo = cost_param[cost_param['value'] == 'apt_geo'][['sector', 'value'] + batiment]
    apt_geo['category'] = 'unique'
    apt_geo = apt_geo[cost_param.columns]
    cout_result = pd.concat([cout_result, apt_geo],
                            ignore_index=True)

    # Professionnels
    prof = cost_param[(cost_param['value'] == 'prof') & (cost_param['category'] == 'ALL')][batiment].reset_index(
        drop=True)
    prof = prof * cct[batiment]
    prof['category'] = 'unique'
    prof['sector'] = secteur
    prof['value'] = 'prof'
    prof = prof[cost_param.columns]
    cout_result = pd.concat([cout_result, prof],
                            ignore_index=True)

    # Evaluator
    eval = cost_param[cost_param['value'] == 'eval'][['sector', 'value'] + batiment]
    eval['category'] = 'unique'
    eval = eval[cost_param.columns]
    cout_result = pd.concat([cout_result, eval],
                            ignore_index=True)

    # Legal Fee
    legal_fee = cost_param[cost_param['value'] == 'legal_fee'][['sector', 'value'] + batiment]
    legal_fee['category'] = 'unique'
    legal_fee = legal_fee[cost_param.columns]
    cout_result = pd.concat([cout_result, legal_fee],
                            ignore_index=True)

    # Professionnal fee divers
    prof_fee_div = cost_param[cost_param['value'] == 'prof_fee_div'][['sector', 'value'] + batiment]
    prof_fee_div['category'] = 'unique'
    prof_fee_div = prof_fee_div[cost_param.columns]
    cout_result = pd.concat([cout_result, prof_fee_div],
                            ignore_index=True)

    # Pub
    pub = cost_param[(cost_param['value'] == 'pub') & (cost_param['category'] == 'ALL')][batiment].reset_index(
        drop=True)
    pub = pub * cct[batiment]
    pub['category'] = 'unique'
    pub['sector'] = secteur
    pub['value'] = 'pub'
    pub = pub[cost_param.columns]
    cout_result = pd.concat([cout_result, pub],
                            ignore_index=True)

    # Construction permit
    construction_permit = \
        cost_param[(cost_param['value'] == 'construction_permit') & (cost_param['category'] == 'ALL')][
            batiment].reset_index(drop=True)
    construction_permit = construction_permit * cct[batiment] / 1000
    construction_permit['category'] = 'unique'
    construction_permit['sector'] = secteur
    construction_permit['value'] = 'construction_permit'
    construction_permit = construction_permit[cost_param.columns]
    cout_result = pd.concat([cout_result, construction_permit],
                            ignore_index=True)

    # Commission vente
    com = cost_param[(cost_param['value'] == 'com') & (cost_param['category'] == 'ALL')].reset_index(drop=True)
    price = table_of_intrant[(table_of_intrant['value'] == 'price') & (table_of_intrant['category'] == 'ALL')][
        batiment].reset_index(drop=True)
    print(price)
    result = com[batiment] * price
    result[['sector', 'value']] = com[['sector', 'value']]
    result['category'] = 'unique'
    result = result[cost_param.columns]

    cout_result = pd.concat([cout_result, result],
                            ignore_index=True)


    # Total soft Cost
    su = cout_result[(cout_result['value'].isin(['apt_geo', 'prof', 'eval', 'legal_fee', 'prof_fee_div', 'pub',
                                                 'construction_permit', 'com']))
                     & (cout_result['category'] == 'unique')][['sector'] + batiment].reset_index(drop=True)

    su = su.groupby('sector').sum().reset_index(drop=True)
    su['category'] = 'partial'
    su['sector'] = secteur
    su['value'] = 'soft cost'
    su = su[cost_param.columns]
    cout_result = pd.concat([cout_result, su],
                            ignore_index=True)


    # Land price
    sup_ter = table_of_intrant[(table_of_intrant['value'] == 'sup_ter') & (table_of_intrant['category'] == 'ALL')][
        batiment].reset_index(drop=True)

    vat = table_of_intrant[(table_of_intrant['value'] == 'vat') & (table_of_intrant['category'] == 'ALL')][
        batiment].reset_index(drop=True)

    price_land = vat.astype(float) * sup_ter.astype(float)

    fm = []
    for value in batiment:
        fm.append(price_land[value].transform(lambda x: apply_mutation_function(x)).values[0])
    # Cout attribution terrain
    result = price_land + fm
    result['category'] = 'unique'
    result["value"] = 'caq_ter'
    result['sector'] = secteur
    result = result[cost_param.columns]

    cout_result = pd.concat([cout_result, result],
                            ignore_index=True)
    #Contribution sociale
    supbtu = table_of_intrant[(table_of_intrant['value'] == 'supbtu') & (table_of_intrant['category'] == 'ALL')][
        batiment].reset_index(drop=True)

    cont_soc = table_of_intrant[(table_of_intrant['value'] == 'cont_soc') & (table_of_intrant['category'] == 'ALL')][
        batiment].reset_index(drop=True)

    result = supbtu.astype(float) * cont_soc.astype(float)
    result['category'] = 'unique'
    result["value"] = 'cont_soc'
    result['sector'] = secteur
    result = result[cost_param.columns]
    cout_result = pd.concat([cout_result, result],
                            ignore_index=True)

    # Frais de parc
    sup_parc = table_of_intrant[(table_of_intrant['value'] == 'sup_parc') & (table_of_intrant['category'] == 'ALL')][
        batiment].reset_index(drop=True)
    result = (sup_parc * price_land * 0.1/ supbtu)
    result['category'] = 'unique'
    result["value"] = 'frais_parc'
    result['sector'] = secteur
    result = result[cost_param.columns]
    cout_result = pd.concat([cout_result, result],
                            ignore_index=True)

    # Decontamination
    decont = table_of_intrant[(table_of_intrant['value'] == 'decont') & (table_of_intrant['category'] == 'ALL')][
        batiment].reset_index(drop=True)
    result = decont.astype(float) * sup_ter.astype(float)
    result['category'] = 'unique'
    result["value"] = 'decont'
    result['sector'] = secteur
    result = result[cost_param.columns]
    cout_result = pd.concat([cout_result, result],
                            ignore_index=True)

    # Sous Total Terrain
    su = cout_result[(cout_result['value'].isin(['caq_ter', 'cont_soc', 'frais_parc', 'decont']))
                     & (cout_result['category'] == 'unique')][['sector'] + batiment].reset_index(drop=True)
    su = su.groupby('sector').sum().reset_index(drop=True)
    su['category'] = 'partial'
    su['sector'] = secteur
    su['value'] = 'financement terrain'
    su = su[cost_param.columns]
    cout_result = pd.concat([cout_result, su], ignore_index=True)

    # Cout total du projet
    tot = cout_result[cout_result['category'] == 'partial'][['sector'] + batiment].reset_index(drop=True)
    tot = tot.groupby('sector').sum().reset_index(drop=True)
    tot['category'] = 'total'
    tot['sector'] = secteur
    tot['value'] = 'cout total du projet'
    tot = tot[cost_param.columns]
    cout_result = pd.concat([cout_result, tot], ignore_index=True)

    # Other Params
    cout_result[cout_result['category'] == 'partial'].to_csv('t.txt')
    cout_result = pd.concat([cout_result, table_of_intrant[table_of_intrant['value'] == 'ntu'],
                            table_of_intrant[table_of_intrant['value'] == 'price']],
                            ignore_index=True)

    return cout_result

if __name__ == '__main__':
    myBook = xlrd.open_workbook(__COUTS_FILES_NAME__)
    intrant_param = get_cb1_characteristics(myBook)
    cost_param = get_building_cost_parameter(myBook)

    intrant_param = get_cb4_characteristics(intrant_param, "Secteur 7", 5389.0, None, 2, None, None)

    intrant_param = intrant_param[['sector', 'category', 'value', 'B1', 'B3', 'B8']]
    intrant_param = intrant_param[(intrant_param['sector'] == 'Secteur 7')]

    cost_param = cost_param[['sector', 'category', 'value', 'B1', 'B3', 'B8']]
    cost_param = cost_param[cost_param['sector'] == 'Secteur 7']

    print(calcul_cout_batiment(intrant_param, cost_param, 'Secteur 7', ['B1', 'B3', 'B8'], myBook)[
              ['category', 'value', 'B8']])
