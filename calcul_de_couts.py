__author__ = 'pougomg'

from obtention_intrant import get_cb1_characteristics
import xlrd
import pandas as pd
import numpy as np
from lexique import __UNITE_TYPE__, __QUALITE_BATIMENT__, __FILES_NAME__, __BATIMENT__, __SECTEUR__
from import_parameter import get_land_param, get_building_cost_parameter


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


def calcul_prix_terrain(densite, superficie):

    terrain_param = get_land_param(myBook)

    augmentation_valeur = (
                1 + terrain_param[terrain_param['Value'] == 'aug valeur'][__BATIMENT__].reset_index(drop=True)).astype(
        float)

    value = (terrain_param[terrain_param['Value'] == 'valeur prox'][__BATIMENT__].reset_index(drop=True) +
             terrain_param[terrain_param['Value'] == 'multi de densite'][__BATIMENT__].reset_index(
                 drop=True) * densite).astype(float)

    mutation = terrain_param[terrain_param['Value'] == 'mutation'][__BATIMENT__].reset_index(drop=True).astype(float)

    prix = np.exp(value) * augmentation_valeur * superficie + mutation
    print(prix)

    # print(prix)


def get_qu(group, dict_of_qu):
    tab = [group[batiment].map(dict_of_qu).values.tolist() for batiment in __BATIMENT__]
    tab = np.array(tab).transpose()
    tab = pd.DataFrame(tab, columns=__BATIMENT__)
    group = group.reset_index()

    return tab


def calcul_cout_batiment(table_of_intrant, myBook):

    cost_param = get_building_cost_parameter(myBook)

    # Coquille
    tc = cost_param[(cost_param['value'] == 'tcq') & (cost_param['category'] == 'ALL')].reset_index(drop=True)
    supths = table_of_intrant[(table_of_intrant['value'] == 'sup_tot_hs') & (table_of_intrant['category'] == 'ALL')][
        __BATIMENT__].reset_index(drop=True)
    result = tc[__BATIMENT__] * supths
    result[['sector', 'value']] = tc[['sector', 'value']]
    result['category'] = 'unique'
    result = result[cost_param.columns]
    cout_result = result

    #TODO: Sous Sol
    tss = cost_param[cost_param['value'] == 'tss'][['sector', 'value'] + __BATIMENT__]
    tss['category'] = 'unique'
    tss = tss[cost_param.columns]
    cout_result = pd.concat([cout_result, tss],
                            ignore_index=True)

    # Travaux Finition unite de marche
    suptu = table_of_intrant[
        (table_of_intrant['value'] == 'suptu') & (table_of_intrant['category'].isin(__UNITE_TYPE__[0:5]))].reset_index(
        drop=True)
    suptu = suptu[__BATIMENT__].groupby(suptu['sector']).sum().reset_index(drop=True)
    tfu = cost_param[(cost_param['value'] == 'tfum') & (cost_param['category'] == 'ALL')].reset_index(drop=True)
    result = tfu[__BATIMENT__] * suptu
    result[['sector', 'value']] = tfu[['sector', 'value']]
    result['category'] = 'unique'
    result = result[cost_param.columns]
    cout_result = pd.concat([cout_result, result],
                            ignore_index=True)

    # TODO: Allocation pour cuisine, salle de bain unite de marche
    val = cost_param[cost_param['value'] == 'all_cuis'][['sector'] + __BATIMENT__]
    val['value'] = 'cuisum'
    val['category'] = 'unique'
    val = val[cost_param.columns]
    cout_result = pd.concat([cout_result, val],
                            ignore_index=True)

    val = cost_param[cost_param['value'] == 'all_sdb'][['sector'] + __BATIMENT__]
    val['value'] = 'saldbum'
    val['category'] = 'unique'
    val = val[cost_param.columns]
    cout_result = pd.concat([cout_result, val],
                            ignore_index=True)

    # --> Total travaux unite de marche

    totbum = cout_result[(cout_result['value'] == 'tfum') | (cout_result['value'] == 'cuisum')
                         | (cout_result['value'] == 'saldbum')]
    totbum = totbum[__BATIMENT__].groupby(totbum['sector']).sum()

    # Travaux Finition unite familiale
    suptu = table_of_intrant[
        (table_of_intrant['value'] == 'suptu') & (table_of_intrant['category'].isin(__UNITE_TYPE__[5:]))].reset_index(
        drop=True)
    suptu = suptu[__BATIMENT__].groupby(suptu['sector']).sum().reset_index(drop=True)
    tfu = cost_param[(cost_param['value'] == 'tfum') & (cost_param['category'] == 'ALL')].reset_index(drop=True)
    result = tfu[__BATIMENT__] * suptu
    result['sector'] = tfu['sector']
    result['value'] = 'tfuf'
    result['category'] = 'unique'
    result = result[cost_param.columns]
    cout_result = pd.concat([cout_result, result],
                            ignore_index=True)

    # TODO: Allocation pour cuisine, salle de bain unite familiale
    val = cost_param[cost_param['value'] == 'all_cuis'][['sector'] + __BATIMENT__]
    val['value'] = 'cuisuf'
    val['category'] = 'unique'
    val = val[cost_param.columns]
    cout_result = pd.concat([cout_result, val],
                            ignore_index=True)

    val = cost_param[cost_param['value'] == 'all_sdb'][['sector'] + __BATIMENT__]
    val['value'] = 'saldbuf'
    val['category'] = 'unique'
    val = val[cost_param.columns]
    cout_result = pd.concat([cout_result, val],
                            ignore_index=True)


    # --> Total travaux unite de marche

    totbuf = cout_result[(cout_result['value'] == 'tfuf') | (cout_result['value'] == 'cuisuf')
                         | (cout_result['value'] == 'saldbuf')]
    totbuf = totbuf[__BATIMENT__].groupby(totbuf['sector']).sum()

    # Couts des finitions
    sh = myBook.sheet_by_name('PCOUTS')
    qu_pos = [88, 4]
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
    qum = qum.groupby('value').apply(get_qu, dict_cost_finitions).reset_index(drop=True)
    totbum = totbum.reset_index(drop=True) * qum
    totbum['category'] = 'unique'
    totbum['sector'] = __SECTEUR__
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
    quf = quf.groupby('value').apply(get_qu, dict_cost_finitions).reset_index(drop=True)
    totbuf = totbuf.reset_index(drop=True) * quf
    totbuf['category'] = 'unique'
    totbuf['sector'] = __SECTEUR__
    totbuf['value'] = 'cfuf'
    totbuf = totbuf[cost_param.columns]
    cout_result = pd.concat([cout_result, totbuf],
                            ignore_index=True)

    # Travaux finitions aires communes
    aire_commune = table_of_intrant[((table_of_intrant['value'] == 'supbtu')
                              |(table_of_intrant['value'] == 'cir'))
                              & (table_of_intrant['category'] == 'ALL')][
        ['sector'] + __BATIMENT__].reset_index(drop=True)

    aire_commune = aire_commune.groupby('sector').prod().reset_index(drop=True)
    tvfac = cost_param[(cost_param['value'] == 'tvfac') & (cost_param['category'] == 'ALL')][__BATIMENT__].reset_index(drop=True)
    tvfac = tvfac * aire_commune
    tvfac['category'] = 'unique'
    tvfac['sector'] = __SECTEUR__
    tvfac['value'] = 'tvfac'
    tvfac = tvfac[cost_param.columns]
    cout_result = pd.concat([cout_result, tvfac],
                            ignore_index=True)

    # ascenceurs
    cout_result = pd.concat([cout_result, cost_param[(cost_param['value'] == 'asc') & (cost_param['category'] == 'ALL')]],
                            ignore_index=True)

    # Cout additionnel Piscine
    pisc = table_of_intrant[(table_of_intrant['value'] == 'pisc')
                              & (table_of_intrant['category'] == 'ALL')][
        __BATIMENT__].reset_index(drop=True)

    pisc.iloc[pisc.iloc[:,:] == 'Non'] = 0
    pisc.iloc[pisc.iloc[:,:] == 'Oui'] = 1

    c_ad_pisc = cost_param[(cost_param['value'] == 'c_ad_pisc') & (cost_param['category'] == 'ALL')][__BATIMENT__].reset_index(drop=True)
    c_ad_pisc = c_ad_pisc * pisc
    c_ad_pisc['category'] = 'unique'
    c_ad_pisc['sector'] = __SECTEUR__
    c_ad_pisc['value'] = 'c_ad_pisc'
    c_ad_pisc = c_ad_pisc[cost_param.columns]
    cout_result = pd.concat([cout_result, c_ad_pisc],
                            ignore_index=True)

    # Cout additionnel chalet urbain
    cub = table_of_intrant[(table_of_intrant['value'] == 'cub')
                              & (table_of_intrant['category'] == 'ALL')][
        __BATIMENT__].reset_index(drop=True)
    cub.iloc[cub.iloc[:,:] == 'Non'] = 0
    cub.iloc[cub.iloc[:,:] == 'Oui'] = 1

    c_ad_cu = cost_param[(cost_param['value'] == 'c_ad_cu') & (cost_param['category'] == 'ALL')][__BATIMENT__].reset_index(drop=True)
    c_ad_cu = c_ad_cu * cub
    c_ad_cu['category'] = 'unique'
    c_ad_cu['sector'] = __SECTEUR__
    c_ad_cu['value'] = 'c_ad_cu'
    c_ad_cu = c_ad_cu[cost_param.columns]
    cout_result = pd.concat([cout_result, c_ad_cu],
                            ignore_index=True)

    # cout additionnel espace commmerciaux
    sup_com = table_of_intrant[(table_of_intrant['value'] == 'sup_com')
                              & (table_of_intrant['category'] == 'ALL')][
        __BATIMENT__].reset_index(drop=True)
    cir = table_of_intrant[(table_of_intrant['value'] == 'cir')
                              & (table_of_intrant['category'] == 'ALL')][
        __BATIMENT__].reset_index(drop=True)
    c_ad_com = cost_param[(cost_param['value'] == 'c_ad_com') & (cost_param['category'] == 'ALL')][__BATIMENT__].reset_index(drop=True)

    c_ad_com = c_ad_com * sup_com * (1 - cir)
    c_ad_com['category'] = 'unique'
    c_ad_com['sector'] = __SECTEUR__
    c_ad_com['value'] = 'c_ad_com'
    c_ad_com = c_ad_com[cost_param.columns]
    cout_result = pd.concat([cout_result, c_ad_com],
                            ignore_index=True)


    # Imprevus sur travaux
    su = cout_result[(cout_result['value'].isin(['tcq', 'tss', 'cfum', 'cfuf', 'tvfac', 'c_ad_pisc', 'c_ad_cu', 'c_ad_com']))
                    & (cout_result['category'] == 'unique')][['sector'] + __BATIMENT__].reset_index(drop=True)

    su = su.groupby('sector').sum().reset_index(drop=True)
    it = cost_param[(cost_param['value'] == 'it') & (cost_param['category'] == 'ALL')][__BATIMENT__].reset_index(drop=True)
    it = su /(1 - it) - su
    it['category'] = 'unique'
    it['sector'] = __SECTEUR__
    it['value'] = 'it'
    it = it[cost_param.columns]
    cout_result = pd.concat([cout_result, it],
                            ignore_index=True)

    # Sous total couts de construction
    cct = su + it
    cct['category'] = 'partial'
    cct['sector'] = __SECTEUR__
    cct['value'] = 'construction cost'
    cct = cct[cost_param.columns]
    cout_result = pd.concat([cout_result, cct],
                            ignore_index=True)

    ### --> SOFT COST

    # Arpenteur-geometre
    apt_geo = cost_param[cost_param['value'] == 'apt_geo'][['sector', 'value'] + __BATIMENT__]
    apt_geo['category'] = 'unique'
    apt_geo = apt_geo[cost_param.columns]
    cout_result = pd.concat([cout_result, apt_geo],
                            ignore_index=True)

    # Professionnels
    prof = cost_param[(cost_param['value'] == 'prof') & (cost_param['category'] == 'ALL')][__BATIMENT__].reset_index(drop=True)
    prof = prof * cct[__BATIMENT__]
    prof['category'] = 'unique'
    prof['sector'] = __SECTEUR__
    prof['value'] = 'prof'
    prof = prof[cost_param.columns]
    cout_result = pd.concat([cout_result, prof],
                            ignore_index=True)

    # Evaluator
    eval = cost_param[cost_param['value'] == 'eval'][['sector', 'value'] + __BATIMENT__]
    eval['category'] = 'unique'
    eval = eval[cost_param.columns]
    cout_result = pd.concat([cout_result, eval],
                            ignore_index=True)

    # Legal Fee
    legal_fee = cost_param[cost_param['value'] == 'legal_fee'][['sector', 'value'] + __BATIMENT__]
    legal_fee['category'] = 'unique'
    legal_fee = legal_fee[cost_param.columns]
    cout_result = pd.concat([cout_result, legal_fee],
                            ignore_index=True)

    # Professionnal fee divers
    prof_fee_div = cost_param[cost_param['value'] == 'prof_fee_div'][['sector', 'value'] + __BATIMENT__]
    prof_fee_div['category'] = 'unique'
    prof_fee_div = prof_fee_div[cost_param.columns]
    cout_result = pd.concat([cout_result, prof_fee_div],
                            ignore_index=True)

    # Pub
    pub = cost_param[(cost_param['value'] == 'pub') & (cost_param['category'] == 'ALL')][__BATIMENT__].reset_index(drop=True)
    pub = pub * cct[__BATIMENT__]
    pub['category'] = 'unique'
    pub['sector'] = __SECTEUR__
    pub['value'] = 'pub'
    pub = pub[cost_param.columns]
    cout_result = pd.concat([cout_result, pub],
                            ignore_index=True)

    # Construction permit
    construction_permit = cost_param[(cost_param['value'] == 'construction_permit') & (cost_param['category'] == 'ALL')][__BATIMENT__].reset_index(drop=True)
    construction_permit = construction_permit * cct[__BATIMENT__] / 1000
    construction_permit['category'] = 'unique'
    construction_permit['sector'] = __SECTEUR__
    construction_permit['value'] = 'pub'
    construction_permit = construction_permit[cost_param.columns]
    cout_result = pd.concat([cout_result, construction_permit],
                            ignore_index=True)

    # Total soft Cost
    su = cout_result[(cout_result['value'].isin(['apt_geo', 'prof', 'eval', 'legal_fee', 'prof_fee_div', 'pub',
                                                 'construction_permit', 'com']))
                    & (cout_result['category'] == 'unique')][['sector'] + __BATIMENT__].reset_index(drop=True)

    su = su.groupby('sector').sum().reset_index(drop=True)
    su['category'] = 'partial'
    su['sector'] = __SECTEUR__
    su['value'] = 'soft cost'
    su = su[cost_param.columns]
    cout_result = pd.concat([cout_result, su],
                            ignore_index=True)

    return cout_result




if __name__ == '__main__':

    myBook = xlrd.open_workbook(__FILES_NAME__)
    intrant_param = get_cb1_characteristics(myBook)
    # calcul_prix_terrain(0,0,0,0)
    print(calcul_cout_batiment(intrant_param, myBook))
