__author__ = 'pougomg'

import numpy as np
import pandas as pd
import xlrd

from import_parameter import get_building_cost_parameter
from lexique import __UNITE_TYPE__, __BATIMENT__, __SECTEUR__,__FILES_NAME__
from obtention_intrant import get_summary_characteristics, get_all_informations


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


def calcul_cout_batiment(secteur: list, batiment: list, table_of_intrant: pd.DataFrame, cost_params: pd.DataFrame) -> pd.DataFrame:

    """
    this function is used to compute the cost of the building given the intrants parameters
    :param secteur: list of sector
    :param batiment: list of building
    :param table_of_intrant: table of CA3, CA4
    :param cost_params: cost parameters
    :return: data frame containing the cost of the building and the usefuls informations to compute the finance.
    """

    ##################################################################################################################
    #
    # HARD COST COMPUTATION
    #
    ##################################################################################################################

    # Coquille
    tc = cost_params[cost_params['value'] == 'tcq'].reset_index(drop=True)
    sup_tot_hs = table_of_intrant[table_of_intrant['value'] == 'sup_tot_hs'].reset_index(drop=True)
    result = tc[batiment].values * sup_tot_hs[batiment]
    result['sector'] = secteur
    result['value'] = 'tcq'
    result['category'] = 'unique'
    result['type'] = 'cost'
    result = result[table_of_intrant.columns]
    table_of_intrant = pd.concat([table_of_intrant, result], ignore_index=True)


    # Sous Sol
    tss = cost_params[cost_params['value'] == 'tss'].reset_index(drop=True)
    sup_ss = table_of_intrant[table_of_intrant['value'] == 'sup_ss'].reset_index(drop=True)
    result = tss[batiment].values * sup_ss[batiment]
    result['sector'] = secteur
    result['value'] = 'tss'
    result['category'] = 'unique'
    result['type'] = 'cost'
    result = result[table_of_intrant.columns]
    table_of_intrant = pd.concat([table_of_intrant, result], ignore_index=True)


    # Travaux Finition unite de marche
    suptu = table_of_intrant[
        (table_of_intrant['value'] == 'suptu') & (table_of_intrant['category'].isin(__UNITE_TYPE__[0:5]))].reset_index(
        drop=True)
    suptu = suptu[batiment].groupby(suptu['sector']).sum().reset_index(drop=True)
    tfu = cost_params[cost_params['value'] == 'tfu'].reset_index(drop=True)
    result = tfu[batiment].values * suptu
    result['sector'] = secteur
    result['value'] = 'tfum'
    result['category'] = 'unique'
    result['type'] = 'cost'
    result = result[table_of_intrant.columns]
    table_of_intrant = pd.concat([table_of_intrant, result], ignore_index=True)


    # Allocation pour cuisine, salle de bain unite de marche
    ntu = table_of_intrant[
        (table_of_intrant['value'] == 'ntu') & (table_of_intrant['category'].isin(__UNITE_TYPE__[0:5]))].reset_index(
        drop=True)


    ntu.loc[ntu['category'].isin(__UNITE_TYPE__[3:5]), batiment] = 1.5 * ntu[batiment]
    ntu = ntu[batiment].groupby(ntu['sector']).sum().reset_index(drop=True)

    val = cost_params[cost_params['value'] == 'all_cuis'].reset_index(drop=True)
    result = val[batiment].values * ntu[batiment]

    result['sector'] = secteur
    result['value'] = 'cuisum'
    result['category'] = 'unique'
    result['type'] = 'cost'
    result = result[table_of_intrant.columns]
    table_of_intrant = pd.concat([table_of_intrant, result], ignore_index=True)


    val = cost_params[cost_params['value'] == 'all_sdb'].reset_index(drop=True)
    val.loc[val['category'].isin(__UNITE_TYPE__[3:5]), batiment] = 1.5 * val[batiment]
    result = val[batiment].values * ntu[batiment]
    result['sector'] = secteur
    result['value'] = 'saldbum'
    result['category'] = 'unique'
    result['type'] = 'cost'
    result = result[table_of_intrant.columns]
    table_of_intrant = pd.concat([table_of_intrant, result], ignore_index=True)


    # Travaux Finition unite abordables
    suptu = table_of_intrant[
        (table_of_intrant['value'] == 'suptu') & (table_of_intrant['category'].isin(__UNITE_TYPE__[5:]))].reset_index(
        drop=True)
    suptu = suptu[batiment].groupby(suptu['sector']).sum().reset_index(drop=True)
    tfu = cost_params[cost_params['value'] == 'tfu'].reset_index(drop=True)

    result = tfu[batiment].values * suptu[batiment]
    result['sector'] = secteur
    result['value'] = 'tfuf'
    result['category'] = 'unique'
    result['type'] = 'cost'
    result = result[table_of_intrant.columns]
    table_of_intrant = pd.concat([table_of_intrant, result], ignore_index=True)

    # Allocation pour cuisine, salle de bain unite de marche
    ntu = table_of_intrant[
        (table_of_intrant['value'] == 'ntu') & (table_of_intrant['category'].isin(__UNITE_TYPE__[5:]))].reset_index(
        drop=True)
    ntu = ntu[batiment].groupby(ntu['sector']).sum().reset_index(drop=True)
    val = cost_params[cost_params['value'] == 'all_cuis'].reset_index(drop=True)
    result = val[batiment].values * ntu[batiment] * 1.5

    result['sector'] = secteur
    result['value'] = 'cuisuf'
    result['category'] = 'unique'
    result['type'] = 'cost'
    result = result[table_of_intrant.columns]
    table_of_intrant = pd.concat([table_of_intrant, result], ignore_index=True)

    val = cost_params[cost_params['value'] == 'all_sdb'].reset_index(drop=True)
    result = val[batiment].values * ntu[batiment]
    result['sector'] = secteur
    result['value'] = 'saldbuf'
    result['category'] = 'unique'
    result['type'] = 'cost'
    result = result[table_of_intrant.columns]
    table_of_intrant = pd.concat([table_of_intrant, result], ignore_index=True)

    # --> Total travaux unite de marche

    totbum = table_of_intrant[(table_of_intrant['value'].isin(['tfum', 'cuisum',  'saldbum'])) &
                              (table_of_intrant['type'] == 'cost')]
    totbum = totbum[batiment].groupby(totbum['sector']).sum().reset_index(drop=True)
    qum = cost_params[cost_params['value'] == 'qum'].reset_index(drop=True)
    result = qum[batiment].values * totbum[batiment]
    result['sector'] = secteur
    result['value'] = 'cfum'
    result['category'] = 'unique'
    result['type'] = 'cost'
    result = result[table_of_intrant.columns]
    table_of_intrant = pd.concat([table_of_intrant, result], ignore_index=True)

    totbum = table_of_intrant[(table_of_intrant['value'].isin(['tfuf', 'cuisuf',  'saldbuf'])) &
                              (table_of_intrant['type'] == 'cost')]
    totbum = totbum[batiment].groupby(totbum['sector']).sum().reset_index(drop=True)
    qum = cost_params[cost_params['value'] == 'quf'].reset_index(drop=True)
    result = qum[batiment].values * totbum[batiment]
    result['sector'] = secteur
    result['value'] = 'cfuf'
    result['category'] = 'unique'
    result['type'] = 'cost'
    result = result[table_of_intrant.columns]
    table_of_intrant = pd.concat([table_of_intrant, result], ignore_index=True)

    # Travaux finitions aires communes
    aire_commune = table_of_intrant[(table_of_intrant['value'].isin(['supbtu', 'cir']))
                                    & (table_of_intrant['category'] == 'ALL')][
        ['sector'] + batiment].reset_index(drop=True)

    aire_commune = aire_commune.groupby('sector').prod().reset_index(drop=True)
    tvfac = cost_params[cost_params['value'] == 'tvfac'][batiment].reset_index(drop=True)
    tvfac = tvfac[batiment].values * aire_commune[batiment]
    tvfac['category'] = 'unique'
    tvfac['sector'] = secteur
    tvfac['value'] = 'tvfac'
    tvfac['type'] = 'cost'
    result = tvfac[table_of_intrant.columns]
    table_of_intrant = pd.concat([table_of_intrant, result], ignore_index=True)

    # ascenceurs
    asc = cost_params[cost_params['value'] == 'asc'][batiment].reset_index(drop=True)
    nba = table_of_intrant[table_of_intrant['value'] == 'nba'][batiment].reset_index(drop=True)
    asc = asc.values * nba
    asc['category'] = 'unique'
    asc['sector'] = secteur
    asc['value'] = 'asc'
    asc['type'] = 'cost'
    result = asc[table_of_intrant.columns]
    table_of_intrant = pd.concat([table_of_intrant, result], ignore_index=True)

    # Cout additionnel Piscine
    pisc = table_of_intrant[(table_of_intrant['value'] == 'pisc') &
                            (table_of_intrant['category'] == 'ALL')][batiment].reset_index(drop=True)
    pisc.replace({'Non': 0, 'Oui': 1}, inplace=True)

    c_ad_pisc = cost_params[cost_params['value'] == 'c_ad_pisc'][batiment].reset_index(drop=True)
    c_ad_pisc = c_ad_pisc.values * pisc
    c_ad_pisc['category'] = 'unique'
    c_ad_pisc['sector'] = secteur
    c_ad_pisc['value'] = 'c_ad_pisc'
    c_ad_pisc['type'] = 'cost'
    result = c_ad_pisc[table_of_intrant.columns]
    table_of_intrant = pd.concat([table_of_intrant, result], ignore_index=True)

    # Cout additionnel chalet urbain
    cub = table_of_intrant[table_of_intrant['value'] == 'cub'][batiment].reset_index(drop=True)
    cub.replace({'Non': 0, 'Oui': 1}, inplace=True)
    c_ad_cu = cost_params[cost_params['value'] == 'c_ad_cu'][batiment].reset_index(drop=True)
    c_ad_cu = c_ad_cu.values* cub
    c_ad_cu['category'] = 'unique'
    c_ad_cu['sector'] = secteur
    c_ad_cu['value'] = 'c_ad_cu'
    c_ad_cu['type'] = 'cost'
    result = c_ad_cu[table_of_intrant.columns]
    table_of_intrant = pd.concat([table_of_intrant, result], ignore_index=True)

    # cout additionnel espace commmerciaux
    sup_com = table_of_intrant[table_of_intrant['value'] == 'sup_com'][batiment].reset_index(drop=True)
    cir = table_of_intrant[table_of_intrant['value'] == 'cir'][batiment].reset_index(drop=True)
    c_ad_com = cost_params[cost_params['value'] == 'c_ad_com'][batiment].reset_index(drop=True)

    c_ad_com = c_ad_com.values * sup_com * (1 - cir)
    c_ad_com['category'] = 'unique'
    c_ad_com['sector'] = secteur
    c_ad_com['value'] = 'c_ad_com'
    c_ad_com['type'] = 'cost'
    result = c_ad_com[table_of_intrant.columns]
    table_of_intrant = pd.concat([table_of_intrant, result], ignore_index=True)

    # Imprevus sur travaux
    value_hard_cost = ['tcq', 'tss', 'cfum', 'cfuf', 'tvfac', 'asc', 'c_ad_pisc', 'c_ad_cu', 'c_ad_com']

    su = table_of_intrant[(table_of_intrant['value'].isin(value_hard_cost)) &
                          (table_of_intrant['category'] == 'unique')][['sector'] + batiment].reset_index(drop=True)

    su = su.groupby('sector').sum().reset_index(drop=True)
    it = cost_params[(cost_params['value'] == 'it')][batiment].reset_index(drop=True)
    it = (1/(1 - it.values)) *  su  - su
    it['category'] = 'unique'
    it['sector'] = secteur
    it['value'] = 'it'
    it['type'] = 'cost'
    result = it[table_of_intrant.columns]
    table_of_intrant = pd.concat([table_of_intrant, result],ignore_index=True)


    # Sous total couts de construction
    cct = su + it
    cct['category'] = 'partial'
    cct['sector'] = secteur
    cct['value'] = 'hard cost'
    cct['type'] = 'cost'
    result = cct[table_of_intrant.columns]
    table_of_intrant = pd.concat([table_of_intrant, result],ignore_index=True)


    ##################################################################################################################
    #
    # SOFT COST
    #
    ##################################################################################################################

    # Arpenteur-geometre
    apt_geo = cost_params[cost_params['value'] == 'apt_geo'][batiment].reset_index(drop=True)
    sup_com = table_of_intrant[table_of_intrant['value'] == 'sup_com'][batiment].reset_index(drop=True)
    sup_com[batiment] = 1
    apt_geo = apt_geo.values * sup_com
    apt_geo['sector'] = secteur
    apt_geo['category'] = 'unique'
    apt_geo['type'] = 'cost'
    apt_geo['value'] = 'apt_geo'
    result = apt_geo[table_of_intrant.columns]
    table_of_intrant = pd.concat([table_of_intrant, result],ignore_index=True)


    # Professionnels
    prof = cost_params[cost_params['value'] == 'prof'][batiment].reset_index(drop=True)
    prof = prof.values * cct[batiment]
    prof['category'] = 'unique'
    prof['sector'] = secteur
    prof['value'] = 'prof'
    prof['type'] = 'cost'
    result = prof[table_of_intrant.columns]
    table_of_intrant = pd.concat([table_of_intrant, result],ignore_index=True)


    # Evaluator
    apt_geo = cost_params[cost_params['value'] == 'eval'][batiment].reset_index(drop=True)
    apt_geo = apt_geo.values * sup_com
    apt_geo['sector'] = secteur
    apt_geo['category'] = 'unique'
    apt_geo['type'] = 'cost'
    apt_geo['value'] = 'eval'
    result = apt_geo[table_of_intrant.columns]
    table_of_intrant = pd.concat([table_of_intrant, result],ignore_index=True)


    # Legal Fee
    apt_geo = cost_params[cost_params['value'] == 'legal_fee'][batiment].reset_index(drop=True)
    apt_geo = apt_geo.values * sup_com
    apt_geo['sector'] = secteur
    apt_geo['category'] = 'unique'
    apt_geo['type'] = 'cost'
    apt_geo['value'] = 'legal_fee'
    result = apt_geo[table_of_intrant.columns]
    table_of_intrant = pd.concat([table_of_intrant, result],ignore_index=True)


    # Professionnal fee divers
    apt_geo = cost_params[cost_params['value'] == 'prof_fee_div'][batiment].reset_index(drop=True)
    apt_geo = apt_geo.values * sup_com
    apt_geo['sector'] = secteur
    apt_geo['category'] = 'unique'
    apt_geo['type'] = 'cost'
    apt_geo['value'] = 'prof_fee_div'
    result = apt_geo[table_of_intrant.columns]
    table_of_intrant = pd.concat([table_of_intrant, result],ignore_index=True)


    # Pub
    pub = cost_params[cost_params['value'] == 'pub'][batiment].reset_index(drop=True)
    pub = pub.values * cct[batiment]
    pub['category'] = 'unique'
    pub['sector'] = secteur
    pub['value'] = 'pub'
    pub['type'] = 'cost'
    result = pub[table_of_intrant.columns]
    table_of_intrant = pd.concat([table_of_intrant, result],ignore_index=True)


    # Construction permit
    construction_permit =  cost_params[cost_params['value'] == 'construction_permit'][batiment].reset_index(drop=True)
    construction_permit = construction_permit.values * cct[batiment] / 1000
    construction_permit['category'] = 'unique'
    construction_permit['sector'] = secteur
    construction_permit['value'] = 'construction_permit'
    construction_permit['type'] = 'cost'
    result = construction_permit[table_of_intrant.columns]
    table_of_intrant = pd.concat([table_of_intrant, result],ignore_index=True)

    # honoraire
    prof = cost_params[cost_params['value'] == 'hon_prom'][batiment].reset_index(drop=True)
    prof = prof.values * cct[batiment]
    prof['category'] = 'unique'
    prof['sector'] = secteur
    prof['value'] = 'hon_prom'
    prof['type'] = 'cost'
    result = prof[table_of_intrant.columns]
    table_of_intrant = pd.concat([table_of_intrant, result],ignore_index=True)


    # Total soft Cost

    entete_sc = ['apt_geo', 'prof', 'eval', 'legal_fee', 'prof_fee_div', 'pub', 'construction_permit', 'com',
                 'hon_prom']

    su = table_of_intrant[(table_of_intrant['value'].isin(entete_sc)) &
                          (table_of_intrant['category'] == 'unique')][['sector'] + batiment].reset_index(drop=True)

    su = su.groupby('sector').sum().reset_index(drop=True)
    su['category'] = 'partial'
    su['sector'] = secteur
    su['value'] = 'soft cost'
    su['type'] = 'cost'
    result = su[table_of_intrant.columns]
    table_of_intrant = pd.concat([table_of_intrant, result],ignore_index=True)




    #################################################################################################################
    #
    # LAND PRICE
    #
    ##################################################################################################################

    # Land price
    sup_ter = table_of_intrant[(table_of_intrant['value'] == 'sup_ter') & (table_of_intrant['category'] == 'ALL')][
        batiment].reset_index(drop=True)

    vat = table_of_intrant[(table_of_intrant['value'] == 'vat') & (table_of_intrant['category'] == 'ALL')][
        batiment].reset_index(drop=True)

    price_land = vat.astype(float) * sup_ter.astype(float)
    price_land['category'] = 'unique'
    price_land["value"] = 'price_land'
    price_land['sector'] = secteur
    price_land['type'] = 'cost'
    result = price_land[table_of_intrant.columns]
    table_of_intrant = pd.concat([table_of_intrant, result],ignore_index=True)


    fm = pd.DataFrame(secteur, columns=['sector'])

    for value in batiment:
        fm.loc[:, value] = price_land[value].transform(lambda x: apply_mutation_function(x)).values

    fm['category'] = 'unique'
    fm["value"] = 'fm'
    fm['sector'] = secteur
    fm['type'] = 'cost'
    result = fm[table_of_intrant.columns]
    table_of_intrant = pd.concat([table_of_intrant, result],ignore_index=True)

    # Cout attribution terrain
    result = price_land[batiment] + fm[batiment]
    result['category'] = 'unique'
    result["value"] = 'caq_ter'
    result['sector'] = secteur
    result['type'] = 'cost'
    result = result[table_of_intrant.columns]
    table_of_intrant = pd.concat([table_of_intrant, result],ignore_index=True)

    #Contribution sociale

    table_of_intrant.loc[table_of_intrant['value'].isin(['contrib_terr_hs', 'contrib_fin', 'contrib_terr_ss']), ['category', 'type']] = \
        [['unique', 'cost']]

    supbtu = table_of_intrant[(table_of_intrant['value'] == 'supbtu') & (table_of_intrant['category'] == 'ALL')][
        batiment].reset_index(drop=True)

    # Frais de parc
    sup_parc = table_of_intrant[(table_of_intrant['value'] == 'sup_parc') & (table_of_intrant['category'] == 'ALL')][
        batiment].reset_index(drop=True)

    result = (sup_parc * price_land * 0.1 / supbtu)
    result['category'] = 'unique'
    result["value"] = 'frais_parc'
    result['sector'] = secteur
    result['type'] = 'cost'
    result = result[table_of_intrant.columns]
    table_of_intrant = pd.concat([table_of_intrant, result],ignore_index=True)


    # Decontamination
    decont = table_of_intrant[(table_of_intrant['value'] == 'decont') & (table_of_intrant['category'] == 'ALL')][
        batiment].reset_index(drop=True)
    result = decont[batiment].astype(float) * sup_ter[batiment].astype(float)
    result['category'] = 'unique'
    result["value"] = 'decont'
    result['sector'] = secteur
    result['type'] = 'cost'
    result = result[table_of_intrant.columns]
    table_of_intrant = pd.concat([table_of_intrant, result],ignore_index=True)

    # Rem
    rem = table_of_intrant[(table_of_intrant['value'] == 'rem') & (table_of_intrant['category'] == 'ALL')][
        batiment].reset_index(drop=True)
    rem.replace({'Oui': 1, 'Non': 0}, inplace=True)
    sup_tot_hs.loc[:, batiment] = sup_tot_hs.loc[: ,batiment].where(sup_tot_hs[batiment] > 2002, 0)
    sup_tot_hs.loc[:, batiment] = sup_tot_hs.loc[: ,batiment].where(cct[batiment] >= 756150, 0)

    result = 10 * sup_tot_hs[batiment].astype(float) * rem[batiment].astype(float)
    result['category'] = 'unique'
    result["value"] = 'rem'
    result['sector'] = secteur
    result['type'] = 'cost'
    result = result[table_of_intrant.columns]
    table_of_intrant = pd.concat([table_of_intrant, result],ignore_index=True)


    # Sous Total Terrain
    su = table_of_intrant[(table_of_intrant['value'].isin(['caq_ter', 'contrib_terr_hs', 'contrib_fin', 'frais_parc',
                                                           'contrib_terr_ss', 'decont', 'rem']))
                          & (table_of_intrant['category'] == 'unique')][['sector'] + batiment].reset_index(drop=True)
    su = su.groupby('sector').sum().reset_index(drop=True)
    su['category'] = 'partial'
    su['sector'] = secteur
    su['value'] = 'acq terrain'
    su['type'] = 'cost'
    result = su[table_of_intrant.columns]
    table_of_intrant = pd.concat([table_of_intrant, result],ignore_index=True)

    # Cout total du projet
    tot = table_of_intrant[table_of_intrant['category'] == 'partial'][['sector'] + batiment].reset_index(drop=True)
    tot = tot.groupby('sector').sum().reset_index(drop=True)
    tot['category'] = 'total'
    tot['sector'] = secteur
    tot['type'] = 'cost'
    tot['value'] = 'cout total du projet'
    tot = tot[table_of_intrant.columns]
    table_of_intrant = pd.concat([table_of_intrant, tot], ignore_index=True)

    # Sous Total Terrain
    su = table_of_intrant[(table_of_intrant['value'].isin(['caq_ter', 'contrib_terr_ss', 'decont']))
                          & (table_of_intrant['category'] == 'unique')][['sector'] + batiment].reset_index(drop=True)
    su = su.groupby('sector').sum().reset_index(drop=True)
    su['category'] = 'partial'
    su['sector'] = secteur
    su['value'] = 'financement terrain'
    su['type'] = 'cost'
    result = su[table_of_intrant.columns]
    table_of_intrant = pd.concat([table_of_intrant, result], ignore_index=True)
    table_of_intrant.loc[table_of_intrant['value'] == 'price', batiment] = 1.1 * table_of_intrant.loc[
                                                                               table_of_intrant['value'] == 'price', batiment]
    price = table_of_intrant[table_of_intrant['value'] == 'price']
    disct_ab = table_of_intrant[table_of_intrant['value'] == 'disct_ab']
    prix_ab = price[batiment].reset_index(drop=True) * (1 - disct_ab[batiment].reset_index(drop=True))
    prix_ab['sector'] = price['sector'].reset_index(drop=True)
    prix_ab['category'] = price['category'].reset_index(drop=True)

    price.loc[price['category'] == __UNITE_TYPE__[-1], batiment] = prix_ab.loc[prix_ab['category'] == __UNITE_TYPE__[-1], batiment]
    price.loc[price['category'] != __UNITE_TYPE__[-1], batiment] = 1.10 * price.loc[price['category'] != __UNITE_TYPE__[-1], batiment]

    n_3cc = table_of_intrant[(table_of_intrant['value'] == 'ntu') & (table_of_intrant['category'] == __UNITE_TYPE__[-1])]
    penth = table_of_intrant[(table_of_intrant['value'] == 'ntu') & (table_of_intrant['category'] == __UNITE_TYPE__[4])]

    prop_ab = table_of_intrant[table_of_intrant['value'] == 'prop_ab']
    ntu = table_of_intrant[(table_of_intrant['value'] == 'ntu') & (table_of_intrant['category'] == 'ALL')]
    tot_ab = ntu[batiment].reset_index(drop=True) * prop_ab[batiment].reset_index(drop=True)
    tot_ab = tot_ab.astype(float).apply(np.ceil)
    prop_rest = (tot_ab - n_3cc[batiment].reset_index(drop=True))/ \
                (ntu[batiment].reset_index(drop=True) - penth[batiment].reset_index(drop=True))
    prop_rest['category'] = __UNITE_TYPE__[0]
    prop_rest['sector'] = secteur
    result = prop_rest.copy()
    for units in __UNITE_TYPE__[1:]:
        result.loc[:, 'category'] = units
        result.loc[:, 'sector'] = secteur
        prop_rest = pd.concat([prop_rest, result], ignore_index=True)
    prop_rest.loc[prop_rest['category'].isin(__UNITE_TYPE__[3:]), batiment] = 0
    prop_rest = prop_rest.sort_values(['sector', 'category'])
    ntu = table_of_intrant[(table_of_intrant['value'] == 'ntu') & (table_of_intrant['category'] != 'ALL')].sort_values(['sector', 'category'])
    nb_ab = prop_rest[batiment].reset_index(drop=True) * ntu[batiment].reset_index(drop=True)
    nb_ab = nb_ab.astype(float).apply(np.round)
    nb_nab = ntu[batiment].reset_index(drop=True) - nb_ab
    nb_ab['category'] = ntu['category'].reset_index(drop=True)
    nb_ab['sector'] = ntu['sector'].reset_index(drop=True)

    nb_nab['category'] = ntu['category'].reset_index(drop=True)
    nb_nab['sector'] = ntu['sector'].reset_index(drop=True)

    price = price.sort_values(['sector', 'category'])
    prix_ab = prix_ab.sort_values(['sector', 'category'])

    price_new = (price[batiment].reset_index(drop=True) * nb_nab[batiment] + prix_ab[batiment] * nb_ab[batiment])
    ntu = nb_nab[batiment] + nb_ab[batiment]
    ntu = ntu.where(ntu != 0, np.nan)
    price_new = price_new.where(ntu == 0, price_new/ntu).fillna(0)
    price_new['category'] = price['category'].reset_index(drop=True)
    price_new['sector'] = price['sector'].reset_index(drop=True)

    price = table_of_intrant[table_of_intrant['value'] == 'price']
    price = price.sort_values(['sector', 'category'])
    price.loc[:, batiment] = price_new.sort_values(['sector', 'category'])[batiment].values
    table_of_intrant.loc[table_of_intrant['value'] == 'price', batiment] = price[batiment]

    return table_of_intrant


def calculate_cost(type, secteur, batiment, params,cost, *args):

    params = get_summary_characteristics(type, secteur, batiment, params, *args)
    return calcul_cout_batiment(secteur, batiment, params, cost)


if __name__ == '__main__':

    myBook = xlrd.open_workbook(__FILES_NAME__)
    x = get_all_informations(myBook)
    cost_params = x[(x['type'].isin(['pcost'])) & (x['sector'] == 'Secteur 1')]
    args = dict()
    # supter = [50000]
    # densite = [10]
    t = calculate_cost('CA3', __SECTEUR__,  __BATIMENT__, x, cost_params, args)
    # params = get_summary_characteristics('CB1', __SECTEUR__[0:4], __BATIMENT__[4:], x, args)
    # calcul_cout_batiment(params,  secteur, batiment)
