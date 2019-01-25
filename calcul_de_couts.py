from obtention_intrant import get_global_intrant

__author__ = 'pougomg'
import xlrd
import pandas as pd
import numpy as np
from lexique import __UNITE_TYPE__, __QUALITE_BATIMENT__
from import_parameter import __FILES_NAME__, __BATIMENT__, __SECTEUR__, get_land_param, get_building_cost_parameter


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


def calcul_cout_batiment(table_of_intrant, myBook):

    cost_param = get_building_cost_parameter(myBook)
    cout_result = pd.DataFrame([], columns=cost_param.columns)

    # Terrain

    den = table_of_intrant[(table_of_intrant['Value'] == 'denm_pu') & (table_of_intrant['Categorie'] == 'ALL')][
        __BATIMENT__].reset_index(drop=True)
    suft = table_of_intrant[(table_of_intrant['Value'] == 'supterrain') & (table_of_intrant['Categorie'] == 'ALL')][
        __BATIMENT__].reset_index(drop=True)
    calcul_prix_terrain(den, suft)
    # Travaux coquille

    tc = cost_param[(cost_param['Value'] == 'tcq') & (cost_param['Categorie'] == 'ALL')].reset_index(drop=True)

    supths = table_of_intrant[(table_of_intrant['Value'] == 'supths') & (table_of_intrant['Categorie'] == 'ALL')][
        __BATIMENT__].reset_index(drop=True)

    result = tc[__BATIMENT__] * supths
    result[['Secteur', 'Categorie', 'Value']] = tc[['Secteur', 'Categorie', 'Value']]
    result = result[cost_param.columns]

    cout_result = result

    # Travaux finitions des unites

    tc = cost_param[(cost_param['Value'] == 'tfu') & (cost_param['Categorie'] == 'ALL')].reset_index(drop=True)

    suptu = table_of_intrant[(table_of_intrant['Value'] == 'suptu') & (table_of_intrant['Categorie'] == 'ALL')][
        __BATIMENT__].reset_index(drop=True)

    result = tc[__BATIMENT__] * suptu
    result[['Secteur', 'Categorie', 'Value']] = tc[['Secteur', 'Categorie', 'Value']]
    result = result[cost_param.columns]
    cout_result = cout_result.append(result)

    # cout additionnel salle d'eau, cuisine

    suptu = table_of_intrant[
        (table_of_intrant['Value'] == 'ntu') & (table_of_intrant['Categorie'].isin(__UNITE_TYPE__[3:]))].reset_index(
        drop=True)

    suptu = suptu[__BATIMENT__].groupby(suptu['Secteur']).sum().reset_index()
    tc = cost_param[(cost_param['Value'] == 'ca_se') & (cost_param['Categorie'] == 'ALL')].reset_index(drop=True)

    result = tc[__BATIMENT__] * suptu
    result[['Secteur', 'Categorie', 'Value']] = tc[['Secteur', 'Categorie', 'Value']]
    result = result[cost_param.columns]
    cout_result = cout_result.append(result)

    tc = cost_param[(cost_param['Value'] == 'ca_gc') & (cost_param['Categorie'] == 'ALL')].reset_index(drop=True)

    result = tc[__BATIMENT__] * suptu
    result[['Secteur', 'Categorie', 'Value']] = tc[['Secteur', 'Categorie', 'Value']]
    result = result[cost_param.columns]
    cout_result = cout_result.append(result)

    # Finition aire commune

    suptu = table_of_intrant[((table_of_intrant['Value'] == 'suptub') |
                              (table_of_intrant['Value'] == 'supescom'))
                             & (table_of_intrant['Categorie'] == 'ALL')].reset_index(drop=True)

    cir = table_of_intrant[(table_of_intrant['Value'] == 'cir')
                           & (table_of_intrant['Categorie'] == 'ALL')][__BATIMENT__].reset_index(drop=True)

    suptu = suptu[__BATIMENT__].groupby(suptu['Secteur']).sum().reset_index()
    tc = cost_param[(cost_param['Value'] == 'tfac') & (cost_param['Categorie'] == 'ALL')].reset_index(drop=True)

    result = tc[__BATIMENT__] * suptu * cir
    result[['Secteur', 'Categorie', 'Value']] = tc[['Secteur', 'Categorie', 'Value']]
    result = result[cost_param.columns]
    cout_result = cout_result.append(result)

    # Ascenceur

    tc = cost_param[(cost_param['Value'] == 'asc') & (cost_param['Categorie'] == 'ALL')].reset_index(drop=True)

    result = tc[__BATIMENT__]
    result[['Secteur', 'Categorie', 'Value']] = tc[['Secteur', 'Categorie', 'Value']]
    result = result[cost_param.columns]
    cout_result = cout_result.append(result)

    # Piscine, Chalet urbain
    cir = table_of_intrant[(table_of_intrant['Value'] == 'pisc')
                           & (table_of_intrant['Categorie'] == 'ALL')][__BATIMENT__].reset_index(drop=True)

    tc = cost_param[(cost_param['Value'] == 'ca_asc_pi') & (cost_param['Categorie'] == 'ALL')].reset_index(drop=True)

    cir = cir.isin(["Oui"]).astype(int)
    result = tc[__BATIMENT__] * cir
    result[['Secteur', 'Categorie', 'Value']] = tc[['Secteur', 'Categorie', 'Value']]
    result = result[cost_param.columns]
    cout_result = cout_result.append(result)

    cir = table_of_intrant[(table_of_intrant['Value'] == 'cub')
                           & (table_of_intrant['Categorie'] == 'ALL')][__BATIMENT__].reset_index(drop=True)

    tc = cost_param[(cost_param['Value'] == 'ca_asc_cu') & (cost_param['Categorie'] == 'ALL')].reset_index(drop=True)

    cir = cir.isin(["Oui"]).astype(int)
    result = tc[__BATIMENT__] * cir
    result[['Secteur', 'Categorie', 'Value']] = tc[['Secteur', 'Categorie', 'Value']]
    result = result[cost_param.columns]
    cout_result = cout_result.append(result)

    # Finition espace commerciaux

    suptu = table_of_intrant[(table_of_intrant['Value'] == 'supescom')
                             & (table_of_intrant['Categorie'] == 'ALL')].reset_index(drop=True)

    cir = table_of_intrant[(table_of_intrant['Value'] == 'cir')
                           & (table_of_intrant['Categorie'] == 'ALL')][__BATIMENT__].reset_index(drop=True)

    tc = cost_param[(cost_param['Value'] == 'ca_esc_b') & (cost_param['Categorie'] == 'ALL')].reset_index(drop=True)

    result = tc[__BATIMENT__] * suptu * (1 - cir)
    result[['Secteur', 'Categorie', 'Value']] = tc[['Secteur', 'Categorie', 'Value']]
    result = result[cost_param.columns]
    cout_result = cout_result.append(result)

    # Imprevu
    sumc = cout_result[__BATIMENT__].groupby(cout_result['Secteur']).sum().reset_index(drop=True)
    tc = cost_param[(cost_param['Value'] == 'it') & (cost_param['Categorie'] == 'ALL')].reset_index(drop=True)
    result = sumc * (tc[__BATIMENT__] / (1 - tc[__BATIMENT__]))
    result[['Secteur', 'Categorie', 'Value']] = tc[['Secteur', 'Categorie', 'Value']]
    result = result[cost_param.columns]
    cout_result = cout_result.append(result)

    # Cout Additionnel
    cout_add = dict()
    sh = myBook.sheet_by_name('Cout')
    for quality in range(len(__QUALITE_BATIMENT__)):
        line = 74 + quality
        for type in range(len(__UNITE_TYPE__)):
            col = 4 + type
            if type > 4:
                col = col - 3
            value = sh.cell(line, col).value + sh.cell(73, col).value
            cout_add[(__UNITE_TYPE__[type], __QUALITE_BATIMENT__[quality])] = value

    sh = myBook.sheet_by_name('Scenarios')
    tab = ajouter_caraterisque_par_type_unite(sh, [], 'qum', [23, 2], False)
    tab = ajouter_caraterisque_par_type_unite(sh, tab, 'quf', [32, 2], False)
    tab = pd.DataFrame(tab, columns=table_of_intrant.columns)

    x = tab[tab['Value'] == 'qum'][__BATIMENT__].values
    res = [tab]

    for type_batiment in __UNITE_TYPE__[0:5]:
        _ = np.copy(x)
        _[_ == __QUALITE_BATIMENT__[0]] = cout_add[(type_batiment, __QUALITE_BATIMENT__[0])]
        _[_ == __QUALITE_BATIMENT__[1]] = cout_add[(type_batiment, __QUALITE_BATIMENT__[1])]
        _[_ == __QUALITE_BATIMENT__[2]] = cout_add[(type_batiment, __QUALITE_BATIMENT__[2])]

        suptu = table_of_intrant[(table_of_intrant['Value'] == 'suptu')
                                 & (table_of_intrant['Categorie'] == type_batiment)][__BATIMENT__].values

        column_name = ['qum' for secteur in __SECTEUR__]
        _ = np.insert(_, 0, column_name, axis=1)
        column_name = [type_batiment for secteur in __SECTEUR__]
        _ = np.insert(_, 0, column_name, axis=1)
        column_name = [secteur for secteur in __SECTEUR__]
        _ = np.insert(_, 0, column_name, axis=1)

        res.append(pd.DataFrame(_, columns=table_of_intrant.columns))

    x = tab[tab['Value'] == 'quf'][__BATIMENT__].values

    for type_batiment in __UNITE_TYPE__[5:]:
        _ = np.copy(x)
        _[_ == __QUALITE_BATIMENT__[0]] = cout_add[(type_batiment, __QUALITE_BATIMENT__[0])]
        _[_ == __QUALITE_BATIMENT__[1]] = cout_add[(type_batiment, __QUALITE_BATIMENT__[1])]
        _[_ == __QUALITE_BATIMENT__[2]] = cout_add[(type_batiment, __QUALITE_BATIMENT__[2])]

        suptu = table_of_intrant[(table_of_intrant['Value'] == 'suptu')
                                 & (table_of_intrant['Categorie'] == type_batiment)][__BATIMENT__].values

        _ = _ * suptu
        column_name = ['qum' for secteur in __SECTEUR__]
        _ = np.insert(_, 0, column_name, axis=1)
        column_name = [type_batiment for secteur in __SECTEUR__]
        _ = np.insert(_, 0, column_name, axis=1)
        column_name = [secteur for secteur in __SECTEUR__]
        _ = np.insert(_, 0, column_name, axis=1)

        res.append(pd.DataFrame(_, columns=table_of_intrant.columns))

    tab = pd.concat(res, ignore_index=True)
    tab = tab[tab['Categorie'] != 'ALL']
    tab = tab[__BATIMENT__].groupby(tab['Secteur']).sum()
    tab = pd.DataFrame(tab, columns=__BATIMENT__)

    tab['Categorie'] = 'ALL'
    tab['Secteur'] = [secteur for secteur in __SECTEUR__]
    tab['Value'] = 'qum'
    tab = tab[cost_param.columns]

    cout_result = cout_result.append(tab)

    sumc = cout_result[cout_result['Value'] != 'it']
    sumc = sumc[__BATIMENT__].groupby(sumc['Secteur']).sum().reset_index(drop=True)
    tc = cost_param[(cost_param['Value'] == 'tx') & (cost_param['Categorie'] == 'ALL')].reset_index(drop=True)
    result = sumc * tc[__BATIMENT__]
    result[['Secteur', 'Categorie', 'Value']] = tc[['Secteur', 'Categorie', 'Value']]
    result = result[cost_param.columns]
    cout_result = cout_result.append(result)

    sumc = cout_result[__BATIMENT__].groupby(cout_result['Secteur']).sum().reset_index(drop=True)
    result = sumc
    result[['Secteur', 'Categorie']] = tc[['Secteur', 'Categorie']]
    result['Value'] = 'cct'
    result = result[cost_param.columns]
    cout_result = cout_result.append(result)

    # SoFT Cost
    tc = cost_param[(cost_param['Value'] == 'aptgeo') & (cost_param['Categorie'] == 'ALL')].reset_index(drop=True)

    result = tc[__BATIMENT__]
    result[['Secteur', 'Categorie', 'Value']] = tc[['Secteur', 'Categorie', 'Value']]
    result = result[cost_param.columns]
    cout_result = cout_result.append(result)

    # Professionnel, Permis de construire, Pub

    cir = cout_result[(cout_result['Value'] == 'cct')
                      & (cout_result['Categorie'] == 'ALL')][__BATIMENT__].reset_index(drop=True)

    tc = cost_param[(cost_param['Value'] == 'pai') & (cost_param['Categorie'] == 'ALL')].reset_index(drop=True)

    result = tc[__BATIMENT__] * cir
    result[['Secteur', 'Categorie', 'Value']] = tc[['Secteur', 'Categorie', 'Value']]
    result = result[cost_param.columns]
    cout_result = cout_result.append(result)

    tc = cost_param[(cost_param['Value'] == 'pub') & (cost_param['Categorie'] == 'ALL')].reset_index(drop=True)

    result = tc[__BATIMENT__] * cir
    result[['Secteur', 'Categorie', 'Value']] = tc[['Secteur', 'Categorie', 'Value']]
    result = result[cost_param.columns]
    cout_result = cout_result.append(result)

    tc = cost_param[(cost_param['Value'] == 'pco') & (cost_param['Categorie'] == 'ALL')].reset_index(drop=True)

    result = tc[__BATIMENT__] * cir / 1000
    result[['Secteur', 'Categorie', 'Value']] = tc[['Secteur', 'Categorie', 'Value']]
    result = result[cost_param.columns]
    cout_result = cout_result.append(result)

    # Evaluateur, Frais legaux, Frais Professionnel Divers

    tc = cost_param[(cost_param['Value'] == 'ev') & (cost_param['Categorie'] == 'ALL')].reset_index(drop=True)

    result = tc[__BATIMENT__]
    result[['Secteur', 'Categorie', 'Value']] = tc[['Secteur', 'Categorie', 'Value']]
    result = result[cost_param.columns]
    cout_result = cout_result.append(result)

    tc = cost_param[(cost_param['Value'] == 'fl') & (cost_param['Categorie'] == 'ALL')].reset_index(drop=True)

    result = tc[__BATIMENT__]
    result[['Secteur', 'Categorie', 'Value']] = tc[['Secteur', 'Categorie', 'Value']]
    result = result[cost_param.columns]
    cout_result = cout_result.append(result)

    tc = cost_param[(cost_param['Value'] == 'fl') & (cost_param['Categorie'] == 'ALL')].reset_index(drop=True)

    result = tc[__BATIMENT__]
    result[['Secteur', 'Categorie', 'Value']] = tc[['Secteur', 'Categorie', 'Value']]
    result = result[cost_param.columns]
    cout_result = cout_result.append(result)

    tc = cost_param[(cost_param['Value'] == 'fpa') & (cost_param['Categorie'] == 'ALL')].reset_index(drop=True)

    result = tc[__BATIMENT__]
    result[['Secteur', 'Categorie', 'Value']] = tc[['Secteur', 'Categorie', 'Value']]
    result = result[cost_param.columns]
    cout_result = cout_result.append(result)

    tc = cost_param[(cost_param['Value'] == 'afc') & (cost_param['Categorie'] == 'ALL')].reset_index(drop=True)

    result = tc[__BATIMENT__]
    result[['Secteur', 'Categorie', 'Value']] = tc[['Secteur', 'Categorie', 'Value']]
    result = result[cost_param.columns]
    cout_result = cout_result.append(result)

    return cout_result





if __name__ == '__main__':

    myBook = xlrd.open_workbook(__FILES_NAME__)
    intrant_param = get_global_intrant(myBook)
    # calcul_prix_terrain(0,0,0,0)
    print(calcul_cout_batiment(__BATIMENT__[7], __SECTEUR__[4], 0, "Base", intrant_param, myBook))
