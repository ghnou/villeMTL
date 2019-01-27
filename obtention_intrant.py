import numpy as np
import pandas as pd
import xlrd

from lexique import __FILES_NAME__, __INTRANT_SHEET__, __BATIMENT__, __SECTEUR__, __UNITE_TYPE__


def ajouter_caraterisque_par_secteur(sh, tab, name, pos, unique):

    for secteur in __SECTEUR__:
        _ = [secteur, 'ALL', name]
        line = pos[0] if unique else __SECTEUR__.index(secteur) + pos[0]
        for batiment in range(len(__BATIMENT__)):
            _.append(sh.cell(line, pos[1] + batiment).value)
        tab.append(_)
    return tab


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


def convert_unity_type_to_sum_sector(group, data):
    print(group.name)
    print(data)
    print(group)
    df = data[__BATIMENT__].mul(group[__BATIMENT__].values.tolist()[0], axis=1)
    # df['category'] = group.name
    # df['sector'] = data['sector']
    # df['value'] = data['value']
    return df.sum()


def convert_unity_type_to_sector(group, data):
    df = data[__BATIMENT__].mul(group[__BATIMENT__].values.tolist()[0], axis=1)
    df['category'] = group.name
    df['sector'] = data['sector']
    df['value'] = data['value']
    return df


def split_unity_type_to_sector(group, data):
    return data


def get_surface(group, dict_of_surface):
    tab = [group[batiment].map(dict_of_surface[group.name]).values.tolist() for batiment in __BATIMENT__]
    tab = np.array(tab).transpose()
    tab = pd.DataFrame(tab, columns=__BATIMENT__)
    tab['category'] = group.name
    group = group.reset_index()
    tab['sector'] = group['sector']
    tab['value'] = group['value']

    return tab


def calculate_total_surface(group):
    df = group[group['value'] == 'ntu'][__BATIMENT__].reset_index(drop=True) * \
         group[group['value'] == 'tum'][__BATIMENT__].reset_index(drop=True)
    df['category'] = group.name
    df['sector'] = group[group['value'] == 'ntu']['sector'].reset_index(drop=True)
    df['value'] = 'suptu'

    return df


def get_mean_brute_surface(group, data):
    df = group[__BATIMENT__].reset_index(drop=True).mul(data[data['sector'] == group.name][__BATIMENT__].values[0],
                                                        axis='columns')
    df['sector'] = group['sector'].reset_index(drop=True)
    df['value'] = 'sup_bru_par_u'
    df['category'] = group.name

    return df


def get_cb1_characteristics(workbook) -> pd.DataFrame:
    """
    This function takes all the parameters of the Intrants sheets and calculate all the characteristics
    of the CB1 sheet.

    :param workbook: Excel Workbook containing all the parameters
    :return: pd.Dataframe of all the characteristics of the CB1 sheet.

    """""

    # Open Intrants sheet and take all the importants parameters
    sh = workbook.sheet_by_name(__INTRANT_SHEET__)
    table_of_intrant = []

    # Table containing the position of all the intrants in the Intrants sheets. Refers to the lexique files
    # to get the definition of the variables.
    tab_of_intrant_pos = [[[10, 3], 'ntu', 's'], [[19, 3], 'nmu_et', 's'], [[55, 3], 'denm_pu', 's'],
                          [[64, 3], 'denm_p', 's'], [[118, 3], 'mp', 'ns'],
                          [[119, 3], 'min_nu', 'ns'], [[120, 3], 'max_nu', 'ns'], [[121, 3], 'min_ne', 'ns'],
                          [[122, 3], 'max_ne', 'ns'], [[123, 3], 'min_ne_ss', 'ns'], [[124, 3], 'max_ne_ss', 'ns'],
                          [[125, 3], 'cir', 'ns'],
                          [[126, 3], 'aec', 'ns'], [[127, 3], 'si', 'ns'], [[128, 3], 'pi_si', 'ns'],
                          [[129, 3], 'ee_ss', 'ns'],
                          [[131, 3], 'cub', 'ns'], [[132, 3], 'sup_cu', 'ns'], [[133, 3], 'supt_cu', 'ns'],
                          [[134, 3], 'pisc', 'ns'], [[136, 3], 'sup_pisc', 'ns'], [[137, 3], 'pp_sup_escom', 'ns'],
                          [[138, 3], 'pp_et_escom', 'ns'], [[139, 3], 'ss_sup_CES', 'ns'],
                          [[140, 3], 'ss_sup_ter', 'ns'],
                          [[141, 3], 'nba', 'ns'], [[142, 3], 'min_max_asc', 'ns'], [[143, 3], 'tap', 'ns'],
                          [[37, 3], 'tum', 's'], [[46, 3], 'tuf', 's']]

    for value in tab_of_intrant_pos:

        if value[2] == 's':
            table_of_intrant = ajouter_caraterisque_par_secteur(sh, table_of_intrant, value[1], value[0], False)
        elif value[2] == 'ns':
            table_of_intrant = ajouter_caraterisque_par_secteur(sh, table_of_intrant, value[1], value[0], True)
        else:
            x = 0

    entete = ['sector', 'category', 'value'] + __BATIMENT__
    table_of_intrant = pd.DataFrame(table_of_intrant, columns=entete)

    # TODO : Add the pptu given the scenarios

    # Add number of unity by unity type
    t = ajouter_caraterisque_par_type_unite(sh, [], 'pptu', [100, 3], False)
    t = pd.DataFrame(t, columns=entete)
    ntu = table_of_intrant[table_of_intrant['value'] == 'ntu'][__BATIMENT__ + ['sector', 'value']]
    result = t.groupby('sector').apply(convert_unity_type_to_sector, ntu).reset_index(drop=True)
    result = result[entete]
    table_of_intrant = pd.concat([table_of_intrant, result], ignore_index=True)

    # Add unity Superficie

    # Get surface for small, medium and bug unity
    dict_of_surface = dict()
    for type in range(len(__UNITE_TYPE__)):
        d = dict()
        line = 146 + type
        if type > 4:
            line += 2
        for col in range(3):
            d[sh.cell(145, col + 3).value] = sh.cell(line, col + 3).value
        dict_of_surface[__UNITE_TYPE__[type]] = d

    tum = table_of_intrant[table_of_intrant['value'] == 'tum']
    t = pd.DataFrame(__UNITE_TYPE__, columns=['category'])
    result = t.groupby('category').apply(split_unity_type_to_sector,
                                         tum[__BATIMENT__ + ['sector', 'value']]).reset_index()
    result = result[entete]
    result = result.groupby('category').apply(get_surface, dict_of_surface).reset_index(drop=True)
    result = result[entete]
    table_of_intrant = table_of_intrant[table_of_intrant['value'] != 'tum']
    table_of_intrant = pd.concat([table_of_intrant, result], ignore_index=True)

    # Total Surface by unity type
    tum = table_of_intrant[(table_of_intrant['value'] == 'tum') | (table_of_intrant['value'] == 'ntu') & (
            table_of_intrant['category'] != 'ALL')]
    result = tum.groupby('category').apply(calculate_total_surface).reset_index(drop=True)
    result = result[entete]
    table_of_intrant = pd.concat([table_of_intrant, result], ignore_index=True)

    # Total Surface
    result = result[__BATIMENT__].groupby(result['sector']).sum().reset_index()
    result['category'] = 'ALL'
    result['value'] = 'suptu'
    result = result[entete]

    # Total Brute Surface
    cir = table_of_intrant[(table_of_intrant['value'] == 'cir') & (table_of_intrant['category'] == 'ALL')][
        __BATIMENT__].reset_index(drop=True)
    supbut = result[__BATIMENT__] / (1 - cir)
    supbut['category'] = 'ALL'
    supbut['value'] = 'supbtu'
    supbut['sector'] = result['sector']
    supbut = supbut[entete]

    ntu = table_of_intrant[(table_of_intrant['value'] == 'ntu') & (table_of_intrant['category'] == 'ALL')]
    nmu_et = table_of_intrant[(table_of_intrant['value'] == 'nmu_et') & (table_of_intrant['category'] == 'ALL')]
    pp_et_escom = table_of_intrant[
        (table_of_intrant['value'] == 'pp_et_escom') & (table_of_intrant['category'] == 'ALL')]

    # Brute Surface per units
    sup_bru_u = supbut[__BATIMENT__] / ntu[__BATIMENT__].reset_index(drop=True)
    sup_bru_u['category'] = 'ALL'
    sup_bru_u['value'] = 'sup_bru_par_u'
    sup_bru_u['sector'] = supbut['sector']
    sup_bru_u = sup_bru_u[entete]

    # Brute surface of 1 floor
    nmu_par_etage = sup_bru_u[__BATIMENT__] * nmu_et[__BATIMENT__].reset_index(drop=True)
    nmu_par_etage['category'] = 'ALL'
    nmu_par_etage['value'] = 'sup_bru_one_floor'
    nmu_par_etage['sector'] = supbut['sector']
    nmu_par_etage = nmu_par_etage[entete]

    # Commerce Surface
    sup_com = nmu_par_etage[__BATIMENT__] * pp_et_escom[__BATIMENT__].reset_index(drop=True)
    sup_com['category'] = 'ALL'
    sup_com['value'] = 'sup_com'
    sup_com['sector'] = supbut['sector']
    sup_com = sup_com[entete]

    # Calculate Brute Surface for common area
    supt_cu = table_of_intrant[(table_of_intrant['value'] == 'supt_cu') & (table_of_intrant['category'] == 'ALL')][
        __BATIMENT__].reset_index(drop=True)
    supt_cu = supt_cu / (1 - cir)
    supt_cu['category'] = 'ALL'
    supt_cu['value'] = 'supbt_cu'
    supt_cu['sector'] = supbut['sector']
    supt_cu = supt_cu[entete]

    # Total surface HS
    sup_tot_hs = supt_cu[__BATIMENT__].reset_index(drop=True) + sup_com[__BATIMENT__].reset_index(drop=True) + \
                 supbut[__BATIMENT__].reset_index(drop=True)

    sup_tot_hs['category'] = 'ALL'
    sup_tot_hs['value'] = 'sup_tot_hs'
    sup_tot_hs['sector'] = supbut['sector']
    sup_tot_hs = sup_tot_hs[entete]
    table_of_intrant = pd.concat([table_of_intrant, result, supbut, supt_cu, sup_bru_u,
                                  nmu_par_etage, sup_com, sup_tot_hs],
                                 ignore_index=True)

    # Proportion in term of total surface
    suptu = table_of_intrant[(table_of_intrant['value'] == 'suptu') & (table_of_intrant['category'] != 'ALL')]
    result = suptu[__BATIMENT__].astype(float).groupby(suptu['category']).mean()
    suptu = table_of_intrant[(table_of_intrant['value'] == 'suptu') & (table_of_intrant['category'] == 'ALL')]
    suptu = suptu[__BATIMENT__].mean().tolist()
    result = result.div(suptu, axis='columns')
    result['category'] = 'ALL'
    result['value'] = 'ppts'
    result['sector'] = __UNITE_TYPE__
    result = result[entete]
    table_of_intrant = pd.concat([table_of_intrant, result],
                                 ignore_index=True)
    # Land surface
    denm_p = table_of_intrant[(table_of_intrant['value'] == 'denm_p') & (table_of_intrant['category'] == 'ALL')]
    result = sup_tot_hs[__BATIMENT__].reset_index(drop=True) / denm_p[__BATIMENT__].reset_index(drop=True)
    result['category'] = 'ALL'
    result['value'] = 'land_surface_p'
    result['sector'] = supbut['sector']
    result = result[entete]
    table_of_intrant = pd.concat([table_of_intrant, result],
                                 ignore_index=True)

    # CES
    result = nmu_par_etage[__BATIMENT__] / result[__BATIMENT__]
    result['category'] = 'ALL'
    result['value'] = 'ces'
    result['sector'] = supbut['sector']
    result = result[entete]
    table_of_intrant = pd.concat([table_of_intrant, result],
                                 ignore_index=True)

    return table_of_intrant


def get_cb3_characteristics(workbook) -> pd.DataFrame:
    """
        This function takes all the parameters of the Intrants sheets and calculate all the characteristics
        of the CB31 sheet.

        :param workbook: Excel Workbook containing all the parameters
        :return: pd.Dataframe of all the characteristics of the CB1 sheet.

        """""

    # Open Intrants sheet and take all the importants parameters
    sh = workbook.sheet_by_name(__INTRANT_SHEET__)
    table_of_intrant = []

    # Table containing the position of all the intrants in the Intrants sheets. Refers to the lexique files
    # to get the definition of the variables.
    tab_of_intrant_pos = [[[28, 3], 'land_surface_p', 's'], [[55, 3], 'den_max_bat', 's'],
                          [[64, 3], 'den_max_per', 's'], [[91, 3], 'ces', 's'],
                          [[118, 3], 'mp', 'ns'],
                          [[119, 3], 'min_nu', 'ns'], [[120, 3], 'max_nu', 'ns'], [[121, 3], 'min_ne', 'ns'],
                          [[122, 3], 'max_ne', 'ns'], [[123, 3], 'min_ne_ss', 'ns'], [[124, 3], 'max_ne_ss', 'ns'],
                          [[125, 3], 'cir', 'ns'],
                          [[126, 3], 'aec', 'ns'], [[127, 3], 'si', 'ns'], [[128, 3], 'pi_si', 'ns'],
                          [[129, 3], 'ee_ss', 'ns'],
                          [[131, 3], 'cub', 'ns'], [[132, 3], 'sup_cu', 'ns'], [[133, 3], 'supt_cu', 'ns'],
                          [[134, 3], 'pisc', 'ns'], [[136, 3], 'sup_pisc', 'ns'], [[137, 3], 'pp_sup_escom', 'ns'],
                          [[138, 3], 'pp_et_escom', 'ns'], [[139, 3], 'ss_sup_CES', 'ns'],
                          [[140, 3], 'ss_sup_ter', 'ns'],
                          [[141, 3], 'nba', 'ns'], [[142, 3], 'min_max_asc', 'ns'], [[143, 3], 'tap', 'ns'],
                          [[37, 3], 'tum', 's'], [[46, 3], 'tuf', 's']]

    for value in tab_of_intrant_pos:

        if value[2] == 's':
            table_of_intrant = ajouter_caraterisque_par_secteur(sh, table_of_intrant, value[1], value[0], False)
        elif value[2] == 'ns':
            table_of_intrant = ajouter_caraterisque_par_secteur(sh, table_of_intrant, value[1], value[0], True)
        else:
            x = 0

    entete = ['sector', 'category', 'value'] + __BATIMENT__
    table_of_intrant = pd.DataFrame(table_of_intrant, columns=entete)

    land_surface_p = table_of_intrant[
        (table_of_intrant['value'] == 'land_surface_p') & (table_of_intrant['category'] == 'ALL')]
    den_max_per = table_of_intrant[
        (table_of_intrant['value'] == 'den_max_per') & (table_of_intrant['category'] == 'ALL')]

    # Total surface HS
    result = land_surface_p[__BATIMENT__].reset_index(drop=True) * den_max_per[__BATIMENT__].reset_index(drop=True)
    result['category'] = 'ALL'
    result['value'] = 'sup_tot_hs'
    result['sector'] = land_surface_p['sector']
    result = result[entete]
    table_of_intrant = pd.concat([table_of_intrant, result],
                                 ignore_index=True)

    # Brute surface of 1 floor
    ces = table_of_intrant[(table_of_intrant['value'] == 'ces') & (table_of_intrant['category'] == 'ALL')]
    result = land_surface_p[__BATIMENT__].reset_index(drop=True) * ces[__BATIMENT__].reset_index(drop=True)
    result['category'] = 'ALL'
    result['value'] = 'sup_bru_one_floor'
    result['sector'] = land_surface_p['sector']
    result = result[entete]
    table_of_intrant = pd.concat([table_of_intrant, result],
                                 ignore_index=True)
    # Total Brute Surface
    cir = table_of_intrant[(table_of_intrant['value'] == 'cir') & (table_of_intrant['category'] == 'ALL')][
        __BATIMENT__].reset_index(drop=True)

    # --> Surface Brute common surface
    supt_cu = table_of_intrant[(table_of_intrant['value'] == 'supt_cu') & (table_of_intrant['category'] == 'ALL')][
        __BATIMENT__].reset_index(drop=True)
    supt_cu = supt_cu / (1 - cir)

    # --> Surface commerce
    pp_et_escom = table_of_intrant[
        (table_of_intrant['value'] == 'pp_et_escom') & (table_of_intrant['category'] == 'ALL')]
    sup_com = result[__BATIMENT__] * pp_et_escom[__BATIMENT__].reset_index(drop=True)

    sup_tot_hs = table_of_intrant[(table_of_intrant['value'] == 'sup_tot_hs') & (table_of_intrant['category'] == 'ALL')]
    result = sup_tot_hs[__BATIMENT__].reset_index(drop=True) - sup_com - supt_cu
    result['category'] = 'ALL'
    result['value'] = 'supbtu'
    result['sector'] = land_surface_p['sector']
    result = result[entete]
    table_of_intrant = pd.concat([table_of_intrant, result],
                                 ignore_index=True)

    # Number of Floor
    pptu = ajouter_caraterisque_par_type_unite(sh, [], 'pptu', [100, 3], False)
    pptu = pd.DataFrame(pptu, columns=entete)

    # Get surface for small, medium and bug unity
    dict_of_surface = dict()
    for type in range(len(__UNITE_TYPE__)):
        d = dict()
        line = 146 + type
        if type > 4:
            line += 2
        for col in range(3):
            d[sh.cell(145, col + 3).value] = sh.cell(line, col + 3).value
        dict_of_surface[__UNITE_TYPE__[type]] = d

    # --> Surface brute per unit
    tum = table_of_intrant[table_of_intrant['value'] == 'tum']
    t = pd.DataFrame(__UNITE_TYPE__, columns=['category'])
    result = t.groupby('category').apply(split_unity_type_to_sector,
                                         tum[__BATIMENT__ + ['sector', 'value']]).reset_index()
    result = result[entete]
    result = result.groupby('category').apply(get_surface, dict_of_surface).reset_index(drop=True)
    result = result[entete]
    table_of_intrant = table_of_intrant[table_of_intrant['value'] != 'tum']
    table_of_intrant = pd.concat([table_of_intrant, result], ignore_index=True)

    result = result.groupby('category').apply(get_mean_brute_surface, pptu).reset_index(drop=True)
    # print(result)
    result = (result[__BATIMENT__].groupby(result['sector']).sum() / (1 - cir.values)).reset_index()

    result['category'] = 'ALL'
    result['value'] = 'sup_bru_par_u'
    result = result[entete]
    table_of_intrant = pd.concat([table_of_intrant, result],
                                 ignore_index=True)

    # --> Floor Hs
    supbtu = table_of_intrant[(table_of_intrant['value'] == 'supbtu') & (table_of_intrant['category'] == 'ALL')]
    sup_bru_one_floor = table_of_intrant[
        (table_of_intrant['value'] == 'sup_bru_one_floor') & (table_of_intrant['category'] == 'ALL')]
    floor_hs = supbtu[__BATIMENT__].reset_index(drop=True) / sup_bru_one_floor[__BATIMENT__].reset_index(drop=True)

    # --> Floor Commercial
    floor_com = sup_com.reset_index() / sup_bru_one_floor[__BATIMENT__].reset_index()

    # --> Floor chalet
    chalet_floor = supt_cu.reset_index() * (1 - cir) / sup_bru_one_floor[__BATIMENT__].reset_index()

    floor = floor_hs + floor_com + chalet_floor
    # table_of_intrant = table_of_intrant[table_of_intrant['value'] != 'tum']
    # table_of_intrant = pd.concat([table_of_intrant, result], ignore_index=True)

    # Mean number units per floor
    sup_bru_par_u = table_of_intrant[
        (table_of_intrant['value'] == 'sup_bru_par_u') & (table_of_intrant['category'] == 'ALL')]
    nmu_et = sup_bru_one_floor[__BATIMENT__].reset_index(drop=True) / sup_bru_par_u[__BATIMENT__].reset_index(drop=True)
    nmu_et['category'] = 'ALL'
    nmu_et['value'] = 'nmu_et'
    nmu_et['sector'] = sup_bru_par_u['sector'].reset_index(drop=True)
    nmu_et = nmu_et[entete]
    table_of_intrant = pd.concat([table_of_intrant, nmu_et],
                                 ignore_index=True)

    # Number of units
    ntu = nmu_et[__BATIMENT__].reset_index(drop=True) * floor_hs
    ntu['category'] = 'ALL'
    ntu['value'] = 'ntu'
    ntu['sector'] = sup_bru_par_u['sector'].reset_index(drop=True)
    ntu = ntu[entete]
    table_of_intrant = pd.concat([table_of_intrant, result],
                                 ignore_index=True)

    # Proportion in term of total surface
    # --> Add number of unity by unity type
    t = ajouter_caraterisque_par_type_unite(sh, [], 'pptu', [100, 3], False)
    t = pd.DataFrame(t, columns=entete)
    ntu = ntu[__BATIMENT__ + ['sector', 'value']]
    result = t.groupby('sector').apply(convert_unity_type_to_sector, ntu).reset_index(drop=True)
    result = result[entete]
    table_of_intrant = pd.concat([table_of_intrant, result], ignore_index=True)

    # --> Total Surface by unity type
    tum = table_of_intrant[(table_of_intrant['value'] == 'tum') | (table_of_intrant['value'] == 'ntu') & (
            table_of_intrant['category'] != 'ALL')]
    result = tum.groupby('category').apply(calculate_total_surface).reset_index(drop=True)
    result = result[entete]
    table_of_intrant = pd.concat([table_of_intrant, result], ignore_index=True)

    # --> Total Surface
    result = result[__BATIMENT__].groupby(result['sector']).sum().reset_index()
    result['category'] = 'ALL'
    result['value'] = 'suptu'
    result = result[entete]
    table_of_intrant = pd.concat([table_of_intrant, result],
                                 ignore_index=True)

    suptu = table_of_intrant[(table_of_intrant['value'] == 'suptu') & (table_of_intrant['category'] != 'ALL')]
    result = suptu[__BATIMENT__].astype(float).groupby(suptu['category']).mean()
    suptu = table_of_intrant[(table_of_intrant['value'] == 'suptu') & (table_of_intrant['category'] == 'ALL')]
    suptu = suptu[__BATIMENT__].mean().tolist()

    result = result.div(suptu, axis='columns')
    result['category'] = 'ALL'
    result['value'] = 'ppts'
    result['sector'] = ['Secteur ' + str(i) for i in range(1, 8)]
    result = result[entete]
    print(result)
    table_of_intrant = pd.concat([table_of_intrant, result],
                                 ignore_index=True)


if __name__ == '__main__':
    myBook = xlrd.open_workbook(__FILES_NAME__)
    get_cb3_characteristics(myBook)
