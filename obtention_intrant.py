import numpy as np
import pandas as pd
import xlrd

from lexique import __INTRANT_SHEET__, __PRICE_SHEET__, \
    __BATIMENT__, __SECTEUR__, __UNITE_TYPE__, __SCENARIO_SHEET__, __FILES_NAME__, __COUT_SHEET__, __QUALITE_BATIMENT__, \
    __ECOULEMENT_SHEET__, __FINANCE_PARAM_SHEET__


def ajouter_caraterisque_par_secteur(sh, tab, name, pos, category, unique):

    for secteur in __SECTEUR__:
        _ = [secteur, category, name]
        line = pos[0] if unique else __SECTEUR__.index(secteur) + pos[0]
        for batiment in range(len(__BATIMENT__)):
            value = sh.cell(line, pos[1] + batiment).value
            value = 0 if value == '' else value
            _.append(value)
        tab.append(_)
    return tab


def ajouter_caraterisque_par_type_unite(sh, tab, name, pos, unique):
    for unite in __UNITE_TYPE__:
        _ = ['ALL', unite, name]
        line = pos[0] if unique else __UNITE_TYPE__.index(unite) + pos[0]
        for batiment in range(len(__BATIMENT__)):
            value = sh.cell(line, pos[1] + batiment).value
            value = 0 if value == "" else value
            _.append(value)
        tab.append(_)
    return tab


def convert_unity_type_to_sum_sector(group, data):

    df = data[__BATIMENT__].mul(group[__BATIMENT__].values.tolist()[0], axis=1)
    return df.sum()


def convert_unity_type_to_sector(group, data, batim):
    sector = group.name
    group = group.copy()
    index = group.index
    v = data[data['sector'] == sector]
    group.loc[:, batim] = group[batim].reset_index(drop=True).mul(v[batim].values[0]).set_index(index)
    group.loc[:, 'value'] = 'ntu'

    return group


def split_unity_type_to_sector(group, data):
    return data


def calculate_price(group):
    t = group[group['value'] == 'ntu'][__BATIMENT__].reset_index(drop=True) * group[group['value'] == 'price'][
        __BATIMENT__].reset_index(drop=True)
    t['sector'] = group[group['value'] == 'ntu'].reset_index()['sector']
    return t


def get_surface(group, dict_of_surface):
    tab = [group[batiment].map(dict_of_surface[group.name]).values.tolist() for batiment in __BATIMENT__]
    tab = np.array(tab).transpose()
    tab = pd.DataFrame(tab, columns=__BATIMENT__)
    tab['category'] = group.name
    group = group.reset_index()
    tab['sector'] = group['sector']
    tab['value'] = group['value']

    return tab


def calculate_total_surface(group, batiment):
    group = group.copy()
    group.loc[group['value'] == 'tum', batiment] = group[batiment].prod().values
    group.loc[group['value'] == 'tum', 'value'] = 'suptu'

    return group[group['value'] == 'suptu']


def calculate_total_unit(group, batiment):
    df = group[group['value'] == 'suptu'][batiment].reset_index(drop=True) / \
         group[group['value'] == 'tum'][batiment].reset_index(drop=True)
    return df


def get_mean_brute_surface(group, data):
    df = group[__BATIMENT__].reset_index(drop=True).mul(data[data['sector'] == group.name][__BATIMENT__].values[0],
                                                        axis='columns')
    df['sector'] = group['sector'].reset_index(drop=True)
    df['value'] = 'sup_bru_par_u'
    df['category'] = group.name

    return df


def get_all_informations(workbook) -> pd.DataFrame:

    """
    this function is used to import all the useful variables to compute caracteristic, cost and finance.
    :param workbook: Excel File containing all the information
    :return: Data frame of all the inputs
    """
    ###################################################################################################################
    #
    # Open Intrants sheet and take all the important parameters
    #
    ###################################################################################################################

    sh = workbook.sheet_by_name(__INTRANT_SHEET__)
    table_of_intrant = []

    # Table containing the position of all the intrants in the Intrants sheets. Refers to the lexique files
    # to get the definition of the variables.
    tab_of_intrant_pos = [[[10, 3], 'ntu', 's'], [[19, 3], 'nmu_et', 's'], [[46, 3], 'vat', 's'],
                          [[55, 3], 'denm_p', 's'], [[82, 3], 'mp', 'ns'],
                          [[83, 3], 'min_nu', 'ns'], [[84, 3], 'max_nu', 'ns'], [[85, 3], 'min_ne', 'ns'],
                          [[86, 3], 'max_ne', 'ns'], [[87, 3], 'min_ne_ss', 'ns'], [[88, 3], 'max_ne_ss', 'ns'],
                          [[89, 3], 'cir', 'ns'],
                          [[90, 3], 'aec', 'ns'], [[91, 3], 'si', 'ns'], [[92, 3], 'pi_si', 'ns'],
                          [[93, 3], 'ee_ss', 'ns'], [[94, 3], 'pi_ee', 'ns'],
                          [[95, 3], 'cub', 'ns'], [[96, 3], 'sup_cu', 'ns'], [[97, 3], 'supt_cu', 'ns'],
                          [[98, 3], 'pisc', 'ns'], [[99, 3], 'sup_pisc', 'ns'], [[101, 3], 'pp_sup_escom', 'ns'],
                          [[102, 3], 'pp_et_escom', 'ns'], [[103, 3], 'ss_sup_CES', 'ns'],
                          [[104, 3], 'ss_sup_ter', 'ns'],
                          [[105, 3], 'nba', 'ns'], [[106, 3], 'min_max_asc', 'ns'], [[107, 3], 'tap', 'ns'],]

    # Get intrant parameters
    for value in tab_of_intrant_pos:
        if value[2] == 's':
            table_of_intrant = ajouter_caraterisque_par_secteur(sh, table_of_intrant, value[1], value[0], 'ALL', False)
        elif value[2] == 'ns':
            table_of_intrant = ajouter_caraterisque_par_secteur(sh, table_of_intrant, value[1], value[0], 'ALL', True)
        else:
            x = 0

    # Define Dataframe for the data
    entete = ['sector', 'category', 'value'] + __BATIMENT__
    table_of_intrant = pd.DataFrame(table_of_intrant, columns=entete)

    # Replace taille unite de marche par les superficie
    dict_of_surface = dict()
    for type in range(len(__UNITE_TYPE__)):
        d = dict()
        line = 110 + type
        if type > 4:
            line += 2
        for col in range(3):
            d[sh.cell(109, col + 3).value] = sh.cell(line, col + 3).value
        dict_of_surface[__UNITE_TYPE__[type]] = d

    for units in __UNITE_TYPE__[0: 5]:
        t = ajouter_caraterisque_par_secteur(sh, [], 'tum', [28, 3], units, False)
        t = pd.DataFrame(t, columns=entete)
        t.replace(dict_of_surface[units], inplace=True)
        table_of_intrant = pd.concat([table_of_intrant, t])

    for units in __UNITE_TYPE__[5:]:
        t = ajouter_caraterisque_par_secteur(sh, [], 'tum', [37, 3], units, False)
        t = pd.DataFrame(t, columns=entete)
        t.replace(dict_of_surface[units], inplace=True)
        table_of_intrant = pd.concat([table_of_intrant, t])

    # Ajout proportion en terme unite et proportion en terme de surface

    t = ajouter_caraterisque_par_type_unite(sh, [], 'pptu', [64, 3], False)
    t = pd.DataFrame(t, columns=entete)
    for secteur in __SECTEUR__:
        t.loc[:, 'sector'] = secteur
        table_of_intrant = pd.concat([table_of_intrant, t])

    t = ajouter_caraterisque_par_type_unite(sh, [], 'ppts', [73, 3], False)
    t = pd.DataFrame(t, columns=entete)
    for secteur in __SECTEUR__:
        t.loc[:, 'sector'] = secteur
        table_of_intrant = pd.concat([table_of_intrant, t])

    table_of_intrant['type'] = 'intrants'


    ####################################################################################################################
    #
    # Open Scenarios sheet and take all the important parameters
    #
    ###################################################################################################################

    sh = workbook.sheet_by_name(__SCENARIO_SHEET__)

    # Contribution sociale
    f7 = sh.cell(6, 6).value
    c27 = sh.cell(29, 2).value
    c30 = sh.cell(32, 2).value

    if f7 == c30 and f7 == c27:
        v = [45, 32]
    elif f7 != c30 and f7 == c27:
        v = [45, 33]
    elif f7 == c30 and f7 != c27:
        v = [55, 32]
    else:
        v = [55, 33]

    result = []

    for line in range(len(__SECTEUR__)):
        prop = sh.cell(v[1], 3).value
        _ = [__SECTEUR__[line], 'ALL', 'cont_soc']
        for col in range(len(__BATIMENT__)):
            _.append(float(sh.cell(v[0] + line, 4).value) * float(prop))
        result.append(_)

    entete = ['sector', 'category', 'value'] + __BATIMENT__
    result = pd.DataFrame(result, columns=entete)
    result['type'] = 'scenarios'
    table_of_intrant = pd.concat([table_of_intrant, result])

    # Frais de parc
    fp_exig = sh.cell(65, 2).value
    result = np.ones((7, 8))
    result = pd.DataFrame(result, columns=__BATIMENT__)
    result['category'] = 'ALL'
    result['value'] = 'parc'
    result['sector'] = __SECTEUR__
    result = result[entete]
    result['type'] = 'scenarios'
    result.replace({1: fp_exig}, inplace=True)
    table_of_intrant = pd.concat([table_of_intrant, result], ignore_index=True)

    # REM
    REM = sh.cell(69, 2).value
    result = np.ones((7, 8))
    result = pd.DataFrame(result, columns=__BATIMENT__)
    result['category'] = 'ALL'
    result['value'] = 'rem'
    result['sector'] = __SECTEUR__
    result = result[entete]
    result['type'] = 'scenarios'
    result.replace({1: REM}, inplace=True)
    table_of_intrant = pd.concat([table_of_intrant, result], ignore_index=True)

    # Decontamination
    decont = sh.cell(73, 2).value
    result = -1 * decont * np.ones((7, 8))
    result = pd.DataFrame(result, columns=__BATIMENT__)
    result['category'] = 'ALL'
    result['value'] = 'decont'
    result['sector'] = __SECTEUR__
    result = result[entete]
    result['type'] = 'scenarios'
    table_of_intrant = pd.concat([table_of_intrant, result], ignore_index=True)


    tab_of_intrant_pos = [[[83, 2], 'qum'], [[92, 3], 'quf']]
    t = []
    for value in tab_of_intrant_pos:
        t = ajouter_caraterisque_par_secteur(sh, t, value[1], value[0], 'ALL', False)
    qum = pd.DataFrame(t, columns=entete)


    table_of_intrant = get_cb1_characteristic(__SECTEUR__, __BATIMENT__, table_of_intrant)


    ###################################################################################################################
    #
    # Open price sheet and take all the important parameters
    #
    ###################################################################################################################

    sh = workbook.sheet_by_name(__PRICE_SHEET__)

    t = []

    for line in range(len(__UNITE_TYPE__)):
        pos = line * 9 + 4
        t = ajouter_caraterisque_par_secteur(sh, t, 'price', [pos, 2], __UNITE_TYPE__[line], False)

    entete = ['sector', 'category', 'value'] + __BATIMENT__
    t = pd.DataFrame(t, columns=entete)
    t['type'] = 'price'
    t = t[table_of_intrant.columns]
    table_of_intrant = pd.concat([table_of_intrant, t])

    tab_of_intrant_pos = [[[68, 2], 'stat']]
    t = []
    for value in tab_of_intrant_pos:
        t = ajouter_caraterisque_par_secteur(sh, t, value[1], value[0], 'ALL', False)
    t = pd.DataFrame(t, columns=entete)
    t['type'] = 'financial'
    t = t[table_of_intrant.columns]
    table_of_intrant = pd.concat([table_of_intrant, t])

    ###################################################################################################################
    #
    # Open costs sheet and take all the important parameters
    #
    ###################################################################################################################

    sh = workbook.sheet_by_name(__COUT_SHEET__)

    tab_cost = []

    tab_params = [
        #Construction
        [[6, 4], 'tcq'], [[7, 4], 'tss'], [[8, 4], 'tfu'], [[9, 4], 'all_cuis'], [[10, 4], 'all_sdb'],
        [[11, 4], 'tvfac'], [[12, 4], 'asc'], [[13, 4], 'c_ad_pisc'], [[14, 4], 'c_ad_cu'], [[15, 4], 'c_ad_com'],
        [[16, 4], 'it']
        #Soft cost
        , [[19, 4], 'apt_geo'], [[20, 4], 'prof'], [[21, 4], 'eval'], [[22, 4], 'legal_fee'], [[23, 4], 'prof_fee_div'],
        [[24, 4], 'pub'], [[25, 4], 'construction_permit'], [[26, 4], 'com'], [[27, 4], 'hon_prom']]


    for value in tab_params:
        tab_cost = ajouter_caraterisque_par_secteur(sh, tab_cost, value[1], value[0], 'ALL', True)

    entete = ['sector', 'category', 'value'] + __BATIMENT__
    t = pd.DataFrame(tab_cost, columns=entete)
    t['type'] = 'pcost'
    t = t[table_of_intrant.columns]
    table_of_intrant = pd.concat([table_of_intrant, t])

    d_ = dict()
    for line in range(len(__QUALITE_BATIMENT__)):
        d_[__QUALITE_BATIMENT__[line]] = sh.cell(35 + line, 4).value
    qum.replace(d_, inplace=True)

    qum['type'] = 'pcost'
    qum = qum[table_of_intrant.columns]
    table_of_intrant = pd.concat([table_of_intrant, qum])

    ###################################################################################################################
    #
    # Open price sheet and take all the important parameters
    #
    ###################################################################################################################

    sh = workbook.sheet_by_name(__ECOULEMENT_SHEET__)
    tab_of_intrant_pos = [[[4, 2], 'ecob3mo'], [[13, 2], 'ecob3mo'], [[22, 2], 'ecob3mo'], [[31, 2], 'ecob3mo'],
                          [[40, 2], 'ecob3mo'], [[49, 2], 'ecob3mo'], [[58, 2], 'ecob3mo'], ]
    t = []
    for pos in range(len(tab_of_intrant_pos)):
        value = tab_of_intrant_pos[pos]
        t = ajouter_caraterisque_par_secteur(sh, t, value[1], value[0], __UNITE_TYPE__[pos], False)
        value[0][1] = value[0][1] + 10
        t = ajouter_caraterisque_par_secteur(sh, t, 'ecoa3mo', value[0], __UNITE_TYPE__[pos], False)

    t = pd.DataFrame(t, columns=entete)
    t['type'] = 'financial'
    t = t[table_of_intrant.columns]
    table_of_intrant = pd.concat([table_of_intrant, t])

    sh = workbook.sheet_by_name(__FINANCE_PARAM_SHEET__)

    tab_of_fin_pos = [[[21, 2], 'dm_1'], [[22, 2], 'dm_prev'], [[16, 2], 'nv_min_prev_av_deb'], [[24, 2], 'dur_moy_const'],
                      [[6, 2], 'eq_terr'], [[7, 2], 'pp_prev'], [[13, 2], 'interet_terrain']]

    t = []
    # Get intrant parameters
    for value in tab_of_fin_pos:
        t = ajouter_caraterisque_par_secteur(sh, t, value[1], value[0], 'ALL', True)


    entete = ['sector', 'category', 'value'] + __BATIMENT__
    t = pd.DataFrame(t, columns=entete)
    t['type'] = 'financial'
    t = t[table_of_intrant.columns]
    table_of_intrant = pd.concat([table_of_intrant, t])


    return table_of_intrant.reset_index(drop=True)


def get_cb1_characteristic(secteur, batiment, table_of_intrant):
    ###################################################################################################################
    #
    # Filter variables for the computations
    #
    ###################################################################################################################

    entete = ['type', 'sector', 'category', 'value'] + batiment

    # input_variable = ['ntu', 'nmu_et', 'sup_ter', 'tum', 'vat', 'denm_p', 'pptu', 'ppts', 'mp', 'min_nu', 'max_nu',
    #                   'min_ne', 'max_ne',
    #                   'min_ne_ss', 'max_ne_ss', 'cir', 'aec', 'si', 'pi_si', 'ee_ss', 'pi_ee', 'cub', 'sup_cu',
    #                   'supt_cu', 'pisc', 'sup_pisc', 'pp_sup_escom', 'pp_et_escom', 'ss_sup_CES', 'ss_sup_ter', 'nba',
    #                   'min_max_asc', 'tap', 'price', 'cont_soc', 'parc', 'decont']
    #
    # table_of_intrant = table_of_intrant[(table_of_intrant['value'].isin(input_variable)) &
    #                                     (table_of_intrant['sector'].isin(secteur))][entete]

    table_of_intrant = table_of_intrant[entete]

    ###################################################################################################################
    #
    # Important variables
    #
    ###################################################################################################################

    cir = table_of_intrant[table_of_intrant['value'] == 'cir']
    aec = table_of_intrant[table_of_intrant['value'] == 'aec']
    si = table_of_intrant[table_of_intrant['value'] == 'si']
    pi_si = table_of_intrant[table_of_intrant['value'] == 'pi_si']
    ee_ss = table_of_intrant[table_of_intrant['value'] == 'ee_ss']
    pi_ee = table_of_intrant[table_of_intrant['value'] == 'pi_ee']
    nmu_et = table_of_intrant[table_of_intrant['value'] == 'nmu_et']
    pp_et_escom = table_of_intrant[table_of_intrant['value'] == 'pp_et_escom']
    denm_p = table_of_intrant[table_of_intrant['value'] == 'denm_p']
    parc = table_of_intrant[table_of_intrant['value'] == 'parc']
    parc.replace({'Oui': 1, "Non": 0}, inplace=True)

    ###################################################################################################################
    #
    # nombre total unite, superficies.
    #
    ###################################################################################################################

    # nombre total unite par type unite

    ntu = table_of_intrant[table_of_intrant['value'] == 'ntu']
    pptu = table_of_intrant[table_of_intrant['value'] == 'pptu']
    result = pptu.groupby(pptu['sector']).apply(convert_unity_type_to_sector, ntu, batiment).reset_index(drop=True)
    table_of_intrant = pd.concat([table_of_intrant, result[entete]])

    # add superfice unite

    x = table_of_intrant[(table_of_intrant['value'] == 'ntu') |
                         (table_of_intrant['value'] == 'tum') &
                         (table_of_intrant['category'] != 'ALL')]

    result = x.groupby(['category', 'sector']).apply(calculate_total_surface, batiment).reset_index(drop=True)

    table_of_intrant = pd.concat([table_of_intrant, result[entete]])

    x = result.groupby('sector')[batiment].sum()
    x['type'] = 'intrants'
    x['category'] = 'ALL'
    x['value'] = 'suptu'
    x.reset_index(inplace=True)
    x = x[entete]
    table_of_intrant = pd.concat([table_of_intrant, x])

    # superficie brute unites

    supbtu = x[batiment].reset_index() / (1 - cir[batiment].astype(float).reset_index())
    supbtu['category'] = 'ALL'
    supbtu['value'] = 'supbtu'
    supbtu['sector'] = x['sector']
    supbtu['type'] = 'intrants'
    result = supbtu[entete]
    table_of_intrant = pd.concat([table_of_intrant, result])

    # Brute Surface per units
    sup_bru_u = result[batiment] / ntu[batiment].reset_index()
    sup_bru_u['category'] = 'ALL'
    sup_bru_u['value'] = 'sup_bru_par_u'
    sup_bru_u['sector'] = result['sector']
    sup_bru_u['type'] = 'intrants'
    result = sup_bru_u[entete]
    table_of_intrant = pd.concat([table_of_intrant, result])

    # Brute surface of 1 floor
    sup_bru_one_floor = sup_bru_u[batiment] * nmu_et[batiment].reset_index()
    sup_bru_one_floor['category'] = 'ALL'
    sup_bru_one_floor['value'] = 'sup_bru_one_floor'
    sup_bru_one_floor['sector'] = sup_bru_u['sector']
    sup_bru_one_floor['type'] = 'intrants'
    result = sup_bru_one_floor[entete]
    table_of_intrant = pd.concat([table_of_intrant, result])

    # Commerce Surface
    sup_com = result[batiment] * pp_et_escom[batiment].reset_index()
    sup_com['category'] = 'ALL'
    sup_com['value'] = 'sup_com'
    sup_com['sector'] = result['sector']
    sup_com['type'] = 'intrants'
    result = sup_com[entete]
    table_of_intrant = pd.concat([table_of_intrant, result])

    # Calculate Brute Surface for common area
    supt_cu = table_of_intrant[table_of_intrant['value'] == 'supt_cu'][batiment].reset_index()
    supt_cu = supt_cu / (1 - cir[batiment].reset_index())
    supt_cu['category'] = 'ALL'
    supt_cu['value'] = 'supbt_cu'
    supt_cu['sector'] = result['sector']
    supt_cu['type'] = 'intrants'
    result = supt_cu[entete]
    table_of_intrant = pd.concat([table_of_intrant, result])

    # Total surface HS
    sup_tot_hs = supt_cu[batiment] + sup_com[batiment] + supbtu[batiment]
    sup_tot_hs['category'] = 'ALL'
    sup_tot_hs['value'] = 'sup_tot_hs'
    sup_tot_hs['sector'] = result['sector']
    sup_tot_hs['type'] = 'intrants'
    result = sup_tot_hs[entete]
    table_of_intrant = pd.concat([table_of_intrant, result])

    # Land surface
    sup_ter = sup_tot_hs[batiment] / denm_p[batiment].reset_index()
    sup_ter['category'] = 'ALL'
    sup_ter['value'] = 'sup_ter'
    sup_ter['sector'] = sup_tot_hs['sector']
    sup_ter['type'] = 'intrants'
    result = sup_ter[entete]
    table_of_intrant = pd.concat([table_of_intrant, result])

    # CES
    ces = sup_bru_one_floor[batiment] / sup_ter[batiment]
    ces['category'] = 'ALL'
    ces['value'] = 'ces'
    ces['sector'] = sup_ter['sector']
    ces['type'] = 'intrants'
    result = ces[entete]
    table_of_intrant = pd.concat([table_of_intrant, result])

    # superfice sous sol
    x = (ntu[batiment].reset_index() * (ee_ss[batiment].reset_index() * pi_ee[batiment].reset_index()
                                        + si[batiment].reset_index() * pi_si[batiment].reset_index())) \
        / (1 - cir[batiment].reset_index())

    sup_ss = (x + sup_tot_hs[batiment].reset_index()) / (1 - aec[batiment].reset_index()) - sup_tot_hs[
        batiment].reset_index()
    sup_ss['category'] = 'ALL'
    sup_ss['value'] = 'sup_ss'
    sup_ss['sector'] = sup_ter['sector']
    sup_ss['type'] = 'intrants'
    result = sup_ss[entete]
    table_of_intrant = pd.concat([table_of_intrant, result])

    # Superficie parc
    sup = table_of_intrant[
        (table_of_intrant['value'] == 'suptu') & (table_of_intrant['category'].isin(__UNITE_TYPE__[0:3]))]
    sup = sup[batiment].groupby(sup['sector']).sum().reset_index(drop=True)
    v = table_of_intrant[(table_of_intrant['value'] == 'ntu') & (table_of_intrant['category'] == 'ALL')][
        batiment].reset_index(drop=True)
    v.where(ntu > 2, 0, inplace=True)
    v.where(ntu == 0, 1, inplace=True)
    result = sup * (1 + cir[batiment].reset_index(drop=True)) * v
    result['category'] = 'ALL'
    result['value'] = 'sup_parc'
    result['sector'] = secteur
    result['type'] = 'intrants'
    result = result[entete]
    table_of_intrant = pd.concat([table_of_intrant, result], ignore_index=True)

    return table_of_intrant


def get_ca_characteristic(secteur, batiment, table_of_intrant):
    ###################################################################################################################
    #
    # Filter variables for the computations
    #
    ###################################################################################################################

    entete = ['sector', 'category', 'value'] + batiment + ['type']

    input_variable = ['ntu', 'nmu_et', 'sup_ter', 'tum', 'vat', 'denm_p', 'ces', 'pptu', 'mp', 'min_nu', 'max_nu',
                      'min_ne', 'max_ne',
                      'min_ne_ss', 'max_ne_ss', 'cir', 'aec', 'si', 'pi_si', 'ee_ss', 'pi_ee', 'cub', 'sup_cu',
                      'supt_cu', 'pisc', 'sup_pisc', 'pp_sup_escom', 'pp_et_escom', 'ss_sup_CES', 'ss_sup_ter', 'nba',
                      'min_max_asc', 'tap', 'price', 'cont_soc', 'parc', 'decont', 'rem']

    table_of_intrant = table_of_intrant[(table_of_intrant['value'].isin(input_variable)) &
                                        (table_of_intrant['sector'].isin(secteur))][entete]

    ###################################################################################################################
    #
    # Important variables
    #
    ###################################################################################################################

    cir = table_of_intrant[table_of_intrant['value'] == 'cir']
    aec = table_of_intrant[table_of_intrant['value'] == 'aec']
    si = table_of_intrant[table_of_intrant['value'] == 'si']
    pi_si = table_of_intrant[table_of_intrant['value'] == 'pi_si']
    ee_ss = table_of_intrant[table_of_intrant['value'] == 'ee_ss']
    pi_ee = table_of_intrant[table_of_intrant['value'] == 'pi_ee']
    nmu_et = table_of_intrant[table_of_intrant['value'] == 'nmu_et']
    pp_et_escom = table_of_intrant[table_of_intrant['value'] == 'pp_et_escom']
    denm_p = table_of_intrant[table_of_intrant['value'] == 'denm_p']
    parc = table_of_intrant[table_of_intrant['value'] == 'parc']
    sup_ter = table_of_intrant[table_of_intrant['value'] == 'sup_ter']
    parc.replace({'Oui': 1, "Non": 0}, inplace=True)

    ###################################################################################################################
    #
    # nombre total unite, superficies.
    #
    ###################################################################################################################

    # nombre total unite par type unite
    mask = (table_of_intrant['value'] == 'ntu') & (table_of_intrant['category'] != 'ALL')
    table_of_intrant.loc[mask, batiment] = table_of_intrant[mask][batiment].astype(float).round(0).astype(int)

    t = table_of_intrant[mask]
    ntu = t.groupby('sector')[batiment].sum().reset_index()
    mask = (table_of_intrant['value'] == 'ntu') & (table_of_intrant['category'] == 'ALL')
    table_of_intrant.loc[mask, ['sector'] + batiment] = ntu.values

    x = table_of_intrant[(table_of_intrant['value'] == 'ntu') |
                         (table_of_intrant['value'] == 'tum') &
                         (table_of_intrant['category'] != 'ALL')]

    result = x.groupby(['category', 'sector']).apply(calculate_total_surface, batiment).reset_index(drop=True)

    table_of_intrant = pd.concat([table_of_intrant, result[entete]])
    x = result.groupby('sector')[batiment].sum()
    x['type'] = 'intrants'
    x['category'] = 'ALL'
    x['value'] = 'suptu'
    x.reset_index(inplace=True)
    x = x[entete]
    table_of_intrant = pd.concat([table_of_intrant, x])

    # superficie brute unites

    supbtu = x[batiment].reset_index() / (1 - cir[batiment].astype(float).reset_index())
    supbtu['category'] = 'ALL'
    supbtu['value'] = 'supbtu'
    supbtu['sector'] = secteur
    supbtu['type'] = 'intrants'
    result = supbtu[entete]
    table_of_intrant = pd.concat([table_of_intrant, result])

    # Brute Surface per units
    sup_bru_u = result[batiment] / ntu[batiment].reset_index()
    sup_bru_u['category'] = 'ALL'
    sup_bru_u['value'] = 'sup_bru_par_u'
    sup_bru_u['sector'] = secteur
    sup_bru_u['type'] = 'intrants'
    result = sup_bru_u[entete]
    table_of_intrant = pd.concat([table_of_intrant, result])

    # Brute surface of 1 floor
    sup_bru_one_floor = sup_bru_u[batiment] * nmu_et[batiment].reset_index()
    sup_bru_one_floor['category'] = 'ALL'
    sup_bru_one_floor['value'] = 'sup_bru_one_floor'
    sup_bru_one_floor['sector'] = secteur
    sup_bru_one_floor['type'] = 'intrants'
    result = sup_bru_one_floor[entete]
    table_of_intrant = pd.concat([table_of_intrant, result])

    # Commerce Surface
    sup_com = result[batiment] * pp_et_escom[batiment].reset_index()
    sup_com['category'] = 'ALL'
    sup_com['value'] = 'sup_com'
    sup_com['sector'] = secteur
    sup_com['type'] = 'intrants'
    result = sup_com[entete]
    table_of_intrant = pd.concat([table_of_intrant, result])

    # Calculate Brute Surface for common area
    supt_cu = table_of_intrant[table_of_intrant['value'] == 'supt_cu'][batiment].reset_index()
    supt_cu = supt_cu / (1 - cir[batiment].reset_index())
    supt_cu['category'] = 'ALL'
    supt_cu['value'] = 'supbt_cu'
    supt_cu['sector'] = secteur
    supt_cu['type'] = 'intrants'
    result = supt_cu[entete]
    table_of_intrant = pd.concat([table_of_intrant, result])

    # Total surface HS
    sup_tot_hs = supt_cu[batiment] + sup_com[batiment] + supbtu[batiment]
    sup_tot_hs['category'] = 'ALL'
    sup_tot_hs['value'] = 'sup_tot_hs'
    sup_tot_hs['sector'] = secteur
    sup_tot_hs['type'] = 'intrants'
    result = sup_tot_hs[entete]
    table_of_intrant = pd.concat([table_of_intrant, result])

    # Proportion in term of total surface
    suptu = table_of_intrant[(table_of_intrant['value'] == 'suptu') & (table_of_intrant['category'] != 'ALL')]
    result = suptu[batiment].astype(float).groupby(suptu['category']).mean()
    suptu = table_of_intrant[(table_of_intrant['value'] == 'suptu') & (table_of_intrant['category'] == 'ALL')]
    suptu = suptu[batiment].mean().tolist()
    result = result.div(suptu, axis='columns')
    result['value'] = 'ppts'
    result['type'] = 'intrants'
    result.reset_index(inplace=True)

    for sect in secteur:
        result['sector'] = sect
        result = result[entete]
        table_of_intrant = pd.concat([table_of_intrant, result])

    # superfice sous sol
    x = (ntu[batiment].reset_index() * (ee_ss[batiment].reset_index() * pi_ee[batiment].reset_index()
                                        + si[batiment].reset_index() * pi_si[batiment].reset_index())) \
        / (1 - cir[batiment].reset_index())

    sup_ss = (x + sup_tot_hs[batiment].reset_index()) / (1 - aec[batiment].reset_index()) - sup_tot_hs[
        batiment].reset_index()
    sup_ss['category'] = 'ALL'
    sup_ss['value'] = 'sup_ss'
    sup_ss['sector'] = secteur
    sup_ss['type'] = 'intrants'
    result = sup_ss[entete]
    table_of_intrant = pd.concat([table_of_intrant, result])

    # Superficie parc
    sup = table_of_intrant[
        (table_of_intrant['value'] == 'suptu') & (table_of_intrant['category'].isin(__UNITE_TYPE__[0:3]))]
    sup = sup[batiment].groupby(sup['sector']).sum().reset_index(drop=True)
    v = table_of_intrant[(table_of_intrant['value'] == 'ntu') & (table_of_intrant['category'] == 'ALL')][
        batiment].reset_index(drop=True)
    v = v.where(ntu > 2, 0)
    v = v.where(v == 0, 1)
    result = sup * (1 + cir[batiment].reset_index(drop=True)) * v

    result['category'] = 'ALL'
    result['value'] = 'sup_parc'
    result['sector'] = secteur
    result['type'] = 'intrants'
    result = result[entete]
    table_of_intrant = pd.concat([table_of_intrant, result], ignore_index=True)

    # nombre etage
    nbr_etage_hs = supbtu[batiment].reset_index(drop=True) / sup_ter[batiment].reset_index(drop=True)
    ces = table_of_intrant[table_of_intrant['value'] == 'ces']
    nbr_etage_hs = nbr_etage_hs / ces[batiment].reset_index(drop=True)
    nbr_etage_hs['category'] = 'ALL'
    nbr_etage_hs['value'] = 'nbr_etage_hs'
    nbr_etage_hs['sector'] = secteur
    nbr_etage_hs['type'] = 'intrants'
    result = nbr_etage_hs[entete]
    table_of_intrant = pd.concat([table_of_intrant, result], ignore_index=True)

    # densi
    dens = sup_tot_hs[batiment].reset_index(drop=True) / sup_ter[batiment].reset_index(drop=True)
    dens['category'] = 'ALL'
    dens['value'] = 'dens'
    dens['sector'] = secteur
    dens['type'] = 'intrants'
    result = dens[entete]
    table_of_intrant = pd.concat([table_of_intrant, result], ignore_index=True)

    return table_of_intrant


def get_cb3_characteristic(secteur, batiment, table_of_intrant, *args):
    ###################################################################################################################
    #
    # Filter variables for the computations
    #
    ###################################################################################################################

    entete = ['type', 'sector', 'category', 'value'] + batiment

    input_variable = ['sup_ter', 'tum', 'vat', 'denm_p', 'ces', 'pptu', 'mp', 'min_nu', 'max_nu',
                      'min_ne', 'max_ne', 'min_ne_ss', 'max_ne_ss', 'cir', 'aec', 'si', 'pi_si', 'ee_ss', 'pi_ee',
                      'cub', 'sup_cu', 'supt_cu', 'pisc', 'sup_pisc', 'pp_sup_escom', 'pp_et_escom', 'ss_sup_CES',
                      'ss_sup_ter', 'nba', 'min_max_asc', 'tap', 'price', 'cont_soc', 'parc', 'decont', 'rem']

    table_of_intrant = table_of_intrant[(table_of_intrant['value'].isin(input_variable)) &
                                        (table_of_intrant['sector'].isin(secteur))][entete]
    ###################################################################################################################
    #
    # Take input parameter for the computations.
    #
    ###################################################################################################################

    intrant_value = args[0]

    for value in intrant_value:
        t = np.ones((len(secteur), len(batiment)))
        v = np.array(intrant_value[value])
        input = np.multiply(t, v)
        table_of_intrant.loc[table_of_intrant['value'] == value, batiment] = input

    ###################################################################################################################
    #
    # Filter variables for the computations
    #
    ###################################################################################################################

    sup_ter = table_of_intrant[table_of_intrant['value'] == 'sup_ter']
    denm_p = table_of_intrant[table_of_intrant['value'] == 'denm_p']
    ces = table_of_intrant[table_of_intrant['value'] == 'ces']
    cir = table_of_intrant[table_of_intrant['value'] == 'cir']
    aec = table_of_intrant[table_of_intrant['value'] == 'aec']
    si = table_of_intrant[table_of_intrant['value'] == 'si']
    pi_si = table_of_intrant[table_of_intrant['value'] == 'pi_si']
    ee_ss = table_of_intrant[table_of_intrant['value'] == 'ee_ss']
    pi_ee = table_of_intrant[table_of_intrant['value'] == 'pi_ee']
    pp_et_escom = table_of_intrant[table_of_intrant['value'] == 'pp_et_escom']
    parc = table_of_intrant[table_of_intrant['value'] == 'parc']
    parc.replace({'Oui': 1, 'Non': 0}, inplace=True)

    ###################################################################################################################
    #
    # Total surface HS, Surface Brute 1 floor.
    #
    ###################################################################################################################

    # Total surface HS
    result = sup_ter[batiment].reset_index(drop=True).astype(float) * denm_p[batiment].reset_index(drop=True)
    result['category'] = 'ALL'
    result['value'] = 'sup_tot_hs'
    result['sector'] = secteur
    result['type'] = 'intrants'
    result = result[entete]
    table_of_intrant = pd.concat([table_of_intrant, result], ignore_index=True)

    # Brute surface of 1 floor
    result = sup_ter[batiment].reset_index(drop=True).astype(float) * ces[batiment].reset_index(drop=True)
    result['category'] = 'ALL'
    result['value'] = 'sup_bru_one_floor'
    result['sector'] = secteur
    result['type'] = 'intrants'
    result = result[entete]
    table_of_intrant = pd.concat([table_of_intrant, result],
                                 ignore_index=True)

    ###################################################################################################################
    #
    # Total Brute surface, commerce, chalet urbain, total units surface.
    #
    ###################################################################################################################

    # Surface Brute common surface
    supt_cu = table_of_intrant[table_of_intrant['value'] == 'supt_cu']
    supt_cu = supt_cu[batiment].reset_index(drop=True) / (1 - cir[batiment].reset_index(drop=True))

    # --> Surface commerce
    sup_com = result[batiment] * pp_et_escom[batiment].reset_index(drop=True)

    sup_tot_hs = table_of_intrant[table_of_intrant['value'] == 'sup_tot_hs']
    result = sup_tot_hs[batiment].reset_index(drop=True) - sup_com - supt_cu
    suptu = result * (1 - cir[batiment].reset_index(drop=True))

    result['category'] = 'ALL'
    result['value'] = 'supbtu'
    result['sector'] = secteur
    result['type'] = 'intrants'
    result = result[entete]

    sup_com['category'] = 'ALL'
    sup_com['value'] = 'sup_com'
    sup_com['sector'] = secteur
    sup_com['type'] = 'intrants'
    sup_com = sup_com[entete]

    suptu['category'] = 'ALL'
    suptu['value'] = 'suptu'
    suptu['sector'] = secteur
    suptu['type'] = 'intrants'
    suptu = suptu[entete]
    table_of_intrant = pd.concat([table_of_intrant, result, sup_com, suptu], ignore_index=True)

    # --> Surface brute per unit
    tum = table_of_intrant[table_of_intrant['value'] == 'tum'].sort_values(by=['sector', 'category'])
    pptu = table_of_intrant[table_of_intrant['value'] == 'pptu'].sort_values(by=['sector', 'category'])
    result = (pptu[batiment].reset_index(drop=True) * tum[batiment].reset_index(drop=True))
    result['sector'] = tum['sector'].reset_index(drop=True)
    result = result.groupby('sector').sum().reset_index(drop=True) / (1 - cir[batiment].reset_index(drop=True))
    result['category'] = 'ALL'
    result['value'] = 'sup_bru_par_u'
    result['sector'] = secteur
    result['type'] = 'intrants'
    result = result[entete]
    table_of_intrant = pd.concat([table_of_intrant, result], ignore_index=True)

    # --> Floor Hs
    supbtu = table_of_intrant[(table_of_intrant['value'] == 'supbtu')]
    sup_bru_one_floor = table_of_intrant[table_of_intrant['value'] == 'sup_bru_one_floor']
    floor_hs = supbtu[batiment].reset_index(drop=True) / sup_bru_one_floor[batiment].reset_index(drop=True)

    # --> Floor Commercial
    floor_com = sup_com[batiment].reset_index(drop=True) / sup_bru_one_floor[batiment].reset_index(drop=True)

    # --> Floor chalet
    chalet_floor = supt_cu[batiment].reset_index(drop=True) * (1 - cir[batiment].reset_index(drop=True)) / \
                   sup_bru_one_floor[batiment].reset_index(drop=True)

    floor = floor_hs + floor_com + chalet_floor

    # Mean number units per floor
    sup_bru_par_u = table_of_intrant[table_of_intrant['value'] == 'sup_bru_par_u']
    nmu_et = sup_bru_one_floor[batiment].reset_index(drop=True) / sup_bru_par_u[batiment].reset_index(drop=True)
    nmu_et['category'] = 'ALL'
    nmu_et['value'] = 'nmu_et'
    nmu_et['sector'] = secteur
    nmu_et['type'] = 'intrants'
    nmu_et = nmu_et[entete]
    table_of_intrant = pd.concat([table_of_intrant, nmu_et],
                                 ignore_index=True)

    # Number of units
    ntu = nmu_et[batiment].reset_index(drop=True) * floor_hs
    ntu = ntu.round(2)
    ntu['category'] = 'ALL'
    ntu['value'] = 'ntu'
    ntu['sector'] = secteur
    ntu['type'] = 'intrants'
    ntu = ntu[entete]
    table_of_intrant = pd.concat([table_of_intrant, ntu],
                                 ignore_index=True)

    # nombre total unite par type unite
    pptu = table_of_intrant[table_of_intrant['value'] == 'pptu']
    result = pptu.groupby(pptu['sector']).apply(convert_unity_type_to_sector, ntu, batiment).reset_index(drop=True)
    table_of_intrant = pd.concat([table_of_intrant, result[entete]])

    # add superfice unite
    x = table_of_intrant[(table_of_intrant['value'] == 'ntu') |
                         (table_of_intrant['value'] == 'tum') &
                         (table_of_intrant['category'] != 'ALL')]

    result = x.groupby(['category', 'sector']).apply(calculate_total_surface, batiment).reset_index(drop=True)

    table_of_intrant = pd.concat([table_of_intrant, result[entete]])

    # Proportion in term of total surface
    suptu = table_of_intrant[(table_of_intrant['value'] == 'suptu') & (table_of_intrant['category'] != 'ALL')]
    result = suptu[batiment].astype(float).groupby(suptu['category']).mean()
    suptu = table_of_intrant[(table_of_intrant['value'] == 'suptu') & (table_of_intrant['category'] == 'ALL')]
    suptu = suptu[batiment].mean().tolist()
    result = result.div(suptu, axis='columns')
    result['value'] = 'ppts'
    result['type'] = 'intrants'
    result.reset_index(inplace=True)

    for sect in secteur:
        result['sector'] = sect
        result = result[entete]
        table_of_intrant = pd.concat([table_of_intrant, result])

    # superfice sous sol
    x = (ntu[batiment].reset_index() * (ee_ss[batiment].reset_index() * pi_ee[batiment].reset_index()
                                        + si[batiment].reset_index() * pi_si[batiment].reset_index())) \
        / (1 - cir[batiment].reset_index())

    sup_ss = (x + sup_tot_hs[batiment].reset_index()) / (1 - aec[batiment].reset_index()) - sup_tot_hs[
        batiment].reset_index()
    sup_ss['category'] = 'ALL'
    sup_ss['value'] = 'sup_ss'
    sup_ss['sector'] = secteur
    sup_ss['type'] = 'intrants'
    result = sup_ss[entete]
    table_of_intrant = pd.concat([table_of_intrant, result])

    # Superficie parc
    sup = table_of_intrant[
        (table_of_intrant['value'] == 'suptu') & (table_of_intrant['category'].isin(__UNITE_TYPE__[0:3]))]
    sup = sup[batiment].groupby(sup['sector']).sum().reset_index(drop=True)
    v = table_of_intrant[(table_of_intrant['value'] == 'ntu') & (table_of_intrant['category'] == 'ALL')][
        batiment].reset_index(drop=True)
    v.where(ntu > 2, 0, inplace=True)
    v.where(ntu == 0, 1, inplace=True)
    result = sup * (1 + cir[batiment].reset_index(drop=True)) * v

    result['category'] = 'ALL'
    result['value'] = 'sup_parc'
    result['sector'] = secteur
    result['type'] = 'intrants'
    result = result[entete]
    table_of_intrant = pd.concat([table_of_intrant, result], ignore_index=True)

    return table_of_intrant


def get_ca3_characteristic(secteur, batiment, table_of_intrant):
    ###################################################################################################################
    #
    # Filter variables for the computations
    #
    ###################################################################################################################

    entete = ['type', 'sector', 'category', 'value'] + batiment

    input_variable = ['ntu', 'nmu_et', 'tum', 'vat', 'denm_p', 'ces', 'pptu', 'mp', 'min_nu', 'max_nu', 'min_ne',
                      'max_ne',
                      'min_ne_ss', 'max_ne_ss', 'cir', 'aec', 'si', 'pi_si', 'ee_ss', 'pi_ee', 'cub', 'sup_cu',
                      'supt_cu', 'pisc', 'sup_pisc', 'pp_sup_escom', 'pp_et_escom', 'ss_sup_CES', 'ss_sup_ter', 'nba',
                      'min_max_asc', 'tap', 'price', 'cont_soc', 'parc', 'decont']

    table_of_intrant = table_of_intrant[(table_of_intrant['value'].isin(input_variable)) &
                                        (table_of_intrant['sector'].isin(secteur))][entete]

    ###################################################################################################################
    #
    # Important variables
    #
    ###################################################################################################################

    cir = table_of_intrant[table_of_intrant['value'] == 'cir']
    aec = table_of_intrant[table_of_intrant['value'] == 'aec']
    si = table_of_intrant[table_of_intrant['value'] == 'si']
    pi_si = table_of_intrant[table_of_intrant['value'] == 'pi_si']
    ee_ss = table_of_intrant[table_of_intrant['value'] == 'ee_ss']
    pi_ee = table_of_intrant[table_of_intrant['value'] == 'pi_ee']
    nmu_et = table_of_intrant[table_of_intrant['value'] == 'nmu_et']
    pp_et_escom = table_of_intrant[table_of_intrant['value'] == 'pp_et_escom']
    denm_p = table_of_intrant[table_of_intrant['value'] == 'denm_p']
    parc = table_of_intrant[table_of_intrant['value'] == 'parc']
    parc.replace({'Oui': 1, "Non": 0}, inplace=True)

    ###################################################################################################################
    #
    # nombre total unite, superficies.
    #
    ###################################################################################################################

    # nombre total unite par type unite
    mask = (table_of_intrant['value'] == 'ntu') & (table_of_intrant['category'] != 'ALL')
    table_of_intrant.loc[mask, batiment] = table_of_intrant[mask][batiment].astype(float).round(0).astype(int)

    t = table_of_intrant[mask]
    ntu = t.groupby('sector')[batiment].sum().reset_index()
    mask = (table_of_intrant['value'] == 'ntu') & (table_of_intrant['category'] == 'ALL')
    table_of_intrant.loc[mask, ['sector'] + batiment] = ntu.values

    x = table_of_intrant[(table_of_intrant['value'] == 'ntu') |
                         (table_of_intrant['value'] == 'tum') &
                         (table_of_intrant['category'] != 'ALL')]

    result = x.groupby(['category', 'sector']).apply(calculate_total_surface, batiment).reset_index(drop=True)

    table_of_intrant = pd.concat([table_of_intrant, result[entete]])

    x = result.groupby('sector')[batiment].sum()
    x['type'] = 'intrants'
    x['category'] = 'ALL'
    x['value'] = 'suptu'
    x.reset_index(inplace=True)
    x = x[entete]
    table_of_intrant = pd.concat([table_of_intrant, x])

    # superficie brute unites

    supbtu = x[batiment].reset_index() / (1 - cir[batiment].astype(float).reset_index())
    supbtu['category'] = 'ALL'
    supbtu['value'] = 'supbtu'
    supbtu['sector'] = x['sector']
    supbtu['type'] = 'intrants'
    result = supbtu[entete]
    table_of_intrant = pd.concat([table_of_intrant, result])

    # Brute Surface per units
    sup_bru_u = result[batiment] / ntu[batiment].reset_index()
    sup_bru_u['category'] = 'ALL'
    sup_bru_u['value'] = 'sup_bru_par_u'
    sup_bru_u['sector'] = result['sector']
    sup_bru_u['type'] = 'intrants'
    result = sup_bru_u[entete]
    table_of_intrant = pd.concat([table_of_intrant, result])

    # Brute surface of 1 floor
    sup_bru_one_floor = sup_bru_u[batiment] * nmu_et[batiment].reset_index()
    sup_bru_one_floor['category'] = 'ALL'
    sup_bru_one_floor['value'] = 'sup_bru_one_floor'
    sup_bru_one_floor['sector'] = sup_bru_u['sector']
    sup_bru_one_floor['type'] = 'intrants'
    result = sup_bru_one_floor[entete]
    table_of_intrant = pd.concat([table_of_intrant, result])

    # Commerce Surface
    sup_com = result[batiment] * pp_et_escom[batiment].reset_index()
    sup_com['category'] = 'ALL'
    sup_com['value'] = 'sup_com'
    sup_com['sector'] = result['sector']
    sup_com['type'] = 'intrants'
    result = sup_com[entete]
    table_of_intrant = pd.concat([table_of_intrant, result])

    # Calculate Brute Surface for common area
    supt_cu = table_of_intrant[table_of_intrant['value'] == 'supt_cu'][batiment].reset_index()
    supt_cu = supt_cu / (1 - cir[batiment].reset_index())
    supt_cu['category'] = 'ALL'
    supt_cu['value'] = 'supbt_cu'
    supt_cu['sector'] = result['sector']
    supt_cu['type'] = 'intrants'
    result = supt_cu[entete]
    table_of_intrant = pd.concat([table_of_intrant, result])

    # Total surface HS
    sup_tot_hs = supt_cu[batiment] + sup_com[batiment] + supbtu[batiment]
    sup_tot_hs['category'] = 'ALL'
    sup_tot_hs['value'] = 'sup_tot_hs'
    sup_tot_hs['sector'] = result['sector']
    sup_tot_hs['type'] = 'intrants'
    result = sup_tot_hs[entete]
    table_of_intrant = pd.concat([table_of_intrant, result])

    # Proportion in term of total surface
    suptu = table_of_intrant[(table_of_intrant['value'] == 'suptu') & (table_of_intrant['category'] != 'ALL')]
    result = suptu[batiment].astype(float).groupby(suptu['category']).mean()
    suptu = table_of_intrant[(table_of_intrant['value'] == 'suptu') & (table_of_intrant['category'] == 'ALL')]
    suptu = suptu[batiment].mean().tolist()
    result = result.div(suptu, axis='columns')
    result['value'] = 'ppts'
    result['type'] = 'intrants'
    result.reset_index(inplace=True)

    for sect in secteur:
        result['sector'] = sect
        result = result[entete]
        table_of_intrant = pd.concat([table_of_intrant, result])

    # Land surface
    sup_ter = sup_tot_hs[batiment] / denm_p[batiment].reset_index()
    sup_ter['category'] = 'ALL'
    sup_ter['value'] = 'sup_ter'
    sup_ter['sector'] = sup_tot_hs['sector']
    sup_ter['type'] = 'intrants'
    result = sup_ter[entete]
    table_of_intrant = pd.concat([table_of_intrant, result])

    # superfice sous sol
    x = (ntu[batiment].reset_index() * (ee_ss[batiment].reset_index() * pi_ee[batiment].reset_index()
                                        + si[batiment].reset_index() * pi_si[batiment].reset_index())) \
        / (1 - cir[batiment].reset_index())

    sup_ss = (x + sup_tot_hs[batiment].reset_index()) / (1 - aec[batiment].reset_index()) - sup_tot_hs[
        batiment].reset_index()
    sup_ss['category'] = 'ALL'
    sup_ss['value'] = 'sup_ss'
    sup_ss['sector'] = sup_ter['sector']
    sup_ss['type'] = 'intrants'
    result = sup_ss[entete]
    table_of_intrant = pd.concat([table_of_intrant, result])

    # Superficie parc
    sup = table_of_intrant[
        (table_of_intrant['value'] == 'suptu') & (table_of_intrant['category'].isin(__UNITE_TYPE__[0:3]))]
    sup = sup[batiment].groupby(sup['sector']).sum().reset_index(drop=True)
    v = table_of_intrant[(table_of_intrant['value'] == 'ntu') & (table_of_intrant['category'] == 'ALL')][
        batiment].reset_index(drop=True)
    v.where(ntu > 2, 0, inplace=True)
    v.where(ntu == 0, 1, inplace=True)
    result = sup * (1 + cir[batiment].reset_index(drop=True)) * v

    result['category'] = 'ALL'
    result['value'] = 'sup_parc'
    result['sector'] = secteur
    result['type'] = 'intrants'
    result = result[entete]
    table_of_intrant = pd.concat([table_of_intrant, result], ignore_index=True)
    table_of_intrant.to_excel('ca3.xlsx')

    return table_of_intrant


def get_cb4_characteristic(secteur, batiment, table_of_intrant, *args):
    ###################################################################################################################
    #
    # Filter variables for the computations
    #
    ###################################################################################################################

    entete = ['type', 'sector', 'category', 'value'] + batiment

    input_variable = ['sup_ter', 'tum', 'vat', 'denm_p', 'ces', 'ppts', 'mp', 'min_nu', 'max_nu',
                      'min_ne', 'max_ne', 'min_ne_ss', 'max_ne_ss', 'cir', 'aec', 'si', 'pi_si', 'ee_ss', 'pi_ee',
                      'cub', 'sup_cu', 'supt_cu', 'pisc', 'sup_pisc', 'pp_sup_escom', 'pp_et_escom', 'ss_sup_CES',
                      'ss_sup_ter', 'nba', 'min_max_asc', 'tap', 'price', 'cont_soc', 'parc', 'decont', 'rem']

    table_of_intrant = table_of_intrant[(table_of_intrant['value'].isin(input_variable)) &
                                        (table_of_intrant['sector'].isin(secteur))][entete]
    ###################################################################################################################
    #
    # Take input parameter for the computations.
    #
    ###################################################################################################################

    intrant_value = args[0]

    for value in intrant_value:
        t = np.ones((len(secteur), len(batiment)))
        v = np.array(intrant_value[value])
        input = np.multiply(t, v)
        table_of_intrant.loc[table_of_intrant['value'] == value, batiment] = input

    ###################################################################################################################
    #
    # Filter variables for the computations
    #
    ###################################################################################################################

    sup_ter = table_of_intrant[table_of_intrant['value'] == 'sup_ter']
    denm_p = table_of_intrant[table_of_intrant['value'] == 'denm_p']
    ces = table_of_intrant[table_of_intrant['value'] == 'ces']
    cir = table_of_intrant[table_of_intrant['value'] == 'cir']
    aec = table_of_intrant[table_of_intrant['value'] == 'aec']
    si = table_of_intrant[table_of_intrant['value'] == 'si']
    pi_si = table_of_intrant[table_of_intrant['value'] == 'pi_si']
    ee_ss = table_of_intrant[table_of_intrant['value'] == 'ee_ss']
    pi_ee = table_of_intrant[table_of_intrant['value'] == 'pi_ee']
    nmu_et = table_of_intrant[table_of_intrant['value'] == 'nmu_et']
    pp_et_escom = table_of_intrant[table_of_intrant['value'] == 'pp_et_escom']
    parc = table_of_intrant[table_of_intrant['value'] == 'parc']
    parc.replace({'Oui': 1, 'Non': 0}, inplace=True)

    ###################################################################################################################
    #
    # Total surface HS, Surface Brute 1 floor.
    #
    ###################################################################################################################

    # Total surface HS
    result = sup_ter[batiment].reset_index(drop=True).astype(float) * denm_p[batiment].reset_index(drop=True)
    result['category'] = 'ALL'
    result['value'] = 'sup_tot_hs'
    result['sector'] = secteur
    result['type'] = 'intrants'
    result = result[entete]
    table_of_intrant = pd.concat([table_of_intrant, result], ignore_index=True)

    # Brute surface of 1 floor
    result = sup_ter[batiment].reset_index(drop=True).astype(float) * ces[batiment].reset_index(drop=True)
    result['category'] = 'ALL'
    result['value'] = 'sup_bru_one_floor'
    result['sector'] = secteur
    result['type'] = 'intrants'
    result = result[entete]
    table_of_intrant = pd.concat([table_of_intrant, result],
                                 ignore_index=True)

    ###################################################################################################################
    #
    # Total Brute surface, commerce, chalet urbain, total units surface.
    #
    ###################################################################################################################

    # Surface Brute common surface
    supt_cu = table_of_intrant[table_of_intrant['value'] == 'supt_cu']
    supt_cu = supt_cu[batiment].reset_index(drop=True) / (1 - cir[batiment].reset_index(drop=True))

    # --> Surface commerce
    sup_com = result[batiment] * pp_et_escom[batiment].reset_index(drop=True)

    sup_tot_hs = table_of_intrant[table_of_intrant['value'] == 'sup_tot_hs']
    result = sup_tot_hs[batiment].reset_index(drop=True) - sup_com - supt_cu
    suptu = result * (1 - cir[batiment].reset_index(drop=True))

    result['category'] = 'ALL'
    result['value'] = 'supbtu'
    result['sector'] = secteur
    result['type'] = 'intrants'
    result = result[entete]

    sup_com['category'] = 'ALL'
    sup_com['value'] = 'sup_com'
    sup_com['sector'] = secteur
    sup_com['type'] = 'intrants'
    sup_com = sup_com[entete]

    suptu['category'] = 'ALL'
    suptu['value'] = 'suptu'
    suptu['sector'] = secteur
    suptu['type'] = 'intrants'
    suptu = suptu[entete]
    table_of_intrant = pd.concat([table_of_intrant, result, sup_com, suptu], ignore_index=True)

    # Calcul des superfices totales des unites selon la typologie
    ppts = table_of_intrant[table_of_intrant['value'] == 'ppts']
    suptu = table_of_intrant[table_of_intrant['value'] == 'suptu']
    result = ppts.groupby('sector').apply(convert_unity_type_to_sector, suptu, batiment).reset_index(drop=True)
    result = result[entete]
    result.loc[:, 'value'] = 'suptu'
    table_of_intrant = pd.concat([table_of_intrant, result], ignore_index=True)

    # Nombre d'unite

    ntu = table_of_intrant[(table_of_intrant['value'] == 'tum') | (table_of_intrant['value'] == 'suptu') & (
            table_of_intrant['category'] != 'ALL')]
    result = ntu.groupby(['sector', 'category']).apply(calculate_total_unit, batiment).reset_index()
    result['value'] = 'ntu'
    result['type'] = 'intrants'
    result = result[entete]
    table_of_intrant = pd.concat([table_of_intrant, result], ignore_index=True)

    result = result[batiment].groupby(result['sector']).sum()
    result['category'] = 'ALL'
    result['value'] = 'ntu'
    result['type'] = 'intrants'
    result['sector'] = secteur
    result = result[entete]
    table_of_intrant = pd.concat([table_of_intrant, result], ignore_index=True)

    return table_of_intrant


def get_cb1_characteristics(workbook) -> pd.DataFrame:

    global fp_exig

    """
        This function takes all the parameters of the Intrants sheets and calculate all the characteristics
        of the CB1 sheet.
        :param workbook: Excel Workbook containing all the parameters
        To compute the results of CB1 we import from intrants sheets the following values:
        1- 10, nombre total d'unites: ntu;
        2- 19, nombre moyen d'unite par etage: nmu_et;
        3- 37, taille des unites de marche: tum;
        4- 46, taille des unites familiales: tuf;
        5- 55, valeur de terrain: vat;
        6- 64, densite maximale permise: denm_p;
        7- 100, proportion en terme de nombre d'unite: pptu;
        8- 126, circulation (hors sol et sous sol)-%: cir;
        9- 127, autres espaces communs: aec;
        10- 128, stationnement interieur: si
        11- 129, pied carre par stationnement interieur: pi_si;
        12- 130, espaces d'entreposage sous sol: ee_ss;
        13- 131, pied carre par espace entreposage: pi_ee;
        :return: pd.Dataframe of all the characteristics of the CB1 sheet.
        1- sup_ter: superficie de terrain;
        2- tum: taille des unites;
        3- tuf: taille des unites familiales;
        4- vat: valeur de terrain;
        5- denm_p: densite maximale permise;
        6- ces: coefficient d' emprise au sol;
        7- pptu: proprtion en terme unite;
        8- cir: circulation (hors sol et sous sol)-%;
        9- aec: autres espaces communs;
        10- si: stationnement interieur;
        11- pi_si: pied carre par stationnement interieur;
        12- ee_ss: espaces d'entreposage sous sol;
        13- pi_ee: pied carre par espace entreposage;
        14- min_ne: min nombre etages;
        15- max_ne: max nombre etages;
        16- suptu: superficie totale des unites-Type;
        17- supbtu: superficie brute unites (unite + circulation);
        18- sup_com: superficie commerce;
        19- sup_tot_hs: superficie totale hors sol;
        20- pisc: piscine (non incluse);
        21- sup_ss: superfice sous sol;
        22- ppts: Proportion en terme de superficie totale;
        23- ntu: nombre total d'unites-type;
        24- supt_cu: superficie chalet urbain
        25- pp_et_escom: proportion un etage espace commercial
        26- pptu: proportion en terme d'unite
        27- cub: chalet urbain presence
        28- price: price per type of units in each sector
        29- cont_soc: Part de contribution sociale
        30- sup_parc: superifcie parc
        31- decont: incitatif decontamination.
    """""

    # Open Intrants sheet and take all the importants parameters
    sh = workbook.sheet_by_name(__INTRANT_SHEET__)
    table_of_intrant = []

    # Table containing the position of all the intrants in the Intrants sheets. Refers to the lexique files
    # to get the definition of the variables.
    tab_of_intrant_pos = [[[10, 3], 'ntu', 's'], [[19, 3], 'nmu_et', 's'], [[55, 3], 'vat', 's'],
                          [[64, 3], 'denm_p', 's'], [[118, 3], 'mp', 'ns'],
                          [[119, 3], 'min_nu', 'ns'], [[120, 3], 'max_nu', 'ns'], [[121, 3], 'min_ne', 'ns'],
                          [[122, 3], 'max_ne', 'ns'], [[123, 3], 'min_ne_ss', 'ns'], [[124, 3], 'max_ne_ss', 'ns'],
                          [[125, 3], 'cir', 'ns'],
                          [[126, 3], 'aec', 'ns'], [[127, 3], 'si', 'ns'], [[128, 3], 'pi_si', 'ns'],
                          [[129, 3], 'ee_ss', 'ns'], [[130, 3], 'pi_ee', 'ns'],
                          [[131, 3], 'cub', 'ns'], [[132, 3], 'sup_cu', 'ns'], [[133, 3], 'supt_cu', 'ns'],
                          [[134, 3], 'pisc', 'ns'], [[136, 3], 'sup_pisc', 'ns'], [[137, 3], 'pp_sup_escom', 'ns'],
                          [[138, 3], 'pp_et_escom', 'ns'], [[139, 3], 'ss_sup_CES', 'ns'],
                          [[140, 3], 'ss_sup_ter', 'ns'],
                          [[141, 3], 'nba', 'ns'], [[142, 3], 'min_max_asc', 'ns'], [[143, 3], 'tap', 'ns'],
                          [[37, 3], 'tum', 's'], [[46, 3], 'tuf', 's']]

    # Get intrant parameters
    for value in tab_of_intrant_pos:
        if value[2] == 's':
            table_of_intrant = ajouter_caraterisque_par_secteur(sh, table_of_intrant, value[1], value[0], 'ALL', False)
        elif value[2] == 'ns':
            table_of_intrant = ajouter_caraterisque_par_secteur(sh, table_of_intrant, value[1], value[0], 'ALL', True)
        else:
            x = 0

    # Define Dataframe for the data
    entete = ['sector', 'category', 'value'] + __BATIMENT__
    table_of_intrant = pd.DataFrame(table_of_intrant, columns=entete)

    # TODO : Add the pptu given the scenarios

    # Add number of unity by unity type

    t = ajouter_caraterisque_par_type_unite(sh, [], 'pptu', [100, 3], False)
    t = pd.DataFrame(t, columns=entete)
    x = ajouter_caraterisque_par_type_unite(sh, [], 'ppts', [109, 3], False)
    x = pd.DataFrame(x, columns=entete)

    table_of_intrant = pd.concat([table_of_intrant, t, x], ignore_index=True)


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
    result['value'] = 'sup_ter'
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
    # superfice sous sol


    cir = table_of_intrant[(table_of_intrant['value'] == 'cir') & (table_of_intrant['category'] == 'ALL')][
        __BATIMENT__].reset_index(drop=True).astype(float)

    aec = table_of_intrant[(table_of_intrant['value'] == 'aec') & (table_of_intrant['category'] == 'ALL')][
        __BATIMENT__].reset_index(drop=True).astype(float)

    si = table_of_intrant[(table_of_intrant['value'] == 'si') & (table_of_intrant['category'] == 'ALL')][
        __BATIMENT__].reset_index(drop=True).astype(float)

    pi_si = table_of_intrant[(table_of_intrant['value'] == 'pi_si') & (table_of_intrant['category'] == 'ALL')][
        __BATIMENT__].reset_index(drop=True).astype(float)

    ee_ss = table_of_intrant[(table_of_intrant['value'] == 'ee_ss') & (table_of_intrant['category'] == 'ALL')][
        __BATIMENT__].reset_index(drop=True).astype(float)

    pi_ee = table_of_intrant[(table_of_intrant['value'] == 'pi_ee') & (table_of_intrant['category'] == 'ALL')][
        __BATIMENT__].reset_index(drop=True).astype(float)

    ntu = table_of_intrant[(table_of_intrant['value'] == 'ntu') & (table_of_intrant['category'] == 'ALL')][
        __BATIMENT__].reset_index(drop=True).astype(float)

    sup_tot_hs = table_of_intrant[(table_of_intrant['value'] == 'sup_tot_hs') & (table_of_intrant['category'] == 'ALL')][
        __BATIMENT__].reset_index(drop=True).astype(float)

    result = (ntu * (ee_ss * pi_ee + si * pi_si)/(1-cir) + sup_tot_hs)/(1-aec) - sup_tot_hs
    result['category'] = 'ALL'
    result['value'] = 'sup_ss'
    result['sector'] = __SECTEUR__
    result = result[entete]
    table_of_intrant = pd.concat([table_of_intrant, result],
                                 ignore_index=True)

    # Prix Maison
    sh = workbook.sheet_by_name(__PRICE_SHEET__)
    # Get intrant parameters
    x = []
    for pos in range(len(__UNITE_TYPE__)):
        x = ajouter_caraterisque_par_secteur(sh, x, 'price', [4 + pos * 9, 2], __UNITE_TYPE__[pos], False)

    x = pd.DataFrame(x, columns=entete)
    table_of_intrant = pd.concat([table_of_intrant, x],
                                 ignore_index=True)

    # Calcul total revenue
    tot = table_of_intrant[((table_of_intrant['value'] == 'ntu') | (table_of_intrant['value'] == 'price'))
                           & (table_of_intrant['category'] != 'ALL')]

    result = (tot.groupby(tot['category']).apply(calculate_price).reset_index(drop=True))
    result = result.groupby('sector').sum().reset_index()


    result['category'] = 'ALL'
    result['value'] = 'price'
    result = result[entete]
    table_of_intrant = pd.concat([table_of_intrant, result],
                                 ignore_index=True)

    # Contribution sociale
    sh = workbook.sheet_by_name(__SCENARIO_SHEET__)
    f7 = sh.cell(6, 5).value
    c27 = sh.cell(26, 2).value
    c30 = sh.cell(29, 2).value
    # =SI('Scénarios'!$C$27 = 'Scénarios'!$F$7;
    # SI('Scénarios'!$C$30 = 'Scénarios'!$F$7;
    # 'Scénarios'!$D43 * CFINAL!D377 * 'Scénarios'!$D$30;
    # 'Scénarios'!$D43 * CFINAL!D377 * 'Scénarios'!$D$31);
    #
    # SI('Scénarios'!$C$30 = 'Scénarios'!$F$7;
    # 'Scénarios'!$D53 * CFINAL!D377 * 'Scénarios'!$D$30;
    # 'Scénarios'!$D53 * CFINAL!D377 * 'Scénarios'!$D$31))
    if f7 == c30 and f7 == c27:
        v = [42, 29]
    elif f7 != c30 and f7 == c27:
        v = [42, 30]
    elif f7 == c30 and f7 != c27:
        v = [52, 29]
    else:
        v = [52, 30]
    result = []
    for line in range(len(__SECTEUR__)):
        prop = sh.cell(v[1], 3).value
        _ = [__SECTEUR__[line], 'ALL', 'cont_soc']
        for col in range(len(__BATIMENT__)):
            _.append(float(sh.cell(v[0] + line, 3).value) * float(prop))
        result.append(_)

    result = pd.DataFrame(result, columns=entete)

    table_of_intrant = pd.concat([table_of_intrant, result],
                                 ignore_index=True)
    # Frais de parc
    fp_exig = sh.cell(62, 2).value
    if fp_exig == 'Oui':
        sup = table_of_intrant[(table_of_intrant['value'] == 'suptu')
                               & (table_of_intrant['category'].isin(__UNITE_TYPE__[0:3]))]
        sup = sup[__BATIMENT__].groupby(sup['sector']).sum().reset_index(drop=True)
        cir = 1 + table_of_intrant[(table_of_intrant['value'] == 'cir')][__BATIMENT__].reset_index(drop=True)
        ntu = table_of_intrant[(table_of_intrant['value'] == 'ntu')
                               & (table_of_intrant['category'] == 'ALL')][__BATIMENT__].reset_index(drop=True)
        ntu.where(ntu > 2, 0, inplace=True)
        ntu.where(ntu == 0, 1, inplace=True)
        result = sup * cir * ntu

        result['category'] = 'ALL'
        result['value'] = 'sup_parc'
        result['sector'] = __SECTEUR__
        result = result[entete]
        table_of_intrant = pd.concat([table_of_intrant, result],
                                 ignore_index=True)
    else:
        result = np.zeros((7, 8))
        result = pd.DataFrame(result, columns=__BATIMENT__)
        result['category'] = 'ALL'
        result['value'] = 'sup_parc'
        result['sector'] = __SECTEUR__
        result = result[entete]
        table_of_intrant = pd.concat([table_of_intrant, result],
                                 ignore_index=True)

    # Decontamination
    decont = sh.cell(67, 2).value
    result = -1*decont * np.ones((7, 8))
    result = pd.DataFrame(result, columns=__BATIMENT__)
    result['category'] = 'ALL'
    result['value'] = 'decont'
    result['sector'] = __SECTEUR__
    result = result[entete]
    table_of_intrant = pd.concat([table_of_intrant, result],
                             ignore_index=True)

    # 1- sup_ter: superficie de terrain (will be removed if the information is provided);
    # 2- tum: taille des unites;
    # 3- tuf: taille des unites familiales;
    # 4- vat: valeur de terrain;
    # 5- denm_p: densite maximale permise;
    # 6- ces: coefficient d' emprise au sol;
    # 7- pptu: proprtion en terme unite;
    # 8- cir: circulation (hors sol et sous sol)-%;
    # 9- aec: autres espaces communs;
    # 10- si: stationnement interieur;
    # 11- pi_si: pied carre par stationnement interieur;
    # 12- ee_ss: espaces d'entreposage sous sol;
    # 13- pi_ee: pied carre par espace entreposage;
    # 14- min_ne: min nombre etages;
    # 15- max_ne: max nombre etages;
    # 16- suptu: superficie totale des unites-Type;
    # 17- supbtu: superficie brute unites (unite + circulation);
    # 18- sup_com: superficie commerce;
    # 19- sup_tot_hs: superficie totale hors sol;
    # 20- pisc: piscine (non incluse);
    # 21- sup_ss: superfice sous sol;
    # 22- ppts: Proportion en terme de superficie totale;
    # 23- ntu: nombre total d'unites-type;

    value_to_return = ['sup_ter', 'tum', 'tuf', 'vat', 'denm_p', 'ces', 'pptu', 'cir', 'aec', 'si', 'pi_si', 'ee_ss',
                       'pi_ee', 'min_ne', 'max_ne', 'suptu', 'supbtu', 'sup_com', 'sup_tot_hs', 'pisc', 'sup_ss', 'ppts',
                       'ntu', 'supt_cu', 'pp_et_escom', 'pptu', 'cub', 'price', 'cont_soc', 'sup_parc', 'decont']

    return table_of_intrant[table_of_intrant['value'].isin(value_to_return)]


def get_cb3_characteristics(table_of_intrant, *args) -> pd.DataFrame:

    """
        This function takes all the parameters of from the CB1 and calculate all the characteristics
        of the CB3 sheet.

        :param
        data: Dataframe containing all the intrants informations and the calculations of CB1.
        args: Parameter to specify the input varialble for computation, namely superficie de terrain, densite maximale
        permise, CES, nombre etage min and max. If the args are not provided the calculations would be made with the
        default values in the Intrants sheets.
        args = ['secteur', 'superficie terrain', 'densite, ces, min_ne, max_ne]
        The Data countains the following variables for Intrants:
        1- sup_ter: superficie de terrain (will be removed if the information is provided);
        2- tum: taille des unites;
        3- tuf: taille des unites familiales;
        4- vat: valeur de terrain;
        5- denm_p: densite maximale permise;
        6- ces: coefficient d' emprise au sol;
        7- pptu: proprtion en terme unite;
        8- cir: circulation (hors sol et sous sol)-%;
        9- aec: autres espaces communs;
        10- si: stationnement interieur;
        11- pi_si: pied carre par stationnement interieur;
        12- ee_ss: espaces d'entreposage sous sol;
        13- pi_ee: pied carre par espace entreposage;
        14- min_ne: min nombre etages;
        15- max_ne: max nombre etages;
        16- suptu: superficie totale des unites-Type;
        17- supbtu: superficie brute unites (unite + circulation);
        18- sup_com: superficie commerce;
        20- pisc: piscine (non incluse);
        21- sup_ss: superfice sous sol;
        22- ppts: Proportion en terme de superficie totale;
        supt_cu

        :return: pd.Dataframe of all the characteristics of the CB1 sheet.
    """""
    input_var = ['sector', 'sup_ter', 'denm_p', 'ces', 'min_ne', 'max_ne']

    entete = ['sector', 'category', 'value'] + __BATIMENT__

    secteur = args[0]
    sup_ter = args[1]
    denm_p = args[2]
    ces = args[3]
    min_ne = args[4]
    max_ne = args[5]
    pptu = table_of_intrant[table_of_intrant['value'] == 'pptu'].sort_values(by=['sector'])
    if secteur is not None:
        table_of_intrant = table_of_intrant[(table_of_intrant['sector'] == secteur) & (table_of_intrant['value'] != 'pptu')]
    drop_value = ['ntu', 'sup_tot_hs', 'supbtu', 'sup_com', 'suptu', 'sup_ss']

    for value in range(1, len(args)):
        if args[value] is not None:
            drop_value.append(input_var[value])

    table_of_intrant = table_of_intrant[table_of_intrant['value'].isin(drop_value) == False]

    if sup_ter is not None:
        v = pd.DataFrame([np.append([secteur, 'ALL', 'sup_ter'], sup_ter*np.ones(len(__BATIMENT__)))],
                         columns = table_of_intrant.columns)
        table_of_intrant = pd.concat([table_of_intrant, v], ignore_index=True)

    if denm_p is not None:
        v = pd.DataFrame([np.append([secteur, 'ALL', 'denm_p'], denm_p*np.ones(len(__BATIMENT__)))],
                         columns = table_of_intrant.columns)
        table_of_intrant = pd.concat([table_of_intrant, v], ignore_index=True)

    if ces is not None:
        v = pd.DataFrame([np.append([secteur, 'ALL', 'ces'], ces*np.ones(len(__BATIMENT__)))],
                         columns = table_of_intrant.columns)
        table_of_intrant = pd.concat([table_of_intrant, v], ignore_index=True)

    if min_ne is not None:
        v = pd.DataFrame([np.append([secteur, 'ALL', 'min_ne'], min_ne*np.ones(len(__BATIMENT__)))],
                         columns = table_of_intrant.columns)
        table_of_intrant = pd.concat([table_of_intrant, v], ignore_index=True)

    if max_ne is not None:
        v = pd.DataFrame([np.append([secteur, 'ALL', 'max_ne'], max_ne*np.ones(len(__BATIMENT__)))],
                         columns = table_of_intrant.columns)
        table_of_intrant = pd.concat([table_of_intrant, v], ignore_index=True)


    land_surface_p = table_of_intrant[
        (table_of_intrant['value'] == 'sup_ter') & (table_of_intrant['category'] == 'ALL')]
    den_max_per = table_of_intrant[
        (table_of_intrant['value'] == 'denm_p') & (table_of_intrant['category'] == 'ALL')]

    # Total surface HS
    result = land_surface_p[__BATIMENT__].reset_index(drop=True).astype(float) * den_max_per[__BATIMENT__].reset_index(drop=True)
    result['category'] = 'ALL'
    result['value'] = 'sup_tot_hs'
    result['sector'] = secteur
    result = result[entete]
    table_of_intrant = pd.concat([table_of_intrant, result], ignore_index=True)

    # Brute surface of 1 floor
    ces = table_of_intrant[(table_of_intrant['value'] == 'ces') & (table_of_intrant['category'] == 'ALL')]
    result = land_surface_p[__BATIMENT__].reset_index(drop=True).astype(float) * ces[__BATIMENT__].reset_index(
        drop=True)
    result['category'] = 'ALL'
    result['value'] = 'sup_bru_one_floor'
    result['sector'] = secteur
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
    suptu = result * (1 - cir)

    result['category'] = 'ALL'
    result['value'] = 'supbtu'
    result['sector'] = secteur
    result = result[entete]

    sup_com['category'] = 'ALL'
    sup_com['value'] = 'sup_com'
    sup_com['sector'] = secteur
    sup_com = sup_com[entete]

    suptu['category'] = 'ALL'
    suptu['value'] = 'suptu'
    suptu['sector'] = secteur
    suptu = suptu[entete]
    table_of_intrant = pd.concat([table_of_intrant, result, sup_com, suptu], ignore_index=True)


    # --> Surface brute per unit
    tum = table_of_intrant[table_of_intrant['value'] == 'tum'].sort_values(by=['category'])
    result = (pptu[__BATIMENT__].reset_index(drop=True) * tum[__BATIMENT__].reset_index(drop=True)).sum()/(1-cir)
    result['category'] = 'ALL'
    result['value'] = 'sup_bru_par_u'
    result['sector'] =secteur
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
    ntu = ntu.round(2)
    ntu['category'] = 'ALL'
    ntu['value'] = 'ntu'
    ntu['sector'] = sup_bru_par_u['sector'].reset_index(drop=True)
    ntu = ntu[entete]

    table_of_intrant = pd.concat([table_of_intrant, result],
                                 ignore_index=True)


    # Proportion in term of total surface
    # --> Add number of unity by unity type

    ntu = ntu[__BATIMENT__ + ['sector', 'value']]
    result = pptu.groupby('sector').apply(convert_unity_type_to_sector, ntu).reset_index(drop=True)
    result = result[entete]
    print(result)
    table_of_intrant = pd.concat([table_of_intrant, result], ignore_index=True)

    result = result[__BATIMENT__].groupby(result['sector']).sum().reset_index()
    result['category'] = 'ALL'
    result['value'] = 'ntu'
    result = result[entete]
    print(result)
    table_of_intrant = pd.concat([table_of_intrant, result],
                                 ignore_index=True)

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

    # superfice sous sol

    cir = table_of_intrant[(table_of_intrant['value'] == 'cir') & (table_of_intrant['category'] == 'ALL')][
        __BATIMENT__].reset_index(drop=True).astype(float)

    aec = table_of_intrant[(table_of_intrant['value'] == 'aec') & (table_of_intrant['category'] == 'ALL')][
        __BATIMENT__].reset_index(drop=True).astype(float)

    si = table_of_intrant[(table_of_intrant['value'] == 'si') & (table_of_intrant['category'] == 'ALL')][
        __BATIMENT__].reset_index(drop=True).astype(float)

    pi_si = table_of_intrant[(table_of_intrant['value'] == 'pi_si') & (table_of_intrant['category'] == 'ALL')][
        __BATIMENT__].reset_index(drop=True).astype(float)

    ee_ss = table_of_intrant[(table_of_intrant['value'] == 'ee_ss') & (table_of_intrant['category'] == 'ALL')][
        __BATIMENT__].reset_index(drop=True).astype(float)

    pi_ee = table_of_intrant[(table_of_intrant['value'] == 'pi_ee') & (table_of_intrant['category'] == 'ALL')][
        __BATIMENT__].reset_index(drop=True).astype(float)

    ntu = table_of_intrant[(table_of_intrant['value'] == 'ntu') & (table_of_intrant['category'] == 'ALL')][
        __BATIMENT__].reset_index(drop=True).astype(float)

    sup_tot_hs = \
        table_of_intrant[(table_of_intrant['value'] == 'sup_tot_hs') & (table_of_intrant['category'] == 'ALL')][
            __BATIMENT__].reset_index(drop=True).astype(float)

    result = (ntu * (ee_ss * pi_ee + si * pi_si) / (1 - cir) + sup_tot_hs) / (1 - aec) - sup_tot_hs
    result['category'] = 'ALL'
    result['value'] = 'sup_ss'
    result['sector'] = secteur
    result = result[entete]

    table_of_intrant = pd.concat([table_of_intrant, result], ignore_index=True)
    table_of_intrant.drop(table_of_intrant[(table_of_intrant['value'] == 'price') &
                                           (table_of_intrant['category'] == 'ALL')].index, inplace=True)
    table_of_intrant.drop(table_of_intrant[table_of_intrant['value'] == 'sup_parc'].index, inplace=True)

    # Price

    # Calcul total revenue
    tot = table_of_intrant[((table_of_intrant['value'] == 'ntu') | (table_of_intrant['value'] == 'price'))
                           & (table_of_intrant['category'] != 'ALL')]

    result = (tot.groupby(tot['category']).apply(calculate_price).reset_index(drop=True))
    result = result.groupby('sector').sum().reset_index()

    result['category'] = 'ALL'
    result['value'] = 'price'
    result = result[entete]
    table_of_intrant = pd.concat([table_of_intrant, result], ignore_index=True)

    if fp_exig == 'Oui':
        sup = table_of_intrant[(table_of_intrant['value'] == 'suptu')
                               & (table_of_intrant['category'].isin(__UNITE_TYPE__[0:3]))]
        sup = sup[__BATIMENT__].groupby(sup['sector']).sum().reset_index(drop=True)
        cir = 1 + table_of_intrant[(table_of_intrant['value'] == 'cir')][__BATIMENT__].reset_index(drop=True)
        ntu = table_of_intrant[(table_of_intrant['value'] == 'ntu')
                               & (table_of_intrant['category'] == 'ALL')][__BATIMENT__].reset_index(drop=True)
        ntu.where(ntu > 2, 0, inplace=True)
        ntu.where(ntu == 0, 1, inplace=True)
        result = sup * cir * ntu

        result['category'] = 'ALL'
        result['value'] = 'sup_parc'
        result['sector'] = secteur
        result = result[entete]
        table_of_intrant = pd.concat([table_of_intrant, result],
                                     ignore_index=True)
    else:
        result = np.zeros((7, 8))
        result = pd.DataFrame(result, columns=__BATIMENT__)
        result['category'] = 'ALL'
        result['value'] = 'sup_parc'
        result['sector'] = secteur
        result = result[entete]
        table_of_intrant = pd.concat([table_of_intrant, result],
                                     ignore_index=True)
    print(table_of_intrant[table_of_intrant['value'].isin(['sup_tot_hs', 'sup_ss', 'suptu', 'ntu', 'supbtu', 'cir',
                                                           'pisc', 'cub', 'sup_com', 'decont', 'sup_parc', 'cont_soc',
                                                           'vat', 'sup_ter', 'price'])][['category', 'value', 'B1']])
    return table_of_intrant[table_of_intrant['value'].isin(['sup_tot_hs', 'sup_ss', 'suptu', 'ntu', 'supbtu', 'cir',
                                                            'pisc', 'cub', 'sup_com', 'decont', 'sup_parc', 'cont_soc',
                                                            'vat', 'sup_ter', 'price'])]


def get_summary_characteristics(type, secteur, batiment, table_of_intrant, *args):

    entete = ['type', 'sector', 'category', 'value'] + batiment

    other_intrant = table_of_intrant[table_of_intrant['type'].isin(['pcost', 'financial'])]
    other_intrant = other_intrant[other_intrant['sector'].isin(secteur)][entete]


    if type == 'CB1':
        value =  table_of_intrant[table_of_intrant['sector'].isin(secteur)][entete]
    elif type == 'CA1':
        value = get_ca_characteristic(__SECTEUR__, __BATIMENT__, table_of_intrant)
    elif type == 'CB3':
        value =  get_cb3_characteristic(secteur, batiment, table_of_intrant, *args)
    elif type == 'CB4':
        value =  get_cb4_characteristic(secteur, batiment, table_of_intrant, *args)
    elif type == 'CA3':
        cb3 = get_cb3_characteristic(secteur, batiment, table_of_intrant, *args)
        value = get_ca_characteristic(secteur, batiment, cb3)
    elif type == 'CA4':
        cb4 = get_cb4_characteristic(secteur, batiment, table_of_intrant, *args)
        value = get_ca_characteristic(secteur, batiment, cb4)
    else:
        raise ('The input value must be: CB1, CB3, CB4, CA1, CA3 or CA4')

    return pd.concat([value[entete], other_intrant], ignore_index=True)

def get_intrants(type, secteur, batiment, table_of_intrant, *args):

    if type == 'CB1':
        value =  table_of_intrant[table_of_intrant['sector'].isin(secteur)]
    elif type == 'CA1':
        value = get_ca_characteristic(__SECTEUR__, __BATIMENT__, table_of_intrant)
    elif type == 'CB3':
        value =  get_cb3_characteristic(secteur, batiment, table_of_intrant, *args)
    elif type == 'CB4':
        value =  get_cb4_characteristic(secteur, batiment, table_of_intrant, *args)
    elif type == 'CA3':
        cb3 = get_cb3_characteristic(secteur, batiment, table_of_intrant, *args)
        value = get_ca_characteristic(secteur, batiment, cb3)
    elif type == 'CA4':
        cb4 = get_cb4_characteristic(secteur, batiment, table_of_intrant, *args)
        value = get_ca_characteristic(secteur, batiment, cb4)
    else:
        raise ('The input value must be: CB1, CB3, CB4, CA1, CA3 or CA4')

    return value


if __name__ == '__main__':
    myBook = xlrd.open_workbook(__FILES_NAME__)
    x = get_all_informations(myBook)

    args = dict()
    supter = [50000]
    densite = [10]
    t = get_summary_characteristics('CA3', __SECTEUR__, __BATIMENT__, x, args)
    t.to_excel('test.xlsx')
    # for ter in supter:
    #     for dens in densite:
    #         args = dict()
    #         v = np.ones((len(__SECTEUR__), 1))
    #         args['sup_ter'] = ter * v
    #         args['denm_p'] = dens * v
    #         result = get_summary_characteristics('CA3', __SECTEUR__, __BATIMENT__, x, args)
    #         result = result[
    #             (result['value'].isin(['sup_ter', 'ntu', 'dens', 'nbr_etage_hs'])) & (result['category'] == 'ALL')]
    # print(result)
    # result.to_csv('x.txt')
    # x = get_summary_characteristics('CA3', __SECTEUR__, __BATIMENT__, x, args)
    # x.to_excel('out.xlsx')

    # args['sup_ter'] = [[88827],[5000]]
    # args['denm_p'] = [[1.075], [10]]
