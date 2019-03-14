import math
import numpy as np
import pandas as pd
import xlrd

from lexique import __INTRANT_SHEET__, __PRICE_SHEET__, \
    __BATIMENT__, __SECTEUR__, __UNITE_TYPE__, __SCENARIO_SHEET__, __FILES_NAME__, __COUT_SHEET__, __QUALITE_BATIMENT__, \
    __ECOULEMENT_SHEET__, __FINANCE_PARAM_SHEET__, __PROP_SHEET__

#######################################################################################################################
#
# Important Function for the computations
#
#######################################################################################################################


def ajouter_caraterisque_par_secteur(sh, tab, name, pos, category, unique):
    """
    Cette fonctin est utilisee pour aller chercher les informations de chaque secteur et batiment pour des intrants
    :param sh: sheets ou se retrouve l'intrant
    :param tab:
    :param name:
    :param pos:
    :param category:
    :param unique:
    :return:
    """
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


def validate_batiment(data, ntu):

    sector = data.name
    group = data.copy()
    nu = ntu[ntu['sector'] == sector].fillna(-10)
    nu = nu.set_index('sector').transpose()
    nu = nu[nu[sector] <= 0].index

    group.loc[group['value'] != 'go_no_go', nu] = np.nan

    return group


def go_no_go(group, header):

    data = group.copy()

    # min_ne = data[data['value'] == 'min_ne'][header]
    # max_ne = data[data['value'] == 'max_ne'][header]
    # floor = data[data['value'] == 'floor'][header]
    # print('Floor')
    # print(min_ne)
    # print(max_ne)
    # print(floor)
    # floor = floor.where((floor >= min_ne.values) & (floor <= max_ne.values), 0)
    # floor = floor.where(floor == 0, 1)
    # print(floor)
    # print('')
    # min_ne = data[data['value'] == 'min_ne_ss'][header]
    # max_ne = data[data['value'] == 'max_ne_ss'][header]
    # floor_ss = data[data['value'] == 'floor_ss'][header]
    #
    # print('Floor SS')
    # print(min_ne)
    # print(max_ne)
    # print(floor_ss)
    # floor_ss = floor_ss.where(floor_ss != 0, max_ne.values)
    # floor_ss = floor_ss.where((floor_ss >= min_ne.values) & (floor_ss <= max_ne.values), 0)
    # floor_ss = floor_ss.where(floor_ss == 0, 1)
    # print(floor_ss)
    # print('')
    # dens = data[data['value'] == 'dens'][header]
    # denm_p = data[data['value'] == 'denm_t'][header]

    # print('Densite')
    # print(dens)
    # print(denm_p)

    # dens = dens.where(dens <= denm_p.values, 0)
    # dens = dens.where(dens == 0, 1)
    # print(dens)
    # print('')

    # print('CES')
    ces = data[data['value'] == 'ces'][header]
    # print(ces)

    ces = ces.where(ces > 0.8, 1)
    ces = ces.where(ces == 1, 0)
    # print(ces)
    # print('')

    go =  ces.reset_index(drop=True)
    ntu = data[data['value'] == 'ntu'][header]
    go = go.where(ntu.values > 0, 0)

    # go['total'] = go.sum(axis=1)
    # floor['total'] = floor.sum(axis=1)
    # print(floor[floor['total'] == 0]['total'].count())

    # print('GO')
    # print(go)
    # print('')
    return go

#######################################################################################################################
#
# Compute the CB1, CB3, CA3
#
#######################################################################################################################


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
                          [[105, 3], 'nba', 'ns'], [[106, 3], 'min_max_asc', 'ns'], [[107, 3], 'tap', 'ns'],
                          [[122, 3], 'denm_t', 's']]

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

    # TODO: Change for number units and sector
    sh = workbook.sheet_by_name(__PROP_SHEET__)

    # Comment this for 5 to  50 units
    t = ajouter_caraterisque_par_type_unite(sh, [], 'pptu', [3, 1], False)
    t = pd.DataFrame(t, columns=entete)
    for secteur in __SECTEUR__:
        t.loc[:, 'sector'] = secteur
        table_of_intrant = pd.concat([table_of_intrant, t])

    t = ajouter_caraterisque_par_type_unite(sh, [], 'ppts', [12, 1], False)
    t = pd.DataFrame(t, columns=entete)
    for secteur in __SECTEUR__:
        t.loc[:, 'sector'] = secteur
        table_of_intrant = pd.concat([table_of_intrant, t])

    # For 50 units and more apply this

    # t = ajouter_caraterisque_par_type_unite(sh, [], 'pptu', [3, 19], False)
    # t = pd.DataFrame(t, columns=entete)
    # for secteur in __SECTEUR__[:-1]:
    #     t.loc[:, 'sector'] = secteur
    #     table_of_intrant = pd.concat([table_of_intrant, t])
    #
    # t = ajouter_caraterisque_par_type_unite(sh, [], 'ppts', [12, 19], False)
    # t = pd.DataFrame(t, columns=entete)
    # for secteur in __SECTEUR__[:-1]:
    #     t.loc[:, 'sector'] = secteur
    #     table_of_intrant = pd.concat([table_of_intrant, t])
    #
    # t = ajouter_caraterisque_par_type_unite(sh, [], 'pptu', [3, 10], False)
    # t = pd.DataFrame(t, columns=entete)
    # for secteur in __SECTEUR__[6:]:
    #     t.loc[:, 'sector'] = secteur
    #     table_of_intrant = pd.concat([table_of_intrant, t])
    #
    # t = ajouter_caraterisque_par_type_unite(sh, [], 'ppts', [12, 10], False)
    # t = pd.DataFrame(t, columns=entete)
    # for secteur in __SECTEUR__[6:]:
    #     t.loc[:, 'sector'] = secteur
    #     table_of_intrant = pd.concat([table_of_intrant, t])


    table_of_intrant['type'] = 'intrants'


    ####################################################################################################################
    #
    # Open Scenarios sheet and take all the important parameters
    #
    ###################################################################################################################

    sh = workbook.sheet_by_name(__SCENARIO_SHEET__)

    # Contribution sociale
    entete = ['sector', 'category', 'value'] + __BATIMENT__

    terr_ss = sh.cell(23, 3).value
    terr_hs = sh.cell(24, 3).value

    contrib_terr_ss = np.array([[sh.cell(i, 8).value for batim in __BATIMENT__] for i in range(29, 36)]) * terr_ss
    contrib_terr_hs = np.array([[sh.cell(i, 8).value for batim in __BATIMENT__] for i in range(40, 47)]) * terr_hs
    contrib_fin = np.array([[sh.cell(i, 3).value for batim in __BATIMENT__] for i in range(54, 61)]) * terr_ss

    contrib_terr_ss = pd.DataFrame(contrib_terr_ss, columns = __BATIMENT__)
    contrib_terr_ss['category'] = 'ALL'
    contrib_terr_ss['value'] = 'contrib_terr_ss'
    contrib_terr_ss['sector'] = __SECTEUR__
    contrib_terr_ss['type'] = 'scenarios'
    result = contrib_terr_ss[table_of_intrant.columns]
    result.loc[result['sector'] == __SECTEUR__[6], __BATIMENT__] = 0
    table_of_intrant = pd.concat([table_of_intrant, result], ignore_index=True)

    contrib_terr_hs = pd.DataFrame(contrib_terr_hs, columns = __BATIMENT__)
    contrib_terr_hs['category'] = 'ALL'
    contrib_terr_hs['value'] = 'contrib_terr_hs'
    contrib_terr_hs['sector'] = __SECTEUR__
    contrib_terr_hs['type']  = 'scenarios'
    result = contrib_terr_hs[table_of_intrant.columns]
    result.loc[result['sector'] != __SECTEUR__[6], __BATIMENT__] = 0
    table_of_intrant = pd.concat([table_of_intrant, result], ignore_index=True)

    contrib_fin = pd.DataFrame(contrib_fin, columns = __BATIMENT__)
    contrib_fin['category'] = 'ALL'
    contrib_fin['value'] = 'contrib_fin'
    contrib_fin['sector'] = __SECTEUR__
    contrib_fin['type']  = 'scenarios'
    result = contrib_fin[table_of_intrant.columns]
    table_of_intrant = pd.concat([table_of_intrant, result], ignore_index=True)



    # result = pd.DataFrame(result, columns=entete)
    # result['type'] = 'scenarios'
    # table_of_intrant = pd.concat([table_of_intrant, result])

    # Frais de parc
    # TODO:  Set Parc to oui
    fp_exig = sh.cell(64, 2).value
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
    REM = sh.cell(66, 2).value
    result = np.ones((7, 8))
    result = pd.DataFrame(result, columns=__BATIMENT__)
    result['category'] = 'ALL'
    result['value'] = 'rem'
    result['sector'] = __SECTEUR__
    result = result[entete]
    result['type'] = 'scenarios'
    result.replace({1: REM}, inplace=True)

    # TODO: Scenarios 1-2 REM Only for sector 7
    result.loc[result['sector'] != __SECTEUR__[6], __BATIMENT__] = 'Non'
    table_of_intrant = pd.concat([table_of_intrant, result], ignore_index=True)

    # Decontamination
    decont = sh.cell(70, 2).value
    result = -1 * decont * np.ones((7, 8))
    result = pd.DataFrame(result, columns=__BATIMENT__)
    result['category'] = 'ALL'
    result['value'] = 'decont'
    result['sector'] = __SECTEUR__
    result = result[entete]
    result['type'] = 'scenarios'
    table_of_intrant = pd.concat([table_of_intrant, result], ignore_index=True)


    tab_of_intrant_pos = [[[80, 2], 'qum'], [[89, 3], 'quf']]
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
    t['type'] = 'intrants'
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
                      [[6, 2], 'eq_terr'], [[7, 2], 'pp_prev'], [[14, 2], 'interet_terrain'], [[13, 2], 'interet_projet']]

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


def get_cb1_characteristic(secteur: list, batiment: list, table_of_intrant: pd.DataFrame) -> pd.DataFrame:

    """
        This function takes all the parameters of the Intrants sheets and calculate all the characteristics
        of the CB1 sheet.

        :param secteur: list of sector
        :param batiment: list of batiment
        :param table_of_intrant: table of the intrants

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
    """
    ###################################################################################################################
    #
    # Filter variables for the computations
    #
    ###################################################################################################################

    entete = ['type', 'sector', 'category', 'value'] + batiment

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
    # ces = sup_bru_one_floor[batiment] / sup_ter[batiment]
    # ces['category'] = 'ALL'
    # ces['value'] = 'ces'
    # ces['sector'] = sup_ter['sector']
    # ces['type'] = 'intrants'
    # result = ces[entete]
    # table_of_intrant = pd.concat([table_of_intrant, result])

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


def get_ca_characteristic(secteur: list, batiment: list, table_of_intrant: pd.DataFrame) ->pd.DataFrame:

    """
    This function take a table of CB1, CB3 or CB4 and compute the CA table to round the number of units.

    :param secteur: list of sector
    :param batiment: list of buildings
    :param table_of_intrant: table of intrant CB
    :return: data frame
    """
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
                      'min_max_asc', 'tap', 'price', 'cont_soc', 'parc', 'decont', 'rem', 'stat', 'denm_t', 'ces_m',
                      'contrib_terr_ss', 'contrib_terr_hs', 'contrib_fin']

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
    ntu = t[batiment].groupby(t['sector']).sum().reset_index()


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
    result = sup * (1 + cir[batiment].reset_index(drop=True)) * v * parc[batiment].reset_index(drop=True)

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

    ################################################################################################
    #
    # Contribution Sociale 'contrib_terr_ss', 'contrib_terr_hs', 'contrib_fin'
    #
    ################################################################################################

    x = (supbtu[batiment].reset_index(drop=True) + supt_cu[batiment].reset_index(drop=True))
    contrib_terr_ss = table_of_intrant[(table_of_intrant['value'] == 'contrib_terr_ss')]
    contrib_terr_hs = table_of_intrant[(table_of_intrant['value'] == 'contrib_terr_hs')]
    contrib_fin = table_of_intrant[(table_of_intrant['value'] == 'contrib_fin')]

    contrib_terr_ss = contrib_terr_ss[batiment].reset_index(drop=True) * x
    contrib_terr_ss = contrib_terr_ss.where(ntu[batiment] >= 150, 0)

    contrib_terr_hs = contrib_terr_hs[batiment].reset_index(drop=True) * x
    contrib_terr_hs = contrib_terr_hs.where(ntu[batiment] >= 150, 0)

    contrib_fin = contrib_fin[batiment].reset_index(drop=True) * x
    contrib_fin = contrib_fin.where(ntu[batiment] <= 149, 0)


    table_of_intrant.loc[table_of_intrant['value'] == 'contrib_terr_ss', batiment] = contrib_terr_ss
    table_of_intrant.loc[table_of_intrant['value'] == 'contrib_terr_hs', batiment] = contrib_terr_hs
    table_of_intrant.loc[table_of_intrant['value'] == 'contrib_fin', batiment] = contrib_fin



    # Go No Go
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
    x = floor
    floor = floor.fillna(-10).astype(int)
    x = (x - floor)*sup_bru_one_floor[batiment].reset_index(drop=True)
    x = x.fillna(-10).astype(int)
    x = x.where(x>0, 0)
    x = x.where(x==0, 1)
    floor = floor + x
    floor['category'] = 'ALL'
    floor['value'] = 'floor'
    floor['sector'] = secteur
    floor['type'] = 'intrants'
    result = floor[entete]
    table_of_intrant = pd.concat([table_of_intrant, result], ignore_index=True)

    ces = sup_bru_one_floor[batiment].reset_index(drop=True) / sup_ter[batiment].reset_index(drop=True)
    ces['value'] = 'ces'
    ces['category'] = 'ALL'
    ces['sector'] = secteur
    ces['type'] = 'intrants'
    ces = ces[entete]

    sup_ces = table_of_intrant[table_of_intrant['value'] == 'ss_sup_CES']
    x = sup_bru_one_floor[batiment].reset_index(drop=True) * sup_ces[batiment].reset_index(drop=True)
    y = sup_ter[batiment].reset_index(drop=True) * sup_ces[batiment].reset_index(drop=True)

    x.loc[:, __BATIMENT__[3:]] = y.loc[:, __BATIMENT__[3:]]
    floor_ss = sup_ss[batiment].reset_index(drop=True) / x
    floor_ss = floor_ss.astype(float).apply(np.round, 1)

    floor_ss['category'] = 'ALL'
    floor_ss['value'] = 'floor_ss'
    floor_ss['sector'] = secteur
    floor_ss['type'] = 'intrants'
    result = floor_ss[entete]
    table_of_intrant = pd.concat([table_of_intrant, result], ignore_index=True)
    value = table_of_intrant[(table_of_intrant['value'].isin(['min_ne', 'max_ne', 'ces_m', 'min_ne_ss', 'max_ne_ss', 'denm_t', 'ntu', 'floor', 'floor_ss', 'dens', 'ntu'])) &
                             (table_of_intrant['category'] == 'ALL')]

    value = pd.concat([value, ces], ignore_index=True)

    value.loc[value['value'] == 'max_ne', 'max_ne'] = value['value'].where(value['value'] != 0, 35)
    result = value[['value'] + batiment].groupby(value['sector']).apply(go_no_go, batiment)

    result.loc[:, 'type'] = 'go_no_go'
    result.loc[:, 'category'] = 'ALL'
    result.loc[:, 'sector'] = secteur
    result.loc[:, 'value'] = 'go_no_go'
    result = result[entete]
    table_of_intrant = pd.concat([table_of_intrant, result],ignore_index=True)

    # Validate number of units
    mask = (table_of_intrant['value'] == 'ntu') & (table_of_intrant['category'] == 'ALL')
    ntu = table_of_intrant[mask][['sector'] + batiment]
    ntu.loc[:, batiment] = ntu.where(ntu[batiment] > 0, np.nan)
    table_of_intrant = table_of_intrant.groupby('sector').apply(validate_batiment, ntu).reset_index(drop=True)

    # table_of_intrant.to_excel('test.xlsx')

    return table_of_intrant


def get_cb3_characteristic(secteur: list, batiment: list, table_of_intrant: pd.DataFrame, *args) ->pd.DataFrame:

    ###################################################################################################################
    #
    # Filter variables for the computations
    #
    ###################################################################################################################

    entete = ['type', 'sector', 'category', 'value'] + batiment

    input_variable = ['sup_ter', 'tum', 'vat', 'denm_p', 'ces', 'pptu', 'mp', 'min_nu', 'max_nu',
                      'min_ne', 'max_ne', 'min_ne_ss', 'max_ne_ss', 'cir', 'aec', 'si', 'pi_si', 'ee_ss', 'pi_ee',
                      'cub', 'sup_cu', 'supt_cu', 'pisc', 'sup_pisc', 'pp_sup_escom', 'pp_et_escom', 'ss_sup_CES',
                      'ss_sup_ter', 'nba', 'min_max_asc', 'tap', 'price', 'cont_soc', 'parc', 'decont', 'rem', 'stat',
                      'denm_t', 'ces_m', 'contrib_terr_ss', 'contrib_terr_hs', 'contrib_fin']

    table_of_intrant = table_of_intrant[(table_of_intrant['value'].isin(input_variable)) &
                                        (table_of_intrant['sector'].isin(secteur))][entete]
    ###################################################################################################################
    #
    # Take input parameter for the computations.
    #
    ###################################################################################################################

    # args is as dictionnariy containing the new input we want to change (sup_ter, vat, denm_p). If those values a 0 we
    # use the default value.

    intrant_value = args[0]

    for value in intrant_value:
        t = np.ones((len(secteur), len(batiment)))
        v = np.array(intrant_value[value])
        input = np.multiply(t, v)
        if value == 'max_ne':
            x = table_of_intrant.loc[table_of_intrant['value'] == value, batiment]
            input = x.where(x < input, input)
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
    max_ne = table_of_intrant[table_of_intrant['value'] == 'max_ne']

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
    result = sup_ter[batiment].reset_index(drop=True).astype(float) * denm_p[batiment].reset_index(drop=True) / \
             max_ne[batiment].reset_index(drop=True)
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
    result = sup * (1 + cir[batiment].reset_index(drop=True)) * v * parc[batiment].reset_index(drop=True)

    result['category'] = 'ALL'
    result['value'] = 'sup_parc'
    result['sector'] = secteur
    result['type'] = 'intrants'
    result = result[entete]
    table_of_intrant = pd.concat([table_of_intrant, result], ignore_index=True)

    return table_of_intrant


def get_cb4_characteristic(secteur: list, batiment: list, table_of_intrant: pd.DataFrame, *args) ->pd.DataFrame:
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


#######################################################################################################################
#
# Return results
#
#######################################################################################################################


def get_summary_characteristics(type, secteur, batiment, table_of_intrant, *args) ->pd.DataFrame:

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
    # args['sup_ter'] = [[37673.685]]
    # args['denm_p'] = [[1.5080235]]
    # args['max_ne'] = [[2]]
    # args['vat'] = [[43]]
    t = get_summary_characteristics('CA3', __SECTEUR__, __BATIMENT__, x, args)
