import numpy as np
import pandas as pd
import xlrd

from lexique import __COUTS_FILES_NAME__, __INTRANT_SHEET__, __PRICE_SHEET__, \
    __BATIMENT__, __SECTEUR__, __UNITE_TYPE__, __SCENARIO_SHEET__


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
        _ = [unite, 'ALL', name]
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


def convert_unity_type_to_sector(group, data):
    df = data[__BATIMENT__].mul(group[__BATIMENT__].values.tolist()[0], axis=1)

    df['category'] = group.name
    df['sector'] = data['sector']
    df['value'] = data['value']
    return df


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


def calculate_total_surface(group):
    df = group[group['value'] == 'ntu'][__BATIMENT__].reset_index(drop=True) * \
         group[group['value'] == 'tum'][__BATIMENT__].reset_index(drop=True)
    df['category'] = group.name
    df['sector'] = group[group['value'] == 'ntu']['sector'].reset_index(drop=True)
    df['value'] = 'suptu'

    return df


def calculate_total_unit(group):
    df = group[group['value'] == 'suptu'][__BATIMENT__].reset_index(drop=True) / \
         group[group['value'] == 'tum'][__BATIMENT__].reset_index(drop=True)
    df['category'] = group.name
    df['sector'] = group[group['value'] == 'tum']['sector'].reset_index(drop=True)
    df['value'] = 'ntu'

    return df


def get_mean_brute_surface(group, data):
    df = group[__BATIMENT__].reset_index(drop=True).mul(data[data['sector'] == group.name][__BATIMENT__].values[0],
                                                        axis='columns')
    df['sector'] = group['sector'].reset_index(drop=True)
    df['value'] = 'sup_bru_par_u'
    df['category'] = group.name

    return df


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


def get_cb4_characteristics(table_of_intrant, *args) -> pd.DataFrame:
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
        
        sup_tot_hs add
    sup_ss 
    suptu add
    ntu add
    supbtu add
    cir in
    pisc in
    cub in
    sup_com add
    decont in
    sup_parc in
    cont_soc in
    vat in 
    sup_ter add  

        """""

    input_var = ['sector', 'sup_ter', 'denm_p', 'ces', 'min_ne', 'max_ne']

    entete = ['sector', 'category', 'value'] + __BATIMENT__

    secteur = args[0]
    sup_ter = args[1]
    denm_p = args[2]
    ces = args[3]
    min_ne = args[4]
    max_ne = args[5]
    ppts = table_of_intrant[table_of_intrant['value'] == 'ppts'].sort_values(by=['sector'])
    if secteur is not None:
        table_of_intrant = table_of_intrant[(table_of_intrant['sector'] == secteur)]
    drop_value = ['ntu', 'sup_tot_hs', 'supbtu', 'sup_com', 'suptu', 'sup_ss']

    for value in range(1, len(args)):
        if args[value] is not None:
            drop_value.append(input_var[value])

    table_of_intrant = table_of_intrant[table_of_intrant['value'].isin(drop_value) == False]

    if sup_ter is not None:
        v = pd.DataFrame([np.append([secteur, 'ALL', 'sup_ter'], sup_ter * np.ones(len(__BATIMENT__)))],
                         columns=table_of_intrant.columns)
        table_of_intrant = pd.concat([table_of_intrant, v], ignore_index=True)

    if denm_p is not None:
        v = pd.DataFrame([np.append([secteur, 'ALL', 'denm_p'], denm_p * np.ones(len(__BATIMENT__)))],
                         columns=table_of_intrant.columns)
        table_of_intrant = pd.concat([table_of_intrant, v], ignore_index=True)

    if ces is not None:
        v = pd.DataFrame([np.append([secteur, 'ALL', 'ces'], ces * np.ones(len(__BATIMENT__)))],
                         columns=table_of_intrant.columns)
        table_of_intrant = pd.concat([table_of_intrant, v], ignore_index=True)

    if min_ne is not None:
        v = pd.DataFrame([np.append([secteur, 'ALL', 'min_ne'], min_ne * np.ones(len(__BATIMENT__)))],
                         columns=table_of_intrant.columns)
        table_of_intrant = pd.concat([table_of_intrant, v], ignore_index=True)

    if max_ne is not None:
        v = pd.DataFrame([np.append([secteur, 'ALL', 'max_ne'], max_ne * np.ones(len(__BATIMENT__)))],
                         columns=table_of_intrant.columns)
        table_of_intrant = pd.concat([table_of_intrant, v], ignore_index=True)

    land_surface_p = table_of_intrant[
        (table_of_intrant['value'] == 'sup_ter') & (table_of_intrant['category'] == 'ALL')]
    den_max_per = table_of_intrant[
        (table_of_intrant['value'] == 'denm_p') & (table_of_intrant['category'] == 'ALL')]

    # Superficice brute totale hors sol

    result = land_surface_p[__BATIMENT__].reset_index(drop=True).astype(float) * den_max_per[__BATIMENT__].reset_index(
        drop=True).astype(float)
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

    # Calcul des superfices totales des unites selon la typologie
    suptu = table_of_intrant[table_of_intrant['value'] == 'suptu'][__BATIMENT__ + ['sector', 'value']]
    result = ppts.groupby('sector').apply(convert_unity_type_to_sector, suptu).reset_index(drop=True)
    result = result[entete]
    table_of_intrant = pd.concat([table_of_intrant, result], ignore_index=True)

    # Nombre d'unite
    # Total Surface by unity type

    ntu = table_of_intrant[(table_of_intrant['value'] == 'tum') | (table_of_intrant['value'] == 'suptu') & (
            table_of_intrant['category'] != 'ALL')]
    result = ntu.groupby('category').apply(calculate_total_unit).reset_index(drop=True)
    result = result[entete]
    table_of_intrant = pd.concat([table_of_intrant, result], ignore_index=True)

    result = result[__BATIMENT__].groupby(result['sector']).sum()
    result['category'] = 'ALL'
    result['value'] = 'ntu'
    result['sector'] = secteur
    result = result[entete]

    table_of_intrant = pd.concat([table_of_intrant, result], ignore_index=True)

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

    return table_of_intrant[table_of_intrant['value'].isin(['sup_tot_hs', 'sup_ss', 'suptu', 'ntu', 'supbtu', 'cir',
                                                            'pisc', 'cub', 'sup_com', 'decont', 'sup_parc', 'cont_soc',
                                                            'vat', 'sup_ter', 'price'])]



if __name__ == '__main__':
    myBook = xlrd.open_workbook(__COUTS_FILES_NAME__)
    data = get_cb1_characteristics(myBook)
    get_cb3_characteristics(data, "Secteur 7", 5389.0, None, None, None, None)
