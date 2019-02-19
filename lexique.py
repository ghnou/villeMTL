__COUTS_FILES_NAME__ = 'outils cout.xlsx'
__FINANCE_FILES_NAME__ = 'finance.xlsx'

__FILES_NAME__ = 'templates.xlsx'

__SCENARIO_SHEET__ = 'scenarios'
__INTRANT_SHEET__ = 'intrants'
__PRICE_SHEET__ = 'prix'
__COUT_SHEET__ = 'couts'
__ECOULEMENT_SHEET__ = 'ecoulement'
__FINANCE_PARAM_SHEET__ = 'finance'

__BATIMENT__ = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']
__SECTEUR__ = ['Secteur 1', 'Secteur 2', 'Secteur 3', 'Secteur 4', 'Secteur 5', 'Secteur 6', 'Secteur 7']
__UNITE_TYPE__ = ['Studios', '1cc', '2cc', '3cc', 'Penthouse', '2cc fam', '3cc fam']
__QUALITE_BATIMENT__ = ['Base', 'Moyenne', 'Elevee']
import pandas as pd

def get_cb1_characteristic(secteur, batiment, table_of_intrant):

    ###################################################################################################################
    #
    # Filter variables for the computations
    #
    ###################################################################################################################

    entete = ['type', 'sector', 'category', 'value'] + batiment

    input_variable = ['ntu', 'nmu_et', 'tum', 'vat', 'denm_p', 'pptu','mp', 'min_nu', 'max_nu', 'min_ne', 'max_ne',
                      'min_ne_ss', 'max_ne_ss', 'cir', 'aec', 'si', 'pi_si', 'ee_ss','pi_ee', 'cub', 'sup_cu',
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

    ntu = table_of_intrant[table_of_intrant['value'] == 'ntu']
    pptu = table_of_intrant[table_of_intrant['value'] == 'pptu']
    result = pptu.groupby(pptu['sector']).apply(convert_unity_type_to_sector, ntu, batiment).reset_index(drop=True)
    table_of_intrant = pd.concat([table_of_intrant, result[entete]])

    # add superfice unite

    x = table_of_intrant[(table_of_intrant['value'] == 'ntu')|
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
    sup_ter = sup_tot_hs[batiment]/ denm_p[batiment].reset_index()
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
        /(1-cir[batiment].reset_index())

    sup_ss = (x + sup_tot_hs[batiment].reset_index())/(1-aec[batiment].reset_index()) - sup_tot_hs[batiment].reset_index()
    sup_ss['category'] = 'ALL'
    sup_ss['value'] = 'sup_ss'
    sup_ss['sector'] = sup_ter['sector']
    sup_ss['type'] = 'intrants'
    result = sup_ss[entete]
    table_of_intrant = pd.concat([table_of_intrant, result])


    # Superficie parc
    sup = table_of_intrant[(table_of_intrant['value'] == 'suptu') & (table_of_intrant['category'].isin(__UNITE_TYPE__[0:3]))]
    sup = sup[batiment].groupby(sup['sector']).sum().reset_index(drop=True)
    v = table_of_intrant[(table_of_intrant['value'] == 'ntu') & (table_of_intrant['category'] == 'ALL')][batiment].reset_index(drop=True)
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
