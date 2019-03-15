import time

__author__ = 'pougomg'
from calcul_de_couts import calcul_cout_batiment
from calcul_financier import calcul_detail_financier
from obtention_intrant import get_all_informations, get_summary_characteristics, get_ca_characteristic, \
    get_cb3_characteristic
import pandas as pd
import numpy as np
import xlrd
import collections
import os
import time
import multiprocessing

data_for_simulation = collections.namedtuple('data_for_simulation', [
    'data',
    'cost_params',
    'financials_params',
    'scenario'])


from lexique import __FILES_NAME__, __SECTEUR__, __BATIMENT__, __UNITE_TYPE__


def prix_terrain(secteur, dens):


    if secteur == __SECTEUR__[0]:
        return 35 if dens < 1 else 43

    elif secteur == __SECTEUR__[1]:
        if dens < 1.4:
            return 48
        elif dens>= 1.4 and dens < 3.5:
            return 7.4939* dens + 32.387
        elif dens>= 3.5 and dens < 10:
            return 26.483 * dens - 29.979
        else:
            return None
    elif secteur == __SECTEUR__[2]:
        return 0.3544 * dens ** 3 - 2.9431 * dens ** 2 + 22.754 * dens + 63.88

    elif secteur == __SECTEUR__[3]:
        return 3.3413 * dens ** 2  - 2.8718 * dens + 118.96

    elif secteur == __SECTEUR__[4]:
        return 4.7471 * dens ** 2   - 16.738 * dens + 167.29

    elif secteur == __SECTEUR__[5]:
        if dens < 2:
            return 193
        else:
            return 0.4548 * dens ** 3 - 1.8238 * dens ** 2  + 1.418 * dens + 201.32

    elif secteur == __SECTEUR__[6] :
        if dens < 2:
            return 285
        else:
            return 55.546 * dens + 73.131


def get_financials_results(z, *params):

    vat = z
    table_intrant, secteur, batiment, to_optimize, value = params

    print(batiment)
    print(vat)

    entete = ['type', 'sector', 'category', 'value'] + batiment
    table_intrant = table_intrant[entete]
    table_intrant.loc[table_intrant['value'] == 'vat', batiment] = vat

    cost_table = calcul_cout_batiment(table_intrant,  secteur, batiment)
    fin_table = calcul_detail_financier(cost_table, secteur, batiment,  120)

    r = fin_table.loc[fin_table[fin_table['value'] == to_optimize].index[0], batiment]

    return r


def function_to_optimize(z, *params):
    value = params[4]

    r = get_financials_results(z, *params)
    r = r[r.astype(float).idxmax(skipna=True)]
    r = 1000 if np.isnan(r) else r
    print(r)
    print('')
    return np.abs(value-r)


def n(params):
    x, y, z, k = params
    if x*y*z*k == 200:

        value = x + y + z + k
    else:
        value =  1000

    return value


def get_land_informations():

    """Open the files and imports all the land parameters"""
    couleur_secteur = {}

    couleur = ['Jaune', 'Vert', 'Bleu pâle', 'Bleu', 'Mauve', 'Rouge', 'Noir']

    for pos in range(len(__SECTEUR__)):
        couleur_secteur[couleur[pos]] = __SECTEUR__[pos]

    terrain_dev = pd.read_excel(__FILES_NAME__, sheet_name='terrains')

    header_dict = {'SuperficieTerrain_Pi2': 'sup_ter', 'COS max formule': 'denm_p', 'couleur': 'sector',
                   'Valeur terrain p2 PROVISOIRE': 'vat_', 'etages_max': 'max_ne', 'etages_min': 'min_ne'}
    terrain_dev.rename(columns = header_dict, inplace=True)
    terrain_dev.loc[:, 'sector'] = terrain_dev['sector'].replace(couleur_secteur)

    terrain_dev['vat'] = terrain_dev[['sector', 'sup_ter', 'denm_p']].apply(lambda row: prix_terrain(*row[['sector', 'denm_p']]), axis = 1)

    terrain_dev = terrain_dev[['ID', 'sup_ter', 'denm_p', 'sector', 'vat', 'max_ne', 'min_ne']]

    return terrain_dev


def get_summary(params):

    print('Start Process: ', os.getpid())

    def get_summary_value(group):
        global x
        data = group.copy()
        # print(data.columns)
        id_batiment = data.loc[:, 'ID'].values[0]
        sup_ter = data.loc[:, 'sup_ter'].values[0]
        denm_p = data.loc[:, 'denm_p'].values[0]
        vat = data.loc[:, 'vat'].values[0]
        min_ne = data.loc[:, 'min_ne'].values[0]
        max_ne = data.loc[:, 'max_ne'].values[0]
        max_ne = 35 if max_ne == 0 else max_ne
        sector = data.loc[:, 'sector'].values[0]

        args = dict()
        args['sup_ter'] = [[sup_ter]]
        args['denm_p'] = [[denm_p]]
        args['vat'] = [[vat]]
        # args['min_ne'] = [[min_ne]]
        args['max_ne'] = [[max_ne]]
        params  = x[x['sector'] == sector]
        params.loc[:, 'sector'] = id_batiment
        result = get_cb3_characteristic([id_batiment], __BATIMENT__, params, args)

        return result

    data = params.data
    scenario = params.scenario
    cost_params = params.cost_params
    financials_params = params.financials_params
    cb3 = data.groupby('ID').apply(get_summary_value).reset_index(drop=True)

    data.drop(['sector'], axis=1, inplace=True)

    if scenario:
        data.rename(columns={'ID': 'sector'}, inplace=True)
        ca3 = get_ca_characteristic(cb3['sector'].unique(), __BATIMENT__, cb3, data[['sector', 'pv_batiment']])
    else:
        ca3 = get_ca_characteristic(cb3['sector'].unique(), __BATIMENT__, cb3)
    # args=dict()
    # cb3 = get_cb3_characteristic(__SECTEUR__, __BATIMENT__, x, args)
    # ca3 = get_ca_characteristic(cb3['sector'].unique(), __BATIMENT__, cb3)
    print('Intrants completed for process: ', os.getpid())

    # Add cost intrants.
    cost_table = calcul_cout_batiment(cb3['sector'].unique(), __BATIMENT__, ca3, cost_params)
    print('Cost completed for process: ', os.getpid())

    result = calcul_detail_financier(cb3['sector'].unique(), __BATIMENT__, 120, cost_table, financials_params)
    print('Finance completed for process: ', os.getpid())
    print(result.head(10))
    print('')

    # Get financials
    return result


global x
myBook = xlrd.open_workbook(__FILES_NAME__)
x = get_all_informations(myBook)

def get_stat(terrain_dev, files):


    def best_building(data):

        group = data.copy()
        group['benef'] = group['revenus totaux'] - group['cout du projet']
        group = group[group['marge beneficiaire'].fillna(-1000) > 12]

        if group.shape[0] == 0:
            return group[['batiment', 'Nombre unites', 'supbtu', 'sup_bru_one_floor', 'marge beneficiaire', 'TRI'] +
                         __UNITE_TYPE__]
        id_ = group['benef'].fillna(-1000).idxmax()
        group = group.loc[id_, :].to_frame().transpose()
        group = group[['batiment', 'Nombre unites', 'supbtu', 'sup_bru_one_floor',  'marge beneficiaire', 'TRI'] +
                      __UNITE_TYPE__]

        return group

    ##################################################################################################################
    #
    # Open Files
    #
    ##################################################################################################################

    terr = terrain_dev.drop_duplicates(['sup_ter', 'denm_p', 'sector', 'vat', 'max_ne', 'min_ne']).reset_index(drop=True)

    data = np.load(files).item()
    header = data['header']
    data = data['data']
    data = pd.DataFrame(data, columns=header)
    # print(data['marge beneficiaire'].describe())
    # data.to_excel('t1.xlsx')
    # Uncomment this for the filters
    # data = data[(data['Nombre unites'] <= 49) & (data['Nombre unites'] > 4)]

    go = data.groupby('sector')['batiment'].count().reset_index()
    go.rename(columns={'batiment': 'go', 'sector': 'ID'}, inplace=True)
    go['ID'] = go['ID'].astype(int)
    terr = pd.merge(terr, go, 'inner', on=['ID'])

    # Get data for the best building Choice
    best_batiment = data.groupby('sector').apply(best_building).reset_index(level=1, drop=True).reset_index()
    best_batiment.rename(columns={'sector': 'ID'}, inplace=True)
    best_batiment['ID'] = best_batiment['ID'].astype(int)
    terr = pd.merge(terr, best_batiment, 'inner', on=['ID'])
    terr.drop(['ID'], axis=1, inplace=True)

    terrain_dev = pd.merge(terrain_dev, terr, 'inner', on=['sup_ter', 'denm_p', 'sector', 'vat', 'max_ne', 'min_ne'])
    terrain_dev['go'] = terrain_dev['go'].fillna(0)
    terrain_dev.to_excel('t1.xlsx')

    header = ['ID', 'go', 'sector', 'batiment', 'sup_ter', 'marge beneficiaire', 'Nombre unites']
    # Write all the lands results in the files
    resultat_total = terrain_dev[header]
    resultat_total = resultat_total[resultat_total['go'] > 0]
    resultat_total[['sup_ter', 'marge beneficiaire', 'Nombre unites']] = resultat_total[['sup_ter', 'marge beneficiaire', 'Nombre unites']].astype(float)
    distrib_total = resultat_total.groupby(['sector', 'batiment'])[['sup_ter', 'marge beneficiaire', 'Nombre unites']].describe()


    header = ['ID', 'sector', 'sup_ter', 'vat', 'denm_p', 'max_ne', 'min_ne', 'go', 'batiment', 'Nombre unites',
              'marge beneficiaire', 'TRI']

    x = terrain_dev[terrain_dev['marge beneficiaire'].isna() == False][header]
    distrib_caract = x[['sup_ter', 'Nombre unites', 'marge beneficiaire']].astype(float).describe().reset_index()
    distrib_caract['Description'] = 'Distribution des caracterisques pour tous les terrains developpables'

    distrib_caract_ = x[x['marge beneficiaire'].astype(float) > 15][['sup_ter', 'Nombre unites', 'marge beneficiaire']].astype(float).describe().reset_index()
    distrib_caract_['Description'] = 'Distribution des caracterisques pour tous les terrains de marge beneficiare > 15%.'

    distrib_caract_1 = x[x['marge beneficiaire'].astype(float) > 18][['sup_ter', 'Nombre unites', 'marge beneficiaire']].astype(float).describe().reset_index()
    distrib_caract_1['Description'] = 'Distribution des caracterisques pour tous les terrains de marge beneficiare > 18%.'

    distrib_caract_2 = x[x['marge beneficiaire'].astype(float) > 20][['sup_ter', 'Nombre unites', 'marge beneficiaire']].astype(float).describe().reset_index()
    distrib_caract_2['Description'] = 'Distribution des caracterisques pour tous les terrains de marge beneficiare > 20%.'


    distrib_caract = pd.concat([distrib_caract, distrib_caract_, distrib_caract_1, distrib_caract_2], ignore_index=True)
    distrib_caract.rename(columns={'index': 'Value'}, inplace=True)
    distrib_caract.set_index(['Description', 'Value'], inplace=True)

    x = terrain_dev[terrain_dev['marge beneficiaire'].fillna(-1000) > 12]
    bkdu = x.groupby(['sector', 'batiment'])[['Nombre unites'] + __UNITE_TYPE__].sum()

    # with pd.ExcelWriter('scenario 50 to 300.xlsx') as writer:  # doctest: +SKIP
    #     distrib_total.to_excel(writer, sheet_name='Distribution totale')
    #     distrib_caract.to_excel(writer, sheet_name='Distribution des carac')
    #     bkdu.to_excel(writer, sheet_name='breakdown by units type')


    # d = dict()
    # d['data'] = terrain_dev
    # d['header'] = terrain_dev.columns
    # np.save('benchmark', d)


def get_statistics(terrain_dev, files):

    ##################################################################################################################
    #
    # Open Files
    #
    ##################################################################################################################

    terr = terrain_dev.drop_duplicates(['sup_ter', 'denm_p', 'sector', 'vat', 'max_ne', 'min_ne']).reset_index(drop=True)
    data = np.load(files).item()
    header = data['header']
    data = data['data']
    data = pd.DataFrame(data, columns=header)

    # Uncomment this for the filters
    # data = data[(data['Nombre unites'] <= 50) & (data['Nombre unites'] >0)]

    go = data.groupby('sector')['batiment'].count().reset_index()
    go.rename(columns={'batiment': 'go', 'sector': 'ID'}, inplace=True)
    go['ID'] = go['ID'].astype(int)
    terr = pd.merge(terr, go, 'left', on=['ID'])

    def land_sup(secteur, sup_ter):

        if secteur == __SECTEUR__[0]:
            return 1 if sup_ter> 3379 and sup_ter <= 145375*1.1 else  0

        elif secteur == __SECTEUR__[1]:
            return 1 if sup_ter> 2961 and sup_ter <= 23030*1.1 else  0

        elif secteur == __SECTEUR__[2]:
            return 1 if sup_ter> 2174 and sup_ter <= 21026*1.1 else  0

        elif secteur == __SECTEUR__[3]:
            return 1 if sup_ter> 1959 and sup_ter <= 21026*1.1 else  0

        elif secteur == __SECTEUR__[4]:
            return 1 if sup_ter> 1747 and sup_ter <= 21026*1.1 else  0

        elif secteur == __SECTEUR__[5]:
            return 1 if sup_ter> 1707 and sup_ter <= 17522*1.1 else  0

        elif secteur == __SECTEUR__[6]:
            return 1 if sup_ter> 1232 and sup_ter <= 16008*1.1 else  0

    def best_building(data):

        group = data.copy()
        group['benef'] = group['revenus totaux'] - group['cout du projet']
        group = group[group['marge beneficiaire'].fillna(-1000) > 12]

        if group.shape[0] == 0:
            return group[['batiment', 'Nombre unites', 'supbtu', 'sup_bru_one_floor', 'marge beneficiaire', 'TRI'] +
                         __UNITE_TYPE__]
        id_ = group['benef'].fillna(-1000).idxmax()
        group = group.loc[id_, :].to_frame().transpose()
        group = group[['batiment', 'Nombre unites', 'supbtu', 'sup_bru_one_floor',  'marge beneficiaire', 'TRI'] +
                      __UNITE_TYPE__]

        return group

    def second_building(data):
        group = data.copy()

        id_ = group['marge beneficiaire'].fillna(-1000).idxmax()
        group = group[group.index != id_]

        count = group['batiment'].count()
        header = pd.MultiIndex.from_product([['Second Choix'],
                                             group.columns])
        if count > 0:
            id_ = group['marge beneficiaire'].fillna(-1000).idxmax()
            group = group.loc[id_, :].to_frame().transpose()
            group = group[['batiment', 'Nombre unites', 'marge beneficiaire', 'TRI']]

        else:
            group = group[['batiment', 'Nombre unites', 'marge beneficiaire', 'TRI']]

        return group

    ##################################################################################################################
    #
    # Add result to the terain data
    #
    ##################################################################################################################

    # Get data for the best building Choice
    best_batiment = data.groupby('sector').apply(best_building).reset_index(level=1, drop=True).reset_index()
    best_batiment.rename(columns={'sector': 'ID'}, inplace=True)
    best_batiment['ID'] = best_batiment['ID'].astype(int)
    terr = pd.merge(terr, best_batiment, 'left', on=['ID'])

    # Get data for the second building Choice
    second_batiment = data.groupby('sector').apply(second_building).reset_index(level=1, drop=True).reset_index()
    second_batiment.rename(columns={'sector': 'ID'}, inplace=True)
    second_batiment['ID'] = second_batiment['ID'].astype(int)
    terr = pd.merge(terr, second_batiment, 'left', on=['ID'])

    # TODO: add ID to join for the unique land
    terrain_dev = pd.merge(terrain_dev, terr, 'left', on=['sup_ter', 'denm_p', 'sector', 'vat', 'max_ne', 'min_ne'])
    terrain_dev['go'] = terrain_dev['go'].fillna(0)
    terrain_dev.rename(columns={'ID_x': 'ID'}, inplace=True)

    #################################################################################################################
    #
    # Write all the land results on Spread sheets
    #
    ##################################################################################################################

    header = ['ID', 'sector', 'sup_ter', 'vat', 'denm_p', 'max_ne', 'min_ne', 'go', 'batiment_x', 'Nombre unites_x', 'supbtu', 'sup_bru_one_floor',
              'marge beneficiaire_x', 'TRI_x', 'batiment_y', 'Nombre unites_y', 'marge beneficiaire_y', 'TRI_y']

    header_val = ['sup_ter', 'vat', 'denm_p', 'max_ne', 'min_ne', 'Nombre unites_x',
                  'marge beneficiaire_x', 'TRI_x', 'Nombre unites_y', 'marge beneficiaire_y',
                  'TRI_y']
    # Write all the lands results in the files
    resultat_total = terrain_dev[header]
    resultat_total = resultat_total[resultat_total['go'] > 0]
    resultat_total[header_val] = resultat_total[header_val].astype(float)
    terrain_dev[header_val] = terrain_dev[header_val].astype(float)
    distrib_total = resultat_total.groupby(['sector', 'batiment_x'])[['sup_ter', 'marge beneficiaire_x', 'Nombre unites_x']].describe()


     #################################################################################################################
    #
    # Distribution des superficie
    #
    ##################################################################################################################

    header = ['ID', 'sector', 'sup_ter', 'vat', 'denm_p', 'max_ne', 'min_ne', 'go', 'batiment_x', 'Nombre unites_x',
              'marge beneficiaire_x', 'TRI_x']

    # terrain_dev = terrain_dev.drop_duplicates(['sup_ter', 'denm_p', 'sector', 'vat', 'max_ne', 'min_ne']).reset_index(drop=True)

    x = terrain_dev[terrain_dev['marge beneficiaire_x'].isna() == False][header]
    distrib_caract = x[['sup_ter', 'Nombre unites_x', 'marge beneficiaire_x']].astype(float).describe().reset_index()
    distrib_caract['Description'] = 'Distribution des caracterisques pour tous les terrains developpables'

    distrib_caract_ = x[x['marge beneficiaire_x'].astype(float) > 15][['sup_ter', 'Nombre unites_x', 'marge beneficiaire_x']].astype(float).describe().reset_index()
    distrib_caract_['Description'] = 'Distribution des caracterisques pour tous les terrains de marge beneficiare > 15%.'

    distrib_caract_1 = x[x['marge beneficiaire_x'].astype(float) > 18][['sup_ter', 'Nombre unites_x', 'marge beneficiaire_x']].astype(float).describe().reset_index()
    distrib_caract_1['Description'] = 'Distribution des caracterisques pour tous les terrains de marge beneficiare > 18%.'

    distrib_caract_2 = x[x['marge beneficiaire_x'].astype(float) > 20][['sup_ter', 'Nombre unites_x', 'marge beneficiaire_x']].astype(float).describe().reset_index()
    distrib_caract_2['Description'] = 'Distribution des caracterisques pour tous les terrains de marge beneficiare > 20%.'

    distrib_caract_3 = x[x['marge beneficiaire_x'].astype(float) > 12][
        ['sup_ter', 'Nombre unites_x', 'marge beneficiaire_x']].astype(float).describe().reset_index()
    distrib_caract_3[
        'Description'] = 'Distribution des caracterisques pour tous les terrains de marge beneficiare > 12%.'
    x = x.sort_values(['marge beneficiaire_x', 'sup_ter'])
    pd.concat([x.head(50), x.tail(50)], ignore_index=True).to_excel('tail.xlsx')

    distrib_caract = pd.concat([distrib_caract,distrib_caract_3, distrib_caract_, distrib_caract_1, distrib_caract_2], ignore_index=True)
    distrib_caract.rename(columns={'index': 'Value'}, inplace=True)
    distrib_caract.set_index(['Description', 'Value'], inplace=True)

    t_ = x[x['marge beneficiaire_x'].astype(float) > 12].groupby('sector')[['sup_ter']].describe().reset_index()
    t_['Description'] = 'Nombre total de terrain pour MB > 12%'

    t = x[x['marge beneficiaire_x'].astype(float) > 15].groupby('sector')[['sup_ter']].describe().reset_index()
    t['Description'] = 'Nombre total de terrain pour MB > 15%'

    t_1 = x[x['marge beneficiaire_x'].astype(float) > 18].groupby('sector')[['sup_ter']].describe().reset_index()
    t_1['Description'] = 'Nombre total de terrain pour MB > 18%'

    t_2 = x[x['marge beneficiaire_x'].astype(float) > 20].groupby('sector')[['sup_ter']].describe().reset_index()
    t_2['Description'] = 'Nombre total de terrain pour MB > 20%'

    t = pd.concat([t_, t, t_1, t_2], ignore_index=True)
    t.set_index(['Description', 'sector'], inplace=True)



     #################################################################################################################
    #
    # BreakDown nombre unites developpables par secteur
    #
    ##################################################################################################################

    terrain_dev = terrain_dev[['sector', 'batiment_x', 'marge beneficiaire_x', 'Nombre unites_x'] + __UNITE_TYPE__]
    terrain_dev[__UNITE_TYPE__] = terrain_dev[__UNITE_TYPE__].astype(float)

    x = terrain_dev[terrain_dev['marge beneficiaire_x'].fillna(-1000) > 15]
    bkdu = x.groupby(['sector', 'batiment_x'])[['Nombre unites_x'] + __UNITE_TYPE__].sum()


     #################################################################################################################
    #
    # Group number of units per margin
    #
    ##################################################################################################################

    # terrain_dev['marge'] = pd.cut(terrain_dev['marge beneficiaire_x'],
    #                               [terrain_dev['marge beneficiaire_x'].min(), 10, 15, 20, 25, 30, 35, terrain_dev['marge beneficiaire_x'].max()]).values.add_categories('0').fillna(
    #     '0')
    #
    # def get_sum(group):
    #     data = group.copy()
    #     data = data.groupby(['marge']).sum().reset_index()
    #     return data.set_index('marge').transpose()
    #
    # terrain_dev = terrain_dev[terrain_dev['marge'] != '0']
    # gupm = terrain_dev.groupby(['sector', 'batiment_x'])[['Nombre unites_x', 'marge']].apply(get_sum)

    ##################################################################################################################
    #
    # Write result in the output files
    #
    ###################################################################################################################

    with pd.ExcelWriter('output.xlsx') as writer:  # doctest: +SKIP
        resultat_total.to_excel(writer, sheet_name='Choix par terrain')
        distrib_total.to_excel(writer, sheet_name='Distribution totales')
        distrib_caract.to_excel(writer, sheet_name='Distribution des carac')
        bkdu.to_excel(writer, sheet_name='breakdown by units type')
        # gupm.to_excel(writer, sheet_name='Total units per return range')
        t.to_excel(writer, sheet_name='Stratified')


def get_simulations(terrain_dev, scenario, files):

    cost_params = x[(x['type'].isin(['pcost'])) & (x['sector'] == 'Secteur 1')]
    finance_params = x[(x['type'].isin(['financial'])) & (x['sector'] == 'Secteur 1')]

    print(terrain_dev.groupby('sector')['sector'].count())
    terr = terrain_dev.drop_duplicates(['sup_ter', 'denm_p', 'sector', 'vat', 'max_ne', 'min_ne']).reset_index(drop=True)
    print(terr.describe())
    intervall = np.array_split(terr.index, 16)
    params = ()
    # params = data_for_simulation(data=terr.head(25),
    #                              cost_params=cost_params,
    #                              financials_params=finance_params,
    #                              scenario=scenario)
    # print(get_summary(params))
    #
    for value in intervall:
        params += data_for_simulation(data=terr.loc[value, :],
                                     cost_params=cost_params,
                                     financials_params=finance_params,
                                      scenario=scenario),

    pool = multiprocessing.Pool(16)
    result = pool.map(get_summary, params)
    pool.close()
    pool.join()

    result = pd.concat(result, ignore_index=True)
    di = dict()

    di['header'] = result.columns
    di['data'] = result
    np.save(files, di)


if __name__ == '__main__':

    start = time.time()

    # myBook = xlrd.open_workbook(__FILES_NAME__)
    # x = get_all_informations(myBook)
    terrain_dev = get_land_informations()
    get_simulations(terrain_dev, False, 'resultat simulation.npy')


    data = np.load('resultat simulation.npy').item()
    header = data['header']
    data = data['data']
    data = pd.DataFrame(data, columns=header)
    print(data.columns)
    # data = data[(data['Nombre unites'] < 301) & (data['Nombre unites'] >= 50)]
    # data.to_excel('t.xlsx')
    # data = data[['ID', 'sup_ter', 'denm_p', 'sector', 'vat', 'max_ne', 'min_ne', 'batiment']]
    # data.rename(columns={'batiment': 'pv_batiment'}, inplace=True)
    # get_simulations(data, True, 'scenario 1 50 to 300 units theta.npy')

    # terrain_dev = get_land_informations()
    # get_stat(terrain_dev, 'scenario 1 50 to 300 units.npy')

    end = time.time()

    print(end - start)


