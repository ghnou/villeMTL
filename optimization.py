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
from scipy.stats import poisson
from lexique import __FILES_NAME__, __SECTEUR__, __BATIMENT__, __UNITE_TYPE__
from pathlib import Path

my_path = Path(__file__).parent.resolve()
data_for_simulation = collections.namedtuple('data_for_simulation', [
    'data',
    'cost_params',
    'financials_params',
    'scenario'])

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


def get_simulations(terrain_dev, scenario):

    cost_params = x[(x['type'].isin(['pcost'])) & (x['sector'] == 'Secteur 1')]
    finance_params = x[(x['type'].isin(['financial'])) & (x['sector'] == 'Secteur 1')]

    terr = terrain_dev.drop_duplicates(['sup_ter', 'denm_p', 'sector', 'vat', 'max_ne', 'min_ne']).reset_index(drop=True)
    # print(terr.describe())
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

    return result


def join_result_with_terrain(terrain_dev, data, scenarios, benchmark):

    def best_building(data, to_maximize):

        group = data.copy()
        group['benef'] = group['revenus totaux'] - group['cout du projet']
        group = group[group['marge beneficiaire'].fillna(-1000) > 12]

        if group.shape[0] == 0:
            return group
        id_ = group[to_maximize].fillna(-1000).idxmax()
        group = group.loc[id_, :].to_frame().transpose()

        return group

    ##################################################################################################################
    #
    # Open Files
    #
    ##################################################################################################################

    terr = terrain_dev.drop_duplicates(['sup_ter', 'denm_p', 'sector', 'vat', 'max_ne', 'min_ne']).reset_index(drop=True)

    # Uncomment this for the filters
    if not scenarios and benchmark:
        data = data[(data['Nombre unites'] < 301) & (data['Nombre unites'] > 4)]

    go = data.groupby('sector')['batiment'].count().reset_index()
    go.rename(columns={'batiment': 'go', 'sector': 'ID'}, inplace=True)
    go['ID'] = go['ID'].astype(int)
    terr = pd.merge(terr, go, 'inner', on=['ID'])

    # Get data for the best building Choice
    to_maximize = 'benef'
    data.rename(columns={'sector': 'ID'}, inplace=True)

    if scenarios:
        best_batiment = data
    else:
        best_batiment = data.groupby('ID').apply(best_building, to_maximize).reset_index(drop=True).reset_index()

    best_batiment['ID'] = best_batiment['ID'].astype(int)
    terr = pd.merge(terr, best_batiment, 'inner', on=['ID'])
    terr.drop(['ID'], axis=1, inplace=True)

    terrain_dev = pd.merge(terrain_dev, terr, 'inner', on=['sup_ter', 'denm_p', 'sector', 'vat', 'max_ne', 'min_ne'])
    terrain_dev['go'] = terrain_dev['go'].fillna(0)

    return terrain_dev


def save_file(data, files, *args):

    if files == 'benchmark.npy':
        d = dict()
        d['data'] = data
        d['header'] = data.columns
        np.save(files, d)
    elif args[0][1] == 'n_rem' and args[0][0] == 1:
        path = str(my_path) + '/scenario 1/n_rem/' + files
        d = dict()
        d['data'] = data
        d['header'] = data.columns
        np.save(path, d)
    elif args[0][1] == 'rem' and args[0][0] == 1:
        path = str(my_path) + '/scenario 1/rem/' + files
        d = dict()
        d['data'] = data
        d['header'] = data.columns
        np.save(path, d)


def read_file(files, *args):

    if files == 'benchmark.npy':
        data = np.load(files).item()
        return pd.DataFrame(data['data'], columns=data['header'])
    elif args[0][1] == 'n_rem' and args[0][0] == 1:
        path = str(my_path) + '/scenario 1/n_rem/' + files
        data = np.load(path).item()
        return pd.DataFrame(data['data'], columns=data['header'])
    elif args[0][1] == 'rem' and args[0][0] == 1:
        path = str(my_path) + '/scenario 1/rem/' + files
        data = np.load(path).item()
        return pd.DataFrame(data['data'], columns=data['header'])


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

    cost_table = calcul_cout_batiment(table_intrant,  secteur, batiment, CASE, PRICE_INCREASE)
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

    print('Intrants completed for process: ', os.getpid())

    # Add cost intrants.
    cost_table = calcul_cout_batiment(cb3['sector'].unique(), __BATIMENT__, ca3, cost_params, CASE, PRICE_INCREASE)
    print('Cost completed for process: ', os.getpid())

    result = calcul_detail_financier(cb3['sector'].unique(), __BATIMENT__, 120, cost_table, financials_params)
    print('Finance completed for process: ', os.getpid())
    print('')

    # Get financials
    return result


def get_poisson(data):

    def residual_land_value(fin_ter, ct_prj, rev_t,  contri):

        ret = 0.12
        v = (rev_t - (1 + ret) * (ct_prj - fin_ter + contri))/(1 + ret)
        v = v/(fin_ter - contri)
        return v

    def calculate_proba(res_val):

        return poisson.cdf(k=5 * res_val, mu=4, loc=0)

    data['residual value'] = data.apply(lambda x: residual_land_value(*x[['financement terrain', 'cout du projet',
                                                                          'revenus totaux', 'contrib_terr_ss']]), axis=1)
    data['proba'] = data.apply(lambda x: calculate_proba(*x[['residual value']]), axis=1)

    return data


def join_scenario_result(files_name, *args):

    files = 'scenario 1 5 to 49 units'
    files_ = 'scenario 1 50 to 300 units'
    for percent in ['', ' 4%', ' 7%', ' 10%']:

        data = read_file(files + percent + '.npy', args[0])
        data_ = read_file(files_ + percent + '.npy', args[0])

        result = pd.concat([data, data_], ignore_index=True)
        di = dict()
        di['header'] = result.columns
        di['data'] = result
        save_file(result, files_name + percent, args[0])


def put_result_in_panel(scenarios_files, args):

    d = dict()
    header = {'contrib_fin': 'contribution financiere',
              'contrib_terr_hs': 'contribution terrain hors sol',
              'contrib_terr_ss': 'contribution terrain sur sol'}

    def get_strat(sector, batiment):
        if sector in __SECTEUR__[:2]:
            if batiment in __BATIMENT__[:3]:
                return 1
            if batiment in __BATIMENT__[3:6]:
                return 2
            if batiment in __BATIMENT__[6:]:
                return 3
        if sector in __SECTEUR__[2:6]:
            if batiment in __BATIMENT__[:3]:
                return 4
            if batiment in __BATIMENT__[3:6]:
                return 5
            if batiment in __BATIMENT__[6:]:
                return 6
        if sector in __SECTEUR__[6:]:
            if batiment in __BATIMENT__[:3]:
                return 7
            if batiment in __BATIMENT__[3:6]:
                return 8
            if batiment in __BATIMENT__[6:]:
                return 9

    # Benchmark
    benchmark = read_file('benchmark.npy')
    benchmark['set'] = 'Small Building'
    benchmark.loc[benchmark['Nombre unites'] >= 50, 'set'] = 'Big Building'
    benchmark['stratification'] = benchmark.apply(lambda x: get_strat(*x[['sector', 'batiment']]), axis=1)
    benchmark['Expect Nombre unites'] = benchmark['Nombre unites'] * benchmark['proba']
    benchmark.rename(columns=header, inplace=True)
    d['Benchmark'] = benchmark.sort_values(['ID'])

    # Market Value
    for value in ['', ' 4%', ' 7%', ' 10%']:
        data = read_file(scenarios_files + value + '.npy', args)
        data['Social'] = data[['contrib_fin', 'contrib_terr_hs', 'contrib_terr_ss']].sum(axis=1)
        data['Expect Nombre unites'] = data['Nombre unites'] * data['proba']
        data.rename(columns=header, inplace=True)
        d['Market Price' + value] = data.sort_values(['ID'])

    return pd.concat(d, axis=1)


def get_scenario_impact(data):

    idx = pd.IndexSlice
    weight_big_building = [0, 0.29, 0.1, 0, 0.33, 0.16, 0, 0.06, 0.06]
    weight_small_building = [0.2, 0.38, 0, 0.12, 0.23, 0, 0.01, 0.06, 0]

    draw = dict()
    for i in range(len(weight_big_building)):
        draw[('Big Building', i + 1)] = int(65 * weight_big_building[i])
    for i in range(len(weight_big_building)):
        draw[('Small Building', i + 1)] = int(80 * weight_small_building[i])

    def random_draw(group, building, draw):
        strat = group.name
        number = draw[(building, strat)]

        return group.sample(n=number)

    def split_set(group):

        building = group.name
        t = []
        for sim in range(1000):
            r = group.groupby(group['Benchmark', 'stratification']).apply(random_draw, building, draw).reset_index(drop=True)
            r = r.sum().to_frame().transpose()
            r['Sample', 'ID'] = 'Sample ' + str(sim + 1)
            r['building', 'type'] = group.name
            t.append(r)
        return pd.concat(t, ignore_index=True)

    header = ['set', 'stratification', 'proba', 'Nombre unites', 'Expect Nombre unites', 'Social',
              'contribution financiere', 'contribution terrain hors sol',  'contribution terrain sur sol',
              'residual value', 'Nombre unites']
    data = data.loc[:, idx[:, header]]
    data = data.groupby(data['Benchmark', 'set']).apply(split_set).reset_index(drop=True)

    # data.set_index(['building'], inplace=True)
    data = data.loc[:, idx[:, ['ID', 'Expect Nombre unites', 'Social', 'contribution financiere',
                               'contribution terrain hors sol',  'contribution terrain sur sol',
                               'residual value', 'type']]]
    data.set_index(['building']).to_excel('poisson.xlsx')

    data.loc[:, idx[:, ['Expect Nombre unites', 'Social', 'contribution financiere',
                               'contribution terrain hors sol',  'contribution terrain sur sol',
                               'residual value']]].groupby(data['building', 'type']).sum().to_excel('total_poisson.xlsx')
    # print(data)


def get_statistics_for_simulation_results(data, name):

    def get_result(data):

        ###############################################################################################################
        #
        # Comparaison des carateristiques
        #
        ###############################################################################################################

        header = ['sup_ter', 'Nombre unites', 'marge beneficiaire']
        data[header] = data[header].astype(float)
        description = 'Distribution des caracterisques pour tous les terrains developpables'
        distrib_caract = dict()
        distrib_caract[description + ' (ALL).'] = data[['sup_ter', 'Nombre unites', 'marge beneficiaire']].astype(
            float).describe().reset_index()
        for value in [[' (MB > 12%.)', 12], [' (MB > 15%.)', 15], [' (MB > 18%.)', 18], [' (MB > 20%.)', 20]]:
            distrib_caract[description + value[0]] = data[data['marge beneficiaire'] >= value[1]][header].astype(
                float).describe().reset_index()

        distrib_caract = pd.concat(distrib_caract).reset_index(drop=True, level=1)
        distrib_caract = distrib_caract.reset_index()
        distrib_caract.rename(columns={'index': 'Value', 'level_0': 'Description'}, inplace=True)
        distrib_caract.set_index(['Description', 'Value'], inplace=True)

        #################################################################################################################
        #
        # Distribution des carateristiques
        #
        ##################################################################################################################

        distrib_total = data.groupby(['sector', 'batiment'])[['sup_ter', 'marge beneficiaire', 'Nombre unites']].describe()

        #################################################################################################################
        #
        # BreakDown nombre unites developpables par secteur
        #
        ##################################################################################################################

        terrain_dev = data[['sector', 'batiment', 'marge beneficiaire', 'Nombre unites'] + __UNITE_TYPE__]
        terrain_dev[__UNITE_TYPE__] = terrain_dev[__UNITE_TYPE__].astype(float)

        x = terrain_dev[terrain_dev['marge beneficiaire'] > 12]
        bkdu = x.groupby(['sector', 'batiment'])[['Nombre unites'] + __UNITE_TYPE__].sum()


        ##################################################################################################################
        #
        # Return result
        #
        ###################################################################################################################

        return [distrib_caract, distrib_total]

    return [name,
            get_result(data),
            get_result(data[(data['Nombre unites'] <= 300) & (data['Nombre unites'] >= 50)]),
            get_result(data[(data['Nombre unites'] <= 49) & (data['Nombre unites'] >= 5)])]


def write_in_excel_files(data, writer):

    data[1][0].to_excel(writer, sheet_name='Comp caract ' + data[0] + ' ALL')
    data[1][1].to_excel(writer, sheet_name='caract terr ' + data[0] + ' ALL')

    data[2][0].to_excel(writer, sheet_name='Comp caract ' + data[0] + ' 50-300')
    data[2][1].to_excel(writer, sheet_name='caract terr ' + data[0] + ' 50-300')

    data[3][0].to_excel(writer, sheet_name='Comp caract ' + data[0] + ' 5-49')
    data[3][1].to_excel(writer, sheet_name='caract ' + data[0] + ' 5-49')


CASE = 1
PRICE_INCREASE = 0
global x
myBook = xlrd.open_workbook(__FILES_NAME__)
x = get_all_informations(myBook, CASE)

if __name__ == '__main__':

    start = time.time()
    terrain_dev = get_land_informations()

    ################################################################################################################
    #
    # Make simulation and apply scenario
    #
    ################################################################################################################

    # Get benchmark
    # result = get_simulations(terrain_dev, False)
    # result = join_result_with_terrain(terrain_dev, result, False, True)
    # result = get_poisson(result)
    # save_file(result, 'benchmark.npy')

    ###############################################################################################################
    #
    # Apply scenario
    #
    ##############################################################################################################

    # Scenario  5 to 49 units
    # data = read_file('benchmark.npy')
    # files = 'scenario 1 5 to 49 units 10%.npy'
    # data = data[(data['Nombre unites'] < 50) & (data['Nombre unites'] >= 5)]
    # data = data[['ID', 'sup_ter', 'denm_p', 'sector', 'vat', 'max_ne', 'min_ne', 'batiment']]
    # data.rename(columns={'batiment': 'pv_batiment'}, inplace=True)
    # result = get_simulations(data, True)
    # result = join_result_with_terrain(terrain_dev, result, True, True)
    # result = get_poisson(result)
    # save_file(result, files, [1, 'rem'])

    # Scenario  50 to 300 units
    # data = read_file('benchmark.npy')
    # files = 'scenario 1 50 to 300 units 10%.npy'
    # data = data[(data['Nombre unites'] < 301) & (data['Nombre unites'] >= 50)]
    # data = data[['ID', 'sup_ter', 'denm_p', 'sector', 'vat', 'max_ne', 'min_ne', 'batiment']]
    # data.rename(columns={'batiment': 'pv_batiment'}, inplace=True)
    # result = get_simulations(data, True)
    # result = join_result_with_terrain(terrain_dev, result, True, True)
    # result = get_poisson(result)
    # save_file(result, files, [1, 'rem'])

    ###############################################################################################################
    #
    # Join scenario result
    #
    ##############################################################################################################

    # join_scenario_result('scenario 1', [1, 'n_rem'])
    # join_scenario_result('scenario 1', [1, 'rem'])

    ###############################################################################################################
    #
    # Put results in panel
    #
    ##############################################################################################################

    # Put all the scenario result in panel
    # result = put_result_in_panel(scenarios_files='scenario 1', args=[1, 'rem'])
    # get_scenario_impact(result)

    ###############################################################################################################
    #
    # Tear Description sheets
    #
    ###############################################################################################################

    # benchmark = read_file('benchmark.npy')
    # benchmark = get_statistics_for_simulation_results(benchmark, 'benchmark')
    # scenario_nrem = read_file('scenario 1.npy', [1, 'n_rem'])
    # scenario_nrem = get_statistics_for_simulation_results(scenario_nrem, 'scen no rem')
    # scenario_rem = read_file('scenario 1.npy', [1, 'rem'])
    # scenario_rem = get_statistics_for_simulation_results(scenario_rem, 'scen rem')

    # with pd.ExcelWriter('resultat scenario 1 avec rem.xlsx') as writer:  # doctest: +SKIP
    #     write_in_excel_files(benchmark, writer)
    #     write_in_excel_files(scenario_nrem, writer)
    #     write_in_excel_files(scenario_rem, writer)

    end = time.time()
    print(end - start)


