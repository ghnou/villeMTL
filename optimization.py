import time

__author__ = 'pougomg'
from calcul_de_couts import calcul_cout_batiment
from calcul_financier import calcul_detail_financier
from obtention_intrant import get_all_informations, get_summary_characteristics, get_ca_characteristic, \
    get_cb3_characteristic
from scipy import optimize
import pandas as pd
import numpy as np
import xlrd
import collections
import multiprocessing
import os

data_for_simulation = collections.namedtuple('data_for_simulation',[
    'data',
    'cost_params',
    'financials_params',])


from lexique import __FILES_NAME__, __SECTEUR__, __BATIMENT__

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

        value =  x + y + z + k
    else:
        value =  1000

    return value

def get_land_informations():

    couleur_secteur = {}

    couleur = ['Jaune', 'Vert', 'Bleu pâle', 'Bleu', 'Mauve', 'Rouge', 'Noir']

    for pos in range(len(__SECTEUR__)):
        couleur_secteur[couleur[pos]] = __SECTEUR__[pos]

    terrain_dev = pd.read_excel(__FILES_NAME__, sheet_name='terrains')

    header_dict = {'SuperficieTerrain_Pi2': 'sup_ter', 'COS max formule': 'denm_p', 'couleur': 'sector',
                   'Valeur terrain p2 PROVISOIRE': 'vat', 'etages_max': 'max_ne', 'etages_min': 'min_ne'}
    terrain_dev.rename(columns = header_dict, inplace=True)

    terrain_dev = terrain_dev[['ID', 'sup_ter', 'denm_p', 'sector', 'vat', 'max_ne', 'min_ne']]

    terrain_dev.loc[:, 'sector'] = terrain_dev['sector'].replace(couleur_secteur)

    return terrain_dev


def get_summary(params):

    print('Start Process: ', os.getpid())
    def get_summary_value(group):

        data = group.copy()

        id_batiment = data.loc[:, 'ID'].values[0]
        sup_ter = data.loc[:, 'sup_ter'].values[0]
        denm_p = data.loc[:, 'denm_p'].values[0]
        vat = data.loc[:, 'vat'].values[0]
        min_ne = data.loc[:, 'min_ne'].values[0]
        max_ne = data.loc[:, 'max_ne'].values[0]
        sector = data.loc[:, 'sector'].values[0]

        args = dict()
        args['sup_ter'] = [[sup_ter]]
        args['denm_p'] = [[denm_p]]
        args['vat'] = [[vat]]
        args['min_ne'] = [[min_ne]]
        args['max_ne'] = [[max_ne]]
        params = x[x['sector'] == sector]
        params.loc[:, 'sector'] = id_batiment
        result = get_cb3_characteristic([id_batiment], __BATIMENT__, params, args)

        return result

    data = params.data
    cost_params = params.cost_params
    financials_params = params.financials_params

    cb3 = data.groupby('ID').apply(get_summary_value).reset_index(drop=True)
    args = dict()
    # cb3 = get_cb3_characteristic(__SECTEUR__, __BATIMENT__, x, args)
    ca3 = get_ca_characteristic(cb3['sector'].unique(), __BATIMENT__, cb3)
    print('Intrants completed for process: ', os.getpid())

    # Add cost intrants.
    cost_table = calcul_cout_batiment(cb3['sector'].unique(), __BATIMENT__, ca3, cost_params)
    print('Cost completed for process: ', os.getpid())

    # Get financials
    return calcul_detail_financier(cb3['sector'].unique(), __BATIMENT__, 120, cost_table, financials_params)


if __name__ == '__main__':

    start = time.time()

    myBook = xlrd.open_workbook(__FILES_NAME__)
    x = get_all_informations(myBook)
    cost_params = x[(x['type'].isin(['pcost'])) & (x['sector'] == 'Secteur 1')]
    finance_params = x[(x['type'].isin(['financial'])) & (x['sector'] == 'Secteur 1')]

    terrain_dev = get_land_informations()
    terr = terrain_dev.drop_duplicates(['sup_ter', 'denm_p', 'sector', 'vat', 'max_ne', 'min_ne']).reset_index(drop=True).head(250)
    intervall = np.array_split(terr.index, 16)
    params = ()
    params = data_for_simulation(data=terr,
                                 cost_params=cost_params,
                                 financials_params=finance_params)
    get_summary(params)

    # for value in intervall:
    #
    #     params += data_for_simulation(data=terr.loc[value, :],
    #                                  cost_params=cost_params,
    #                                  financials_params=finance_params),
    #
    # pool = multiprocessing.Pool(16)
    # result = pool.map(get_summary, params)
    # pool.close()
    # pool.join()
    #
    # end = time.time()
    #
    # print(end - start)

