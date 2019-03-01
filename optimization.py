__author__ = 'pougomg'
from calcul_de_couts import calcul_cout_batiment
from calcul_financier import calcul_detail_financier
from obtention_intrant import get_all_informations, get_summary_characteristics
from scipy import optimize
import numpy as np
import xlrd
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



if __name__ == '__main__':

    myBook = xlrd.open_workbook(__FILES_NAME__)
    x = get_all_informations(myBook)
    args = dict()

    args['sup_ter'] = [[15000]]
    args['denm_p'] = [[3]]
    args['vat'] = [[85]]

    objectif = 0.15
    to_optimize = 'TRI'

    table_intrant = get_summary_characteristics('CA3', __SECTEUR__[6:], __BATIMENT__, x, args)
    # params = (table_intrant, __SECTEUR__[6:], __BATIMENT__, to_optimize, objectif)
    # base_value = get_financials_results(args['vat'][0][0], *params)
    #
    # batiment = base_value.astype(float).idxmax(skipna=True)
    # base_value = base_value[batiment]
    #
    # if base_value > objectif:
    #     rranges = (slice(args['vat'][0][0], 1000, 200),)
    #     params = (table_intrant, __SECTEUR__[6:], [batiment], to_optimize, objectif)
    #     resbrute = optimize.brute(function_to_optimize, ranges=rranges,args=params, full_output=True, finish=None)
    #
    #     rranges = (slice(resbrute[0]-200, resbrute[0] + 200, 50),)
    #     resbrute = optimize.brute(function_to_optimize, ranges=rranges,args=params, full_output=True, finish=None)
    #
    #     rranges = (slice(resbrute[0]-50, resbrute[0] + 50, 10),)
    #     resbrute = optimize.brute(function_to_optimize, ranges=rranges,args=params, full_output=True, finish=None)
    #
    # else:
    #     rranges = (slice(0, args['vat'][0][0], 200),)
    #     params = (table_intrant, __SECTEUR__[6:], [batiment], to_optimize, objectif)
    #     resbrute = optimize.brute(function_to_optimize, ranges=rranges,args=params, full_output=True, finish=None)



    rranges = (slice(0, 50, 1), slice(0, 50, 1), slice(0, 50, 1), slice(0, 50, 1))
    resbrute = optimize.brute(n, ranges=rranges, finish=None)
    print(resbrute[0])
    print(resbrute[1])