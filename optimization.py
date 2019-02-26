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

    entete = ['type', 'sector', 'category', 'value'] + [batiment]
    table_intrant = table_intrant[entete]
    table_intrant.loc[table_intrant['value'] == 'vat', [batiment]] = vat

    cost_table = calcul_cout_batiment(table_intrant,  secteur, [batiment])
    fin_table = calcul_detail_financier(cost_table, secteur, [batiment],  120)

    r = fin_table.loc[fin_table[fin_table['value'] == to_optimize].index[0], batiment]
    r = 1000 if np.isnan(r) else r

    return (r, value)


def function_to_optimize(z, *params):

    r, value = get_financials_results(z, *params)
    return np.abs(value-r)



if __name__ == '__main__':

    myBook = xlrd.open_workbook(__FILES_NAME__)
    x = get_all_informations(myBook)
    args = dict()

    args['sup_ter'] = [[15000]]
    args['denm_p'] = [[3]]
    args['vat'] = [[100]]

    table_intrant = get_summary_characteristics('CA3', __SECTEUR__[6:], __BATIMENT__, x, args)

    base_value = get_financials_results(args['vat'][0][0], *params)


    rranges = (slice(0, 1000, 50),)

    params = (table_intrant, __SECTEUR__[6:], 'B8', 'TRI', 0.3)
    print(function_to_optimize(400, *params))

    resbrute = optimize.brute(function_to_optimize, ranges=rranges,args=params, full_output=True, finish=None)
    print(resbrute[0])
    print(resbrute[1])
