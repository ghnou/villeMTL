__author__ = 'pougomg'
import xlrd
import pandas as pd
import numpy as np
from import_parameter import __FILES_NAME__,__BATIMENT__, __SECTEUR__, get_land_param, get_building_cost_parameter


def Calcul_prix_terrain(batim, secteur, ensemble, myBook):

    terrain_param = get_land_param(myBook)
    entete = terrain_param[0]
    terrain_param = terrain_param[1]
    terrain_param = pd.DataFrame(terrain_param, columns=entete)
    terrain_param = terrain_param[(terrain_param['Secteur'] == secteur) & (terrain_param['Batiment'] == batim)]

    __DENSITE__ = [55, 3]
    ligne_pos =  __SECTEUR__.index(secteur)
    ligne_pos = ligne_pos - 1 if ligne_pos > 0 else ligne_pos
    densite = myBook.sheet_by_name('Intrants').cell(__DENSITE__[0] + ligne_pos ,
                                                    __DENSITE__[1] + __BATIMENT__.index(batim)).value

    __SUP__ = [28, 3]
    superficie = myBook.sheet_by_name('Intrants').cell(__SUP__[0] + __SECTEUR__.index(secteur),
                                                    __SUP__[1] + __BATIMENT__.index(batim)).value


    terrain_param['cout'] = (1 + terrain_param['aug valeur'])*np.exp(terrain_param['valeur prox'] +
                                                                     terrain_param['multi de densite']*densite)*superficie + terrain_param['mutation']
    terrain_param = terrain_param.reset_index()
    return terrain_param['cout'].values

def Calcul_cout_batiment(batim, secteur, ensemble, quality, myBook):

    cost_param = get_building_cost_parameter(myBook)
    entete = cost_param[0]
    cost_param = pd.DataFrame(cost_param[1], columns=entete)
    cost_param = cost_param[(cost_param["Batiment"] == batim) & (cost_param["quality"] == quality)].reset_index()
    print(cost_param)

    tab_of_mesure = []
    __SUP_HS__ = [400, 3]
    sup_hs = myBook.sheet_by_name('Intrants').cell(__SUP_HS__[0] + __SECTEUR__.index(secteur) ,
                                                    __SUP_HS__[1] + __BATIMENT__.index(batim)).value
    tab_of_mesure.append(sup_hs)

    __SUP_TU__ = [355, 3]
    sup_tu = myBook.sheet_by_name('Intrants').cell(__SUP_TU__[0] + __SECTEUR__.index(secteur) ,
                                                    __SUP_TU__[1] + __BATIMENT__.index(batim)).value
    tab_of_mesure.append(sup_tu)

    __SallE__ = [[193, 3], [202, 3], [211, 3], [220, 3]]
    sup_ea = 0

    for value in __SallE__:
        sup_ea += myBook.sheet_by_name('Intrants').cell(value[0] + __SECTEUR__.index(secteur) ,
                                                    value[1] + __BATIMENT__.index(batim)).value

    tab_of_mesure.append(sup_ea)

    __SallGC__ = [[193, 3], [202, 3], [211, 3], [220, 3]]
    sup_gc = 0

    for value in __SallGC__:
        sup_gc += myBook.sheet_by_name('Intrants').cell(value[0] + __SECTEUR__.index(secteur) ,
                                                    value[1] + __BATIMENT__.index(batim)).value

    tab_of_mesure.append(sup_gc)

    sup_ac = myBook.sheet_by_name('Intrants').cell(364+ __SECTEUR__.index(secteur) ,
                                                    3 + __BATIMENT__.index(batim)).value * myBook.sheet_by_name('Intrants').cell(114 ,3 + __BATIMENT__.index(batim)).value

    sup_ac += myBook.sheet_by_name('Intrants').cell(391+ __SECTEUR__.index(secteur) ,
                                                    3 + __BATIMENT__.index(batim)).value * myBook.sheet_by_name('Intrants').cell(114 ,3 + __BATIMENT__.index(batim)).value

    tab_of_mesure.append(sup_ac)
    tab_of_mesure.append(1)

    if myBook.sheet_by_name('Intrants').cell(123 ,3 + __BATIMENT__.index(batim)).value == 'Non':
        tab_of_mesure.append(0)
    else:
        tab_of_mesure.append(1)

    if myBook.sheet_by_name('Intrants').cell(120, 3 + __BATIMENT__.index(batim)).value == 'Non':
        tab_of_mesure.append(0)
    else:
        tab_of_mesure.append(1)


    cad_esp_com = myBook.sheet_by_name('Intrants').cell(391+ __SECTEUR__.index(secteur) ,
                                                    3 + __BATIMENT__.index(batim)).value *(1- myBook.sheet_by_name('Intrants').cell(114 ,3 + __BATIMENT__.index(batim)).value)


    tab_of_mesure.append(cad_esp_com)

    tab_of_mesure.append( myBook.sheet_by_name('Intrants').cell(292 + __SECTEUR__.index(secteur) ,
                                                    3 + __BATIMENT__.index(batim)).value)

    tab_of_mesure.append( myBook.sheet_by_name('Intrants').cell(238 + __SECTEUR__.index(secteur) ,
                                                    3 + __BATIMENT__.index(batim)).value)

    tab_of_mesure.append( myBook.sheet_by_name('Intrants').cell(247 + __SECTEUR__.index(secteur) ,
                                                    3 + __BATIMENT__.index(batim)).value +
                          myBook.sheet_by_name('Intrants').cell(337 + __SECTEUR__.index(secteur) ,
                                                    3 + __BATIMENT__.index(batim)).value)

    tab_of_mesure.append( myBook.sheet_by_name('Intrants').cell(256 + __SECTEUR__.index(secteur) ,
                                                    3 + __BATIMENT__.index(batim)).value +
                          myBook.sheet_by_name('Intrants').cell(346 + __SECTEUR__.index(secteur) ,
                                                    3 + __BATIMENT__.index(batim)).value)

    tab_of_mesure.append( myBook.sheet_by_name('Intrants').cell(265 + __SECTEUR__.index(secteur) ,
                                                    3 + __BATIMENT__.index(batim)).value)
    for v in range(0 , len(entete)):
        print(v, entete[v])
    entete_cout_construction = [entete[1], entete[2], entete[3], entete[4], entete[5], entete[6], entete[7], entete[8],
                                entete[9], entete[24],entete[25],entete[26],entete[27],entete[28]]

    print(len(entete_cout_construction))

    tab = cost_param[entete_cout_construction].values
    print(tab)
    print(tab_of_mesure)
    result = tab * tab_of_mesure

    for r in result:
        for t in r:
            print(t)
    print(len(entete_cout_construction))





if __name__ == '__main__':

    myBook = xlrd.open_workbook(__FILES_NAME__)
    print(Calcul_cout_batiment(__BATIMENT__[7], __SECTEUR__[4],0,"Base", myBook))
    # for v in __SECTEUR__:
    #     for ba in __BATIMENT__:
    #         Calcul_prix_terrain(ba, v,0,myBook)

