"""Cette """
import math

import numpy as np
import pandas as pd
import xlrd

from lexique import __COUT_SHEET__, __BATIMENT__, \
    __SECTEUR__, __UNITE_TYPE__


def ajouter_caraterisque_par_secteur(sh, tab, name, pos, unique):
    for secteur in __SECTEUR__:
        _ = [secteur, 'ALL', name]
        line = pos[0] if unique else __SECTEUR__.index(secteur) + pos[0]
        for batiment in range(len(__BATIMENT__)):
            value = sh.cell(line, pos[1] + batiment).value
            value = 0 if value == "" else value
            _.append(value)
        tab.append(_)
    return tab

def get_house_price(batim, secteur, myBook):
    """Cette fonction est utilise pour creer une table contenant les prix des batiments par secteur
    :parameter: myBook Table de donnees contenant tous les parametres.
    :return DataFrame de prix par batiment et par secteur"""

    tab = []
    for nu in range(len(__UNITE_TYPE__)):
        pr = myBook.sheet_by_name('Prix').cell(3 + 9*nu + __SECTEUR__.index(secteur), 2 + __BATIMENT__.index(batim)).value
        tab.append([__UNITE_TYPE__[nu], pr])
    return tab


def get_building_cost_parameter(myBook):

    sh = myBook.sheet_by_name(__COUT_SHEET__)
    tab_cost = []

    tab_params = [
        #Construction
        [[60, 4], 'tcq'], [[61, 4], 'tss'], [[62, 4], 'tfum'], [[63, 4], 'all_cuis'],[[64, 4], 'all_sdb'],
        [[65, 4], 'tvfac'], [[66, 4], 'asc'], [[67, 4], 'c_ad_pisc'], [[68, 4], 'c_ad_cu'], [[69, 4], 'c_ad_com'],
        [[70, 4], 'it']
        #Soft cost
        ,[[73, 4], 'apt_geo'], [[74, 4], 'prof'], [[75, 4], 'eval'], [[76, 4], 'legal_fee'], [[77, 4], 'prof_fee_div'],
        [[78, 4], 'pub'], [[79, 4], 'construction_permit'], [[80, 4], 'com'],

                  ]

    for value in tab_params:
        tab_cost = ajouter_caraterisque_par_secteur(sh, tab_cost, value[1], value[0], True)
    return pd.DataFrame(tab_cost, columns=['sector', 'category', 'value'] + __BATIMENT__)




def get_land_param(myBook):

    sh = myBook.sheet_by_name(__COUT_SHEET__)

    __VALEUR_PROX__ = 4
    tab_land_param = []
    entete = ['Secteur', 'Value'] + __BATIMENT__

    v = ['valeur prox', 'multi de densite', 'aug valeur', 'mutation',
              'cout add']

    for line in range(__VALEUR_PROX__, __VALEUR_PROX__ + 7):
        _ = [__SECTEUR__[line - 4], 'valeur prox']
        for batiment in __BATIMENT__:
            if line == __VALEUR_PROX__:
                value = sh.cell(line, 4).value
            else:
                value = sh.cell(line, 4).value + sh.cell(__VALEUR_PROX__, 4).value
            _.append(value)
        tab_land_param.append(_)

    t = np.ones((7, 9))
    mult_densite = sh.cell(13, 4).value
    mult_densite = t * mult_densite
    column_name = ['multi de densite' for secteur in __SECTEUR__]
    mult_densite = np.insert(mult_densite.astype(object), 0, column_name, axis=1)
    column_name = [secteur for secteur in __SECTEUR__]
    mult_densite = np.insert(mult_densite, 0, column_name, axis=1)

    valeur = sh.cell(14, 4).value
    valeur = t * valeur
    column_name = ['aug valeur' for secteur in __SECTEUR__]
    valeur = np.insert(valeur.astype(object), 0, column_name, axis=1)
    column_name = [secteur for secteur in __SECTEUR__]
    valeur = np.insert(valeur, 0, column_name, axis=1)
    tab_land_param = tab_land_param + mult_densite.tolist() + valeur.tolist()


    __FM__ = 17
    for line in range(__FM__, __FM__ + 7):
        _ = [__SECTEUR__[line - __FM__], 'mutation']
        for batiment in range(len(__BATIMENT__)):
            value = sh.cell(line, 4).value
            _.append(value)
        tab_land_param.append(_)

    __CR__ = 26
    for line in range(__CR__, __CR__ + 7):
        _ = [__SECTEUR__[line - __CR__], 'cout add']
        for batiment in range(len(__BATIMENT__)):
            value = sh.cell(line, batiment + 4).value
            _.append(value)
        tab_land_param.append(_)

    return pd.DataFrame(tab_land_param, columns=entete)

def get_nombre_unite(batim, secteur, myBook):

    tab = []
    for nu in range(len(__UNITE_TYPE__)):
        nbu = myBook.sheet_by_name('Intrants').cell(166 + 9*nu + __SECTEUR__.index(secteur), 3 + __BATIMENT__.index(batim)).value
        nbuecoul = myBook.sheet_by_name('Ecoulement').cell(4 + 9*nu + __SECTEUR__.index(secteur), 2 + __BATIMENT__.index(batim)).value
        tab.append([__UNITE_TYPE__[nu], nbu, nbuecoul, math.ceil(nbu/nbuecoul)])
    return tab
if __name__ == '__main__':

    myBook = xlrd.open_workbook('ville_MTL_templates.xlsx')
    # print(get_price_parameter(myBook))
    # get_building_cost_parameter(myBook)
    print(get_land_param(myBook))