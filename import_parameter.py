"""Cette """
import math

import numpy as np
import pandas as pd
import xlrd

__FILES_NAME__ = 'ville_MTL_templates.xlsx'
__BATIMENT__ = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9']
__SECTEUR__ = ['Secteur 1', 'Secteur 2', 'Secteur 3', 'Secteur 4', 'Secteur 5', 'Secteur 6', 'Secteur 7']
__UNITE_TYPE__ = ['Studios', '1cc', '2cc', '3cc', 'Penthouse', '2cc fam', '3cc fam']
__COUT_SHEET__ = 'Cout'
__PRIX_SHEET__ = 'Prix'


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
    entete = []

    __CONSTRUCTION_LINE__ = 40
    entete.append('Batiment')
    tab_construction = [[[41, 4], 'tcq'], [[42, 4], 'tfu'], [[43, 4], 'ca_se'], [[44, 4], 'ca_gc'], [[45, 4], 'tfac'],
                        [[46, 4], 'asc'], [[47, 4], 'ca_asc_pi'], [[48, 4], 'ca_asc_cu'],
                        [[49, 4], 'ca_esc_b'], [[50, 4], 'it'], [[67, 4], 'tx'], [[53, 4], 'aptgeo'], [[54, 4], 'pai']
        , [[55, 4], 'ev'], [[56, 4], 'fl'], [[57, 4], 'fp'], [[58, 4], 'pub'], [[59, 4], 'pco'], [[60, 4], 'fpa']
        , [[61, 4], 'cad'], [[62, 4], 'afc'], [[63, 4], 'clv']]

    for value in tab_construction:
        tab_cost = ajouter_caraterisque_par_secteur(sh, tab_cost, value[1], value[0], True)

    return pd.DataFrame(tab_cost, columns=['Secteur', 'Categorie', 'Value'] + __BATIMENT__)

    for col in range(4, 13):
        tab = [__BATIMENT__[col - 4]]
        for line in range(__CONSTRUCTION_LINE__ + 1, __CONSTRUCTION_LINE__ + 11):
            value = sh.cell(line, col).value
            value = 0 if value == '' else value
            tab.append(value)
            if col == 4:
                entete.append(sh.cell(line, 2).value)
        tab_cost.append(tab)

    __SOFT_COST__ = 52

    for col in range(4, 13):
        for line in range(__SOFT_COST__ + 1, __SOFT_COST__ + 12):
            value = sh.cell(line, col).value
            value = 0 if value == '' else value
            tab_cost[col-4].append(value)
            if col == 4:
                entete.append(sh.cell(line, 2).value)

    __TAXES__ = 67

    for col in range(4, 13):
        value = sh.cell(__TAXES__, col).value
        value = 0 if value == '' else value
        tab_cost[col-4].append(value)

    entete.append('Taxes')
    tab_total_cost = []
    entete.append('quality')

    __COUT_ADDITIONNEL__ = 72
    __QUALITE_BATIMENT__ = ['Base', 'Moyenne', 'Elevee']

    for line in range(__COUT_ADDITIONNEL__ + 2, __COUT_ADDITIONNEL__ + 5):
        quality = [__QUALITE_BATIMENT__[line - __COUT_ADDITIONNEL__ - 2]]

        for col in range(4, 9):
            value = sh.cell(line, col).value + sh.cell(73, col).value
            value = 0 if value == '' else value
            quality.append(value)
            if line == __COUT_ADDITIONNEL__ + 2:
                entete.append(sh.cell(71, col).value)

        for value in tab_cost:
            tab_total_cost.append(value + quality)

    return pd.DataFrame(tab_total_cost, columns=entete)


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