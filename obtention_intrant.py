import numpy as np
import pandas as pd
import xlrd

from lexique import __FILES_NAME__, __INTRANT_SHEET__, __BATIMENT__, __SECTEUR__, __UNITE_TYPE__


def ajouter_caraterisque_par_secteur(sh, tab, name, pos, unique):
    for secteur in __SECTEUR__:
        _ = [secteur, 'ALL', name]
        line = pos[0] if unique else __SECTEUR__.index(secteur) + pos[0]
        for batiment in range(len(__BATIMENT__)):
            _.append(sh.cell(line, pos[1] + batiment).value)
        tab.append(_)
    return tab


def ajouter_caraterisque_par_type_unite(sh, tab, name, pos, unique):
    for unite in __UNITE_TYPE__:
        _ = [unite, 'ALL', name]
        line = pos[0] if unique else __UNITE_TYPE__.index(unite) + pos[0]
        for batiment in range(len(__BATIMENT__)):
            _.append(sh.cell(line, pos[1] + batiment).value)
        tab.append(_)
    return tab


def get_global_intrant(myBook):
    sh = myBook.sheet_by_name(__INTRANT_SHEET__)
    table_of_intrant = []
    tab_of_intrant_pos = [[[6, 3], 'ntu', 's'], [[15, 3], 'nmu_et', 's'], [[42, 3], 'denm_pu', 's'],
                          [[51, 3], 'denm_p', 's'], [[69, 3], 'mp', 'ns'],
                          [[70, 3], 'min_nu', 'ns'], [[71, 3], 'max_nu', 'ns'], [[72, 3], 'min_ne', 'ns'],
                          [[73, 3], 'max_ne', 'ns'], [[74, 3], 'min_ne_ss', 'ns'], [[75, 3], 'max_ne_ss', 'ns'],
                          [[76, 3], 'cir', 'ns'],
                          [[77, 3], 'aec', 'ns'], [[78, 3], 'si', 'ns'], [[79, 3], 'pi_si', 'ns'],
                          [[80, 3], 'ee_ss', 'ns'],
                          [[82, 3], 'cub', 'ns'], [[83, 3], 'sup_cu', 'ns'], [[84, 3], 'supt_cu', 'ns'],
                          [[85, 3], 'pisc', 'ns'], [[87, 3], 'sup_pisc', 'ns'], [[88, 3], 'pp_sup_escom', 'ns'],
                          [[89, 3], 'pp_et_escom', 'ns'], [[90, 3], 'ss_sup_CES', 'ns'], [[91, 3], 'ss_sup_ter', 'ns'],
                          [[92, 3], 'nba', 'ns'], [[93, 3], 'min_max_asc', 'ns'], [[94, 3], 'tap', 'ns'],
                          [[24, 3], 'tum', 's'], [[33, 3], 'tuf', 's']]

    for value in tab_of_intrant_pos:

        if value[2] == 's':
            table_of_intrant = ajouter_caraterisque_par_secteur(sh, table_of_intrant, value[1], value[0], False)
        elif value[2] == 'ns':
            table_of_intrant = ajouter_caraterisque_par_secteur(sh, table_of_intrant, value[1], value[0], True)
        else:
            x = 0

    entete = ['Secteur', 'Categorie', 'Value'] + __BATIMENT__
    table_of_intrant = pd.DataFrame(table_of_intrant, columns=entete)

    # Ajouter nombre unites par type d'unites

    t = ajouter_caraterisque_par_type_unite(sh, [], 'pptu', [60, 3], False)
    t = pd.DataFrame(t, columns=entete)
    print(table_of_intrant.shape)
    x = table_of_intrant[table_of_intrant['Value'] == 'ntu'][__BATIMENT__].values
    res = [table_of_intrant]
    for type_batiment in __UNITE_TYPE__:
        _ = (x * t[t['Secteur'] == type_batiment][__BATIMENT__].values)

        column_name = ['ntu' for secteur in __SECTEUR__]
        _ = np.insert(_, 0, column_name, axis=1)
        column_name = [type_batiment for secteur in __SECTEUR__]
        _ = np.insert(_, 0, column_name, axis=1)
        column_name = [secteur for secteur in __SECTEUR__]
        _ = np.insert(_, 0, column_name, axis=1)

        res.append(pd.DataFrame(_, columns=entete))

    table_of_intrant = pd.concat(res, ignore_index=True)
    print(table_of_intrant.shape)

    # Ajout Superficie des unites

    tum = dict()
    for type in range(len(__UNITE_TYPE__)):
        line = 97 + type
        if type > 4:
            line += 2
        for col in range(3):
            tum[(__UNITE_TYPE__[type], sh.cell(96, col + 3).value)] = sh.cell(line, col + 3).value

    res = [table_of_intrant]

    x = table_of_intrant[table_of_intrant['Value'] == 'tum'][__BATIMENT__].values
    for type_batiment in __UNITE_TYPE__[0:5]:
        _ = np.copy(x)
        _[_ == 'Grande'] = tum[(type_batiment, 'Grande')]
        _[_ == 'Moyenne'] = tum[(type_batiment, 'Moyenne')]
        _[_ == 'Petite'] = tum[(type_batiment, 'Petite')]

        column_name = ['tum' for secteur in __SECTEUR__]
        _ = np.insert(_, 0, column_name, axis=1)
        column_name = [type_batiment for secteur in __SECTEUR__]
        _ = np.insert(_, 0, column_name, axis=1)
        column_name = [secteur for secteur in __SECTEUR__]
        _ = np.insert(_, 0, column_name, axis=1)

        res.append(pd.DataFrame(_, columns=entete))

    x = table_of_intrant[table_of_intrant['Value'] == 'tuf'][__BATIMENT__].values
    for type_batiment in __UNITE_TYPE__[5:]:
        _ = np.copy(x)
        _[_ == 'Grande'] = tum[(type_batiment, 'Grande')]
        _[_ == 'Moyenne'] = tum[(type_batiment, 'Moyenne')]
        _[_ == 'Petite'] = tum[(type_batiment, 'Petite')]

        column_name = ['tuf' for secteur in __SECTEUR__]
        _ = np.insert(_, 0, column_name, axis=1)
        column_name = [type_batiment for secteur in __SECTEUR__]
        _ = np.insert(_, 0, column_name, axis=1)
        column_name = [secteur for secteur in __SECTEUR__]
        _ = np.insert(_, 0, column_name, axis=1)

        res.append(pd.DataFrame(_, columns=entete))
    table_of_intrant = pd.concat(res, ignore_index=True)

    print(table_of_intrant.shape)

    # Calcul superfice totale unite, Superficie Brute

    res = [table_of_intrant]

    for type_batiment in __UNITE_TYPE__:
        ntu = table_of_intrant[(table_of_intrant['Value'] == 'ntu') & (table_of_intrant['Categorie'] == type_batiment)][
            __BATIMENT__].values
        if type_batiment in __UNITE_TYPE__[0:5]:
            tum = \
            table_of_intrant[(table_of_intrant['Value'] == 'tum') & (table_of_intrant['Categorie'] == type_batiment)][
                __BATIMENT__].values
        else:
            tum = \
            table_of_intrant[(table_of_intrant['Value'] == 'tuf') & (table_of_intrant['Categorie'] == type_batiment)][
                __BATIMENT__].values

        _ = ntu * tum
        column_name = ['suptu' for secteur in __SECTEUR__]
        _ = np.insert(_, 0, column_name, axis=1)
        column_name = [type_batiment for secteur in __SECTEUR__]
        _ = np.insert(_, 0, column_name, axis=1)
        column_name = [secteur for secteur in __SECTEUR__]
        _ = np.insert(_, 0, column_name, axis=1)

        res.append(pd.DataFrame(_, columns=entete))

    table_of_intrant = pd.concat(res, ignore_index=True)

    suptu_all = table_of_intrant[(table_of_intrant['Value'] == 'suptu') | (table_of_intrant['Value'] == 'supt_cu')][
        ['Value', 'Secteur'] + __BATIMENT__].groupby(['Value', 'Secteur']).sum()
    print(suptu_all)
    cir = 1 - table_of_intrant[(table_of_intrant['Value'] == 'cir')][__BATIMENT__].values
    cir = np.append(cir, cir, axis=0)
    suptubu_all = suptu_all.values / cir

    suptu_all = pd.DataFrame(suptu_all, columns=__BATIMENT__).reset_index()
    suptubu_all = pd.DataFrame(suptubu_all, columns=__BATIMENT__).reset_index()
    print('')
    print(suptubu_all)
    suptu_all['Value'] = 'suptu'
    suptu_all['Categorie'] = 'ALL'
    suptubu_all['Value'] = 'suptu'
    suptubu_all['Categorie'] = 'ALL'
    suptubu_all['Secteur'] = suptu_all['Secteur']
    suptu_all = suptu_all[entete]
    suptubu_all = suptubu_all[entete]

    table_of_intrant = pd.concat([table_of_intrant, suptu_all, suptubu_all], ignore_index=True)

    # print(x)


if __name__ == '__main__':
    myBook = xlrd.open_workbook(__FILES_NAME__)
    get_global_intrant(myBook)
