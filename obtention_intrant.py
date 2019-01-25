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
            value = sh.cell(line, pos[1] + batiment).value
            value = 0 if value == "" else value
            _.append(value)
        tab.append(_)
    return tab


def get_global_intrant(myBook):

    sh = myBook.sheet_by_name(__INTRANT_SHEET__)
    table_of_intrant = []
    tab_of_intrant_pos = [[[6, 3], 'ntu', 's'], [[15, 3], 'nmu_et', 's'], [[42, 3], 'denm_pu', 's'],
                          [[51, 3], 'denm_p', 's'], [[60, 3], 'pptnu', 'ns'], [[69, 3], 'mp', 'ns'],
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
    _ = suptu_all.reset_index()[['Value', 'Secteur']]
    cir = 1 - table_of_intrant[(table_of_intrant['Value'] == 'cir')][__BATIMENT__].values
    cir = np.append(cir, cir, axis=0)
    suptubu_all = suptu_all.values / cir

    suptu_all = pd.DataFrame(suptu_all, columns=__BATIMENT__).reset_index()
    suptubu_all = pd.DataFrame(suptubu_all, columns=__BATIMENT__).reset_index()
    suptubu_all['Secteur'] = _['Secteur']
    suptubu_all['Value'] = _['Value'] + 'b'
    suptu_all['Value'] = _['Value']
    suptu_all['Categorie'] = 'ALL'
    suptubu_all['Categorie'] = 'ALL'
    suptu_all = suptu_all[entete]
    suptubu_all = suptubu_all[entete]

    table_of_intrant = pd.concat([table_of_intrant, suptu_all, suptubu_all], ignore_index=True)

    "superficie Brute par unite"
    x = table_of_intrant[(table_of_intrant['Value'] == 'suptub') & (table_of_intrant['Categorie'] == 'ALL')]
    _ = x[['Secteur', 'Categorie']].reset_index()

    x = x[__BATIMENT__].values
    ntu = table_of_intrant[(table_of_intrant['Value'] == 'ntu') & (table_of_intrant['Categorie'] == 'ALL')][
        __BATIMENT__].values
    nmu_et = table_of_intrant[(table_of_intrant['Value'] == 'nmu_et') & (table_of_intrant['Categorie'] == 'ALL')][
        __BATIMENT__].values
    escom = table_of_intrant[(table_of_intrant['Value'] == 'pp_et_escom') & (table_of_intrant['Categorie'] == 'ALL')][
        __BATIMENT__].values

    suptubu = x / ntu
    supetbu = suptubu * nmu_et
    supescom = supetbu * escom
    suptubu = pd.DataFrame(suptubu, columns=__BATIMENT__).reset_index()
    supetbu = pd.DataFrame(supetbu, columns=__BATIMENT__).reset_index()
    supescom = pd.DataFrame(supescom, columns=__BATIMENT__).reset_index()
    suptubu[['Secteur', 'Categorie']] = _[['Secteur', 'Categorie']]
    supetbu[['Secteur', 'Categorie']] = _[['Secteur', 'Categorie']]
    supescom[['Secteur', 'Categorie']] = _[['Secteur', 'Categorie']]
    supetbu['Value'] = 'supetbu'
    suptubu['Value'] = 'suptubu'
    supescom['Value'] = 'supescom'
    suptubu = suptubu[entete]
    supetbu = supetbu[entete]
    supescom = supescom[entete]
    table_of_intrant = pd.concat([table_of_intrant, suptubu, supetbu, supescom], ignore_index=True)

    x = table_of_intrant[((table_of_intrant['Value'] == 'suptub') | (table_of_intrant['Value'] == 'supt_cub') |
                          (table_of_intrant['Value'] == 'supescom')) & (table_of_intrant['Categorie'] == 'ALL')]

    x = pd.DataFrame(x[__BATIMENT__].groupby(x["Secteur"]).sum(), columns=__BATIMENT__).reset_index()
    x["Value"] = "supths"
    x["Categorie"] = "ALL"
    x = x[entete]
    table_of_intrant = pd.concat([table_of_intrant, x], ignore_index=True)

    "Calcul densite et superficie du terrain"
    _ = x[__BATIMENT__].values
    ntu = table_of_intrant[(table_of_intrant['Value'] == 'denm_p') & (table_of_intrant['Categorie'] == 'ALL')][
        __BATIMENT__].values

    supterrain = _ / ntu
    densite = _ / supterrain

    supterrain = pd.DataFrame(supterrain, columns=__BATIMENT__).reset_index()
    supterrain["Value"] = "supterrain"
    supterrain["Categorie"] = "ALL"
    supterrain['Secteur'] = x['Secteur']
    supterrain = supterrain[entete]
    table_of_intrant = pd.concat([table_of_intrant, supterrain], ignore_index=True)

    ntu = table_of_intrant[(table_of_intrant['Value'] == 'suptu')][['Categorie'] + __BATIMENT__]
    # print(ntu)
    ntu[__BATIMENT__] = ntu[__BATIMENT__].astype(float)
    ntu = ntu.groupby(['Categorie']).mean()
    # ntu = ntu.set_index('Categorie')

    for value in __UNITE_TYPE__:
        ntu.loc[value] = ntu.loc[value] / ntu.loc['ALL']

    ntu = ntu.reset_index()
    x = pd.DataFrame(ntu, columns=['Categorie'] + __BATIMENT__).reset_index()
    x["Value"] = "pptst"
    x["Secteur"] = "ALL"
    x = x[entete]
    # print(x)
    table_of_intrant = pd.concat([table_of_intrant, x], ignore_index=True)

    return table_of_intrant



if __name__ == '__main__':
    myBook = xlrd.open_workbook(__FILES_NAME__)
    get_global_intrant(myBook)
