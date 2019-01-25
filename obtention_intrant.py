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


def convert_unity_type_to_sector(group, data):
        df = data[__BATIMENT__].mul(group[__BATIMENT__].values.tolist()[0] , axis=1)
        df['category'] = group.name
        df['sector'] = data['sector']
        df['value'] = data['value']
        return df


def split_unity_type_to_sector(group , data):
    return data


def get_surface(group, dict_of_surface):

    tab = [group[batiment].map(dict_of_surface[group.name]).values.tolist() for batiment in __BATIMENT__]
    tab = np.array(tab).transpose()
    tab = pd.DataFrame(tab, columns=__BATIMENT__)
    tab['category'] = group.name
    group = group.reset_index()
    tab['sector'] = group['sector']
    tab['value'] = group['value']

    return tab

def calculate_total_surface(group):

    df = group[group['value'] == 'ntu'][__BATIMENT__].reset_index(drop=True) * \
         group[group['value'] == 'tum'][__BATIMENT__].reset_index(drop=True)
    df['category'] = group.name
    df['sector'] = group[group['value'] == 'ntu']['sector'].reset_index(drop=True)
    df['value'] = 'suptu'

    return df

def get_cb1_characteristics(workbook) -> pd.DataFrame:

    """
    This function takes all the parameters of the Intrants sheets and calculate all the characteristics
    of the CB1 sheet.

    :param workbook: Excel Workbook containing all the parameters
    :return: pd.Dataframe of all the characteristics of the CB1 sheet.

    """""

    #Open Intrants sheet and take all the importants parameters
    sh = workbook.sheet_by_name(__INTRANT_SHEET__)
    table_of_intrant = []

    #Table containing the position of all the intrants in the Intrants sheets. Refers to the lexique files
    #to get the definition of the variables.
    tab_of_intrant_pos = [[[10, 3], 'ntu', 's'], [[19, 3], 'nmu_et', 's'], [[55, 3], 'denm_pu', 's'],
                          [[64, 3], 'denm_p', 's'],[[118, 3], 'mp', 'ns'],
                          [[119, 3], 'min_nu', 'ns'], [[120, 3], 'max_nu', 'ns'], [[121, 3], 'min_ne', 'ns'],
                          [[122, 3], 'max_ne', 'ns'], [[123, 3], 'min_ne_ss', 'ns'], [[124, 3], 'max_ne_ss', 'ns'],
                          [[125, 3], 'cir', 'ns'],
                          [[126, 3], 'aec', 'ns'], [[127, 3], 'si', 'ns'], [[128, 3], 'pi_si', 'ns'],
                          [[129, 3], 'ee_ss', 'ns'],
                          [[131, 3], 'cub', 'ns'], [[132, 3], 'sup_cu', 'ns'], [[133, 3], 'supt_cu', 'ns'],
                          [[134, 3], 'pisc', 'ns'], [[136, 3], 'sup_pisc', 'ns'], [[137, 3], 'pp_sup_escom', 'ns'],
                          [[138, 3], 'pp_et_escom', 'ns'], [[139, 3], 'ss_sup_CES', 'ns'], [[140, 3], 'ss_sup_ter', 'ns'],
                          [[141, 3], 'nba', 'ns'], [[142, 3], 'min_max_asc', 'ns'], [[143, 3], 'tap', 'ns'],
                          [[37, 3], 'tum', 's'], [[46, 3], 'tuf', 's']]

    for value in tab_of_intrant_pos:

        if value[2] == 's':
            table_of_intrant = ajouter_caraterisque_par_secteur(sh, table_of_intrant, value[1], value[0], False)
        elif value[2] == 'ns':
            table_of_intrant = ajouter_caraterisque_par_secteur(sh, table_of_intrant, value[1], value[0], True)
        else:
            x = 0

    entete = ['sector', 'category', 'value'] + __BATIMENT__
    table_of_intrant = pd.DataFrame(table_of_intrant, columns=entete)

    #TODO : Add the pptu given the scenarios

    #Add number of unity by unity type
    t = ajouter_caraterisque_par_type_unite(sh, [], 'pptu', [100, 3], False)
    t = pd.DataFrame(t, columns=entete)
    ntu = table_of_intrant[table_of_intrant['value'] == 'ntu'][__BATIMENT__ + ['sector', 'value']]
    result = t.groupby('sector').apply(convert_unity_type_to_sector, ntu).reset_index(drop=True)
    result = result[entete]
    table_of_intrant = pd.concat([table_of_intrant, result], ignore_index=True)

    #Add unity Superficie

    #Get surface for small, medium and bug unity
    dict_of_surface = dict()
    for type in range(len(__UNITE_TYPE__)):
        d = dict()
        line = 146 + type
        if type > 4:
            line += 2
        for col in range(3):
            d[sh.cell(145, col + 3).value] = sh.cell(line, col + 3).value
        dict_of_surface[__UNITE_TYPE__[type]] = d

    tum = table_of_intrant[table_of_intrant['value'] == 'tum']
    t = pd.DataFrame(__UNITE_TYPE__, columns=['category'])
    result = t.groupby('category').apply(split_unity_type_to_sector ,tum[__BATIMENT__ + ['sector', 'value']]).reset_index()
    result = result[entete]
    result = result.groupby('category').apply(get_surface, dict_of_surface).reset_index(drop=True)
    result = result[entete]
    table_of_intrant = table_of_intrant[table_of_intrant['value'] != 'tum']
    table_of_intrant = pd.concat([table_of_intrant, result], ignore_index=True)

    #Total Surface by unity type
    tum = table_of_intrant[(table_of_intrant['value'] == 'tum')|(table_of_intrant['value'] == 'ntu')&(table_of_intrant['category'] != 'ALL')]
    result = tum.groupby('category').apply(calculate_total_surface).reset_index(drop=True)
    result = result[entete]
    table_of_intrant = pd.concat([table_of_intrant, result], ignore_index=True)

    cir = table_of_intrant[(table_of_intrant['value'] == 'cir')&(table_of_intrant['category'] == 'ALL')][__BATIMENT__].reset_index(drop=True)

    print(cir)
    result = result[__BATIMENT__].groupby(result['sector']).sum().reset_index()
    supbut = result[__BATIMENT__]/(1-cir)
    result['category'] = 'ALL'
    result['value'] = 'suptu'
    result = result[entete]

    supbut['category'] = 'ALL'
    supbut['value'] = 'supbtu'
    supbut['sector'] = result['sector']
    supbut = supbut[entete]

    #Calculate Brute Surface for common area
    supt_cu = table_of_intrant[(table_of_intrant['value'] == 'supt_cu')&(table_of_intrant['category'] == 'ALL')][__BATIMENT__].reset_index(drop=True)
    supt_cu = supt_cu/(1 - cir)
    supt_cu['category'] = 'ALL'
    supt_cu['value'] = 'supbt_cu'
    supt_cu['sector'] = supbut['sector']
    supt_cu = supt_cu[entete]

    print(supt_cu)
    table_of_intrant = pd.concat([table_of_intrant, result, supbut, supt_cu], ignore_index=True)

    return
    # def get_surface(group, dict_of_surface_by_sector):

    print(tum.apply(lambda x: get_surface(x, dict_of_surface), axis = 1))
    print(x)
    return
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
    get_cb1_characteristics(myBook)
