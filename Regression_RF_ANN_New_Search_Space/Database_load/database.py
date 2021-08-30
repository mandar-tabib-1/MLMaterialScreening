import pymatgen as pm
import numpy as np
import json
import os, sys
from os.path import isfile


scalars_of_interest = ['atomic_mass', 'electrical_resistivity', 'atomic_radius', 'X']
groups_of_interest  = ['group', 'row']
size_of_these_groups= [    18,      9]
shell_order = ['1s', '2s', '2p', '3s', '3p', '4s', '3d', '4p', '5s', '4d', '5p', '6s', '4f', '5d', '6p', '7s', '5f', '6d', '7p']
tensor_postfix      = ['xx', 'yy', 'zz', 'xy', 'xz', 'yx', 'yz', 'zx', 'zy']

def get_atom_values(elements):
    # initialize all extracted properties
    result = {s : [] for s in scalars_of_interest}
    result['valence_electrons'] = []
    result['electrons'] = []
    for vector in groups_of_interest:
        result[vector] = []

    for symbol, amount in elements.items():
        element = pm.Element(symbol)
        for scalar in scalars_of_interest:
            try:
                result[scalar] += [float(getattr(element, scalar))]*int(amount)
            except ValueError: # typically descriptive string values
                result[scalar] += [np.nan]*int(amount)
            except TypeError:  # None gives this error when passed to float()
                result[scalar] += [np.nan]*int(amount)

        electrons = element.full_electronic_structure # [(1, 's', 2), (2, 's', 2), (2, 'p', 6), ... ]
        outer_shell = max([val[0] for val in electrons])
        result['valence_electrons'] += [sum([val[2] for val in electrons if val[0]==outer_shell])]*int(amount)

        orbits = [0]*len(shell_order)
        for i in range(len(shell_order)):
            for val in electrons:
                if(val[0]==int(shell_order[i][0]) and val[1]==shell_order[i][1]):
                    orbits[i] += val[2]
        result['electrons'].append( orbits )

        for vector, size in zip(groups_of_interest, size_of_these_groups):
            result[vector].append( [amount if i==getattr(element, vector)  else 0 for i in range(1,size+1)] )

    return result


def read_database(filename):
    # open database
    with open('db.json') as db_file:
        json_struct = json.load(db_file)

    results = []
    err = []
    maxerr = 0
    for material in os.listdir():
        if not material[:3] == 'mp-':
            continue
        sigma  = np.loadtxt(material+'/t4me/output/sigma')
        kappae = np.loadtxt(material+'/t4me/output/kappae')
        seebeck= np.loadtxt(material+'/t4me/output/seebeck')
        try:
            zt300  = np.loadtxt(material+'/t4me/output/zt-300.txt')
            print(np.min(zt300[:,1]), np.max(zt300[:,1]))
        except IOError:
            zt300 = None
        sigma  = sigma.reshape(  (15,51,13))
        kappae = kappae.reshape( (15,51,13))
        seebeck= seebeck.reshape((15,51,13))
        compound = [c for c in json_struct if c['material_id']==material]
        compound = compound[0]

        print('Processing {} ({})'.format(compound['pretty_formula'],material))

        # split compound into single elements and fetch information
        all_atoms = get_atom_values(compound['unit_cell_formula'])

        temp = range(100,850,50)
        for T in range(15):
            for j in range(51):
                temperature           = temp[T]
                eV                    = sigma[T,j,0]
                eta                   = sigma[T,j,1]
                n_type_carrier_concen = sigma[T,j,2]
                p_type_carrier_concen = sigma[T,j,3]
                # zt                    = (seebeck[T,j,4:].mean()**2*sigma[T,j,4:].mean()*temp[T] / (1+kappae[T,j,4:].mean()))
                zt                    = (seebeck[T,j,4]**2*sigma[T,j,4]*temp[T] / (1+kappae[T,j,4]))
                zt = zt*1e-12
                # if zt != 0:
                    # zt = 1/zt
                if(T==4 and zt300 is not None):
                    err.append(abs(zt - zt300[j,1]))
                    if abs(zt - zt300[j,1]) > maxerr:
                        maxerr =  abs(zt - zt300[j,1])
                        print('  maxerr', maxerr)
                        print('  zt (compute)', zt)
                        print('  zt (listed) ', zt300[j,1])

                # extract output variables (computed compound properties)
                data = {'density'       : compound['density'],
                        'e_above_hull'  : compound['e_above_hull'],
                        'symmetry_group': compound['spacegroup']['number'],
                        'crystal'       : compound['spacegroup']['crystal_system'],
                        'band_gap'      : compound['band_gap'],
                        'full_formula'  : compound['full_formula'],
                        'pretty_formula': compound['pretty_formula'],
                        'material_id'   : compound['material_id'],
                        'temperature'   : temperature,
                        'eV'            : eV,
                        'eta'           : eta,
                        'n_type_carrier': n_type_carrier_concen,
                        'p_type_carrier': p_type_carrier_concen,
                        'zt'            : zt}

                # extract "nice" scalars of interest
                for s in scalars_of_interest:
                    data[s + ' variance'] = np.var(all_atoms[s])
                    data[s + ' mean']     = np.mean(all_atoms[s])

                # extract "nice" vectors of interest
                for (s,v) in zip(groups_of_interest, size_of_these_groups):
                    data = {**data, **{'Total '+s+' #'+str(i):np.sum([vec_result[i-1] for vec_result in all_atoms[s]]) for i in range(1, v+1)}}

                # extract special-tailored scalars
                data['valence_electrons' + ' variance'] = np.var( all_atoms['valence_electrons'])
                data['valence_electrons' + ' mean']     = np.mean(all_atoms['valence_electrons'])
                data['n_units_in_cell']                 = sum([amount for (symbol, amount) in compound['unit_cell_formula'].items()])

                # extract special-tailored vectors
                for (s,i) in zip(shell_order, range(len(shell_order))):
                    data['Total '+s+' electrons'] = np.sum( [vec_result[i] for vec_result in all_atoms['electrons']] )

                # extract tensor values
                for k in range(9):
                    data['Sigma_'+tensor_postfix[k]]   = sigma[T,j,k+4]
                    data['Kappae_'+tensor_postfix[k]]  = kappae[T,j,k+4]
                    data['Seebeck_'+tensor_postfix[k]] = seebeck[T,j,k+4]

                results.append(data)

    print('Min error: ', np.min(err))
    print('Max error: ', np.max(err))
    print('Avg error: ', np.mean(err))
    print('Std error: ', np.std(err))
    return results

def get_order():
    # organize output in a logical way
    order = ['pretty_formula', 'full_formula', 'material_id', 'temperature', 'eV', 'n_units_in_cell', 'atomic_mass variance', 'atomic_mass mean', 'electrical_resistivity variance', 'electrical_resistivity mean', 'atomic_radius variance', 'atomic_radius mean', 'X variance', 'X mean', 'valence_electrons variance', 'valence_electrons mean']
    for (s,v) in zip(groups_of_interest, size_of_these_groups):
        order += ['Total ' + s + ' #'+str(i) for i in range(1,v+1)]
    for s in shell_order:
        order.append('Total '+s+' electrons')
    order.append('density'       )
    order.append('e_above_hull'  )
    order.append('symmetry_group')
    order.append('crystal')
    order.append('band_gap'      )
    for k in range(9):
        order.append('Sigma_'+tensor_postfix[k])
    for k in range(9):
        order.append('Kappae_'+tensor_postfix[k])
    for k in range(9):
        order.append('Seebeck_'+tensor_postfix[k])
    order.append('zt')

    return order


if __name__ == "__main__":
    # read databse from json file
    results = read_database('db.json')
    print('Number of rows:   ', len(results))
    print('Number of colums: ', len(results[0]))

    # write results to a csv file
    print('Writing results to initial_db.csv')
    with open('initial_db.csv', 'w') as fp:
        for name in get_order():
            fp.write(name+',')
        fp.write('\n')
        for compound in results:
            for name in get_order():
                fp.write(str(compound[name])+',')
            fp.write('\n')

