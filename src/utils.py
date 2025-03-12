import numpy as np

def read_modeparams(fname):
    """Read mode parameters from pkb file
    
    """
    data = np.loadtxt(fname)
    # enn, ell, nu, signu-, signu+, h, sigh-, sigh+, w, sigw-, sigw+, i, sigi-, sigi+
    enn = data[:, 0]
    ell = data[:, 1]
    nu  = data[:, 2]
    height = data[:, 5]
    width  = data[:, 8]
    incl   = data[:, 11]
    sig_nu = 0.5*(data[:, 4] - data[:, 3])
    sig_h  = 0.5*(data[:, 7] - data[:, 6])
    sig_w  = 0.5*(data[:, 10] - data[:, 9])
    sig_i  = 0.5*(data[:, 13] - data[:, 12])
    mode_dict = {'ell':ell, 
                 'enn':enn,
                 'nu': nu,
                 'height': height,
                 'width': width,
                 'incl': incl,
                 'signu': sig_nu,
                 'sigh' : sig_h,
                 'sigw' : sig_w,
                 'sigi' : sig_i,}
    return mode_dict


def read_a2z(fname):
    def combine_GAf_data(gamma, amp, freq, ampl):
        def remove_duplicates(data):
            enn = data[:, 0]
            ell = data[:, 1]
            new_data = []
            for _ell in np.unique(ell):
                maskell = ell==_ell
                uniqn = np.unique(enn[maskell])
                for _enn in uniqn:
                    try:
                        idx = np.where(maskell*(enn==_enn))[0][0]
                        new_data.append([_enn, _ell, *data[idx, 2:]])
                    except IndexError:
                        continue
            return np.array(new_data)
        
        newdata = []
        enns = np.unique([*freq[:, 0].astype('int'), *amp[:, 0].astype('int')])
        ells = freq[:, 1].astype('int')
        for ell in ells:
            _ampl = ampl[ell, 1]
            for enn in enns:
                maskfreq = (freq[:, 0]==enn)*(freq[:, 1]==ell)
                maskamp  = (amp[:, 0]==enn)
                maskgam  = (gamma[:, 0]==enn)
                if maskfreq.sum()==1 and maskamp.sum()==1 and maskgam.sum()==1:
                    _newdata = [enn, ell, *freq[maskfreq, 2],  *(_ampl*amp[maskamp, 2]), *gamma[maskgam, 2]]
                    newdata.append(_newdata)
                else:
                    continue
        newdata = np.array(newdata)
        newdata = remove_duplicates(newdata)
        return newdata


    # Initialize an empty list to store the parsed data
    data = []

    # Open the file and read it line by line
    with open(fname, 'r') as file:
        for line in file:
            # Split the line into components based on spaces
            components = line.strip().split()

            # Convert the components to the appropriate data types
            row = [
                components[0],  # First column (integer)
                components[1],  # Second column (integer)
                components[2],       # Third column (string)
                components[3],       # Fourth column (string)
                float(components[4]),  # Fifth column (float)
                float(components[5]),  # Sixth column (float)
                float(components[6]),  # Seventh column (float)
                float(components[7]),  # Eighth column (float)
                float(components[8])   # Ninth column (float)
            ]

            # Append the parsed row to the data list
            data.append(row)

    # Now `data` contains all the parsed rows
    freq_data = []
    height_data = []
    width_data = []
    ampl_data = []

    for row in data:
        if row[2]=='freq':
            freq_data.append([int(row[0]), int(row[1]), *row[4:]])
        elif row[2]=='height':
            height_data.append([int(row[0]), 0, *row[4:]])
        elif row[2]=='width':
            width_data.append([int(row[0]), 0, *row[4:]])
        elif row[2]=='amp_l':
            ampl_data.append([int(row[1]), float(row[4])])
    freq_data = np.array(freq_data)
    height_data = np.array(height_data)
    width_data = np.array(width_data)
    ampl_data = np.array(ampl_data)

    mode_data = combine_GAf_data(width_data, height_data, freq_data, ampl_data)
    mode_cols = ['enn', 'ell', 'freq', 'amp', 'gamma']
    return mode_data, mode_cols



def read_bgparams(fname):
    with open(fname) as f:
        data = f.read().splitlines()
    
    param = []
    for _data in data[1:]:
        param.append(float(_data.split(' ')[0]))

    # converting from muHz to Hz
    muHz_to_Hz = 1e-6
    param[0] *= 4
    param[1] *= muHz_to_Hz
    param[4] *= muHz_to_Hz
    param[7] *= muHz_to_Hz
    param[8] *= muHz_to_Hz
    return np.array(param)
