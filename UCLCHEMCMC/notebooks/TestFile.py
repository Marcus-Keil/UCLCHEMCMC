import SpectralRadex.radex as radex

radexDic = {'molfile': './SpectralRadex/radex/data/ch-h.dat',
            'tkin': 8.0,
            'tbg': 2.73,
            'cdmol': 101870398119.48259,
            'h2': 651002.7148529647,
            'h': 0.0,
            'e-': 2.6159532012430126e-09,
            'p-h2': 0.0,
            'o-h2': 0.0,
            'h+': 2.6159532012430126e-09,
            'linewidth': 1.0,
            'fmin': 0.0,
            'fmax': 30000000000.0}
#radexDic['p-h2'] = 1.0e-80
#radexDic['o-h2'] = 1.0e-80
dataframe = radex.run(radexDic)
print("Hello")
print(dataframe)

