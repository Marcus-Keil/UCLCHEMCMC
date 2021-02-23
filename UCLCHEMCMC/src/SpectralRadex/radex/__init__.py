from .radexwrap import *
from pandas import DataFrame
import numpy as np

def run(parameters,output_file=None):
	"""
	Run a single RADEX model

	Args:
		parameters: A dictionary containing the RADEX inputs that the user wishes to set,
		all other parameters will use the default values. See :func:``get_default_parameters`` 
		for a list of possible parameters.
		output_file: If not ``None``, the RADEX results are stored to this file in csv format/
	"""
	columns=['E_UP (K)','freq','WAVEL (um)','T_ex','tau',
			'T_R (K)','POP UP','POP LOW', 'FLUX (K*km/s)', 'FLUX (erg/cm2/s)']
	  
	nlines,qup,qlow,output = radex(parameters)
	qlistup = []
	for i in qup.reshape(-1, 6).view('S6'):
		if str(i, 'utf-8').strip() != "\x00\x00\x00\x00\x00\x00":
			qlistup += [str(i, 'utf-8').strip()]
	qlistup = np.asarray(qlistup)
	output = output[:, :nlines]
	output=DataFrame(columns=columns, data=output[:, :len(qlistup)].T)
	output["QN Upper"]=qup.reshape(-1, 6).view('S6')[:len(qlistup)]
	output["QN Lower"]=qlow.reshape(-1, 6).view('S6')[:len(qlistup)]
	output["QN Upper"]=output["QN Upper"].map(lambda x: str(x, 'utf-8')).str.strip()
	output["QN Lower"]=output["QN Lower"].map(lambda x: str(x, 'utf-8')).str.strip()
	output = output.drop(output[output["QN Upper"] == ""].index)
	if output_file is not None:
		output.to_csv(output_file, index=False)
	return output


def get_default_parameters():
	"""
	Get the default RADEX parameters as a dictionary, this largely serves as an example for the
	input required for :func:`run`.
	"""
	parameters={
		"molfile":"co.dat",
		"tkin":30.0,
		"tbg":2.73,
		"cdmol":1.0e13,
		"h2":1.0e5,
		"h":0.0,
		"e-":0.0,
		"p-h2":0.0,
		"o-h2":0.0,
		"h+":0.0,
		"linewidth":1.0,
		"fmin":0.0,
		"fmax":3.0e10
	}
	return parameters