import numpy as np

def downsample(x, delta, axis=0):
	if len(x.shape)==1:
		return x[::delta]
	else:
		if axis == 0:
			return x[::delta, :]
		elif axis ==1:
			return x[:,::delta]