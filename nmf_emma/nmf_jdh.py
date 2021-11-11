'''
Scripts for performing NMF end-member mixing following Shaughnessy et al. (2021)
'''

#import packages
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

from itertools import combinations
from numpy.linalg import eig
from sklearn.decomposition import NMF
from functools import partial
from multiprocessing import Pool

#MAKE THIS A MODULE IN MY CODE PACKAGE!
from copkmeans.cop_kmeans import cop_kmeans

#helper functions:

#define function to apply to each column
def _keep(x, n):
	'''
	Function for keeping only the n best-fitting results per sample

	Parameters
	----------
	x : pd.Series
		Each column of the unstacked sse matrix

	n : int
		Number of best-fitting results to keep for column x

	Returns
	-------
	tf : pd.Series
		Boolean series that is true where the fits should be kept and false 
		where they should be dropped.
	'''

	st = np.sort(x.dropna()) #drop nans and sort
	thresh = st[n] #get threshold
	tf = x <= thresh #find where less or equal to threshold

	return tf

#Step 1: Normalize data to analyte of interest and generate bootstrapped df
def normbs(df, nums, denom, nbs = 5000, logged = True):
	'''
	Normalizes input data, extracts relevant columns, and generates bootstrapped
	dataframe.

	Parameters
	----------
	df : pd.DataFrame
		DataFrame containing all data, with sample names/IDs as the index,
		and with columns containing ``nums'' and ``denom'' strings.

	nums : list
		List of strings containing the column names of each analyte of interest,
		excluding the denominator analyte.

	denom : str
		String of the column name of the denomenator analyte

	nbs : int
		Number of bootstrap datapoints to generate

	logged : boolean
		If True, uses log-normalized data when generating bootstrapped
		dataframe.

	Returns
	-------
	mdnorm : pd.DataFrame
		Resulting normalized dataframe of measured data, shape nxp. All data are
		element of (0,1).

	bdnorm : pd.DataFrame
		Resulting normalized dataframe of bootstrapped data, shape nbsxp. Data
		outside of the measured data range are dropped. All data are element of
		(0,1).

	colmax : pd.Series
		Series of column-wise maximum values, shape 1xp.
	'''

	n = len(df)
	m = len(nums)

	#generate new (normalized) column names
	ncs = [n+'_'+denom for n in nums]

	#extract numerator columns and divide by denominator
	md = df[nums]
	mdd = md.divide(df[denom], axis = 0)
	mdd.columns = ncs

	if logged:

		#calculate column-wise means and covariance matrix for logged data
		d = np.log10(mdd)
		cov = np.cov(d, rowvar = False)
		means = d.mean(axis = 0).values

	else:

		#calculate column-wise means and covariance matrix
		cov = np.cov(mdd, rowvar = False)
		means = mdd.mean(axis = 0).values

	#reseed random state and generate bootstrapped data
	# make matrix twice as long as needed to ensure enough points after dropping
	rng = np.random.RandomState(None) #seed from clock
	bs = rng.multivariate_normal(means, cov, size = 2*nbs)

	if logged:
		bs = 10**bs

	#get bootstrapped data into dataframe
	bsd = pd.DataFrame(bs, columns = ncs)

	#get maximum values for each column and normalize
	colmax = mdd.max()
	colmin = mdd.min()

	mdnorm = mdd.divide(colmax)
	bdnormm = bsd.divide(colmax)

	#remove bootstrapped points outside of range and only retain first nbs
	mask = (bsd <= colmax) & (bsd >= colmin)
	ind = np.where(mask.sum(axis = 1) == m)[0]

	#randomly select nbs points from the masked index array
	bsind = rng.choice(ind, size = nbs, replace = False)

	bdnorm = bdnormm.iloc[bsind,:]
	bdnorm = bdnorm.reset_index()
	bdnorm = bdnorm.drop('index', axis = 1) #drop old 'index' column

	return mdnorm, bdnorm, colmax

#Step 2: Determine number of end-members using PCA
def calc_nems(df, var_exp = 0.95, whiten = True):
	'''
	Function to determine the number of end-members needed to explain a given
	amount of variance in the data.

	Parameters
	----------
	df : pd.DataFrame
		Dataframe containing all data, where each row is a sample and each
		column is a variable. Shape nxm.

	var_exp : float
		The fraction of variance needed to be explained by the resulting model.
		Float element of (0,1).

	whiten : Boolean
		Tells the funciton whether or not to whiten the data (i.e., subtract
		column-wise means and divide by column-wise standard deviations).

	Returns
	-------
	nems : int
		Number of end-members needed.
	'''

	#whiten if needed
	if whiten:

		mus = df.mean(axis = 0)
		sigmas = df.std(axis = 0)

		X = (df - mus).divide(sigmas)

	else:

		X = df

	#calculate XTX matrix
	XTX = np.dot(X.T, X)

	#calculate eigenvectors and eigenvalues
	l, V = eig(XTX)

	#sort eigenvectors and eigenvalues
	i = np.argsort(l)[::-1]

	ls = l[i]
	Vs = V[:,i]

	#determine number of eigenvalues retained
	cve = np.cumsum(ls)/np.sum(ls)

	#add one for indexing, one for the fact that 2 is the minimum (for 1 PC)
	nems = np.where(cve >= var_exp)[0][0] + 2

	return nems

#Step 3: Bootstrap data, omitting values outside of measured range
# done in normbs function

#Step 4: Perform NMF to get W and H matrices
def nmf_emma(bdf, mdf, nems, sd = None, stuc_err = 0.05):
	'''
	Function to perform non-negative matrix factorization on streamwater data
	and return end-member compositions and samples for which the model meets
	the sum-to-unity constraint.

	Parameters
	----------
	bdf : pd.DataFrame
		Dataframe of scaled bootstrapped data, generated using ``normbs''. This
		is the data that is fit to generate the model. Shape nbs x m.

	mdf : pd.DataFrame
		Dataframe of scaled measured data, generated using ``normbs''. The model
		is then applied to this dataframe. Shape n x m.

	nems : int
		Number of end-members to consider. Can be estimated using ``calc_nems''.

	sd : None or int
		The seed to use for random state. Defaults to ``None''.

	stuc_err : float
		Sum-to-unity constraint error, as a fraction. For example, if `stuc_err'
		is set to 0.05, then samples in which the model meets the sum-to-unity
		constraint between 0.95 and 1.05 are retained. Defaults to ``0.05''.

	Returns
	-------
	fems : pd.DataFrame
		Dataframe containing fractional abundances of each end-member for each
		sample in which the sum-to-unity constraint was met. Shape ns x p, where
		ns is the number of solved samples.

	ems : pd.DataFrame
		Dataframe containing the composition of each end-member. Shape p x m.

	See Also
	--------
	nmf_emma_mc
		Function to perform ``nmf_emma'' iteratively (Monte Carlo solution).

	'''

	#make NMF model
	m = NMF(n_components = nems,
			init = 'random',
			solver = 'cd',
			random_state = sd,
			max_iter = 10000,
			tol = 1e-4,
			)

	#fit NMF model ("solve")
	mod = m.fit(bdf)

	#apply NMF model to measured data ("transform")
	W = mod.transform(mdf)
	H = mod.components_

	#extract samples where sum-to-unity constraint is within threshold range
	sow = W.sum(axis = 1)
	ind = np.where((sow >= 1 - stuc_err) & (sow <= 1 + stuc_err))[0]

	#get name lists for making dataframes
	enames = ['em_'+str(e+1) for e in range(nems)]
	cnames = mdf.columns

	#store as dataframes
	fems = pd.DataFrame(W[ind,:],
						index = ind,
						columns = enames,
						)

	ems = pd.DataFrame(H,
					   index = enames,
					   columns = cnames,
					   )

	return fems, ems

#Step 5: Store results for samples that meet sum-to-unity constraint
# done in nmf_emma function

#Step 6: Repeat steps 4-5 n_iter times
def nmf_emma_mc(bdf, mdf, nems, ni, stuc_err = 0.05):
	'''
	Function to perform Monte Carlo non-negative matrix factorization on
	streamwater data and return resulting fractional contributions and end-
	member compositions for each iteration. Uses multiprocessing to parallelize
	iteration solutions.

	Parameters
	----------
	bdf : pd.DataFrame
		Dataframe of scaled bootstrapped data, generated using ``normbs''. This
		is the data that is fit to generate the model. Shape nbs x m.

	mdf : pd.DataFrame
		Dataframe of scaled measured data, generated using ``normbs''. The model
		is then applied to this dataframe. Shape n x m.

	nems : int
		Number of end-members to consider. Can be estimated using ``calc_nems''.

	ni : int
		Number of iterations to perform.

	stuc_err : float
		Sum-to-unity constraint error, as a fraction. For example, if `stuc_err'
		is set to 0.05, then samples in which the model meets the sum-to-unity
		constraint between 0.95 and 1.05 are retained. Defaults to ``0.05''.

	Returns
	-------
	femsmc : pd.DataFrame
		Dataframe containing fractional abundances of each end-member for each
		sample in which the sum-to-unity constraint was met for each iteration.
		Returns dataframe with multi-index, with the first level as the 
		iteration number and the second level as the samples for which the sum-
		to-unity constraint was met for that iteration; contains p columns of
		fractional contribution of each end member.

	emsmc : pd.DataFrame
		Dataframe containing the composition of each end-member for each
		iteration. Returns dataframe with multi-index, with the first level as
		the iteration and the second level as the end-member. Contains p columns
		and ni*3 rows, where ni is the number of iterations that solved at least
		one sample within the sum-to-unity constraint.

	Notes
	-----
	End-members are *not* sorted! That is, the "split-rule" constraint has not
	yet been applied. So, for example, em_1 for iteration 1 may not be the same
	as em_1 for iteration 2.

	See Also
	--------
	nmf_emma
		Function to perform a single iteration.

	'''

	#make partial function
	f = partial(nmf_emma, bdf, mdf, nems, stuc_err = stuc_err)

	#seed uniquely each time
	rng = np.random.default_rng()
	s0 = int(rng.uniform(low=0, high=1e6, size=1))
	svec = np.arange(s0,s0+ni)

	#parallelize
	with Pool() as p: #with no arg, uses all available cores)
		il = p.map(f, svec)

		#necessary?
		p.close()
		p.join()

	#extract fems and ems
	fl, el = list(zip(*il))

	#add iteration number to each sub dataframe
	for i in range(ni):
		fl[i]['iter'] = i
		el[i]['iter'] = i

	#concatenate lists of dfs into overall df
	flist = pd.concat(fl)
	elist = pd.concat(el)

	#reindex flist by iteration
	ft = flist.reset_index()
	ft = ft.rename(columns = {'index':'sample'})
	ft = ft.set_index(['iter','sample'])
	femsmc = ft.sort_index(axis = 0)

	#reindex elist by iteration
	et = elist.reset_index()
	et = et.rename(columns = {'index':'em'})
	et = et.set_index(['iter','em'])

	#only keep em results that matched at least one sample
	emsmc = et.loc[femsmc.index.levels[0]]

	return femsmc, emsmc

#Step 7: Sort solutions by SSE and save best fitting 5% for each sample
def sse_keep(femsmc, emsmc, mdnorm, cutoff = 0.05):
	'''
	Sorts model results by their || model - data || fit and retains only the
	best-fitting models for each sample.

	Parameters
	----------
	femsmc : pd.DataFrame
		Dataframe containing fractional abundances of each end-member for each
		sample in which the sum-to-unity constraint was met for each iteration.
		Returns dataframe with multi-index, with the first level as the 
		iteration number and the second level as the samples for which the sum-
		to-unity constraint was met for that iteration; contains p columns of
		fractional contribution of each end member.

	emsmc : pd.DataFrame
		Dataframe containing the composition of each end-member for each
		iteration. Returns dataframe with multi-index, with the first level as
		the iteration and the second level as the end-member. Contains p columns
		and ni*3 rows, where ni is the number of iterations that solved at least
		one sample within the sum-to-unity constraint.

	mdnorm : pd.DataFrame
		Resulting normalized dataframe of measured data, shape nxp. All data are
		element of (0,1).

	cutoff : float
		The fraction of best-fitting models to keep; e.g., if ``cutoff = 0.05``,
		then the 5% of best fitting models for each sample are retained.
		Defaults to ``0.05``.

	Returns
	-------
	ssek : pd.Series
		Series of the sum of squared errors for all retained solutions. Length
		``ns''.

	femsk : pd.DataFrame
		Dataframe of the fractional contributions by each end-member for all
		retained solutions. Shape ``ns'' x ``nem''.

	emsk : pd.DataFrame
		Dataframe of the end-member compositions for all retained solutions.
		Shape ``3*nm'' x ``nsol''.

	nsave : pd.Series
		Series of the number of model solutions saved for each sample.

	'''

	#get fractional abundance matrix into the right shape
	#shape = ns x 3*nis, where ns is the total number of samples fit and nis is
	# the number of iterations that fit at least one sample.
	X = femsmc.reset_index()
	X['temp'] = X['iter']
	X = X.set_index(['temp','iter','sample'])
	X = X.unstack(0)
	X = X.swaplevel(axis=1)
	X = X.T.sort_index(level=0).T
	X = X.fillna(0) #make all NaNs zero

	#get design matrix, A
	#shape = 3*nis x na, where na is the number of analytes
	A = emsmc

	#calculate estimated analyte values, Bhat, as A*X = Bhat
	#shape = ns x na
	Bhat = X.dot(A)

	i,j = list(zip(*Bhat.index))

	#extract measured values
	B = mdnorm.loc[list(j)]
	B.index = Bhat.index

	#calculate sse (length = ns)
	sse = ((B-Bhat)**2).sum(axis=1)

	#for each sample, only retain best-fitting models

	#unstack into nis x ns df, where ns is the number of unique samples fit
	sseus = sse.unstack()

	#get total number of fits per sample
	nts = sseus.count()

	#get the number to be saved for each sample
	nsave = (cutoff*nts).astype(int)

	#pre-define boolean matrix of keep/not
	dfkeep = pd.DataFrame(index = sseus.index, columns = sseus.columns)

	#for each sample, save n best fitting models
	for col, n in zip(sseus, nsave):
		dfkeep[col] = _keep(sseus[col], n)

	#restack into series with heirarchical index
	ssek = sseus[dfkeep].stack()

	#extract values to keep from femsms and emsmc
	femsk = femsmc.loc[ssek.index]

	ik = ssek.index.get_level_values(0)
	emsk = emsmc.loc[set(ik)].sort_index()

	return ssek, femsk, emsk, nsave+1

#Step 8: Ensure end-members are in same order using cop-kmeans ("split rule")
def splitrule_sort(femsk, emsk):
	'''
	Function for sorting the "split rule", i.e., for dealing with the fact that
	ems can be reported in different order

	Parameters
	----------
	femsk : pd.DataFrame
		Dataframe of the fractional contributions by each end-member for all
		retained solutions. Shape ``ns'' x ``nem''.

	emsk : pd.DataFrame
		Dataframe of the end-member compositions for all retained solutions.
		Shape ``3*nm'' x ``nsol''.


	Returns
	-------
	femsks : pd.DataFrame
		Sorted version of femsk, such that each end-member is now consistent
		across all samples.
		

	emsks : pd.DataFrame
		Sorted version of emsk, such that each end-member is now consistent
		across all samples.
	'''

	#extract values from inputs
	ns, nems = np.shape(femsk)
	tnm, nsol = np.shape(emsk)
	nm = int(tnm/3) #number of models

	#get list of rows within each model
	mrs = [np.arange(i,i+nems) for i in np.arange(0,tnm,nems)]

	#now get all combinations within each model
	pl = [list(combinations(m,2)) for m in mrs]
	cl = [p for em in pl for p in em]

	#perform constrained kmeans clustering
	clus, cent = cop_kmeans(
		dataset = emsk.values, 
		k = nems, 
		cl = cl
		)

	#save cluster results into series
	sclus = pd.Series(clus, index = emsk.index)

	#get into shape of femsk
	sclusu = sclus.unstack() #unstack
	t = pd.Series(index = femsk.index,
		dtype = int,
		name = 't'
		) #dummy series 

	#get ordered indices
	sinds = sclusu.join(t, how = 'inner')[sclusu.columns] #ordered indices

	#sort each row and save as new df of sorted femsk
	srtd = [femsk.iloc[i,sinds.iloc[i,:]].values for i in range(ns)]
	femsks = pd.DataFrame(
		srtd, 
		index = femsk.index, 
		columns = femsk.columns
		)

	#now sort emsk
	emstrs = ['em_'+str(i+1) for i in clus] #sorted em indices
	emsks = emsk.reset_index()
	emsks['em'] = emstrs #replace with sorted strings
	emsks = emsks.set_index(['iter','em']) #get back to multiindex
	emsks = emsks.sort_index(level = 1).sort_index(level = 0) #reset em sorting


	return femsks, emsks

#Step 9: Take mean and std. dev. of w and h results for each sample
def summary(femsks, emsks):
	'''
	Function to take sorted results and summarize them into a single dataframe
	with mean, std. dev., count.

	Parameters
	----------
	femsks : pd.DataFrame
		Sorted version of femsk, such that each end-member is now consistent
		across all samples.
		

	emsks : pd.DataFrame
		Sorted version of emsk, such that each end-member is now consistent
		across all samples.

	Returns
	-------
	sum_table : pd.DataFrame
		Summary table, with each row being a single sample. Columns are the
		fractional contributions, end-member compositions, and model solution
		count for each sample, including means and std. devs.
	'''

	#pre-allocate dataframe
	sams = femsks.index.levels[1]
	fcols = ['f_' + e + ms \
		for ms in ['_mean','_std'] \
		for e in femsks.columns]

	ccols = [e +'_'+ s + ms \
		for ms in ['_mean','_std'] \
		for e in femsks.columns \
		for s in emsks.columns]

	cols = fcols + ccols + ['count']

	#get constants
	nf = len(fcols)
	nc = len(ccols)

	sum_table = pd.DataFrame(index = sams, columns = cols)

	#populate fractional contributions
	fgr = femsks.groupby('sample')
	fres = fgr.mean().join(fgr.std(),lsuffix = '_mean', rsuffix = '_std')

	sum_table.iloc[:,:nf] = fres

	#populate end member compositions
	emus = emsks.unstack().sort_index(axis = 1,level = 1)
	ecols = [i[0]+'_'+i[1] for i in emus.columns]
	emus.columns = ecols

	#project onto index with samples for each iteration
	emus = emus.join(femsks,how='inner')[ecols]

	#now groupby and project
	egr = emus.groupby('sample')
	eres = egr.mean().join(egr.std(),lsuffix = '_mean', rsuffix = '_std')

	sum_table.iloc[:,nf:nf+nc] = eres

	#finally, add counts
	cts = fgr.count().mean(axis=1).astype(int)

	sum_table['count'] = cts

	return sum_table


if __name__ == "__main__":

	print('success')

	# tic = time.time()
	# with ThreadPoolExecutor(max_workers = 30) as executor:

	# 	for i in range(100):
	# 		executor.submit(nmf_emma, bdf, mdf, 3, sd = i)

	# toc = time.time() - tic
	# print(toc)



	# tic = time.time()
	# for i in range(100):
	# 	res = nmf_emma(bdf,mdf,3,sd=i)

	# toc = time.time() - tic
	# print(toc)


	# f = partial(nmf_emma, bdf, mdf, nems, stuc_err = stuc_err)

	# #something like this (but it bugs out right now)
 #    with Pool(5) as p:
 #        print(p.map(f, [1, 2, 3]))


	# def pt(a,b,c=5,d=10):

	# 	print('a = {}'.format(a))
	# 	print('b = {}'.format(b))
	# 	print('c = {}'.format(c))
	# 	print('d = {}'.format(d))

	# 	return a

	# f = partial(pt, 2,3, d=5)

	# with Pool(5) as p:
	# 	p.map(f, range(5))


# tic = time.time()
# iterlist = nmf_iter(bdnorm,mdnorm,nems,1000)
# toc = time.time()
# print('time = %.2f' % toc-tic)


	# #get matrix for each iteration
	# A = emsmc.loc[0].unstack()
	# x = femsmc.loc[0]
	# Bhat = np.dot(x,A)
	# B = mdnorm.loc[x.index]
	# se = (B - Bhat)**2
	# sse = se.sum(axis=1)



	# #TESTING CODE
	# # i,j = list(zip(*femsmc.index))
	
	# #reshape femsmc
	# # X = femsmc.unstack(0)
	# # X = X.swaplevel(axis = 1)

	# # #now reorder columns
	# # X = X.T.sort_index(level=0).T
