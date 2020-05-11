import numpy             as     np 
from   .tools            import *

def covgc_time(X, dt, lag, t0):

    """
    [GC, pairs] = covGC_time(X, dt, lag, t0)
    Computes single-trials covariance-based Granger Causality for gaussian variables
    X   = data arranged as sources x timesamples
    dt  = duration of the time window for covariance correlation in samples
    lag = number of samples for the lag within each trial
    t0  = zero time in samples
    GC  = Granger Causality arranged as (number of pairs) x (3 directionalities (pair(:,1)->pair(:,2), pair(:,2)->pair(:,1), instantaneous))
    pairs = indices of sources arranged as number of pairs x 2
    -------------------- Total Granger interdependence ----------------------
    Total Granger interdependence:
    TGI = GC(x,y)
    TGI = sum(GC,2):
    TGI = GC(x->y) + GC(y->x) + GC(x.y)
    TGI = GC(x->y) + GC(y->x) + GC(x.y) = Hycy + Hxcx - Hxxcyy
    This quantity can be defined as the Increment of Total
    Interdependence and it can be calculated from the different of two
    mutual informations as follows
    ----- Relations between Mutual Informarion and conditional entropies ----
    % I(X_i+1,X_i|Y_i+1,Y_i) = H(X_i+1) + H(Y_i+1) - H(X_i+1,Y_i+1)
    Ixxyy   = log(det_xi1) + log(det_yi1) - log(det_xyi1);
    % I(X_i|Y_i) = H(X_i) + H(Y_i) - H(X_i, Y_i)
    Ixy     = log(det_xi) + log(det_yi) - log(det_yxi);
    ITI(np) = Ixxyy - Ixy;
    Reference
    Brovelli A, Chicharro D, Badier JM, Wang H, Jirsa V (2015)
    Copyright of Andrea Brovelli (Jan 2015) - Matlab version -
    Copyright of Andrea Brovelli & Michele Allegra (Jan 2020) - Python version -
    """

    X = np.array(X)

    dt = int(dt)
    t0 = int(t0)

    # Data parameters. Size = sources x time points
    nSo, nTi = X.shape

    ind_t = (t0 - lag) * np.ones((dt, lag+1))

    for i in range(dt):
        for j in range(lag+1):
            ind_t[i, j] = ind_t[i, j] + i + (lag-j)

    ind_t = ind_t.astype(int)

    # Pairs between sources
    nPairs = nSo * (nSo-1)/2
    nPairs = np.int(nPairs)

    # Init
    GC = np.zeros((nPairs, 3))

    pairs = np.zeros((nPairs, 2))

    # Normalisation coefficient for gaussian entropy
    C = np.log(2*np.pi*np.exp(1))

    # Loop over number of pairs
    cc = 0

    for i in range(nSo):
        for j in range(i+1, nSo):

            # Define pairs of channels
            pairs[cc, 0] = i
            pairs[cc, 1] = j


            # Extract data for a given pair of sources
            x = X[i, ind_t]
            y = X[j, ind_t]

            # ---------------------------------------------------------------------
            # Conditional Entropies
            # ---------------------------------------------------------------------
            # Hycy: H(Y_i+1|Y_i) = H(Y_i+1) - H(Y_i)
            det_yi1 = rdet(np.cov(y.T))
            det_yi = rdet(np.cov(y[:, 1:].T))
            Hycy = np.log(det_yi1) - np.log(det_yi)
            # Hxcx: H(X_i+1|X_i) = H(X_i+1) - H(X_i)
            det_xi1 = rdet(np.cov(x.T))
            det_xi = rdet(np.cov(x[:, 1:].T))
            Hxcx = np.log(det_xi1) - np.log(det_xi)
            # Hycx: H(Y_i+1|X_i,Y_i) = H(Y_i+1,X_i,Y_i) - H(X_i,Y_i)
            det_yxi1 = rdet(np.cov(np.column_stack((y, x[:, 1:])).T))
            det_yxi = rdet(np.cov(np.column_stack((y[:, 1:], x[:, 1:])).T))
            Hycx = np.log(det_yxi1) - np.log(det_yxi)
            # Hxcy: H(X_i+1|X_i,Y_i) = H(X_i+1,X_i,Y_i) - H(X_i,Y_i)
            det_xyi1 = rdet(np.cov(np.column_stack((x, y[:, 1:])).T))
            Hxcy = np.log(det_xyi1) - np.log(det_yxi)
            # Hxxcyy: H(X_i+1,Y_i+1|X_i,Y_i) = H(X_i+1,Y_i+1,X_i,Y_i) - H(X_i,Y_i)
            det_xyi1 = rdet(np.cov(np.column_stack((x, y)).T))
            Hxxcyy = np.log(det_xyi1) - np.log(det_yxi)

            # ---------------------------------------------------------------------
            # Granger Causality measures
            # ---------------------------------------------------------------------
            # GC(pairs(:,1)->pairs(:,2))
            GC[cc, 0] = Hycy - Hycx
            # GC(pairs(:,2)->pairs(:,1))
            GC[cc, 1] = Hxcx - Hxcy
            # GC(x.y)
            GC[cc, 2] = Hycx + Hxcy - Hxxcyy

            cc = cc + 1


    return GC#np.column_stack((pairs, GC))