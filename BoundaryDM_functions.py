# %%
import numpy as np
import scipy
import time
import matplotlib.pyplot as plt
#__________________________________________________________________________________________________________
# FUNCTIONS FOR TUNING OF PARAMETERS: epsilon (DM, TGL, SymmetricTGL) and boundary_mask (TGL, Symmetric TGL)
#___________________________________________________________________________________________________________
# Contents: 
#       auto_tune_epsilon,
#       auto_tune_epsilon_boundary,
#       auto_tune_epsilon_TGL
#       auto_tune_epsilon_TGL_mask
#       auto_tune_epsilon_SymmetricTGL
#       auto_tune_epsilon_SymmetricTGL_mask
#____________________________________________________________________________________________________________

#                                       /// auto_tune_epsilon  ///

#returns: epsilon, dimension <---------- epsilon is optimal parameter over a search 
#                                        dimension is estimated dimension of manifold 
#accepts:
#       X: numpy array of size (num_data_points, ambient_dimension)
#       n_neighbors: number of neighbors to construct kernel 
#       lower_range|upper_range|num_epsilon : parameters specifying grid to search for optimal epsilon

def auto_tune_epsilon(X, n_neighbors, lower_range = .0001, upper_range = 1, num_epsilons =  100, verbose = False):
    
    N = X.shape[0]                                      #number of datapoints
    if verbose:
        print("auto tuning begin")
        start = time.time()


    X_tree = scipy.spatial.KDTree(X)
    dist, ind = X_tree.query(x = X,  k = n_neighbors) # get indices and distance of nearest neighbors
   
    #Grid for parameter search, derivative for maximizing
    epsilon_grid = np.linspace(lower_range, upper_range, num_epsilons)
    derivative = np.zeros((num_epsilons,))
    #Search grid
    for epsilon_index in range(num_epsilons-2):

        #Current and next S_eps and epsilon for taking derivatives
        epsilon = epsilon_grid[epsilon_index]
        epsilon_next = epsilon_grid[epsilon_index+1] 
        S_eps = 0
        S_eps_next = 0

        #Sum all elements of Kernel for epsilon and epsilon_next
        for i in range(N):
            S_eps += (np.exp( ( -1/(4*epsilon) ) * dist[i,:]**2 )).sum()
            S_eps_next += (np.exp( ( -1/(4*epsilon_next) ) * dist[i,:]**2 )).sum()

        #Normalize above sum properly       
        S_eps = (1/N)*(1/n_neighbors)*S_eps
        S_eps_next = (1/N)*(1/n_neighbors)*S_eps_next
        
        #Compute derivative 
        derivative[epsilon_index] = ( np.log(S_eps_next) - np.log(S_eps) ) /  (np.log(epsilon_next) - np.log(epsilon) )


    #Find optimal epsilon and approximate dimension 
    epsilon = epsilon_grid[derivative.argmax()]
    dimension = 2 * derivative.max()
     

    if verbose:
        end = time.time()
        print(f"auto-tuning finished. Total time:{end-start}")
        print(f"estimated dimension:{dimension}")
        print(f"final choice of epsilon:{epsilon}")
        fig,ax = plt.subplots()
        ax.plot(epsilon_grid, derivative)
    
    return epsilon, dimension

#__________________________________________________________________________________________________________

#                               /// auto_tune_epsilon_boundary  ///

# returns: epsilon, boundary_mask <---- epsilon is a float. Automatic value for parameter found over search
#                                       boundary_mask is numpy array of shape (num_data_points,). boundary_mask[i] = True iff 
#                                       the point X[i,:] is considered INTERIOR
# accepts: 
#       X:      numpy array of size (num_data_points, ambient_dimension)
#       n_neighbors:        number of neighbors to constuct kernel 
#       truncation:         bool indicating if truncation is to be performed
#       manual_boundary_mask:       bool indicating if boundary_mask is automatic or manually specified 
#       boundary_mask:      None or boolean array of size (num_data_points,). See returing value
#       num_truncation points: how many points to truncate
#       lower_range | upper_range | num_epsilons:       specify grid to search for optimal epsilon

def auto_tune_epsilon_boundary(X, n_neighbors, truncation, manual_boundary_mask = False, boundary_mask = None, num_truncation_points = 'auto',
                                lower_range = 2**(-8), upper_range = 2**(-1), num_epsilons = 10, verbose = False):
    N = X.shape[0]
    if verbose:
        start = time.time()
        print('auto tuning begin')

  
    # Compute boundary mask if needed 
    if truncation and (not manual_boundary_mask):
        Kernel, Moment = construct_kernel_moment(X, epsilon = 'auto', n_neighbors = n_neighbors)
        boundary_mask = construct_boundary_mask(Moment = Moment, N_data_points = N, 
                                                num_truncation_points = num_truncation_points)
    
    #minimize semi_group_error over epsilon_grid
    epsilon_grid = np.linspace(lower_range, upper_range, num_epsilons)
    semi_group_error = np.zeros((num_epsilons,))

    for i,epsilon in enumerate(epsilon_grid):

        Kernel_1, Moment_1 = construct_kernel_moment(X, epsilon, n_neighbors)
        Kernel_1 = normalize_kernel(Kernel_1, truncation = truncation, boundary_mask=boundary_mask)
        Kernel_1 = Kernel_1 @ Kernel_1

        Kernel_2, Moment_2 = construct_kernel_moment(X, 2*epsilon, n_neighbors)
        Kernel_2 = normalize_kernel(Kernel_2, truncation = truncation, boundary_mask=boundary_mask)

        semi_group_error[i] = ( (Kernel_1 - Kernel_2) * (Kernel_1 - Kernel_2) ).sum()
    
    epsilon = epsilon_grid[np.argmin(semi_group_error)]

    if verbose:
        end = time.time()
        print(f"training finished. Total time:{end-start}\n")
        print(f"Selected epsilon = {epsilon}")
        fig, ax = plt.subplots()
        ax.scatter(epsilon_grid, semi_group_error)

    return epsilon

#__________________________________________________________________________________________________________

#                                   /// auto_tune_epsilon_TGL  ///

# returns: epsilon, boundary_mask <---- epsilon is a float. Automatic value for parameter found over search
#                                       boundary_mask is numpy array of shape (num_data_points,). boundary_mask[i] = True iff 
#                                       the point X[i,:] is considered INTERIOR
# accepts: 
#       X:      numpy array of size (num_data_points, ambient_dimension)
#       n_neighbors:        number of neighbors to constuct kernel 
#       num_truncation points: how many points to truncate
#       lower_range | upper_range | num_epsilons:       specify grid to search for optimal epsilon

# This function reconstructs the boundary_mask for each epsilon in the grid.

def auto_tune_epsilon_TGL(X, n_neighbors, num_truncation_points = 'auto',
                                lower_range = 2**(-8), upper_range = 2**(-1), num_epsilons = 10, verbose = False):
    N = X.shape[0]
    if verbose:
        start = time.time()
        print('auto tuning begin. TGL only.')


    epsilon_grid = np.linspace(lower_range, upper_range, num_epsilons)
    semi_group_error = np.zeros((num_epsilons,))

    for i,epsilon in enumerate(epsilon_grid):
        #constructs boundary mask for each epsilon
        Kernel_1, Moment_1 = construct_kernel_moment(X, epsilon, n_neighbors)
        
        boundary_mask = construct_boundary_mask(Moment_1, N_data_points = N, 
                                                  num_truncation_points = num_truncation_points)
        Kernel_1 = normalize_kernel(Kernel_1, truncation = True, boundary_mask=boundary_mask)
        Kernel_1 = Kernel_1 @ Kernel_1


        Kernel_2, Moment_2 = construct_kernel_moment(X, 2*epsilon, n_neighbors)
        Kernel_2 = normalize_kernel(Kernel_2, truncation = True, boundary_mask=boundary_mask)


        semi_group_error[i] = ( (Kernel_1 - Kernel_2) * (Kernel_1 - Kernel_2) ).sum()
    
    epsilon = epsilon_grid[np.argmin(semi_group_error)]

    #reconstuct boundary mask for optimal epsilon
    Kernel, Moment = construct_kernel_moment(X, epsilon = epsilon, n_neighbors = n_neighbors)
    boundary_mask = construct_boundary_mask(Moment, N_data_points = N, 
                                                  num_truncation_points = num_truncation_points)

    if verbose:
        end = time.time()
        print(f"training finished. Total time:{end-start}\n")
        print(f"Selected epsilon = {epsilon}")
        fig, ax = plt.subplots()
        ax.scatter(epsilon_grid, semi_group_error)

    return epsilon, boundary_mask

#_____________________________________________________________________________________________

#                           /// auto_tune_epsilon_TGL_mask  ///

# returns: epsilon <---- epsilon is a float. Automatic value for parameter found over search

# accepts: 
#       X:      numpy array of size (num_data_points, ambient_dimension)
#       n_neighbors:        number of neighbors to constuct kernel 
#       boundary_mask:      boolean array of size (num_data_points,) where boundary_mask[i] = True iff
#                           X[i,:] is considered an INTERIOR point                            
#       lower_range | upper_range | num_epsilons:       specify grid to search for optimal epsilon

def auto_tune_epsilon_TGL_mask(X, n_neighbors, boundary_mask,
                                lower_range = 2**(-8), upper_range = 2**(-1), num_epsilons = 10, verbose = False):
    N = X.shape[0]

    if verbose:
        start = time.time()
        print('auto tuning begin. TGL only. Manual boundary mask.')

  
    epsilon_grid = np.linspace(lower_range, upper_range, num_epsilons)
    semi_group_error = np.zeros((num_epsilons,))
    # construct semi_group_error
    for i,epsilon in enumerate(epsilon_grid):
        
        #does not reconstuct boundary_mask each time - uses given boundary_mask

        Kernel_1, Moment_1 = construct_kernel_moment(X, epsilon, n_neighbors)
        Kernel_1 = normalize_kernel(Kernel_1, truncation = True, boundary_mask=boundary_mask)
        Kernel_1 = Kernel_1 @ Kernel_1

        Kernel_2, Moment_2 = construct_kernel_moment(X, 2*epsilon, n_neighbors)
        Kernel_2 = normalize_kernel(Kernel_2, truncation = True, boundary_mask=boundary_mask)


        semi_group_error[i] = ( (Kernel_1 - Kernel_2) * (Kernel_1 - Kernel_2) ).sum()

 
    epsilon = epsilon_grid[np.argmin(semi_group_error)]     #epsilon minimizing semi-group error


    if verbose:
        end = time.time()
        print(f"training finished. Total time:{end-start}\n")
        print(f"Selected epsilon = {epsilon}")
        fig, ax = plt.subplots()
        ax.scatter(epsilon_grid, semi_group_error)

    return epsilon


#_________________________________________________________________________________________

#                                  /// auto_tune_epsilon_SymmetricTGL  ///

#Analogue of auto_tune_epsilon_TGL but for SymmetricTGL (i.e., loss function is different)

# returns: epsilon, boundary_mask <---- epsilon is a float. Automatic value for parameter found over search
#                                       boundary_mask is numpy array of shape (num_data_points,). boundary_mask[i] = True iff 
#                                       the point X[i,:] is considered INTERIOR
# accepts: 
#       X:      numpy array of size (num_data_points, ambient_dimension)
#       n_neighbors:        number of neighbors to constuct kernel 
#       num_truncation points: how many points to truncate
#       lower_range | upper_range | num_epsilons:       specify grid to search for optimal epsilon

# This function reconstructs the boundary_mask for each epsilon in the grid.

def auto_tune_epsilon_SymmetricTGL(X, n_neighbors, num_truncation_points = 'auto',
                                lower_range = 2**(-8), upper_range = 2**(-1), num_epsilons = 10, verbose = False):
    N = X.shape[0]
    if verbose:
        start = time.time()
        print('auto tuning begin. TGL only.')

    epsilon_grid = np.linspace(lower_range, upper_range, num_epsilons)
    semi_group_error = np.zeros((num_epsilons,))

    #compute semi_group_error as a function of epsilon
    for i,epsilon in enumerate(epsilon_grid):
        
        Kernel_1, Moment_1 = construct_kernel_moment(X, epsilon, n_neighbors)
        
        boundary_mask = construct_boundary_mask(Moment_1, N_data_points = N, 
                                                  num_truncation_points = num_truncation_points)
        
        Kernel_1 = normalize_kernel(Kernel_1, truncation = True, boundary_mask=boundary_mask)
        Kernel_1 = .5 * (Kernel_1 + Kernel_1.transpose())       # <--- Symmetrized Kernel
        Kernel_1 = Kernel_1 @ Kernel_1


        Kernel_2, Moment_2 = construct_kernel_moment(X, 2*epsilon, n_neighbors)
        Kernel_2 = normalize_kernel(Kernel_2, truncation = True, boundary_mask=boundary_mask)
        Kernel_2 = .5 * (Kernel_2 + Kernel_2.transpose())       #<--- Symmetrized Kernel


        semi_group_error[i] = ( (Kernel_1 - Kernel_2) * (Kernel_1 - Kernel_2) ).sum()
    
    #optimize over epsilon
    epsilon = epsilon_grid[np.argmin(semi_group_error)]

    #reconstuct boundary mask with optimal epsilon
    Kernel, Moment = construct_kernel_moment(X, epsilon = epsilon, n_neighbors = n_neighbors)
    boundary_mask = construct_boundary_mask(Moment, N_data_points = N, 
                                                  num_truncation_points = num_truncation_points)

    if verbose:
        end = time.time()
        print(f"training finished. Total time:{end-start}\n")
        print(f"Selected epsilon = {epsilon}")
        fig, ax = plt.subplots()
        ax.scatter(epsilon_grid, semi_group_error)

    return epsilon, boundary_mask


#_________________________________________________________________________________________

#                               /// auto_tune_epsilon_SymmetricTGL_mask  ///

# This is an analogue of auto_tune_epsilon_TGL_mask but for SymmetricTGL

# returns: epsilon <---- epsilon is a float. Automatic value for parameter found over search

# accepts: 
#       X:      numpy array of size (num_data_points, ambient_dimension)
#       n_neighbors:        number of neighbors to constuct kernel 
#       boundary_mask:      boolean array of size (num_data_points,) where boundary_mask[i] = True iff
#                           X[i,:] is considered an INTERIOR point                            
#       lower_range | upper_range | num_epsilons:       specify grid to search for optimal epsilon

def auto_tune_epsilon_SymmetricTGL_mask(X, n_neighbors, boundary_mask,
                                lower_range = 2**(-8), upper_range = 2**(-1), num_epsilons = 10, verbose = False):
    #initilize data/grid for optimizing
    N = X.shape[0]
    epsilon_grid = np.linspace(lower_range, upper_range, num_epsilons)
    semi_group_error = np.zeros((num_epsilons,))

    if verbose:
        start = time.time()
        print('auto tuning begin. TGL only. Manual boundary mask.')

    #Compute semi-group-error for each epsilon
    for i,epsilon in enumerate(epsilon_grid):
        
        Kernel_1, Moment_1 = construct_kernel_moment(X, epsilon, n_neighbors)
        Kernel_1 = normalize_kernel(Kernel_1, truncation = True, boundary_mask=boundary_mask)
        Kernel_1 = Kernel_1 @ Kernel_1
        Kernel_1 = .5 * (Kernel_1 + Kernel_1.tranpose())        #<---- Uses symmetrized kernel

        Kernel_2, Moment_2 = construct_kernel_moment(X, 2*epsilon, n_neighbors)
        Kernel_2 = normalize_kernel(Kernel_2, truncation = True, boundary_mask=boundary_mask)
        Kernel_2 = .5 * (Kernel_2 + Kernel_2.transpose())       #<---- Uses symmetrized kernel


        semi_group_error[i] = ( (Kernel_1 - Kernel_2) * (Kernel_1 - Kernel_2) ).sum()
    
    epsilon = epsilon_grid[np.argmin(semi_group_error)]


    if verbose:
        end = time.time()
        print(f"training finished. Total time:{end-start}\n")
        print(f"Selected epsilon = {epsilon}")
        fig, ax = plt.subplots()
        ax.scatter(epsilon_grid, semi_group_error)

    return epsilon
#__________________________________________________________________________________________________________
#__________________________________________________________________________________________________________
#__________________________________________________________________________________________________________
# FUNCTIONS FOR CONSTRUCTING AND NORMALIZING KERNELS
#___________________________________________________________________________________________________________
# Contents: 
#       construct_kernel_moment
#       symmetric_normalize_kernel and eigen_decomp_from_symmetric
#       normalize_kernel
#_____________________________________________________________________________________________________________

#                                   /// construct_kernel_moment  ///

# returns: Kernel, Moment <--- Kernel is scipy.sparse matrix of size (num_data_points, num_data_points)
#                              Only distances between nearest neighbors are computed
#                              Moment is numpy array of size (N,) and is used to determine boundary_mask automatically

# accepts: 
#       X:      numpy array of size (num_data_points, ambient_dimension)
#       epsilon:        graph bandwidth parameter
#       n_neighbors:        number of neighbors to constuct kernel 

def construct_kernel_moment(X, epsilon, n_neighbors):
    # constants used throughout function
    N = X.shape[0]
    if epsilon == 'auto':
        epsilon = .0001

    sqrt_eps = np.sqrt(epsilon)
    sqrt_pi = np.sqrt(np.pi)

    # find neighbors and distances
    X_tree = scipy.spatial.KDTree(X)
    dist, ind = X_tree.query(x = X, k= n_neighbors)

    # initialize empty matrices
    Kernel = scipy.sparse.lil_matrix((N,N))
    Moment = np.zeros((N,))

    for i in range(N):
        # contruct the i-th row of the kernel 
        exp_dist = (1/N) * np.exp( ( -1/(4*epsilon) ) * dist[i,:]**2 )
        Kernel[i,ind[i,:]] = exp_dist

        # constuct i-th entry of Moment vector
        Moment[i] = sqrt_pi*np.linalg.norm(               
             ( 
                    (np.matmul(
                        np.diag(exp_dist),
                            ((X[i,:] - X[ind[i,:],:])/(2*sqrt_eps))
                                )
                                    ).sum(axis = 0)
                                        ) / (exp_dist.sum())
            )


    # final kernel w nearest neighbors
    Kernel = Kernel.tocsr()
    Kernel = (Kernel + Kernel.transpose())

    # correct kernel for non-uniform sampling density 
    q = 1/Kernel.sum(axis = 1)
    q = np.asarray(q).reshape(-1)
    Q = scipy.sparse.diags(diagonals = q, offsets = 0, format = 'csr')
    Kernel = Q @ Kernel @ Q

    return Kernel, Moment
    
#_______________________________________________________________________

#                           /// symmetric_normalize_kernel and normalize_kernel ///

# returns: normalized_kernel, [normalizer] <--- both are scipy sparse matrices of size (num_data_points, num_data_points)

# accepts: 
#       Kernel:         scipy sparse matrix of size (num_data_points, num_data_points)
#       truncation:     bool,  whether or not to remove truncation points 

#       boundary_mask:      bool array of size (num_data_poitns,)    
#                           If truncating, says which points to truncate

def symmetric_normalize_kernel(Kernel, truncation = False, boundary_mask = None):
    d = 1/Kernel.sum(axis = 1)
    d = np.asarray(d).reshape(-1)
    d = np.sqrt(d) #<-- this computes a symmetric normalization 
                            # equivalent for eigendecomp since  
                            # D^(-1)( I- D^2K )D = I - D K D

    if truncation:
        d = d[boundary_mask]
        Kernel = truncate_kernel(Kernel, boundary_mask) 
        
    normalizer = scipy.sparse.diags(diagonals = d, offsets = 0, format = 'csr')
    normalized_kernel = normalizer @ Kernel @ normalizer

    return normalized_kernel, normalizer

#_____________________________________________________________________________

def normalize_kernel(Kernel, truncation = False, boundary_mask = None):
    d = 1/Kernel.sum(axis = 1)
    d = np.asarray(d).reshape(-1)       #<--- corresponding to nonzymmetric normalization

    if truncation:
        d = d[boundary_mask]
        Kernel = truncate_kernel(Kernel, boundary_mask) 
        
    normalizer = scipy.sparse.diags(diagonals = d, offsets = 0, format = 'csr')
    normalized_kernel = normalizer @ Kernel

    return normalized_kernel

#___________________________________________________________________
#Symmetric normalizer above often paired with the following function:

def eigen_decomp_from_symmetric(symmetric_normalized_kernel, normalizer, epsilon, eigen_number):
    # use symmetric kernel to compute eigendecomp
    untransformed_eigvals, untransformed_eigvects = scipy.sparse.linalg.eigsh(symmetric_normalized_kernel, 
                                                              k = eigen_number, which = 'LM')

    # transform into Laplace-Beltrami eigenvalues 
    eigenvalues = -1*np.log(untransformed_eigvals**(1/epsilon)) 
    eigenvalues = np.round(eigenvalues[::-1],3)     #reorder according to increasing transformed eigenvalues

    # transform into Laplace-Beltrami eigenvectors
    eigenvectors = normalizer @ untransformed_eigvects
    eigenvectors = eigenvectors[:,::-1]             #reorder eigenvectors as well to match

    return eigenvalues, eigenvectors

#__________________________________________________________________________________________________________
#__________________________________________________________________________________________________________
# SOME BOUNDARY ONLY FUNCTIONS: truncating kernels and constructing mask for truncation
#___________________________________________________________________________________________________________
#___________________________________________________________________________________________________________


# This function uses the Moment to estimate which points are close to the boundary

# returns: boundary_mask        <--- boolean array of size (num_data_points,) indicating which points to keep
# accepts:
#       Moment: numpy array of size (num_data_points,) 
#       N_data_points: integer 
#       num_truncation_points:      integer. Number of points to truncate.

def construct_boundary_mask(Moment, N_data_points, num_truncation_points = 'auto'):

    # Compute N_interior dependent on input
    if num_truncation_points == 'auto':
        N_boundary = np.max( [(Moment > .5125).sum(),5])
        N_interior = round(N_data_points-N_boundary)
    else:
        N_interior = (N_data_points - num_truncation_points)


    #Sort moment corresponding to interior points vs. boundary points
    idx = np.argpartition(Moment, N_interior)

    #Create boundary mask where interior points are True
    boundary_mask = np.full((N_data_points,), False, dtype=bool)
    boundary_mask[idx[0:N_interior]] = True

    return boundary_mask
    

# Simple helper function which truncates kernels given mask. 
def truncate_kernel(Kernel, boundary_mask):
    truncated_Kernel = Kernel[:,boundary_mask][boundary_mask,:]
    return truncated_Kernel
    

#______________________________________________________________________________________________________












