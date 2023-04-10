import BoundaryDM_functions
import numpy as np
import matplotlib.pyplot as plt
import scipy

#__________________________________________________________________________________________________

# No boundary, standard Diffusion maps class
# Supports autotuning of epsilon

class DiffusionMaps:

    def __init__(self, epsilon = 'auto', n_neighbors = 64, eigen_number = 10):
        # initialize core atttributes
        self.epsilon = epsilon
        self.n_neighbors = n_neighbors
        self.eigen_number = eigen_number
    
        #__________________________
        # attributes which require fitting 
        self.eigenvalues = None
        self.eigenvectors = None
        self.coordinates = None

    #_______________________________________________________________________________
    #Method Estimates Laplace-Beltami, computes eigendecomposition, 
    # and stores Diffusion Coordinates
    # X: dataset for fitting 
    # epsilon_grid_params: specifies linspace for parameter tuning [ lower bound, upper bound, number of points ]

    def fit(self, X, epsilon_grid_params = [1e-5, 1e-2, 10], verbose = False): 
        N = X.shape[0]
        if self.epsilon == 'auto':
            lower_range = epsilon_grid_params[0]
            upper_range = epsilon_grid_params[1]
            num_epsilons = epsilon_grid_params[2]
            self.epsilon, self.dimension = BoundaryDM_functions.auto_tune_epsilon(X, n_neighbors = self.n_neighbors, 
                                                             lower_range = lower_range, upper_range = upper_range, 
                                                             num_epsilons = num_epsilons, verbose = verbose)
        
        #Use tuned epsilon and neighbors to make unnormalized kernel
        Kernel, Moment = BoundaryDM_functions.construct_kernel_moment(X, epsilon = self.epsilon, n_neighbors = self.n_neighbors)

        #Use boundary mask and kernel to create symmetric kernel for eigenvalue problem
        normalized_kernel, normalizer = BoundaryDM_functions.symmetric_normalize_kernel(Kernel)


        #Compute spectrum and update attributes
        eigenvalues, eigenvectors = BoundaryDM_functions.eigen_decomp_from_symmetric(symmetric_normalized_kernel=normalized_kernel,
                                                      normalizer = normalizer, epsilon = self.epsilon,
                                                      eigen_number = self.eigen_number)
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors


        #Compute diffusion coordinates and update attribtue
        diffusion_coordinates = np.matmul(eigenvectors,np.diag(np.exp(-eigenvalues*self.epsilon)))
        self.coordinates = diffusion_coordinates[:,1:]

    #_______________________________________________________________________________
    #Fits and returns data in new coordinates

    def fit_transform(self, X, epsilon_grid_params = [1e-5, 1e-2, 10], verbose = False):
    
        self.fit(X, epsilon_grid_params, verbose)
    
        return self.coordinates

    #__________________________________________________________________________________
    #Cosmetic methods for easy display of spectal properties

    def display_eigenvector(self, k):
        fig,ax = plt.subplots(figsize = (6,4))
        vector = self.eigenvectors[:,k]
        ax.scatter(np.arange(vector.shape[0]),vector)

    def display_spectrum(self):
        fig,ax = plt.subplots(figsize = (6,4))
        ax.scatter(np.arange(self.eigen_number), self.eigenvalues)
    
#___________________________________________________________________________________





#Truncated version of Diffusion maps algorithm ---> Designed for manifolds with boundary
#Estimates Eigendecomposition of Dirichlet Laplacian 

class TGL:
    
    def __init__(self, epsilon = 'auto', n_neighbors = 64, eigen_number = 10):
        # initialize core atttributes
        self.epsilon = epsilon
        self.n_neighbors = n_neighbors
        self.eigen_number = eigen_number
    
        #__________________________
        # attributes which require fitting 
        self.eigenvalues = None
        self.eigenvectors = None
        self.coordinates = None

    #_____________________________________________________________________________________
    #Estimates eigendecomposition of Dirichlet Laplacian 
    #Stores "diffusion coordinates" based on Dirichlet heat kernel
    #       X: dataset of size (N,n) 
    #       manual_boundary_mask: Whether or not user labels boundary/interior points
    #       boundary_mask: 'auto' or array of size (N,) with boundary_mask[i] = True iff point X[i,:] is far away from interior
    def fit(self, X, 
            manual_boundary_mask = False, boundary_mask = 'auto', num_truncation_points = 'auto',
            epsilon_grid_params = [1e-5, 1e-2, 10], verbose = False):
        N = X.shape[0]
    
    #_______________________________________________________________
    #  PARAMETER TUNING
    #______________________________________________________________

        #case 1: automatic finding of boundary mask and epsilon
        if (self.epsilon == 'auto') and (not manual_boundary_mask):
            lower_range = epsilon_grid_params[0]
            upper_range = epsilon_grid_params[1]
            num_epsilons = epsilon_grid_params[2]
  
            self.epsilon, self.boundary_mask = BoundaryDM_functions.auto_tune_epsilon_TGL(X, n_neighbors = self.n_neighbors, 
                                                                            num_truncation_points = num_truncation_points,
                                lower_range = lower_range, upper_range = upper_range, num_epsilons = num_epsilons, verbose = verbose)

        #case 2: automatically find epsilon given boundary_mask
        elif (self.epsilon == 'auto') and (manual_boundary_mask):
            lower_range = epsilon_grid_params[0]
            upper_range = epsilon_grid_params[1]
            num_epsilons = epsilon_grid_params[2]

            self.boundary_mask =  boundary_mask

            self.epsilon = BoundaryDM_functions.auto_tune_epsilon_TGL_mask(X, n_neighbors = self.n_neighbors,
                                                                                  boundary_mask = self.boundary_mask,
                                lower_range = lower_range, upper_range = upper_range, num_epsilons = num_epsilons, verbose = verbose )
        
        #case 3: find boundary mask only, given epsilon
        elif (self.epsilon != 'auto') and (not manual_boundary_mask):

            Kernel, Moment = BoundaryDM_functions.construct_kernel_moment(X, epsilon = self.epsilon, n_neighbors=self.n_neighbors)

            self.boundary_mask = BoundaryDM_functions.construct_boundary_mask(Moment = Moment, N_data_points = N, 
                                                num_truncation_points = num_truncation_points)
        #case 4: all manual parameters!  
        else:
            self.boundary_mask = boundary_mask

    #_________________________________________________________________________________

        #Use parameters to compute normalized kernel
        Kernel, Moment = BoundaryDM_functions.construct_kernel_moment(X, epsilon = self.epsilon, n_neighbors=self.n_neighbors)
        normalized_kernel, normalizer = BoundaryDM_functions.symmetric_normalize_kernel(Kernel, truncation = True, boundary_mask=self.boundary_mask)

        #Use normalized_kernel to compute laplacian spectum
        eigenvalues, eigenvectors = BoundaryDM_functions.eigen_decomp_from_symmetric(symmetric_normalized_kernel=normalized_kernel,
                                                      normalizer = normalizer, epsilon = self.epsilon,
                                                      eigen_number = self.eigen_number)
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors


        #compute diffusion coordinates and update attribtue
        diffusion_coordinates = np.matmul(eigenvectors,np.diag(np.exp(-eigenvalues*self.epsilon)))
        self.coordinates = diffusion_coordinates[:,0:]

    #______________________________________________________________________

    #Fits and returns data in new coordinates

    def fit_transform(self, X, manual_boundary_mask = False, boundary_mask = 'auto', num_truncation_points = 'auto',
                      epsilon_grid_params = [1e-5, 1e-2, 10], verbose = False):
    
        self.fit(X, manual_boundary_mask = manual_boundary_mask, boundary_mask = boundary_mask, 
                 num_truncation_points = num_truncation_points,
                epsilon_grid_params = epsilon_grid_params, verbose = verbose)
    
        return self.coordinates

    #__________________________________________________________________________________

    #__________________________________________________________________________________
    #Cosmetic methods for easy display of spectal properties

    def display_boundary_points(self, X):
        fig,ax = plt.subplots()
        ax.scatter(X[self.boundary_mask,0],X[self.boundary_mask,1], color = "blue")
        ax.scatter(X[~self.boundary_mask,0],X[~self.boundary_mask,1], color = "orange")

    def display_eigenvector(self, k):
        fig,ax = plt.subplots(figsize = (6,4))
        vector = self.eigenvectors[:,k]
        ax.scatter(np.arange(vector.shape[0]),vector)

    def display_spectrum(self):
        fig,ax = plt.subplots(figsize = (6,4))
        ax.scatter(np.arange(self.eigen_number), self.eigenvalues)

#__________________________________________________________________________________






#                   Constructs Novel estimator developed by J. Wilson Peoples and John Harlim.
#                   Estimates Dirichlet Laplacian by truncating .5 * (L^T + L), 
# w                 here L is a nonsymmetric normalized graph Laplacian

class SymmetricTGL:
    
    def __init__(self, epsilon = 'auto', n_neighbors = 64, eigen_number = 10):
        # initialize core atttributes
        self.epsilon = epsilon
        self.n_neighbors = n_neighbors
        self.eigen_number = eigen_number
    
        #__________________________
        # attributes which require fitting 
        self.eigenvalues = None
        self.eigenvectors = None
        self.coordinates = None

    #___________________________________________________________________________________

    #Estimates eigendecomposition of Dirichlet Laplacian 
    #Stores "diffusion coordinates" based on Dirichlet heat kernel
    #       X: dataset of size (N,n) 
    #       manual_boundary_mask: Whether or not user labels boundary/interior points
    #       boundary_mask: 'auto' or array of size (N,) with boundary_mask[i] = True iff point X[i,:] is far away from interior
    #       epsilon_grid_params: [lowerbound, upperbound, gridsize] for epsilon parameter search.
    def fit(self, X, 
            manual_boundary_mask = False, boundary_mask = 'auto', num_truncation_points = 'auto',
            epsilon_grid_params = [1e-5, 1e-2, 10], verbose = False):
        N = X.shape[0]


    #_______________________________________________________
    #  PARAMETER TUNING
    #_______________________________________________________

        #case 1: automatically tune epsilon and truncation points
        if (self.epsilon == 'auto') and (not manual_boundary_mask):
            lower_range = epsilon_grid_params[0]
            upper_range = epsilon_grid_params[1]
            num_epsilons = epsilon_grid_params[2]
  
            self.epsilon, self.boundary_mask = BoundaryDM_functions.auto_tune_epsilon_SymmetricTGL(X, n_neighbors = self.n_neighbors, 
                                                                            num_truncation_points = num_truncation_points,
                                lower_range = lower_range, upper_range = upper_range, num_epsilons = num_epsilons, verbose = verbose)

        #case 2: tune epsilon given truncation points
        elif (self.epsilon == 'auto') and (manual_boundary_mask):
            lower_range = epsilon_grid_params[0]
            upper_range = epsilon_grid_params[1]
            num_epsilons = epsilon_grid_params[2]

            self.boundary_mask =  boundary_mask
            self.epsilon = BoundaryDM_functions.auto_tune_epsilon_SymmetricTGL_mask(X, n_neighbors = self.n_neighbors,
                                                                                  boundary_mask = self.boundary_mask,
                                lower_range = lower_range, upper_range = upper_range, num_epsilons = num_epsilons, verbose = verbose )
            
        #case 3: find truncation points given epsilon
        elif (self.epsilon != 'auto') and (not manual_boundary_mask):

            Kernel, Moment = BoundaryDM_functions.construct_kernel_moment(X, epsilon = self.epsilon, n_neighbors=self.n_neighbors)
            self.boundary_mask = BoundaryDM_functions.construct_boundary_mask(Moment = Moment, N_data_points = N, 
                                                num_truncation_points = num_truncation_points)
        #case 4: no parameters to be tuned
        else:
            self.boundary_mask = boundary_mask

    #___________________________________________________________

        #Constuct symmetric estimator from Kernel
        Kernel, Moment = BoundaryDM_functions.construct_kernel_moment(X, epsilon = self.epsilon, n_neighbors=self.n_neighbors)
        normalized_kernel = BoundaryDM_functions.normalize_kernel(Kernel, truncation = True, boundary_mask=self.boundary_mask)
        I = scipy.sparse.identity(n = normalized_kernel.shape[0], format = 'csr') 

        symmetric_estimator = (1 / self.epsilon) * (I - .5 * (normalized_kernel + normalized_kernel.transpose()) )


        # Use symmetric estimator to compute spectrum and update attributes
        eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(symmetric_estimator, 
                                                              k = self.eigen_number, which = 'SM')
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors


        #compute diffusion coordinates and update attribtue
        diffusion_coordinates = np.matmul(eigenvectors,np.diag(np.exp(-eigenvalues*self.epsilon)))
        self.coordinates = diffusion_coordinates[:,0:]

    #______________________________________________________________________

    #Fits and returns data in new coordinates

    def fit_transform(self, X, manual_boundary_mask = False, boundary_mask = 'auto', num_truncation_points = 'auto',
                      epsilon_grid_params = [1e-5, 1e-2, 10], verbose = False):
    
        self.fit(X, manual_boundary_mask = manual_boundary_mask, boundary_mask = boundary_mask, 
                 num_truncation_points = num_truncation_points,
                epsilon_grid_params = epsilon_grid_params, verbose = verbose)
    
        return self.coordinates

    #__________________________________________________________________________________

    #__________________________________________________________________________________
    #Cosmetic methods for easy display of spectal properties

    def display_boundary_points(self, X):
        fig,ax = plt.subplots()
        ax.scatter(X[self.boundary_mask,0],X[self.boundary_mask,1], color = "blue")
        ax.scatter(X[~self.boundary_mask,0],X[~self.boundary_mask,1], color = "orange")


    def display_eigenvector(self, k):
        fig,ax = plt.subplots(figsize = (6,4))
        vector = self.eigenvectors[:,k]
        ax.scatter(np.arange(vector.shape[0]),vector)

    def display_spectrum(self):
        fig,ax = plt.subplots(figsize = (6,4))
        ax.scatter(np.arange(self.eigen_number), self.eigenvalues)

#__________________________________________________________________________________



