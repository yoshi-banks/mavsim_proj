import numpy as np

class ParticleFilter:
    def __init__(self, x0_pdf, N):
        """
        Construct an instance of the ParticleFilter class.

        Args:
            fdis (callable): Transition function.
            h (callable): Measurement function.
            R (numpy.ndarray): Measurement noise covariance matrix.
            N (int): Number of particles.
            x0_pdf (scipy.stats.rv_continuous): Initial state distribution.
            process_noise_pdf (scipy.stats.rv_continuous): Process noise distribution.
            pfParams (dict): Particle filter parameters.
        """
        # self.fdis = fdis
        # self.h = h
        # self.R = R
        self.N = N
        self.x = x0_pdf.rvs(size=N).T
        # self.process_noise_pdf = process_noise_pdf
        self.xdim = self.x.shape[0]

    def predict(self, u, t, params, stepParams):
        """
        Predict the state of the system at the next time step.

        Args:
            u: Control input.
            t: Time.
            params: Additional parameters.

        Returns:
            None
        """
        fdis = stepParams['fdis']
        process_noise_pdf = stepParams['process_noise_pdf']

        for i in range(self.N):
            omega = process_noise_pdf.rvs(size=1).reshape(-1, 1)
            self.x[:, i] = fdis(self.x[:, i].reshape(-1, 1), u, omega, t, params).flatten()

    def step(self, y, u, t, params, stepParams):
        """
        Perform one step of the particle filter algorithm.

        Args:
            y (numpy.ndarray): Measurement vector.
            u: Control input.
            t: Time.
            params: Additional parameters.

        Returns:
            None
        """
        fdis = stepParams['fdis']
        h = stepParams['h']
        R = stepParams['R']
        process_noise_pdf = stepParams['process_noise_pdf']
        K = stepParams['K']

        q = np.zeros(self.N)
        x_prior = np.zeros((self.xdim, self.N))
        x_posterior = self.x.copy()

        for i in range(self.N):
            omega = process_noise_pdf.rvs(size=1).reshape(-1, 1)
            x_prior[:, i] = fdis(x_posterior[:, i].reshape(-1, 1), u, omega, t, params).flatten()

            vhat = y - h(x_prior[:, i].reshape(-1, 1), u, np.zeros_like(y), t, params)
            q[i] = 1. / ((2 * np.pi) ** (self.xdim / 2) * np.linalg.det(R) ** 0.5) * \
                   np.exp(-0.5 * vhat.T @ np.linalg.inv(R) @ vhat)

        if np.sum(q) == 0:
            q /= 0.0000001
        q /= np.sum(q) # TODO add a check for q summing to 0

        x_posterior = np.zeros((self.xdim, self.N))
        for i in range(self.N):
            rand_num = np.random.rand()
            qtempsum = 0
            for j in range(self.N):
                qtempsum += q[j]
                if qtempsum >= rand_num:
                    x_posterior[:, i] = x_prior[:, i]
                    E = np.max(x_prior, axis=1) - np.min(x_prior, axis=1)
                    sigma = K * E * self.N ** (-1 / self.xdim)
                    x_posterior[:, i] += sigma * np.random.randn(self.xdim)
                    break

        self.x = x_posterior

    def multi_step(self, ylist, u, t, params, stepParams):
        """
        Perform one step of the particle filter algorithm.

        Args:
            y (numpy.ndarray): Measurement vector.
            u: Control input.
            t: Time.
            params: Additional parameters.

        Returns:
            None
        """
        fdis = stepParams['fdis']
        hlist = stepParams['hlist']
        Rlist = stepParams['Rlist']
        process_noise_pdf = stepParams['process_noise_pdf']
        Klist = stepParams['Klist']

        q = np.zeros(self.N)
        x_prior = np.zeros((self.xdim, self.N))
        x_posterior = self.x.copy()

        num_meas = len(hlist)
        x_new_posterior = np.zeros((num_meas, self.xdim, self.N))
        iota = 0
        for y, h, R, K in zip(ylist, hlist, Rlist, Klist):

            for i in range(self.N):
                omega = process_noise_pdf.rvs(size=1).reshape(-1, 1)
                x_prior[:, i] = fdis(x_posterior[:, i].reshape(-1, 1), u, omega, t, params).flatten()

                vhat = y - h(x_prior[:, i].reshape(-1, 1), u, np.zeros_like(y), t, params)
                q[i] = 1. / ((2 * np.pi) ** (self.xdim / 2) * np.linalg.det(R) ** 0.5) * \
                    np.exp(-0.5 * vhat.T @ np.linalg.inv(R) @ vhat)

            q /= np.sum(q) # TODO add a check for q summing to 0


            for i in range(self.N):
                rand_num = np.random.rand()
                qtempsum = 0
                for j in range(self.N):
                    qtempsum += q[j] 
                    if qtempsum >= rand_num:
                        x_new_posterior[iota, :, i] = x_prior[:, i]
                        E = np.max(x_prior, axis=1) - np.min(x_prior, axis=1)
                        sigma = K * E * self.N ** (-1 / self.xdim)
                        x_new_posterior[iota, :, i] += sigma * np.random.randn(self.xdim)
                        break

            iota += 1

        self.x = x_posterior


    @property
    def x_hat(self):
        """
        Calculate the mean estimate of the state.

        Returns:
            numpy.ndarray: Mean estimate of the state.
        """
        return np.mean(self.x, axis=1)
    
    @property
    def P(self):
        """
        Calculate the covariance matrix of the state estimate.

        Returns:
            numpy.ndarray: Covariance matrix of the state estimate.
        """
        return np.cov(self.x)
