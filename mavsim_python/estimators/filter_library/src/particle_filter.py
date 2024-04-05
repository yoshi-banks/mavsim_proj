import numpy as np

class ParticleFilter:
    def __init__(self, fdis, h, R, N, x0_pdf, process_noise_pdf, pfParams):
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
        self.fdis = fdis
        self.h = h
        self.R = R
        self.N = N
        self.x = x0_pdf.rvs(size=N).T
        self.process_noise_pdf = process_noise_pdf
        self.xdim = self.x.shape[0]
        self.pfParams = pfParams

    def step(self, y, u, t, params):
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
        q = np.zeros(self.N)
        x_prior = np.zeros((self.xdim, self.N))
        x_posterior = self.x.copy()

        for i in range(self.N):
            omega = self.process_noise_pdf.rvs(size=1).reshape(-1, 1)
            x_prior[:, i] = self.fdis(x_posterior[:, i].reshape(-1, 1), u, omega, t, params).flatten()

            vhat = y - self.h(x_prior[:, i].reshape(-1, 1), np.zeros_like(y), t, params)
            q[i] = 1. / ((2 * np.pi) ** (self.xdim / 2) * np.linalg.det(self.R) ** 0.5) * \
                   np.exp(-0.5 * vhat.T @ np.linalg.inv(self.R) @ vhat)

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
                    sigma = self.pfParams['K'] * E * self.N ** (-1 / self.xdim)
                    x_posterior[:, i] += sigma * np.random.randn(self.xdim)
                    break

        self.x = x_posterior

    @property
    def x_hat(self):
        """
        Calculate the mean estimate of the state.

        Returns:
            numpy.ndarray: Mean estimate of the state.
        """
        return np.mean(self.x, axis=1)
