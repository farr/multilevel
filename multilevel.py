r"""Module implementing the multi-level modeling example for teacher
scores in a school district.  The model is that each student score is 

.. math::

  s_{ij} \sim N\left(\mu_j, \sigma_j\right)

where the index :math:`i` labels students and the index :math:`j`
labels teachers.  That is, we assume that the students of a particular
teacher score in a bell curve with a mean and standard deviation
(width) that is teacher-dependent.  We further assume that the
population of teacher means, :math:`\mu_j` is itself drawn from a
Normal distribution with an overall mean :math:`\mu` and standard
deviation :math:`\sigma`:

.. math::

  \mu_j \sim N\left( \mu, \sigma \right)

and that the population of teacher standard deviations are drawn from
a Log-normal distribution with parameters :math:`\mu_\sigma` and
:math:`\sigma_\sigma`:

.. math::

  \log \sigma_j \sim N\left( \mu_\sigma, \sigma_\sigma \right).

Our goal is to use the population of student scores and
student-teacher mappings to fit for :math:`\mu`, :math:`\sigma`,
:math:`\mu_s`, :math:`\sigma_s`, :math:`\mu_j`, and :math:`\sigma_j`.
The effect of the multi-level model (one level for student-teacher,
and one level for teachers themselves) is to smoothly interpolate
between an independent, per-teacher fit to student mean and variance,
and a "pooled" simultaneous fit to all students at once.  See
http://www.stat.columbia.edu/~gelman/research/published/multi2.pdf
(Gelman (2006)) for a simple introduction to multilevel modeling.

"""

import numpy as np
import numpy.random as nr
import scipy.stats as ss

def params_length(nteachers):
    """Returns the length of the parameters array for a given number of
    teachers.

    """

    return 4 + 2*nteachers

def nteachers_from_length(len):
    """Given a length, returns the number of teachers that would produce a
    parameter array of that length.

    """

    return (len-4)/2

def to_params(arr, nteachers):
    r"""Converts a numpy array into an array with named fields:

    :param arr: The array to be converted.

    :param nteachers: The number of teachers to be modeled.

    :return: An array with labels 
    
      ``mu`` 
        The mean of the distribution from which the teacher mean score
        is drawn.

      ``sigma``
        The standard deviation of the distribution from which the
        teacher mean score is drawn.

      ``mus``
        The :math:`\mu` parameter for the lognormal distribution from
        which the teacher standard deviations are drawn.

      ``sigmas``
        The :math:`\sigma` parameter for the lognormal distribution
        from which the teacher standard deviations are drawn.

      ``mu_teacher``
        The array of teacher mean scores (of length ``nteachers``).

      ``sigma_teacher``
        The array of teacher standard deviations (also of length
        ``nteachers``).

    """
    return arr.view(np.dtype([('mu', np.float),
                              ('sigma', np.float),
                              ('mus', np.float),
                              ('sigmas', np.float),
                              ('mu_teacher', np.float, nteachers),
                              ('sigma_teacher', np.float, nteachers)]))

class Posterior(object):
    """Callable object used to compute the posterior probability of
    parameter values."""

    def __init__(self, scores, teachers):
        """Initializes the posterior object with the given scores and teacher
        indices for each student.

        :param scores: An array of scores for each student.

        :param teachers: An array of the teacher index for each
          student.

        """

        self.scores = scores
        self.teachers = teachers
        
        sorted_teachers = np.sort(teachers)
        self.nteachers = 1 + np.count_nonzero(sorted_teachers[1:] != sorted_teachers[:-1])

    def log_prior(self, parameters):
        r"""Log of the prior probability of the parameters.  In addition to the
        restriction that all :math`\sigma` parameters must be
        positive, the prior is given by 

        .. math::

          p(\theta) \propto \frac{1}{\sigma} \frac{1}{\sigma_\sigma^2} \prod_j \frac{1}{\sigma_j}.

        This is the Jeffreys prior (see
        http://en.wikipedia.org/wiki/Jeffreys_prior ) for the scale
        parameters.

        """

        p = to_params(parameters, self.nteachers)

        # Zero probability if any of the sigmas is smaller than zero
        if p['sigma'] <= 0.0:
            return float('-inf')

        if p['sigmas'] <= 0.0:
            return float('-inf')

        if np.any(p['sigma_teacher'] <= 0.0):
            return float('-inf')

        # Now return Jeffreys prior for all the normal distributions
        # p(mu, sigma) ~ 1/sigma, and lognormal, p(mu, sigma) ~
        # 1/sigma^2

        log_p = -np.log(p['sigma']) - 2.0*np.log(p['sigmas'])
        log_p -= np.sum(np.log(p['sigma_teacher']))

        return log_p

    def log_likelihood(self, parameters):
        r"""The log of the likelihood of the given parameters, using the data
        stored in the posterior object.  The likelihood is 

        .. math::

          p\left(\left\{s_{ij}\right\} | \theta \right) \propto \prod_{j} \left[\prod_{i} N\left( s_{ij} | \mu_j, \sigma_j \right) \right] N\left( \mu_j | \mu, \sigma \right) N\left( \log \sigma_j | \mu_\sigma, \sigma_\sigma\right) \frac{1}{\sigma_j}

        That is, a normal PDF for each student score with the
        corresponding teacher mean and standard deviation, a normal
        PDF for each teacher mean with the global teacher mean and
        standard deviation, and a log-normal PDF for each teacher
        standard deviation with the global sigma parameters.

        """

        p = to_params(parameters, self.nteachers)

        # Get the teacher mean and sigma corresponding to each
        # student.  Note that the 'mu_teacher' and 'sigma_teacher'
        # calls return arrays of shape (0, nteachers), so we need to
        # 'squeeze' them into arrays of shape (nteachers,) before
        # indexing.
        mu_js = np.squeeze(p['mu_teacher'])[self.teachers]
        sigma_js = np.squeeze(p['sigma_teacher'])[self.teachers]
        
        # The array of student probabilities: N(sij | mu_j, sigma_j).
        # This could be computed more quickly if we saved the mean and
        # standard deviation of each teacher's students, since we
        # could then compute 
        #
        # log(prob) = sum(-0.5*(x_j-mu_j)^2/(2*sigma_j^2)) - N*log(sigma_j) + const
        #
        # using 
        #
        # log(prob) = -0.5*N*(sigmax_j^2/sigma_j^2 + (mu_j - mux_j)^2/sigma_j^2) - N*log(sigma_j)
        # 
        # where mux_j = 1/N*sum(x_j) and sigmax_j = 1/N*sum((x_j-mux_j)^2).
        # (This decomposition is related to the parallel-axis theorem
        # for the moment of intertia of a solid body about a displaced
        # axis.)  I leave it as an exercise for the reader to
        # implement.
        log_pstudents = ss.norm.logpdf(self.scores, loc=mu_js, scale=sigma_js)

        # The teacher probabilities: normal for the teacher means,
        # log-normal for teacher sigmas.
        log_pteachers = ss.norm.logpdf(np.squeeze(p['mu_teacher']), loc=p['mu'], scale=p['sigma'])
        log_pteachers += ss.lognorm.logpdf(np.squeeze(p['sigma_teacher']), p['sigmas'], scale=np.exp(p['mus']))

        return np.sum(log_pstudents) + np.sum(log_pteachers)

    def __call__(self, params):
        """Make the posterior object a callable function, so
        ``posterior(params)`` returns the log-posterior."""

        log_prior = self.log_prior(params)

        # If prior == 0, we don't need to compute the likelihood (and,
        # indeed, such computation might fail because prior == 0 may
        # indicate some of the parameters outside their natural
        # boundaries).
        if log_prior == float('-inf'):
            return log_prior

        return log_prior + self.log_likelihood(params)
