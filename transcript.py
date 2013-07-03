# Here is a transcript of a Python session where I do the analysis of
# the data set in this directory.  I've editied it extensively, so
# it's possible that some of the commands will not work as written,
# but hopefully this is close.
import emcee
import scipy.stats as ss
import multilevel as m

# Load up the true parameters, just so we have access later
p0 = loadtxt('true-params.dat')
nteachers = p0.shape[0]
p0 = m.to_params(p0, m.nteachers_from_length(nteachers))

# Load the student data
students = loadtxt('students.dat.gz', dtype=[('score', np.float), ('teacher', np.int)])

# Set up the posterior object.
lnpost = m.Posterior(students['score'], students['teacher'])

# Initialize a sampler using 200 walkers, in 48 dimensions
# (m.params_length(nteachers)==48), with the given posterior function.
sampler = emcee.EnsembleSampler(200, 48, lnpost)

# Set up the initial sample in a small ball about p0.
pts = random.multivariate_normal(mean=p0.view(float).reshape((-1,)), cov=1e-3*eye(48), size=200)

# Run for a little while, just to make sure it works:
result = sampler.run_mcmc(pts, 500, thin=10)

# Examine how the mean of the ensemble of points has changed over each
# iteration for each variable (normalized to the overall mean and
# standard deviation).
for k in range(48):
    mu = mean(sampler.chain[:,:,k])
    sigma = std(mean(sampler.chain[:,:,k], axis=0))
    plot((mean(sampler.chain[:,:,k], axis=0)-mu)/sigma)
    
# Just to check that the sampler has a good acceptance fraction---not
# too high (near 1), not too low (near 0).  In this case it was near
# 0.1, which is fine.
sampler.acceptance_fraction

# Now that we've had a bit of burn-in time for the sampler to settle
# down, go ahead and reset, and run for a significant length.
sampler.reset()
result = sampler.run_mcmc(result[0], 500, thin=10)

# Look again, see that the sampler is not really converging very
# fast---the means don't seem to wander randomly, but rather wander in
# a directed fashion.
for k in range(48):
    mu = mean(sampler.chain[:,:,k])
    sigma = std(mean(sampler.chain[:,:,k], axis=0))
    plot((mean(sampler.chain[:,:,k], axis=0)-mu)/sigma)

# Confirm that the autocorrelation length (acor) is very long.    
sampler.acor
sampler.reset()

# Run for a very long time, recording a lot fewer samples (only every
# 100 steps of the chain).
result = sampler.run_mcmc(result[0], 5000, thin=100)
for k in range(48):
    mu = mean(sampler.chain[:,:,k])
    sigma = std(mean(sampler.chain[:,:,k], axis=0))
    plot((mean(sampler.chain[:,:,k], axis=0)-mu)/sigma)
    
# Looking better....
sampler.acor

# Go just a bit further.
result = sampler.run_mcmc(result[0], 1000, thin=100)
sampler.acor

# And now things look pretty good.
for k in range(48):
    mu = mean(sampler.chain[:,:,k])
    sigma = std(mean(sampler.chain[:,:,k], axis=0))
    plot((mean(sampler.chain[:,:,k], axis=0)-mu)/sigma)
    
# Make it possible to access the chain of samples using the column
# names
pchain = m.to_params(sampler.flatchain, nteachers)

# Plot the average mu_j and sigma_j from the chain (in blue), and then
# the true mu_j and sigma_j (in black).
errorbar(range(nteachers), mean(squeeze(pchain['mu_teacher']), axis=0), yerr=mean(squeeze(pchain['sigma_teacher']), axis=0), fmt='.b')
errorbar(range(nteachers), squeeze(p0['mu_teacher']), yerr=squeeze(p0['sigma_teacher']), fmt='.k')
axis(xmin=-1)
axis(xmax=nteachers)
xlabel(r'Teacher')
ylabel(r'$\mu_j \pm \sigma_j$')
savefig('teachers.pdf')

# You can get the plotutils package from
# http://github.com/farr/plotutils ; it does some nifty things for
# you, like automatically choose the histogram bin width according to
# the amount of data (more data = finer bin width, so you can resolve
# more features of the distribution).
import plotutils.plotutils as pu

# Now make a plot of the expected distribution of teacher means
xs = linspace(p0['mu'] - 3.0*p0['sigma'], p0['mu']+3.0*p0['sigma'], 1000)

# This is the true distribution: N(mu_j, sigma_j) with the true values
# of mu_j and sigma_j.
plot(xs, ss.norm.pdf(xs, loc=p0['mu'], scale=p0['sigma']), '-k')

# Now for each sample of mu_j and sigma_j in the chain, we'll compute
# the sample distribution N(mu_j, sigma_j), evaluate it at the xs, and
# average all those values together.  (Make sure you understand how
# this is different from N(<mu_j>, <sigma_j>), which is the single
# distribution evaluated at the average mu_j and sigma_j.  This sort
# of thing, by the way, is why people like Bayesian techniques that
# sample the posterior distribution---you don't have to just plot
# things like parameters; you can plot any *function* of the
# parameters you are interested in.
ys = zeros(xs.shape)
for mu, sigma in zip(pchain['mu'].flatten(), pchain['sigma'].flatten()):
    ys += ss.norm.pdf(xs, loc=mu, scale=sigma)
ys /= 12000.0

# Here's the plot
plot(xs, ys, '-b')
xlabel(r'$\mu_j$')
ylablel(r'$p\left( \mu_j \right)$')
ylabel(r'$p\left( \mu_j \right)$')
savefig('muj.pdf')

# Now just plot some posterior distributions of the mu and sigma
# parameters.
pu.plot_histogram_posterior(pchain['mu'].flatten(), normed=True, histtype='step')
axis(ymax=1)
axis(ymax=0.2
)
axis(ymax=0.07)
axis(ymax=0.06)
axvline(p0['mu'])
axvline(p0['mu'], color='k')
xlabel(r'$\mu$')
ylabel(r'$p(\mu)$')
savefig('mu.pdf')
pu.plot_histogram_posterior(pchain['sigma'].flatten(), normed=True, histtype='step')
axis(ymax=0.07)
axis(ymax=0.1)
axis(ymax=0.09)
axvline(p0['sigma'])
axvline(p0['sigma'], color='k')
xlabel(r'$\sigma$')
ylabel(r'$p(\sigma)$')
savefig('sigma.pdf')
