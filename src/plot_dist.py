import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, gamma, expon, lognorm, weibull_min

# Parameters
normal_params = {'mean': 5, 'std': 1.5}
gamma_params = {'shape': 2.5, 'scale': 1.2}
exponential_params = {'scale': 2}
lognormal_params = {'sigma': 0.7, 'scale': np.exp(1)}
weibull_params = {'shape': 1.5, 'scale': 6}

# X-axis values
x = np.linspace(0, 20, 400)

# Normal Distribution
normal_pdf = norm.pdf(x, loc=normal_params['mean'], scale=normal_params['std'])

# Gamma Distribution
gamma_pdf = gamma.pdf(x, a=gamma_params['shape'], scale=gamma_params['scale'])

# Exponential Distribution
exponential_pdf = expon.pdf(x, scale=exponential_params['scale'])

# Lognormal Distribution
lognormal_pdf = lognorm.pdf(x, s=lognormal_params['sigma'], scale=lognormal_params['scale'])

# Weibull Distribution
weibull_pdf = weibull_min.pdf(x, c=weibull_params['shape'], scale=weibull_params['scale'])

# Plotting
# sns.set(style='whitegrid')  # Set the style for the plots
plt.figure(figsize=(10, 6))

plt.plot(x, normal_pdf, label='Normal')
plt.plot(x, gamma_pdf, label='Gamma')
plt.plot(x, exponential_pdf, label='Exponential')
plt.plot(x, lognormal_pdf, label='Lognormal')
plt.plot(x, weibull_pdf, label='Weibull')

plt.title('Probability Density Functions')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()

plt.show()
