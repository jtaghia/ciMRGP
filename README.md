# Conditionally independent multiresolution Gaussian process

This is our implementation of the conditionally independent multiresolution Gaussian process (ciMRGP) model based on Taghia and Schön (2019). Unlike the common assumption of full independence, ciMRGP assumes conditional independence among Gaussian processes (GPs) across all resolution. 

Jalil Taghia and Thomas B. Schön. Conditionally Independent Multiresolution Gaussian Processes. AISTATS, 2019.

Example:
Consider a multi-output nonlinear regression problem. We have observed some noisy observations. The underlying latent functions exhibit some rapid changes. The goal is to recover the functions from the noisy measurements. 
