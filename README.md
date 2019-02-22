# Conditionally independent multiresolution Gaussian process

This is an implementation of the conditionally independent multiresolution Gaussian process (ciMRGP) model based on Taghia and Schön (2019). Unlike the common assumption of the full independence, ciMRGP assumes conditional independence among Gaussian processes (GPs) across all resolutions. 

References:

Jalil Taghia and Thomas B. Schön. Conditionally Independent Multiresolution Gaussian Processes. AISTATS, 2019.

Note:
The source files are in <./src/> folder. The main file is MRGP.py. Two test examples are provided in <./scripts/tests/>: 
- The first example ciMRGP_vs_fiMRGP.py compares the two different MRGP architectures, namely: fully independent MRGP (fiMRGP) and conditionally independent MRGP (ciMRGP). The only difference between the two model variants is that the former assumes full independence between resolutions and the latter assumes conditional independence. This is simalr to the Figure 2 in (Taghia and Schön, 2019) but a different random seed. 
- The second example GPRBF_vs_ciMRGP_vs_fiMRGP.py compares ciMRGP and fiMRGP against a standard Gaussian process with an RBF kernel. For this example, an RBF kernel is not the best choice as the underlying function is highly nonstationary. But the question is if a multiresolution GP, seen as a collection of smooth GPs, could tackle this problem? 

Hope you find the implemtation helpful, and if so, please cite our paper. Finally, if you have questions, please let me know (jalil.taghia@it.uu.se) 

Jalil Taghia 

(Feb. 22, 2019, Uppsala)

