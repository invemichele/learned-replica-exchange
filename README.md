# Learned replica exchange (LREX)

Code to reproduce the paper [Skipping the replica exchange ladder with normalizing flows](https://pubs.acs.org/doi/10.1021/acs.jpclett.2c03327), JPCL (2022). [[Supporting Information]](https://pubs.acs.org/doi/suppl/10.1021/acs.jpclett.2c03327/suppl_file/jz2c03327_si_001.pdf) [![arXiv](https://img.shields.io/badge/arXiv-2210.14104-b31b1b.svg)](https://arxiv.org/abs/2210.14104) 


The systems studied are: a multidimensional double-well (dw), alanine dipeptide (ala2), and alanine tetrapeptide (ala4).

![GraphicTOC](https://user-images.githubusercontent.com/14904699/200288098-697d8cb9-2b01-48b1-abb9-dcd1902d4aef.png)

## Requirements
- [bgflow](https://github.com/noegroup/bgflow)
- [bgmol](https://github.com/noegroup/bgmol)
- [openmm](https://openmm.org)
- [plumed](https://plumed.org) and [openmm-plumed](https://github.com/openmm/openmm-plumed) for the reference simulations with [OPES](https://www.plumed.org/doc-master/user-doc/html/_o_p_e_s.html)

## Quick start
A self-contained Google Colab notebook is also available: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/invemichele/learned-replica-exchange/blob/main/Colab-LREX_alanine.ipynb)

## Errata corrige
There is an error in the paragraph just after eq 3 of the paper. For two times the words "prior" and "target" have been switched.

Thus $x_p \sim p(x)$ and $x_q \sim q(x)$, and the target continues from $f(x_q)$ and the prior from $f^{-1}(x_p)$, not vice versa as wrongly stated in the paper.
