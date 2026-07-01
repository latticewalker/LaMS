This code implements a complexity estimator for provable dual attacks on Kyber based on three approaches: **PS24**, **QX25**, and **LaMS**. Specifically, **PS24** refers to _Provable Dual Attacks on Learning with Errors_, **QX25** refers to _On the Provable Dual Attack for LWE by Modulus Switching_, and **LaMS** refers to _LaMS: A p-adic Layered Modulus Switching for Provable Dual Attacks on LWE_.
It reports, for Kyber-512 / 768 / 1024, the best attack cost (bits) and the parameters that achieve it. The results of **PS24** is provided by the authors of _Provable Dual Attacks on Learning with Errors_.

## Requirements
* Python 3.8+
* The [lattice-estimator](https://github.com/malb/lattice-estimator) repository

## Running
sage -python Provable_dual_attack_kyber_estimator.py
