# LaMS
This code is a complexity estimator for the provable dual attack on Kyber under three methods: **PS24**, **QX25**, and **LaMS** . It reports, for Kyber-512 / 768 / 1024, the best attack cost (bits) and the parameters that achieve it. The results of **PS24** is provided by the authors of [16].

## Requirements

* Python 3.8+

* `numpy`

* The [lattice-estimator](https://github.com/malb/lattice-estimator) repository

## Running

sage -python dual_attack_kyber_estimator.py
