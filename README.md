# FedGTF-EF/FedGTF-EF-PC

Source code for WWW 2021 paper "Communication Efficient Federated Generalized Tensor Factorization for Collaborative Health Data Analytics".

FedGTF-EF/FedGTF-Ef-PC are communication-efficient generalized federated tensor factorization algorithms that exploit up to three level of communication reduction strategies to the generalized tensor factorization, which is able to reduce the uplink communication cost up to 99.90%.

## Requirements

This code can be run with tensor toolbox version 3.1: https://gitlab.com/tensors/tensor_toolbox/-/releases

## Usage

Run "run_FedGTF-EF-PC.m". You can also change the function to FedGTF_EF and add the $l_1$ norm by changing to FedGTF_EF_prox.

You can swtich other parameter options:

- isBinary: set to 0 for square loss, 1 for Bernoulli Logit loss;

- isLogit: set to 0 for square loss, 1 for Bernoulli Logit loss;

- isCyclic: default as 0, switch to 1 to change from block-randomize to cyclic updates.
