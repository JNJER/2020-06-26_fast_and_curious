# *Fast and curious* : Projet de stage master 1 : modélisation du traitement visuel d'une image fovéale

* This is rendered in https://laurentperrinet.github.io/sciblog/posts/2020-09-28-benchmarking-cnns.html
* See a follow-up in https://github.com/JNJER/2021-04-28_transfer_learning

## getting the submodule to download images

See [doc](https://github.blog/2016-02-01-working-with-submodules/) :

to init:
```
git submodule update --init --recursive
```

to update:
```
git submodule update --recursive --remote
```

## to run without notebook

(for instance on the NVIDIA)

```
python3 experiment_basic.py
python3 experiment_downsample.py
python3 experiment_grayscale.py
```
In case you get `OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.`, run :
```
KMP_DUPLICATE_LIB_OK=TRUE python3 experiment_basic.py
KMP_DUPLICATE_LIB_OK=TRUE python3 experiment_downsample.py
KMP_DUPLICATE_LIB_OK=TRUE python3 experiment_grayscale.py
```
