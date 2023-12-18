#!/bin/bash

# Obvious / static arguments
comp="compute=local"

# EXPERIMENT LAUNCHES
# GO BOTTOM TO TOP

# test mae crop
# ./.python-greene submitit_hydra.py $comp exp=eval_repr name="$(date +%F)-1GPU_eval_concat_image_1M_test"

# ./.python-greene submitit_hydra.py $comp exp=eval_repr name="$(date +%F)-1GPU_eval_concat_time_1M_test"

# ./.python-greene submitit_hydra.py $comp exp=sample_cond_img name="$(date +%F)-test_cond_sampling"

# ./.python-greene submitit_hydra.py $comp exp=sample_cond_time name="$(date +%F)-test_cond_sampling2"

./.python-greene submitit_hydra.py $comp exp=eval_repr_cifar10_img name="$(date +%F)-cifar10_img_eval"

# ./.python-greene submitit_hydra.py $comp exp=eval_repr_cifar10_time name="$(date +%F)-cifar10_time_eval"

# ./.python-greene submitit_hydra.py $comp exp=trainer_uncond name="$(date +%F)-sample_uncond"