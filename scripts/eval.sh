
# Eval
# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_1day exp=eval_repr name="$(date +%F)-1GPU_eval_concat_image_1M"

# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_2hrs exp=eval_repr name="$(date +%F)-1GPU_noload"

# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_2hrs exp=eval_repr_img name="$(date +%F)-img_eval2"

# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_2hrs exp=eval_repr_time name="$(date +%F)-time_eval2"

# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_2hrs exp=eval_repr_cifar10_img name="$(date +%F)-cifar10_img_eval"

# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_2hrs exp=eval_repr_cifar10_time name="$(date +%F)-cifar10_time_eval"

./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_2hrs exp=eval_repr_cifar10_random_init name="$(date +%F)-cifar10_randominit_eval"
