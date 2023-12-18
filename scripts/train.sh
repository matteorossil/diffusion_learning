
# Train with embedding
# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_2hrs exp=trainer_reps name="$(date +%F)-test_1GPU"

# ./.python-greene submitit_hydra.py compute/greene=1x4 compute/greene/node=rtx8000_4days exp=trainer_reps name="$(date +%F)-4GPU_train_time_concat_embed"

./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_1day exp=trainer_reps name="$(date +%F)-1GPU_train_enc_vqvae_time_concat_emb_cifar10_restart"