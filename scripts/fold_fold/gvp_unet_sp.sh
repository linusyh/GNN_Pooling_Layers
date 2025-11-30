export NLAYERS=6
export ENCODER=unet_gvp_enc_dec_add_sparse
export DATASET=fold_fold
export WD=1.44e-5
export LR=5e-4

bash ../run_unet_gvp_bn.sh

