set -x

UP=300
DIM=128
alpha=0.025
lr=0.001

# Train Beibei 
./code/air+ -train_ui ./data/Beibei/processed/purc.txt.air,./data/Beibei/processed/cart.txt.air,./data/Beibei/processed/pv.txt.air,./data/Beibei/processed/all-v.txt.air \
-save ./result_embedding/Beibei.$lr-$alpha-$DIM -update_times $UP -dimension $DIM -worker 50 -l2_reg $lr -init_alpha $alpha

# Train Taobao, Rees46, Ecommerce
# dataset=Taobao
# for dataset in Taobao Rees46 Ecommerce
# do
#     ./code/air+ -train_ui ./data/$dataset/processed/purc.txt.air,./data/$dataset/processed/cart.txt.air,./data/$dataset/processed/pv.txt.air,./data/$dataset/processed/all.txt.air \
#     -save ./result_embedding/$dataset.$lr-$alpha-$DIM -update_times $UP -dimension $DIM -worker 50 -l2_reg $lr -init_alpha $alpha
# done


