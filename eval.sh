# purc
for dataset in Beibei Taobao Rees46 Ecommerce
do
    python ./code/eval.py \
    --dataset $dataset \
    --r1_train purc.txt \
    --r1_test purc_test.txt \
    --r2_train cart.txt \
    --r3_train pv.txt \
    --r1_emb ./result_embedding/$dataset.0.001-0.025-128.0 \
    --all_emb ./result_embedding/$dataset.0.001-0.025-128.3 > result-$dataset-purc
done

# cart
# for dataset in Rees46 Ecommerce
# do
#     python ./code/eval.py \
#     --dataset $dataset \
#     --r1_train cart.txt \
#     --r1_test cart_test.txt \
#     --r2_train purc.txt \
#     --r3_train pv.txt \
#     --r1_emb ./result_embedding/$dataset.0.001-0.025-128.1 \
#     --all_emb ./result_embedding/$dataset.0.001-0.025-128.3 > result-$dataset-cart
# done