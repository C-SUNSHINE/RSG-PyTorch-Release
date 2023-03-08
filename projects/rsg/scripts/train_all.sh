echo Crafting-RSGs

jac-run projects/rsg/scripts/learn_classifier.py -t crafting_rsgs --env craftingworld --data_version v2.primitives.large --mode fast-train --n_epochs 60 --n_train_batches 30 --n_val_batches 10 --train_init --state_machine_type compact --batch_size 5 --test_batch_size 30 --action_cost 0.2 --plot_wrong_sample 0 --lr 1e-3 --train_choice 5 --alpha 1.0 --beta 0.1 --train_choice_uniform --use_tb
jac-run projects/rsg/scripts/learn_classifier.py -t crafting_rsgs --env craftingworld --data_version v2.all.large --mode fast-train --n_epochs 60 --n_train_batches 30 --n_val_batches 10 --train_init --state_machine_type compact --batch_size 5 --test_batch_size 30 --action_cost 0.2 --plot_wrong_sample 0 --lr 1e-3 --train_choice 5 --alpha 1.0 --beta 0.1 --train_choice_uniform --use_tb --cont --start_epoch 31

echo Playroom-RSGs

jac-run projects/rsg/scripts/learn_classifier.py -t crafting_rsgs --env toyrobot --dataset playroom --data_version v2.all.large --mode fast-train --n_epochs 60 --train_init --state_machine_type incompact --batch_size 25 --n_train_batches 30 --n_val_batches 10 --plot_wrong_sample 1 --toy_robot_net point --lr 0.003 --beta 1.0 --alpha 1.0 --action_cost 0.1 --optim sgd --use_tb
jac-run projects/rsg/scripts/learn_classifier.py -t crafting_rsgs --env toyrobot --dataset playroom --data_version v2.all.large --mode fast-train --n_epochs 60 --train_init --state_machine_type incompact --batch_size 25 --n_train_batches 30 --n_val_batches 10 --plot_wrong_sample 1 --toy_robot_net point --lr 0.003 --beta 1.0 --alpha 1.0 --action_cost 0.1 --optim sgd --use_tb --cont --start_epoch 31

echo Crafting-LSTM

jac-run projects/rsg/scripts/learn_classifier.py -t crafting_lstm --env craftingworld --data_version v2.primitives.large --mode fast-train --n_epochs 100 --n_train_batches 300 --n_val_batches 100 --train_init --state_machine_type compact --batch_size 10 --test_batch_size 10 --plot_wrong_sample 0 --skip_our_model --add_baseline_lstm --lr 1e-3 --use_tb --alpha 1.0 --beta 0.0 --optim RMSprop
jac-run projects/rsg/scripts/learn_classifier.py -t crafting_lstm --env craftingworld --data_version v2.all.large --mode fast-train --n_epochs 100 --n_train_batches 300 --n_val_batches 100 --train_init --state_machine_type compact --batch_size 10 --test_batch_size 10 --plot_wrong_sample 0 --skip_our_model --add_baseline_lstm --lr 1e-3 --use_tb --alpha 1.0 --beta 0.0 --optim RMSprop --cont --start_epoch 51

echo Playroom-LSTM

jac-run projects/rsg/scripts/learn_classifier.py -t playroom_lstm --env toyrobot --dataset playroom --data_version v2.single.large --mode fast-train --n_epochs 100 --train_init --state_machine_type incompact --batch_size 64 --n_train_batches 300 --n_val_batches 100 --plot_wrong_sample 1 --toy_robot_net point --lr 0.0001 --beta 0.0 --alpha 1.0 --action_cost 0.1 --optim RMSprop --use_tb --skip_our_model --add_baseline_lstm
jac-run projects/rsg/scripts/learn_classifier.py -t playroom_lstm --env toyrobot --dataset playroom --data_version v2.all.large --mode fast-train --n_epochs 100 --train_init --state_machine_type incompact --batch_size 64 --n_train_batches 300 --n_val_batches 100 --plot_wrong_sample 1 --toy_robot_net point --lr 0.0001 --beta 0.0 --alpha 1.0 --action_cost 0.1 --optim RMSprop --use_tb --skip_our_model --add_baseline_lstm --cont --start_epoch 51

echo Crafting-IRL

jac-run projects/rsg/scripts/learn_classifier.py -t crafting_irl --env craftingworld --data_version v2.primitives.large --mode fast-train --n_epochs 60 --n_train_batches 30 --n_val_batches 10 --train_init --state_machine_type compact --batch_size 32 --test_batch_size 32 --plot_wrong_sample 0 --skip_our_model --add_baseline_irllstmdqn --irl_classify_by_reward --lr 1e-3 --use_tb --alpha 1.0 --beta 0.05 --use_tb
jac-run projects/rsg/scripts/learn_classifier.py -t crafting_irl --env craftingworld --data_version v2.all.large --mode fast-train --n_epochs 60 --n_train_batches 30 --n_val_batches 10 --train_init --state_machine_type compact --batch_size 32 --test_batch_size 32 --plot_wrong_sample 0 --skip_our_model --add_baseline_irllstmdqn --irl_classify_by_reward --lr 1e-3 --use_tb --alpha 1.0 --beta 0.05 --use_tb --cont --start_epoch 31

echo Playroom-IRL

jac-run projects/rsg/scripts/learn_classifier.py -t playroom_irl --env toyrobot --dataset playroom --data_version v2.single.large --mode fast-train --n_epochs 60 --train_init --state_machine_type incompact --batch_size 20 --n_train_batches 30 --n_val_batches 10 --plot_wrong_sample 1 --toy_robot_net point --lr 0.001 --beta 0.1 --alpha 1.0 --action_cost 0.1 --optim adam --use_tb --skip_our_model --add_baseline_irlcont  --irl_classify_by_reward
jac-run projects/rsg/scripts/learn_classifier.py -t playroom_irl --env toyrobot --dataset playroom --data_version v2.all.large --mode fast-train --n_epochs 60 --train_init --state_machine_type incompact --batch_size 20 --n_train_batches 30 --n_val_batches 10 --plot_wrong_sample 1 --toy_robot_net point --lr 0.001 --beta 0.1 --alpha 1.0 --action_cost 0.1 --optim adam --use_tb --skip_our_model --add_baseline_irlcont  --irl_classify_by_reward --cont --start_epoch 31

echo Crafting-BC

jac-run projects/rsg/scripts/learn_classifier.py -t crafting_bc --env craftingworld --data_version v2.primitives.large --mode fast-train --n_epochs 100 --n_train_batches 30 --n_val_batches 10 --train_init --state_machine_type compact --batch_size 32 --test_batch_size 32 --plot_wrong_sample 0 --skip_our_model --add_baseline_bc --lr 1e-3 --use_tb --alpha 1.0 --beta 0.1 --use_tb
jac-run projects/rsg/scripts/learn_classifier.py -t crafting_bc --env craftingworld --data_version v2.all.large --mode fast-train --n_epochs 100 --n_train_batches 30 --n_val_batches 10 --train_init --state_machine_type compact --batch_size 32 --test_batch_size 32 --plot_wrong_sample 0 --skip_our_model --add_baseline_bc --lr 1e-3 --use_tb --alpha 1.0 --beta 0.1 --use_tb --cont --start_epoch 51

echo Playroom-BC

jac-run projects/rsg/scripts/learn_classifier.py -t playroom_bc --env toyrobot --dataset playroom --data_version v2.single.large --mode fast-train --n_epochs 100 --train_init --state_machine_type incompact --batch_size 32 --n_train_batches 30 --n_val_batches 10 --plot_wrong_sample 1 --toy_robot_net point --lr 0.001 --beta 0.1 --alpha 1.0 --action_cost 0.1 --optim adam --use_tb --skip_our_model --add_baseline_bc
jac-run projects/rsg/scripts/learn_classifier.py -t playroom_bc --env toyrobot --dataset playroom --data_version v2.all.large --mode fast-train --n_epochs 100 --train_init --state_machine_type incompact --batch_size 32 --n_train_batches 30 --n_val_batches 10 --plot_wrong_sample 1 --toy_robot_net point --lr 0.001 --beta 0.1 --alpha 1.0 --action_cost 0.1 --optim adam --use_tb --skip_our_model --add_baseline_bc --cont --start_epoch 51

echo Crafting-BC_FSM

jac-run projects/rsg/scripts/learn_classifier.py -t crafting_bcfsm --env craftingworld --data_version v2.primitives.large --mode fast-train --n_epochs 100 --n_train_batches 30 --n_val_batches 10 --train_init --state_machine_type compact --batch_size 32 --test_batch_size 32 --plot_wrong_sample 0 --skip_our_model --add_baseline_seq2seq --lr 1e-3 --use_tb --alpha 1.0 --beta 0.1 --use_tb
jac-run projects/rsg/scripts/learn_classifier.py -t crafting_bcfsm --env craftingworld --data_version v2.all.large --mode fast-train --n_epochs 100 --n_train_batches 30 --n_val_batches 10 --train_init --state_machine_type compact --batch_size 32 --test_batch_size 32 --plot_wrong_sample 0 --skip_our_model --add_baseline_seq2seq --lr 1e-3 --use_tb --alpha 1.0 --beta 0.1 --use_tb --cont --start_epoch 51

echo Playroom-BC_FSM

jac-run projects/rsg/scripts/learn_classifier.py -t playroom_bcfsm --env toyrobot --dataset playroom --data_version v2.single.large --mode fast-train --n_epochs 100 --train_init --state_machine_type incompact --batch_size 32 --n_train_batches 30 --n_val_batches 10 --plot_wrong_sample 1 --toy_robot_net point --lr 0.003 --beta 0.1 --alpha 1.0 --action_cost 0.1 --optim adam --use_tb --skip_our_model --add_baseline_seq2seq
jac-run projects/rsg/scripts/learn_classifier.py -t playroom_bcfsm --env toyrobot --dataset playroom --data_version v2.all.large --mode fast-train --n_epochs 100 --train_init --state_machine_type incompact --batch_size 32 --n_train_batches 30 --n_val_batches 10 --plot_wrong_sample 1 --toy_robot_net point --lr 0.003 --beta 0.1 --alpha 1.0 --action_cost 0.1 --optim adam --use_tb --skip_our_model --add_baseline_seq2seq --cont --start_epoch 51

