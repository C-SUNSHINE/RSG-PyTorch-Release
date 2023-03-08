
echo Crafting-RSGs-Classification-Compositional

jac-run projects/rsg/scripts/learn_classifier.py -t crafting_rsgs --env craftingworld --data_version v2.integrated.large --mode test --n_epochs 60 --n_train_batches 30 --n_val_batches 10 --train_init --state_machine_type compact --batch_size 5 --test_batch_size 30 --action_cost 0.2 --plot_wrong_sample 0 --lr 1e-3 --train_choice 5 --alpha 1.0 --beta 0.1 --train_choice_uniform --use_tb --save_answer classification_com

echo Crafting-RSGs-Classification-Novel

jac-run projects/rsg/scripts/learn_classifier.py -t crafting_rsgs --env craftingworld --data_version v2.novel --mode test --n_epochs 60 --n_train_batches 30 --n_val_batches 10 --train_init --state_machine_type compact --batch_size 5 --test_batch_size 30 --action_cost 0.2 --plot_wrong_sample 0 --lr 1e-3 --train_choice 5 --alpha 1.0 --beta 0.1 --train_choice_uniform --use_tb --save_answer classification_novel

echo Crafting-RSGs-Planning-Compositional

jac-run projects/rsg/scripts/learn_classifier.py -t crafting_rsgs --env craftingworld --data_version v2.integrated.large --mode test --n_epochs 60 --n_train_batches 30 --n_val_batches 10 --train_init --state_machine_type compact --batch_size 5 --test_batch_size 30 --action_cost 0.2 --plot_wrong_sample 0 --lr 1e-3 --train_choice 5 --alpha 1.0 --beta 0.1 --train_choice_uniform --use_tb --plan --save_answer planning_com

echo Crafting-RSGs-Planning-Novel

jac-run projects/rsg/scripts/learn_classifier.py -t crafting_rsgs --env craftingworld --data_version v2.novel --mode test --n_epochs 60 --n_train_batches 30 --n_val_batches 10 --train_init --state_machine_type compact --batch_size 5 --test_batch_size 30 --action_cost 0.2 --plot_wrong_sample 0 --lr 1e-3 --train_choice 5 --alpha 1.0 --beta 0.1 --train_choice_uniform --use_tb --plan --save_answer planning_novel --test_part_label


echo Playroom-RSGs-Classification-Compositional

jac-run projects/rsg/scripts/learn_classifier.py -t playroom_rsgs --env toyrobot --dataset playroom --data_version v2.c2s.large --mode test --n_epochs 60 --train_init --state_machine_type incompact --batch_size 25 --n_train_batches 30 --n_val_batches 10 --plot_wrong_sample 1 --toy_robot_net point --lr 0.003 --beta 1.0 --alpha 1.0 --action_cost 0.1 --optim sgd --use_tb --save_answer classification_com

echo Playroom-RSGs-Classification-Novel

jac-run projects/rsg/scripts/learn_classifier.py -t playroom_rsgs --env toyrobot --dataset playroom --data_version v2.complex.large --mode test --n_epochs 60 --train_init --state_machine_type incompact --batch_size 25 --n_train_batches 30 --n_val_batches 10 --plot_wrong_sample 1 --toy_robot_net point --lr 0.003 --beta 1.0 --alpha 1.0 --action_cost 0.1 --optim sgd --use_tb --save_answer classification_novel

echo Playroom-RSGs-Planning-Compositional

jac-run projects/rsg/scripts/learn_classifier.py -t playroom_rsgs --env toyrobot --dataset playroom --data_version v2.c2s.large --mode test --n_epochs 60 --train_init --state_machine_type incompact --batch_size 25 --n_train_batches 30 --n_val_batches 10 --plot_wrong_sample 1 --toy_robot_net point --lr 0.003 --beta 1.0 --alpha 1.0 --action_cost 0.1 --optim sgd --use_tb --plan --save_answer planning_com

echo Playroom-RSGs-Planning-Novel

jac-run projects/rsg/scripts/learn_classifier.py -t playroom_rsgs --env toyrobot --dataset playroom --data_version v2.complex.large --mode test --n_epochs 60 --train_init --state_machine_type incompact --batch_size 25 --n_train_batches 30 --n_val_batches 10 --plot_wrong_sample 1 --toy_robot_net point --lr 0.003 --beta 1.0 --alpha 1.0 --action_cost 0.1 --optim sgd --use_tb --plan --save_answer planning_novel


echo Crafting-LSTM-Classification-Compositional

jac-run projects/rsg/scripts/learn_classifier.py -t crafting_lstm --env craftingworld --data_version v2.integrated.large --mode test --n_epochs 100 --n_train_batches 300 --n_val_batches 100 --train_init --state_machine_type compact --batch_size 10 --test_batch_size 10 --plot_wrong_sample 0 --skip_our_model --add_baseline_lstm --lr 1e-3 --use_tb --alpha 1.0 --beta 0.0 --optim RMSprop --save_answer classification_com

echo Crafting-LSTM-Classification-Novel

jac-run projects/rsg/scripts/learn_classifier.py -t crafting_lstm --env craftingworld --data_version v2.novel --mode test --n_epochs 100 --n_train_batches 300 --n_val_batches 100 --train_init --state_machine_type compact --batch_size 10 --test_batch_size 10 --plot_wrong_sample 0 --skip_our_model --add_baseline_lstm --lr 1e-3 --use_tb --alpha 1.0 --beta 0.0 --optim RMSprop --save_answer classification_novel


echo Playroom-LSTM-Classification-Compositional

jac-run projects/rsg/scripts/learn_classifier.py -t playroom_lstm --env toyrobot --dataset playroom --data_version v2.c2s.large --mode test --n_epochs 100 --train_init --state_machine_type incompact --batch_size 64 --n_train_batches 300 --n_val_batches 100 --plot_wrong_sample 1 --toy_robot_net point --lr 0.0001 --beta 0.0 --alpha 1.0 --action_cost 0.1 --optim RMSprop --use_tb --skip_our_model --add_baseline_lstm --save_answer classification_com

echo Playroom-LSTM-Classification-Novel

jac-run projects/rsg/scripts/learn_classifier.py -t playroom_lstm --env toyrobot --dataset playroom --data_version v2.complex.large --mode test --n_epochs 100 --train_init --state_machine_type incompact --batch_size 64 --n_train_batches 300 --n_val_batches 100 --plot_wrong_sample 1 --toy_robot_net point --lr 0.0001 --beta 0.0 --alpha 1.0 --action_cost 0.1 --optim RMSprop --use_tb --skip_our_model --add_baseline_lstm --save_answer classification_novel


echo Crafting-IRL-Classification-Compositional

jac-run projects/rsg/scripts/learn_classifier.py -t crafting_irl --env craftingworld --data_version v2.integrated.large --mode test --n_epochs 60 --n_train_batches 30 --n_val_batches 10 --train_init --state_machine_type compact --batch_size 32 --test_batch_size 32 --plot_wrong_sample 0 --skip_our_model --add_baseline_irllstmdqn --irl_classify_by_reward --lr 1e-3 --use_tb --alpha 1.0 --beta 0.05 --use_tb --save_answer classification_com

echo Crafting-IRL-Classification-Novel

jac-run projects/rsg/scripts/learn_classifier.py -t crafting_irl --env craftingworld --data_version v2.novel --mode test --n_epochs 60 --n_train_batches 30 --n_val_batches 10 --train_init --state_machine_type compact --batch_size 32 --test_batch_size 32 --plot_wrong_sample 0 --skip_our_model --add_baseline_irllstmdqn --irl_classify_by_reward --lr 1e-3 --use_tb --alpha 1.0 --beta 0.05 --use_tb --save_answer classification_novel

echo Crafting-IRL-Planning-Compositional

jac-run projects/rsg/scripts/learn_classifier.py -t crafting_irl --env craftingworld --data_version v2.integrated.large --mode test --n_epochs 60 --n_train_batches 30 --n_val_batches 10 --train_init --state_machine_type compact --batch_size 32 --test_batch_size 32 --plot_wrong_sample 0 --skip_our_model --add_baseline_irllstmdqn --irl_classify_by_reward --lr 1e-3 --use_tb --alpha 1.0 --beta 0.05 --use_tb --plan --save_answer planning_com

echo Crafting-IRL-Planning-Novel

jac-run projects/rsg/scripts/learn_classifier.py -t crafting_irl --env craftingworld --data_version v2.novel --mode test --n_epochs 60 --n_train_batches 30 --n_val_batches 10 --train_init --state_machine_type compact --batch_size 32 --test_batch_size 32 --plot_wrong_sample 0 --skip_our_model --add_baseline_irllstmdqn --irl_classify_by_reward --lr 1e-3 --use_tb --alpha 1.0 --beta 0.05 --use_tb --plan --save_answer planning_novel


echo Playroom-IRL-Classification-Compositional

jac-run projects/rsg/scripts/learn_classifier.py -t playroom_irl --env toyrobot --dataset playroom --data_version v2.c2s.large --mode test --n_epochs 60 --train_init --state_machine_type incompact --batch_size 20 --n_train_batches 30 --n_val_batches 10 --plot_wrong_sample 1 --toy_robot_net point --lr 0.001 --beta 0.1 --alpha 1.0 --action_cost 0.1 --optim adam --use_tb --skip_our_model --add_baseline_irlcont  --irl_classify_by_reward --save_answer classification_com

echo Playroom-IRL-Classification-Novel

jac-run projects/rsg/scripts/learn_classifier.py -t playroom_irl --env toyrobot --dataset playroom --data_version v2.complex.large --mode test --n_epochs 60 --train_init --state_machine_type incompact --batch_size 20 --n_train_batches 30 --n_val_batches 10 --plot_wrong_sample 1 --toy_robot_net point --lr 0.001 --beta 0.1 --alpha 1.0 --action_cost 0.1 --optim adam --use_tb --skip_our_model --add_baseline_irlcont  --irl_classify_by_reward --save_answer classification_novel

echo Playroom-IRL-Planning-Compositional

jac-run projects/rsg/scripts/learn_classifier.py -t playroom_irl --env toyrobot --dataset playroom --data_version v2.c2s.large --mode test --n_epochs 60 --train_init --state_machine_type incompact --batch_size 20 --n_train_batches 30 --n_val_batches 10 --plot_wrong_sample 1 --toy_robot_net point --lr 0.001 --beta 0.1 --alpha 1.0 --action_cost 0.1 --optim adam --use_tb --skip_our_model --add_baseline_irlcont  --irl_classify_by_reward --plan --save_answer planning_com

echo Playroom-IRL-Planning-Novel

jac-run projects/rsg/scripts/learn_classifier.py -t playroom_irl --env toyrobot --dataset playroom --data_version v2.complex.large --mode test --n_epochs 60 --train_init --state_machine_type incompact --batch_size 20 --n_train_batches 30 --n_val_batches 10 --plot_wrong_sample 1 --toy_robot_net point --lr 0.001 --beta 0.1 --alpha 1.0 --action_cost 0.1 --optim adam --use_tb --skip_our_model --add_baseline_irlcont  --irl_classify_by_reward --plan --save_answer planning_novel


echo Crafting-BC-Classification-Compositional

jac-run projects/rsg/scripts/learn_classifier.py -t crafting_bc --env craftingworld --data_version v2.integrated.large --mode test --n_epochs 100 --n_train_batches 30 --n_val_batches 10 --train_init --state_machine_type compact --batch_size 32 --test_batch_size 32 --plot_wrong_sample 0 --skip_our_model --add_baseline_bc --lr 1e-3 --use_tb --alpha 1.0 --beta 0.1 --use_tb --save_answer classification_com

echo Crafting-BC-Classification-Novel

jac-run projects/rsg/scripts/learn_classifier.py -t crafting_bc --env craftingworld --data_version v2.novel --mode test --n_epochs 100 --n_train_batches 30 --n_val_batches 10 --train_init --state_machine_type compact --batch_size 32 --test_batch_size 32 --plot_wrong_sample 0 --skip_our_model --add_baseline_bc --lr 1e-3 --use_tb --alpha 1.0 --beta 0.1 --use_tb --save_answer classification_novel

echo Crafting-BC-Planning-Compositional

jac-run projects/rsg/scripts/learn_classifier.py -t crafting_bc --env craftingworld --data_version v2.integrated.large --mode test --n_epochs 100 --n_train_batches 30 --n_val_batches 10 --train_init --state_machine_type compact --batch_size 32 --test_batch_size 32 --plot_wrong_sample 0 --skip_our_model --add_baseline_bc --lr 1e-3 --use_tb --alpha 1.0 --beta 0.1 --use_tb --plan --save_answer planning_com

echo Crafting-BC-Planning-Novel

jac-run projects/rsg/scripts/learn_classifier.py -t crafting_bc --env craftingworld --data_version v2.novel --mode test --n_epochs 100 --n_train_batches 30 --n_val_batches 10 --train_init --state_machine_type compact --batch_size 32 --test_batch_size 32 --plot_wrong_sample 0 --skip_our_model --add_baseline_bc --lr 1e-3 --use_tb --alpha 1.0 --beta 0.1 --use_tb --plan --save_answer planning_novel


echo Playroom-BC-Classification-Compositional

jac-run projects/rsg/scripts/learn_classifier.py -t playroom_bc --env toyrobot --dataset playroom --data_version v2.c2s.large --mode test --n_epochs 100 --train_init --state_machine_type incompact --batch_size 32 --n_train_batches 30 --n_val_batches 10 --plot_wrong_sample 1 --toy_robot_net point --lr 0.001 --beta 0.1 --alpha 1.0 --action_cost 0.1 --optim adam --use_tb --skip_our_model --add_baseline_bc --save_answer classification_com

echo Playroom-BC-Classification-Novel

jac-run projects/rsg/scripts/learn_classifier.py -t playroom_bc --env toyrobot --dataset playroom --data_version v2.complex.large --mode test --n_epochs 100 --train_init --state_machine_type incompact --batch_size 32 --n_train_batches 30 --n_val_batches 10 --plot_wrong_sample 1 --toy_robot_net point --lr 0.001 --beta 0.1 --alpha 1.0 --action_cost 0.1 --optim adam --use_tb --skip_our_model --add_baseline_bc --save_answer classification_novel

echo Playroom-BC-Planning-Compositional

jac-run projects/rsg/scripts/learn_classifier.py -t playroom_bc --env toyrobot --dataset playroom --data_version v2.c2s.large --mode test --n_epochs 100 --train_init --state_machine_type incompact --batch_size 32 --n_train_batches 30 --n_val_batches 10 --plot_wrong_sample 1 --toy_robot_net point --lr 0.001 --beta 0.1 --alpha 1.0 --action_cost 0.1 --optim adam --use_tb --skip_our_model --add_baseline_bc --plan --save_answer planning_com

echo Playroom-BC-Planning-Novel

jac-run projects/rsg/scripts/learn_classifier.py -t playroom_bc --env toyrobot --dataset playroom --data_version v2.complex.large --mode test --n_epochs 100 --train_init --state_machine_type incompact --batch_size 32 --n_train_batches 30 --n_val_batches 10 --plot_wrong_sample 1 --toy_robot_net point --lr 0.001 --beta 0.1 --alpha 1.0 --action_cost 0.1 --optim adam --use_tb --skip_our_model --add_baseline_bc --plan --save_answer planning_novel


echo Crafting-BC_FSM-Classification-Compositional

jac-run projects/rsg/scripts/learn_classifier.py -t crafting_bcfsm --env craftingworld --data_version v2.integrated.large --mode test --n_epochs 100 --n_train_batches 30 --n_val_batches 10 --train_init --state_machine_type compact --batch_size 32 --test_batch_size 32 --plot_wrong_sample 0 --skip_our_model --add_baseline_seq2seq --lr 1e-3 --use_tb --alpha 1.0 --beta 0.1 --use_tb --save_answer classification_com

echo Crafting-BC_FSM-Classification-Novel

jac-run projects/rsg/scripts/learn_classifier.py -t crafting_bcfsm --env craftingworld --data_version v2.novel --mode test --n_epochs 100 --n_train_batches 30 --n_val_batches 10 --train_init --state_machine_type compact --batch_size 32 --test_batch_size 32 --plot_wrong_sample 0 --skip_our_model --add_baseline_seq2seq --lr 1e-3 --use_tb --alpha 1.0 --beta 0.1 --use_tb --save_answer classification_novel

echo Crafting-BC_FSM-Planning-Compositional

jac-run projects/rsg/scripts/learn_classifier.py -t crafting_bcfsm --env craftingworld --data_version v2.integrated.large --mode test --n_epochs 100 --n_train_batches 30 --n_val_batches 10 --train_init --state_machine_type compact --batch_size 32 --test_batch_size 32 --plot_wrong_sample 0 --skip_our_model --add_baseline_seq2seq --lr 1e-3 --use_tb --alpha 1.0 --beta 0.1 --use_tb --plan --save_answer planning_com

echo Crafting-BC_FSM-Planning-Novel

jac-run projects/rsg/scripts/learn_classifier.py -t crafting_bcfsm --env craftingworld --data_version v2.novel --mode test --n_epochs 100 --n_train_batches 30 --n_val_batches 10 --train_init --state_machine_type compact --batch_size 32 --test_batch_size 32 --plot_wrong_sample 0 --skip_our_model --add_baseline_seq2seq --lr 1e-3 --use_tb --alpha 1.0 --beta 0.1 --use_tb --plan --save_answer planning_novel


echo Playroom-BC_FSM-Classification-Compositional

jac-run projects/rsg/scripts/learn_classifier.py -t playroom_bcfsm --env toyrobot --dataset playroom --data_version v2.c2s.large --mode test --n_epochs 100 --train_init --state_machine_type incompact --batch_size 32 --n_train_batches 30 --n_val_batches 10 --plot_wrong_sample 1 --toy_robot_net point --lr 0.003 --beta 0.1 --alpha 1.0 --action_cost 0.1 --optim adam --use_tb --skip_our_model --add_baseline_seq2seq --save_answer classification_com

echo Playroom-BC_FSM-Classification-Novel

jac-run projects/rsg/scripts/learn_classifier.py -t playroom_bcfsm --env toyrobot --dataset playroom --data_version v2.complex.large --mode test --n_epochs 100 --train_init --state_machine_type incompact --batch_size 32 --n_train_batches 30 --n_val_batches 10 --plot_wrong_sample 1 --toy_robot_net point --lr 0.003 --beta 0.1 --alpha 1.0 --action_cost 0.1 --optim adam --use_tb --skip_our_model --add_baseline_seq2seq --save_answer classification_novel

echo Playroom-BC_FSM-Planning-Compositional

jac-run projects/rsg/scripts/learn_classifier.py -t playroom_bcfsm --env toyrobot --dataset playroom --data_version v2.c2s.large --mode test --n_epochs 100 --train_init --state_machine_type incompact --batch_size 32 --n_train_batches 30 --n_val_batches 10 --plot_wrong_sample 1 --toy_robot_net point --lr 0.003 --beta 0.1 --alpha 1.0 --action_cost 0.1 --optim adam --use_tb --skip_our_model --add_baseline_seq2seq --plan --save_answer planning_com

echo Playroom-BC_FSM-Planning-Novel

jac-run projects/rsg/scripts/learn_classifier.py -t playroom_bcfsm --env toyrobot --dataset playroom --data_version v2.complex.large --mode test --n_epochs 100 --train_init --state_machine_type incompact --batch_size 32 --n_train_batches 30 --n_val_batches 10 --plot_wrong_sample 1 --toy_robot_net point --lr 0.003 --beta 0.1 --alpha 1.0 --action_cost 0.1 --optim adam --use_tb --skip_our_model --add_baseline_seq2seq --plan --save_answer planning_novel


echo Crafting-RSGs-Plan-for-final-goal-without-discovered-dependencies

jac-run projects/rsg/scripts/learn_classifier.py -t crafting_rsgs --env craftingworld --data_version v2.plan_search --mode test --n_epochs 100 --n_train_batches 30 --n_val_batches 10 --train_init --state_machine_type compact --batch_size 5 --test_batch_size 1 --action_cost 0.2 --plot_wrong_sample 0 --lr 1e-3 --train_choice 5 --alpha 1.0 --beta 0.1 --train_choice_uniform --use_tb --group test --plan --plan_search dependency_base --search_iter 500

echo Crafting-RSGs-Plan-for-final-goal-with-discovered-dependencies

jac-run projects/rsg/scripts/learn_classifier.py -t crafting_rsgs --env craftingworld --data_version v2.plan_search --mode test --n_epochs 100 --n_train_batches 30 --n_val_batches 10 --train_init --state_machine_type compact --batch_size 5 --test_batch_size 1 --action_cost 0.2 --plot_wrong_sample 0 --lr 1e-3 --train_choice 5 --alpha 1.0 --beta 0.1 --train_choice_uniform --use_tb --group test --plan --plan_search dependency --search_iter 500

echo Crafting-RSGs-Plan-for-final-goal-brute-force

jac-run projects/rsg/scripts/learn_classifier.py -t crafting_rsgs --env craftingworld --data_version v2.plan_search --mode test --n_epochs 100 --n_train_batches 30 --n_val_batches 10 --train_init --state_machine_type compact --batch_size 5 --test_batch_size 1 --action_cost 0.2 --plot_wrong_sample 0 --lr 1e-3 --train_choice 5 --alpha 1.0 --beta 0.1 --train_choice_uniform --use_tb --group test --plan --plan_search brute --search_iter 100000
