REM find optimal batch size
python enwik8_run.py --name=first_layer_batchsize --num_layers=1 --sweeps=0 --units_num=200 --time_steps=60 --drop_output_init=0.7 --drop_output_step=0.1 --drop_state_init=0.7 --drop_state_step=0.1 --drop_emb=0.8 --batch_size=15
python enwik8_run.py --name=first_layer_batchsize --num_layers=1 --sweeps=0 --units_num=200 --time_steps=60 --drop_output_init=0.7 --drop_output_step=0.1 --drop_state_init=0.7 --drop_state_step=0.1 --drop_emb=0.8 --batch_size=30
python enwik8_run.py --name=first_layer_batchsize --num_layers=1 --sweeps=0 --units_num=200 --time_steps=60 --drop_output_init=0.7 --drop_output_step=0.1 --drop_state_init=0.7 --drop_state_step=0.1 --drop_emb=0.8 --batch_size=50
python enwik8_run.py --name=first_layer_batchsize --num_layers=1 --sweeps=0 --units_num=200 --time_steps=60 --drop_output_init=0.7 --drop_output_step=0.1 --drop_state_init=0.7 --drop_state_step=0.1 --drop_emb=0.8 --batch_size=80
python enwik8_run.py --name=first_layer_batchsize --num_layers=1 --sweeps=0 --units_num=200 --time_steps=60 --drop_output_init=0.7 --drop_output_step=0.1 --drop_state_init=0.7 --drop_state_step=0.1 --drop_emb=0.8 --batch_size=100
python enwik8_run.py --name=first_layer_batchsize --num_layers=1 --sweeps=0 --units_num=200 --time_steps=60 --drop_output_init=0.7 --drop_output_step=0.1 --drop_state_init=0.7 --drop_state_step=0.1 --drop_emb=0.8 --batch_size=150



