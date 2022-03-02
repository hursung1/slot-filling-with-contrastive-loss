#############################
####### Slot Filling ########
#############################

## Domain: PlayMusic
sed -i 's/"target_domain": .*/"target_domain": "PlayMusic",/g' config_n_PRSS_0.json
sed -i 's/"loss_key_ratio": .*/"loss_key_ratio": 0,/g' config_n_PRSS_0.json
python main.py config_n_PRSS_0.json
sed -i 's/"loss_key_ratio": 0/"loss_key_ratio": 0.1/g' config_n_PRSS_0.json
python main.py config_n_PRSS_0.json
sed -i 's/"loss_key_ratio": 0.1/"loss_key_ratio": 0.3/g' config_n_PRSS_0.json
python main.py config_n_PRSS_0.json
sed -i 's/"loss_key_ratio": 0.3/"loss_key_ratio": 0.5/g' config_n_PRSS_0.json
python main.py config_n_PRSS_0.json
sed -i 's/"loss_key_ratio": 0.5/"loss_key_ratio": 0.7/g' config_n_PRSS_0.json
python main.py config_n_PRSS_0.json
sed -i 's/"loss_key_ratio": 0.7/"loss_key_ratio": 0.9/g' config_n_PRSS_0.json
python main.py config_n_PRSS_0.json

## Domain: RateBook
sed -i 's/"target_domain": .*/"target_domain": "RateBook",/g' config_n_PRSS_0.json
sed -i 's/"loss_key_ratio": .*/"loss_key_ratio": 0,/g' config_n_PRSS_0.json
python main.py config_n_PRSS_0.json
sed -i 's/"loss_key_ratio": 0/"loss_key_ratio": 0.1/g' config_n_PRSS_0.json
python main.py config_n_PRSS_0.json
sed -i 's/"loss_key_ratio": 0.1/"loss_key_ratio": 0.3/g' config_n_PRSS_0.json
python main.py config_n_PRSS_0.json
sed -i 's/"loss_key_ratio": 0.3/"loss_key_ratio": 0.5/g' config_n_PRSS_0.json
python main.py config_n_PRSS_0.json
sed -i 's/"loss_key_ratio": 0.5/"loss_key_ratio": 0.7/g' config_n_PRSS_0.json
python main.py config_n_PRSS_0.json
sed -i 's/"loss_key_ratio": 0.7/"loss_key_ratio": 0.9/g' config_n_PRSS_0.json
python main.py config_n_PRSS_0.json

## Domain: SearchCreativeWork
sed -i 's/"target_domain": .*/"target_domain": "SearchCreativeWork",/g' config_n_PRSS_0.json
sed -i 's/"loss_key_ratio": .*/"loss_key_ratio": 0,/g' config_n_PRSS_0.json
python main.py config_n_PRSS_0.json
sed -i 's/"loss_key_ratio": 0/"loss_key_ratio": 0.1/g' config_n_PRSS_0.json
python main.py config_n_PRSS_0.json
sed -i 's/"loss_key_ratio": 0.1/"loss_key_ratio": 0.3/g' config_n_PRSS_0.json
python main.py config_n_PRSS_0.json
sed -i 's/"loss_key_ratio": 0.3/"loss_key_ratio": 0.5/g' config_n_PRSS_0.json
python main.py config_n_PRSS_0.json
sed -i 's/"loss_key_ratio": 0.5/"loss_key_ratio": 0.7/g' config_n_PRSS_0.json
python main.py config_n_PRSS_0.json
sed -i 's/"loss_key_ratio": 0.7/"loss_key_ratio": 0.9/g' config_n_PRSS_0.json
python main.py config_n_PRSS_0.json

## Domain: SearchScreeningEvent
sed -i 's/"target_domain": .*/"target_domain": "SearchScreeningEvent",/g' config_n_PRSS_0.json
sed -i 's/"loss_key_ratio": .*/"loss_key_ratio": 0,/g' config_n_PRSS_0.json
python main.py config_n_PRSS_0.json
sed -i 's/"loss_key_ratio": 0/"loss_key_ratio": 0.1/g' config_n_PRSS_0.json
python main.py config_n_PRSS_0.json
sed -i 's/"loss_key_ratio": 0.1/"loss_key_ratio": 0.3/g' config_n_PRSS_0.json
python main.py config_n_PRSS_0.json
sed -i 's/"loss_key_ratio": 0.3/"loss_key_ratio": 0.5/g' config_n_PRSS_0.json
python main.py config_n_PRSS_0.json
sed -i 's/"loss_key_ratio": 0.5/"loss_key_ratio": 0.7/g' config_n_PRSS_0.json
python main.py config_n_PRSS_0.json
sed -i 's/"loss_key_ratio": 0.7/"loss_key_ratio": 0.9/g' config_n_PRSS_0.json
python main.py config_n_PRSS_0.json