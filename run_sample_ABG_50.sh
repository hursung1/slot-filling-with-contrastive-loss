#############################
####### Slot Filling ########
#############################

## Domain: AddToPlaylist
sed -i 's/"target_domain": .*/"target_domain": "AddToPlaylist",/g' config_n_ABG_50.json
sed -i 's/"loss_key_ratio": .*/"loss_key_ratio": 0,/g' config_n_ABG_50.json
python main.py config_n_ABG_50.json
sed -i 's/"loss_key_ratio": 0/"loss_key_ratio": 0.1/g' config_n_ABG_50.json
python main.py config_n_ABG_50.json
sed -i 's/"loss_key_ratio": 0.1/"loss_key_ratio": 0.3/g' config_n_ABG_50.json
python main.py config_n_ABG_50.json
sed -i 's/"loss_key_ratio": 0.3/"loss_key_ratio": 0.5/g' config_n_ABG_50.json
python main.py config_n_ABG_50.json
sed -i 's/"loss_key_ratio": 0.5/"loss_key_ratio": 0.7/g' config_n_ABG_50.json
python main.py config_n_ABG_50.json
sed -i 's/"loss_key_ratio": 0.7/"loss_key_ratio": 0.9/g' config_n_ABG_50.json
python main.py config_n_ABG_50.json

## Domain: BookRestaurant
sed -i 's/"target_domain": .*/"target_domain": "BookRestaurant",/g' config_n_ABG_50.json
sed -i 's/"loss_key_ratio": .*/"loss_key_ratio": 0,/g' config_n_ABG_50.json
python main.py config_n_ABG_50.json
sed -i 's/"loss_key_ratio": 0/"loss_key_ratio": 0.1/g' config_n_ABG_50.json
python main.py config_n_ABG_50.json
sed -i 's/"loss_key_ratio": 0.1/"loss_key_ratio": 0.3/g' config_n_ABG_50.json
python main.py config_n_ABG_50.json
sed -i 's/"loss_key_ratio": 0.3/"loss_key_ratio": 0.5/g' config_n_ABG_50.json
python main.py config_n_ABG_50.json
sed -i 's/"loss_key_ratio": 0.5/"loss_key_ratio": 0.7/g' config_n_ABG_50.json
python main.py config_n_ABG_50.json
sed -i 's/"loss_key_ratio": 0.7/"loss_key_ratio": 0.9/g' config_n_ABG_50.json
python main.py config_n_ABG_50.json

## Domain: GetWeather
sed -i 's/"target_domain": .*/"target_domain": "GetWeather",/g' config_n_ABG_50.json
sed -i 's/"loss_key_ratio": .*/"loss_key_ratio": 0,/g' config_n_ABG_50.json
python main.py config_n_ABG_50.json
sed -i 's/"loss_key_ratio": 0/"loss_key_ratio": 0.1/g' config_n_ABG_50.json
python main.py config_n_ABG_50.json
sed -i 's/"loss_key_ratio": 0.1/"loss_key_ratio": 0.3/g' config_n_ABG_50.json
python main.py config_n_ABG_50.json
sed -i 's/"loss_key_ratio": 0.3/"loss_key_ratio": 0.5/g' config_n_ABG_50.json
python main.py config_n_ABG_50.json
sed -i 's/"loss_key_ratio": 0.5/"loss_key_ratio": 0.7/g' config_n_ABG_50.json
python main.py config_n_ABG_50.json
sed -i 's/"loss_key_ratio": 0.7/"loss_key_ratio": 0.9/g' config_n_ABG_50.json
python main.py config_n_ABG_50.json