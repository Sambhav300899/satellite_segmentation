from model import network
import os
import json
import argparse

if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    ap.add_argument('--config', required = True, help = "path to config file")

    args = vars(ap.parse_args())

    f = open(args['config'], 'r')
    json_data = json.load(f)
    f.close()

    try:
        os.mkdir(json_data['train']['logs'])
        os.mkdir(json_data['train']['model_callbacks'])
    except:
        pass
    finally:
        print ('writing model_callbacks to folder : {}'.format(json_data['train']['model_callbacks']))
        print ('writing logs to folder : {}'.format(json_data['train']['logs']))

    model_path = None

    if os.path.isfile(json_data['train']['final_model_name']) == True:
        model_path = json_data['train']['final_model_name']

    model = network(model_path, json_data['train']['input_shape'])

    model.train(
    json_data['data']['train_data'],
    json_data['data']['polygon_data'],
    json_data['data']['scaler_data'],
    json_data['train']['epochs'],
    json_data['train']['batch_size'],
    json_data['train']['initial_lr'],
    json_data['train']['model_callbacks'],
    json_data['train']['logs'],
    json_data['data']['label']
    )
