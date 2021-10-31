import os
import argparse

root_path = "/mnt/inspurfs/user-fs/zhaoyu/GlyphBERT"
data_path = "/mnt/inspurfs/user-fs/zhaoyu/pretrain_data"


def get_path(p, is_data=False):
    if is_data:
        return os.path.join(data_path, p)
    else:
        return os.path.join(root_path, p)


config = {

    "device": "7",
    "epoch": 15,
    "batch_size": 32,
    "batch_expand_times": 1,
    "warm_up": 0.1,
    "weight_decay": 0,
    "steps_eval": 0.2,
    "start_eval_epoch": 1,
    "exp_times": 3,
    "lr": 1e-5,
    "bmp_path": get_path("data/bmp", is_data=True),
    "vocab_path": get_path("data/vocab_bmp.txt", is_data=True),
    "bert_config_path": get_path("pretrained_model/config.json"),
    "preprocessing": False,
    "use_res2bert": True,
    "cnn_and_embed_mat": True,
    "parallel": None,
    "vocab_size": 18612,
    "save_root": "/mnt/inspurfs/user-fs/zhaoyu/GlyphBERT/downstream/save",
}

parser = argparse.ArgumentParser()
parser.add_argument('--device', default=config.get('device'), type=str, required=False)
parser.add_argument('--epoch', default=config.get('epoch'), type=int, required=False)
parser.add_argument('--batch_size', default=config.get('batch_size'), type=int, required=False)
parser.add_argument('--batch_expand_times', default=config.get('batch_expand_times'), type=int, required=False)
parser.add_argument('--warm_up', default=config.get('warm_up'), type=float, required=False)
parser.add_argument('--weight_decay', default=config.get('weight_decay'), type=float, required=False)
parser.add_argument('--lr', default=config.get('lr'), type=float, required=False)
parser.add_argument('--steps_eval', default=config.get('steps_eval'), type=float, required=False)
parser.add_argument('--start_eval_epoch', default=config.get('start_eval_epoch'), type=int, required=False)
parser.add_argument('--dataset_name', type=str, required=True)
parser.add_argument('--exp_times', default=config.get('exp_times'), type=int, required=False)
parser.add_argument('--pretrained_model_path', type=str, required=True)
parser.add_argument('--use_res2bert', default=config.get('use_res2bert'), type=bool, required=False)
parser.add_argument('--cnn_and_embed_mat', default=config.get('cnn_and_embed_mat'), type=bool, required=False)
parser.add_argument('--state_dict', default=config.get('state_dict'), type=str, required=False)
parser.add_argument('--save_root', default=config.get('save_root'), type=str, required=False)
args = vars(parser.parse_args())

for k in args.keys():
    if k not in config.keys():
        print("add new config key: {}={}".format(k, args[k]))
    config[k] = args[k]

os.environ["CUDA_VISIBLE_DEVICES"] = config['device']

# senti
if config['dataset_name'] == "chnsenti":
    config.update({
        "train_data_path": get_path("./data/senti_train.pkl", is_data=True),
        "dev_data_path": get_path("./data/senti_dev.pkl", is_data=True),
        "test_data_path": get_path("./data/senti_test.pkl", is_data=True),
    })
elif config['dataset_name'] == 'hotel':
    config.update({
        "train_data_path": get_path("./data/hotel_train.pkl", is_data=True),
        "dev_data_path": get_path("./data/hotel_dev.pkl", is_data=True),
        "test_data_path": get_path("./data/hotel_test.pkl", is_data=True),
    })
# few shot
elif config['dataset_name'] == "abs_cls":
    config.update({
        "train_data_path": get_path("./data/abs_cls_train.pkl", is_data=True),
        "dev_data_path": get_path("./data/abs_cls_dev.pkl", is_data=True),
        "test_data_path": get_path("./data/abs_cls_test.pkl", is_data=True),
    })
elif config['dataset_name'] == "onlinesenti_cls":
    config.update({
        "train_data_path": get_path("./data/onlinesenti_cls_train.pkl", is_data=True),
        "dev_data_path": get_path("./data/onlinesenti_cls_dev.pkl", is_data=True),
        "test_data_path": get_path("./data/onlinesenti_cls_test.pkl", is_data=True),
    })

print(config)
print("")
