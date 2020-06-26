# %% [code]
import logging
logging.getLogger().setLevel(logging.NOTSET)

import os
def commit_print(string):
    os.system(f'echo \"{string}\"')

import gc
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout, Average, Concatenate, GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from kaggle_datasets import KaggleDatasets
import transformers
from transformers import TFAutoModel, AutoTokenizer
from tqdm.notebook import tqdm
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, KFold, GroupKFold
import argparse

def head_tail_encode(texts, tokenizer, maxlen=512, head_ratio=0.25):
    enc_di = tokenizer.batch_encode_plus(
        texts, 
        return_attention_masks=False, 
        return_token_type_ids=False,
    )
    sentence_list = enc_di['input_ids']
    head_len = int(maxlen * head_ratio)
    tail_len = maxlen - head_len
    new_sentence_list = []
    for token_list in sentence_list:
        if len(token_list) < maxlen:
            new_token_list = token_list + [tokenizer.pad_token_id] * (maxlen - len(token_list))
        else:
            new_token_list = token_list[:head_len] + token_list[-tail_len:]
        new_sentence_list.append(new_token_list)
    return np.array(new_sentence_list)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def my_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int)
    parser.add_argument('--tot_path', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fold_seed', type=int, default=0)
    parser.add_argument('--loss', default="binary_crossentropy")
    parser.add_argument('--focal_gamma', type=float, default=1.5)
    parser.add_argument('--focal_alpha', type=float, default=0.2)
    parser.add_argument('--focal_ls', type=float, default=0.0)
    parser.add_argument('--cos_min_lr', type=float, default=2e-6)
    parser.add_argument('--cos_max_lr', type=float, default=1e-5)
    parser.add_argument('--cos_lr_tmax', type=int, default=2)
    parser.add_argument('--n_epoch', type=int, default=5)
    parser.add_argument('--album_sample_weight', type=float, default=1.0)
    parser.add_argument('--neg_class_weight', type=float, default=1.0)
    parser.add_argument('--pos_class_weight', type=float, default=2.718)
    parser.add_argument('--checkpoint_metrics', default="val_auc")
    parser.add_argument('--swa', type=str2bool, default=True)
    parser.add_argument('--swa_start_epoch', type=int, default=3)
    parser.add_argument('--swa_lr', type=float, default=2e-6)
    parser.add_argument('--valid_pl', type=str2bool, default=False)
    parser.add_argument('--valid_pl_path', default="")
    parser.add_argument('--trans_back', type=str2bool, default=True)
    args = parser.parse_args()
    print("all arguments value:")
    for k, v in vars(args).items():
        print(k, v, type(v))
    return args
    
from tensorflow.keras import backend as K

def focal_loss(gamma=2., alpha=.2, ls=0.1): #ls is label smooth paramter
    def focal_loss_fixed_true(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    def focal_loss_fixed(y_true, y_pred):
        sum_loss = focal_loss_fixed_true(y_true, y_pred) * (1-ls)
        sum_loss += focal_loss_fixed_true(1-y_true, y_pred) * (ls)
        return sum_loss
    return focal_loss_fixed

def build_model(transformer, loss='binary_crossentropy', max_len=512):
    """
    https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras
    """
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]

    mean_token = GlobalAveragePooling1D()(sequence_output)
    max_token = GlobalMaxPooling1D()(sequence_output)
    all_token = Concatenate(name = "all_token")([mean_token, max_token])
    all_token = Dropout(0.3)(all_token)
    out = Dense(1, activation='sigmoid')(all_token)
    
    
    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(lr=5e-6), loss=loss, metrics=[tf.keras.metrics.AUC()])
    
    return model

def load_tta_files():
    # load multi lang TTA data
    data_dir = "/kaggle/input/jmt-preprocess-v3/"
    valid_mat = {}
    valid_back_mat = {}
    test_mat = {}
    test_back_mat = {}
    for lang in ['en', 'es', 'tr', 'pt', 'ru', 'it', 'fr']:
        print(lang)
        valid_mat[lang] = np.load(data_dir + 'x_valid_'+lang+".npy")
        valid_back_mat[lang] = np.load(data_dir + 'x_valid_back_'+lang+".npy")
        test_mat[lang] = np.load(data_dir + 'x_test_'+lang+".npy")
        test_back_mat[lang] = np.load(data_dir + 'x_test_back_'+lang+".npy")
    return valid_mat, valid_back_mat, test_mat, test_back_mat

def inference_full_tta(model, val_idx, oof_tta_dict, test_tta_dict, valid_mat, valid_back_mat, test_mat, test_back_mat):
    val_ds_dict = {}
    val_ds_back_dict = {}
    test_ds_dict = {}
    test_ds_back_dict = {}
    for lang in ['en', 'es', 'tr', 'pt', 'ru', 'it', 'fr']:
        print(lang)
        # prep data
        val_ds_lang = (tf.data.Dataset.from_tensor_slices(valid_mat[lang][val_idx]).batch(BATCH_SIZE))
        val_ds_back_lang = (tf.data.Dataset.from_tensor_slices(valid_back_mat[lang][val_idx]).batch(BATCH_SIZE))
        # inference
        oof_tta_dict['pred_'+lang][val_idx] = model.predict(val_ds_lang, verbose=1)
        oof_tta_dict['pred_'+lang+"_back"][val_idx] = model.predict(val_ds_back_lang, verbose=1)
        # prep data
        test_ds_lang = (tf.data.Dataset.from_tensor_slices(test_mat[lang]).batch(BATCH_SIZE))
        test_ds_back_lang = (tf.data.Dataset.from_tensor_slices(test_back_mat[lang]).batch(BATCH_SIZE))
        # inference
        test_tta_dict['pred_'+lang] = model.predict(test_ds_lang, verbose=1)
        test_tta_dict['pred_'+lang+"_back"] = model.predict(test_ds_back_lang, verbose=1)
        del val_ds_lang, val_ds_back_lang, test_ds_lang, test_ds_back_lang
        gc.collect()
        gc.collect()

def inference(model, val_idx, oof_tta_dict, test_tta_dict, valid_mat, valid_back_mat, test_mat, test_back_mat):
    val_ds_dict = {}
    val_ds_back_dict = {}
    test_ds_dict = {}
    test_ds_back_dict = {}
    for lang in ['en', 'es', 'tr', 'pt', 'ru', 'it', 'fr']:
        print(lang)
        val_ds_back_lang = (tf.data.Dataset.from_tensor_slices(valid_back_mat[lang][val_idx]).batch(BATCH_SIZE))
        oof_tta_dict['pred_'+lang+"_back"][val_idx] = model.predict(val_ds_back_lang, verbose=1).T[0]
        test_ds_back_lang = (tf.data.Dataset.from_tensor_slices(test_back_mat[lang]).batch(BATCH_SIZE))
        test_tta_dict['pred_'+lang+"_back"] = model.predict(test_ds_back_lang, verbose=1).T[0]
        del val_ds_back_lang, test_ds_back_lang
        gc.collect()
        gc.collect()
        
def inference2(model, val_idx, oof_tta_dict, test_tta_dict, valid_mat, valid_back_mat, test_mat, test_back_mat):
    val_ds_dict = {}
    val_ds_back_dict = {}
    test_ds_dict = {}
    test_ds_back_dict = {}
    for lang in ['en', 'es', 'tr', 'pt', 'ru', 'it', 'fr']:
        print(lang)
        val_ds_back_lang = (tf.data.Dataset.from_tensor_slices(valid_mat[lang][val_idx]).batch(BATCH_SIZE))
        oof_tta_dict['pred_'+lang+"_back"][val_idx] = model.predict(val_ds_back_lang, verbose=1).T[0]
        test_ds_back_lang = (tf.data.Dataset.from_tensor_slices(test_mat[lang]).batch(BATCH_SIZE))
        test_tta_dict['pred_'+lang+"_back"] = model.predict(test_ds_back_lang, verbose=1).T[0]
        del val_ds_back_lang, test_ds_back_lang
        gc.collect()
        gc.collect()

def prepare_tta_storage(val_len, test_len):
    oof_tta_dict = {}
    test_tta_dict = {}
    for lang in ['en', 'es', 'tr', 'pt', 'ru', 'it', 'fr']:
        oof_tta_dict['pred_'+lang+"_back"] = np.zeros(val_len)
        test_tta_dict['pred_'+lang+"_back"] = np.zeros(test_len)
    return oof_tta_dict, test_tta_dict

import math
def build_cosine_annealing_lr(eta_min=3e-6, eta_max=1e-5, T_max=10):
    def lrfn(epoch):
        lr = eta_min + (eta_max - eta_min) * (1 + math.cos(math.pi * epoch / T_max)) / 2
        return lr
    return lrfn

if __name__ == "__main__":
    args = my_parse_args()
    
    # Detect hardware, return appropriate distribution strategy
    try:
        # TPU detection. No parameters necessary if TPU_NAME environment variable is
        # set: this is always the case on Kaggle.
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Running on TPU ', tpu.master())
    except ValueError:
        tpu = None

    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    else:
        # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
        strategy = tf.distribute.get_strategy()

    print("REPLICAS: ", strategy.num_replicas_in_sync)

    AUTO = tf.data.experimental.AUTOTUNE

    # Configuration
    BATCH_SIZE = 8 * strategy.num_replicas_in_sync
    MAX_LEN = 192
    PRESAVED_MODEL = r"/kaggle/input/jmt-save-model-xlmrl"
    TOKENIZER_MODEL = 'jplu/tf-xlm-roberta-large'
    # r"/kaggle/input/jmt-pretrain-tot-0611-esxe-v8-pl/"
    MODEL_PATH = args.tot_path 
    TEST_DATA_PATH = r"/kaggle/input/jmt-preprocess-head-tail-v2/"
    SEED = args.seed
    EPOCHS = args.n_epoch
    ES_METRIC = args.checkpoint_metrics
    
    print(f"training each fold for 2*{EPOCHS} epochs")
    print(f"TOT path: {MODEL_PATH}")
    print(f"using seed {SEED}")
    print(f"splitting 5 folds using fold seed {args.fold_seed}")
    if args.swa:
        print("using SWA in this experiment")
    else:
        print(f"check point metrics {ES_METRIC}")
        
    if args.valid_pl:
        assert args.valid_pl_path!=''
        print(f"using 0.5*(valid pl+valid true) as validation label, valid path: {args.valid_pl_path}")
    
    
    ################ Data prep
    valid = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')
    test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')
    sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')
    valid_album = pd.read_csv('/kaggle/input/nlp-albumentations-xl-v3/valid_album.csv').sort_values(by='id').reset_index(drop=True)
    
    if args.valid_pl:
        valid_pl_pred = pd.read_csv(os.path.join(args.valid_pl_path, "valid_pred.csv"))
        valid['toxic_pl'] = (valid['toxic'] + valid_pl_pred['pred'])/2
        valid_album['toxic_pl'] = (valid_album['toxic'] + valid_pl_pred['pred'])/2
    
    valid = pd.concat([valid,
                       valid_album.rename(columns={'comment_transform':'comment_text'})
                      ],axis=0,ignore_index=True)
    
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
    x_valid = head_tail_encode(valid.comment_text.values, tokenizer, maxlen=MAX_LEN)
    y_valid = valid.toxic.values
    x_test = np.load(os.path.join(TEST_DATA_PATH, 'x_test.npy'))
    
    test_dataset = (
        tf.data.Dataset
        .from_tensor_slices(x_test)
        .batch(BATCH_SIZE)
    )

    ############ CV
    from sklearn.model_selection import train_test_split, KFold, GroupKFold, StratifiedKFold
    NFOLDS = 5
    valid_orig = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')
    valid_orig['label'] = valid_orig.apply(lambda x: str(x['lang']) + str(x['toxic']), axis=1)
    folds = StratifiedKFold(n_splits=NFOLDS, random_state=args.fold_seed, shuffle=True)
    splits = folds.split(valid_orig['comment_text'].values, valid_orig['label'].values)
    skf_split = np.array(list(splits))
    
    ##################
    # variable prep
    oof = np.zeros(int(y_valid.shape[0]/2))
    oof_rank = np.zeros(int(y_valid.shape[0]/2))
    pred = np.zeros(len(x_test))
    pred_rank = np.zeros(len(x_test))
    
    ##################
    # TTA prep
    valid_mat, valid_back_mat, test_mat, test_back_mat = load_tta_files()
    oof_tta_dict, test_tta_dict = prepare_tta_storage(len(valid_orig), len(x_test))

    ###################
    # start CV
    for fold_, (trn_idx, val_idx) in enumerate(skf_split):
        # only train this fold that we specified in the args
        if int(args.fold) != fold_:
            continue
        else:
            commit_print("begin fold " + str(fold_))

        # shuffle validation data by id so val and val_aug are still in the same batch
        id_index_map = valid['id'].reset_index().iloc[list(trn_idx) + list(trn_idx+8000)
                                      ].sort_values(by=['id', 'index']).set_index('id')
        np.random.seed(SEED+fold_)
        random_id_list = np.random.permutation(np.array(pd.unique(id_index_map.index)))
        full_trn_idx = id_index_map.loc[random_id_list]['index'].values
        full_val_idx = val_idx
        tr_sample_weight = np.ones(len(full_trn_idx))
        tr_sample_weight[full_trn_idx > 8000] = args.album_sample_weight
        print("valid albumentations weight %.5f\n"%args.album_sample_weight)
        print("training for %d"%fold_)
        print("training index: ",full_trn_idx)
        print("validation index: ",full_val_idx)
        print('valid-subset lang composition:', valid.iloc[full_val_idx]['lang'].value_counts().to_dict())

        if strategy.num_replicas_in_sync == 1: # for debug purpose
            def build_model(transformer, loss='binary_crossentropy', max_len=512):
                """
                https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras
                """
                input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
                out = Dense(1, activation='sigmoid')(input_word_ids)
                model = Model(inputs=input_word_ids, outputs=out)
                model.compile(Adam(lr=1e-5), loss=loss, metrics=[tf.keras.metrics.AUC()])
                return model
            transformer_layer = [0]
            model = build_model(transformer_layer, max_len=MAX_LEN)
        else:
            with strategy.scope():
                transformer_layer = TFAutoModel.from_pretrained(PRESAVED_MODEL)
                print(f"using loss function {args.loss}")
                if args.loss == "binary_crossentropy":
                    print("using binary crossentropy loss")
                    model = build_model(transformer_layer, loss='binary_crossentropy', max_len=MAX_LEN)
                elif args.loss == 'focal':
                    print(f"using focal loss with gamma: {args.focal_gamma}, alpha: {args.focal_alpha}, label smoothing: {args.focal_ls}")
                    model = build_model(transformer_layer, loss=focal_loss(gamma=args.focal_gamma, alpha=args.focal_alpha, ls=args.focal_ls), max_len=MAX_LEN)
                else:
                    raise ValueError("model loss has to be either focal or binary_crossentropy")

            print("\n Loading model weights \n")
            model.load_weights(os.path.join(MODEL_PATH, "model.h5"))
            print("load weights done\n")
        
        if args.valid_pl:
            print("using valid set pl")
            y_valid_pl = valid.toxic_pl.values
            tr_x, tr_y = x_valid[full_trn_idx,:], y_valid_pl[full_trn_idx]
            vl_x, vl_y = x_valid[full_val_idx,:], y_valid[full_val_idx]
        else:
            tr_x, tr_y = x_valid[full_trn_idx,:], y_valid[full_trn_idx]
            vl_x, vl_y = x_valid[full_val_idx,:], y_valid[full_val_idx]
            
        fold_tr_ds = (
            # refer to xhlulu validation training for the fold
            tf.data.Dataset
            .from_tensor_slices((tr_x, tr_y, tr_sample_weight))
            .batch(BATCH_SIZE)
            .cache()
            .prefetch(AUTO)
        )
        fold_vl_ds = (
            tf.data.Dataset
            .from_tensor_slices((vl_x, vl_y))
            .batch(BATCH_SIZE)
            .cache()
            .prefetch(AUTO)
        )

        fold_checkpoint_path = 'cv_fold_model.h5'
        ########## define callback for checkpoint ensemble
        from tensorflow.keras.callbacks import Callback
        if ES_METRIC == 'val_auc':
            class CustomCheckPoint(Callback):
                def __init__(self):
                    self.max_auc = 0

                def on_epoch_end(self, epoch, logs=None):
                    gc.collect()
                    # save model if val auc improved
                    if logs['val_auc'] > self.max_auc:
                        self.max_auc = logs['val_auc']
                        # note here I'm using global variable to try to avoid memory issue
                        print("auc improved ... saving model weights ...")
                        model.save_weights(fold_checkpoint_path)
                        gc.collect()
        elif ES_METRIC == 'val_loss':
            class CustomCheckPoint(Callback):
                def __init__(self):
                    self.min_loss = 100

                def on_epoch_end(self, epoch, logs=None):
                    gc.collect()
                    # save model if val auc improved
                    if logs['val_loss'] < self.min_loss:
                        self.min_loss = logs['val_loss']
                        # note here I'm using global variable to try to avoid memory issue
                        print("loss improved ... saving model weights ...")
                        model.save_weights(fold_checkpoint_path)
                        gc.collect()

        model_checkpoint_callback = CustomCheckPoint()
        n_steps = tr_x.shape[0] // BATCH_SIZE // EPOCHS
        
        print(f"using cosine annealing lr with lr_min: {args.cos_min_lr}, lr_max: {args.cos_max_lr}, lr_tmax: {args.cos_lr_tmax}")
        _lrfn = build_cosine_annealing_lr(eta_min=args.cos_min_lr, eta_max=args.cos_max_lr, T_max=args.cos_lr_tmax)
        lr_schedule = tf.keras.callbacks.LearningRateScheduler(_lrfn, verbose=1)
        
        from swa.tfkeras import SWA
        swa_cb = SWA(start_epoch=args.swa_start_epoch, 
                  lr_schedule='manual', 
                  swa_lr=args.swa_lr,
                  verbose=1)
        
        if args.swa: 
            CBs = [lr_schedule, swa_cb]
        else:
            CBs = [lr_schedule, model_checkpoint_callback]

        commit_print("start training for fold " + str(fold_))
        model.fit(
            fold_tr_ds.repeat(),
            steps_per_epoch=n_steps,
            validation_data=fold_vl_ds,
            epochs=2*EPOCHS,
            callbacks=CBs,
            class_weight={0:args.neg_class_weight, 1:args.pos_class_weight},
        )
        
        commit_print("finish training for fold " + str(fold_))
        gc.collect()
        if not args.swa:
            print("not swa, loading checkpoint weight for inference")
            model.load_weights(fold_checkpoint_path)
        gc.collect()

        commit_print("running inference for fold " + str(fold_))
        oof_preds = model.predict(fold_vl_ds, verbose=1).T[0]
        print("fold auc:", roc_auc_score(vl_y, oof_preds))
        oof[full_val_idx]=oof_preds
        oof_rank[full_val_idx] = np.array(pd.Series(np.array(oof_preds)).rank(pct=True))
        gc.collect()

        test_preds = model.predict(test_dataset, verbose=1).T[0]
        pred += test_preds/NFOLDS
        pred_rank += np.array(pd.Series(test_preds).rank(pct=True))/NFOLDS
        gc.collect()
        commit_print("finish inference for fold " + str(fold_))
        
        commit_print("saving files for fold " + str(fold_))
        np.save("oof_preds_fold_" + str(fold_), oof_preds)
        np.save("val_idx_fold_" + str(fold_), full_val_idx)
        np.save("pred_fold_" + str(fold_), test_preds)
        
        commit_print("saving TTA for fold " + str(fold_))
        if args.trans_back:
            print("using trans back as TTA")
            inference(model, full_val_idx, oof_tta_dict, test_tta_dict, valid_mat, valid_back_mat, test_mat, test_back_mat)
        else:
            print("using trans as TTA")
            inference2(model, full_val_idx, oof_tta_dict, test_tta_dict, valid_mat, valid_back_mat, test_mat, test_back_mat)
        pd.DataFrame(oof_tta_dict).to_csv("oof_tta_fold_"+str(fold_)+".csv")
        pd.DataFrame(test_tta_dict).to_csv("test_tta_fold_"+str(fold_)+".csv")

        del fold_vl_ds, fold_tr_ds, tr_x, tr_y, vl_x, vl_y, model, transformer_layer, model_checkpoint_callback
        gc.collect()

        # delete h5 at end of fold to speed up ensemble loading
        if os.path.exists(fold_checkpoint_path):
            os.system("rm " + fold_checkpoint_path)