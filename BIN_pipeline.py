import datetime
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from load_data import load_polarity_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from imblearn.over_sampling import RandomOverSampler

from modules.polarity.model import SentimentModel

TRAIN_PATH = 'data/input/SP/tech_shopee_train.csv'
TEST_PATH = 'data/input/SP/tech_shopee_test.csv'


def train_SP_chi2(aspect, seed, X_train, y_train, X_test, y_test, model, save=True):
    print('=' * 20 + ' Training SP with Chi2 ' + aspect + ' ' + (50 - len(' Training SP with Chi2 ')) * '=')
    model.chi2_dict_lda()
    model.lda_dict()
    _y_train = [i.score for i in y_train]
    _y_test = [i.score for i in y_test]
    print('* Representing using Chi2 ...')
    _X_train = model.chi2_represent(X_train)
    _X_test = model.chi2_represent(X_test)
    print('  Representing using Chi2 DONE!')

    print('* Training ...')
    model.train(_X_train, _y_train)
    print('  Training DONE!')
    predict = model.predict(_X_test)

    neg_p, neg_r, neg_f1 = model.evaluate_lda(_y_test, predict, -1)
    pos_p, pos_r, pos_f1 = model.evaluate_lda(_y_test, predict, 1)
    print("- p negative     :", neg_p)
    print("- r negative     :", neg_r)
    print("- F1 negative      :", neg_f1)
    print("- p positive     :", pos_p)
    print("- r positive     :", pos_r)
    print("- F1 positive      :", pos_f1)

    result = pd.DataFrame({'score': [aspect, seed, model.model, neg_p, neg_r, neg_f1, neg_p, neg_r, neg_f1]},
                          index=['Aspect', 'seed', 'Model', 'p-negative', 'r-negative', 'F1-negative', 'p-positive',
                                 'r-positive', 'F1-positive'])

    if save:
        result.to_csv('./data/output/evaluate/SP/lda/SP_result_{m}_{a}_seed{seed}.csv'.format(
            m=model.model,
            a=aspect,
            seed=seed
        ))
    return neg_f1, pos_f1


if __name__ == '__main__':
    test_df = pd.read_csv(TEST_PATH)
    print(test_df.giá.value_counts())
    model = LogisticRegression()
    aspect_list = ['giá', 'dịch_vụ', 'ship', 'hiệu_năng', 'chính_hãng', 'cấu_hình', 'phụ_kiện', 'mẫu_mã']
    neg_f1 = []
    pos_f1 = []
    seed = 0
    domain = 'tech_shopee'
    DOMAIN = 'tech'
    for aspect in aspect_list:
        X_train, y_train = load_polarity_data(path=TRAIN_PATH,
                                              stc_idx_col_name='id',
                                              stc_col_name='cmt',
                                              label_col_name=aspect)

        X_test, y_test = load_polarity_data(path=TEST_PATH,
                                            stc_idx_col_name='id',
                                            stc_col_name='cmt',
                                            label_col_name=aspect)
        vocab_path_chi2 = './data/output/chi2_score_dict/SP_{domain}_{aspect}.csv'.format(domain=domain,aspect=aspect)
        vocab_path_lda = './data/output/lda/total_seed{seed}/{domain}_{aspect}_seed{seed}.csv'.format(domain=DOMAIN,
                                                                                                  seed=seed,
                                                                                                  aspect=aspect)
        SP_model = SentimentModel(vocab_path_chi2, vocab_path_lda, model)
        _neg_f1, _pos_f1 = train_SP_chi2(aspect, seed, X_train, y_train, X_test, y_test, SP_model, save=False)
        neg_f1.append(_neg_f1)
        pos_f1.append(_pos_f1)
    macro_neg_f1 = np.array(neg_f1).mean()
    macro_pos_f1 = np.array(pos_f1).mean()
    print('=' * 20 + ' Performance of SP model ' + (50 - len(' Performance of SP model ')) * '=')
    print("- Macro-F1 negative         :", macro_neg_f1)
    print("- Macro-P positive          :", macro_pos_f1)
    result = pd.DataFrame({'score': [model, macro_neg_f1, macro_pos_f1]},
                          index=['Model', 'Macro-F1 negative', 'Macro-F1 positive'])
    result.to_csv(
        './data/output/evaluate/SP/lda/SP_result_{m}_seed{seed}.csv'.format(m=model, seed=seed))
