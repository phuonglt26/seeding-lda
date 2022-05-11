import numpy as np
from sklearn.model_selection import KFold


def five_folds_cross_validation(inputs, outputs, model, aspectId):
    # Định nghĩa K-Fold CV
    kfold = KFold(n_splits=5, shuffle=True)
    # K-fold Cross Validation model evaluation
    print("Bắt đầu chạy 5-fold cho aspect", aspectId, ":")
    fold_idx = 1
    X = inputs
    Y = outputs
    f1_list = []
    p_list = []
    r_list = []
    for train_ids, val_ids in kfold.split(X, Y):
        model = model
        X_train = [X[train_id] for train_id in train_ids]
        Y_train = [Y[train_id] for train_id in train_ids]
        model.train(X_train, Y_train)
        X_val = [X[val_id] for val_id in val_ids]
        Y_val = [Y[val_id] for val_id in val_ids]
        predicts = model.predict(X_val, aspectId)
        _, _, _, p, r, f1 = model.evaluate(Y_val, predicts)
        p_list.append(p)
        r_list.append(r)
        f1_list.append(f1)
        print("fold", fold_idx, f1)
        print("đã chạy xong fold", fold_idx)
        fold_idx = fold_idx + 1
    print("Đánh giá tổng thể các folds: ")
    print("Trung bình p các folds: ", np.mean(p_list), "(Độ lệch ",
          "+- ", np.std(p_list), " )")
    print("Trung bình r các folds: ", np.mean(r_list), "(Độ lệch ",
          "+- ", np.std(r_list), " )")
    print("Trung bình f1 các folds: ", np.mean(f1_list), "(Độ lệch ",
          "+- ", np.std(f1_list), " )")
    print("chạy xong 5-fold cho aspect", aspectId)
    print("----------------------------------------")
