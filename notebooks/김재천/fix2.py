import codecs
import re
import traceback
import copy
import json

nb_in = (
    r"c:\Users\gorhk\Hotsix_Project\hotsix\notebooks\김재천\모델링_파생변수추가2.ipynb"
)
nb_out = r"c:\Users\gorhk\Hotsix_Project\hotsix\notebooks\김재천\모델링_파생변수추가2_수정.ipynb"

try:
    print("loading from:", nb_in)
    with open(nb_in, "r", encoding="utf-8") as f:
        nb = json.load(f)

    print("processing cells...")

    # 1. Baseline Replace string
    old_base = "thr,r,p,f1,ok = find_best_threshold_constrained(y_test, yp)"
    new_base = """if name == 'XGBoost':
        yp_tr = model.predict_proba(X_train_xgb)[:,1]
    else:
        yp_tr = model.predict_proba(X_train)[:,1]
    thr,_,_,_,_  = find_best_threshold_constrained(y_train, yp_tr)
    
    y_pred = (yp >= thr).astype(int)
    r = recall_score(y_test, y_pred, zero_division=0)
    p = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    ok = r >= 0.80"""

    # 2. S1 Replace
    old_s1 = """thr_s1, r_s1, p_s1, f1_s1, _ = find_best_threshold_constrained(
    y_test, y_prob_s1, target_recall=0.90)"""
    new_s1 = """from sklearn.model_selection import cross_val_predict
yp_oof = cross_val_predict(best_lgb, X_train, y_train, cv=5, method="predict_proba", n_jobs=8)[:, 1]
thr_s1, r_s1_tr, p_s1_tr, f1_s1_tr, _ = find_best_threshold_constrained(y_train, yp_oof, target_recall=0.90)"""

    # 3. Final Replace
    old_final = "thr_final, r_final, p_final, f1_final, ok_final = find_best_threshold_constrained(y_test, y_prob_final)"
    new_final = """yp_tr_s2 = cascade_xgb.predict_proba(X_train_s2)[:, 1]
thr_s2, _, _, _, _ = find_best_threshold_constrained(y_train_s2, yp_tr_s2, target_recall=0.85)

yp_te_s2 = cascade_xgb.predict_proba(X_test_suspect)[:, 1]
y_prob_final[suspect_idx] = yp_te_s2
y_pred_cascade[suspect_idx] = (yp_te_s2 >= thr_s2).astype(int)

r_final = recall_score(y_test, y_pred_cascade, zero_division=0)
p_final = precision_score(y_test, y_pred_cascade, zero_division=0)
f1_final = f1_score(y_test, y_pred_cascade, zero_division=0)
r_s1 = recall_score(y_test, y_pred_s1_test, zero_division=0)
p_s1 = precision_score(y_test, y_pred_s1_test, zero_division=0)
f1_s1 = f1_score(y_test, y_pred_s1_test, zero_division=0)
ok_final = r_final >= 0.80"""

    # 4. S10 Replace
    old_s10 = "thr_new, r_new, p_new, f1_new, _ = find_best_threshold_constrained(y_test, y_prob_lgb_new)"
    new_s10 = """thr_new = oof_thresholds['LightGBM']
y_pred_new = (y_prob_lgb_new >= thr_new).astype(int)
r_new = recall_score(y_test, y_pred_new, zero_division=0)
p_new = precision_score(y_test, y_pred_new, zero_division=0)
f1_new = f1_score(y_test, y_pred_new, zero_division=0)"""

    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            # Remove outputs to prevent Hang!
            cell["outputs"] = []

            src = "".join(cell.get("source", []))

            if old_base in src:
                src = src.replace(old_base, new_base)

            if old_s1 in src:
                src = src.replace(old_s1, new_s1)

            if old_final in src:
                src = src.replace(old_final, new_final)

            if old_s10 in src:
                src = src.replace(old_s10, new_s10)

            src = src.replace("f1_s1, ok_s1", "f1_s1_tr, True")

            # format back
            cell["source"] = [line + "\\n" for line in src.split("\\n")[:-1]] + [
                src.split("\\n")[-1]
            ]

    print("compact saving to:", nb_out)
    with open(nb_out, "w", encoding="utf-8") as f:
        # Prevent any massive indentation allocations
        json.dump(nb, f, ensure_ascii=False)

    print("done!")

except Exception as e:
    traceback.print_exc()
