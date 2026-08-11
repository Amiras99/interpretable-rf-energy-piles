import os
import random
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from SALib.sample import sobol
from SALib.analyze import sobol as sobol_analyze


# ==========================================================
# Reproducibility
# ==========================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# ==========================================================
# Logging
# ==========================================================
logging.basicConfig(level=logging.INFO, format="%(message)s")
plt.style.use("seaborn-v0_8-whitegrid")


def main():

    # ==========================================================
    # Data Loading
    # ==========================================================
    base_path = input("Enter folder path containing data.csv:\n> ").strip().replace('"', '').replace("'", "")
    data_path = os.path.join(base_path, "data.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError("data.csv not found.")

    data = pd.read_csv(data_path)
    logging.info("Data loaded successfully.")

    expected_cols = ['Days', 'Pile_Group', 'S/D', 'L/D', 'IF']
    if not all(col in data.columns for col in expected_cols):
        raise ValueError("Dataset does not contain expected columns.")

    # ==========================================================
    # Features / Target
    # ==========================================================
    categorical_features = ['Pile_Group']
    numerical_features = ['Days', 'S/D', 'L/D']

    X = data[categorical_features + numerical_features]
    y = data['IF']
    strata = data["Days"].astype(str) + "_" + data["Pile_Group"].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, stratify=strata, random_state=SEED
)
    # ==========================================================
    # Preprocessing
    # ==========================================================
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
            ("num", "passthrough", numerical_features)
        ]
    )

    # ==========================================================
    # Model + Pipeline
    # ==========================================================
    rf = RandomForestRegressor(random_state=SEED, n_jobs=-1)

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", rf)
        ]
    )

    # ==========================================================
    # Hyperparameter Grid
    # ==========================================================
    param_grid = {
        "model__n_estimators": [100, 300, 500],
        "model__max_depth": [15, 25, None],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", 1.0]
    }

    # ==========================================================
    # Hyperparameter Optimization using GridSearchCV
    # ==========================================================
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

    grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=inner_cv,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    return_train_score=True
    )

    logging.info("Training Random Forest with GridSearchCV...")
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    joblib.dump(best_model, os.path.join(base_path, "rf_pipeline.pkl"))
    logging.info("Best Hyperparameters:")
    logging.info(grid.best_params_)
    cv_results = pd.DataFrame(grid.cv_results_)

    cv_results.to_csv(
        os.path.join(base_path, "GridSearch_Results.csv"),
       index=False
       )

    logging.info("GridSearch results saved.")
    
    # ==========================================================
    # Evaluation
    # ==========================================================

    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    def smape(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        denominator = np.abs(y_true) + np.abs(y_pred)
        mask = denominator != 0
        return np.mean(2 * np.abs(y_pred[mask] - y_true[mask]) / denominator[mask]) * 100

    train_smape = smape(y_train, y_train_pred)
    test_smape = smape(y_test, y_test_pred)
    
    
    
    
    

    logging.info(f"Train = R2: {train_r2:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, SMAPE: {train_smape:.2f}%")
    logging.info(f"Test = R2: {test_r2:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, SMAPE: {test_smape:.2f}%")
    # ==========================================================
    # Comprehensive Visualization (Legacy Output)
    # ==========================================================
    logging.info("Generating comprehensive legacy visualization...")

    from sklearn.inspection import permutation_importance

    # Permutation Feature Importance
    X_test_transformed = best_model.named_steps["preprocess"].transform(X_test)

    perm = permutation_importance(
        best_model.named_steps["model"],
        X_test_transformed,
        y_test,
        n_repeats=20,
        random_state=SEED,
        n_jobs=-1
        )

    preproc = best_model.named_steps["preprocess"]
    feature_names = preproc.get_feature_names_out()

    idx = np.argsort(perm.importances_mean)[::-1][:10]

    # Comprehensive Figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        "Performance Analysis of Random Forest Model",
        fontsize=16,
        fontweight="bold"
        )

    # 1. Actual vs Predicted (Train)
    axes[0, 0].scatter(y_train, y_train_pred, alpha=0.6, edgecolors="k", s=50)
    axes[0, 0].plot(
        [y_train.min(), y_train.max()],
        [y_train.min(), y_train.max()],
        "r--", lw=2
        )
    axes[0, 0].set_xlabel("Actual IF")
    axes[0, 0].set_ylabel("Predicted IF")
    axes[0, 0].set_title("Actual vs Predicted (Train)")
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Actual vs Predicted (Test)
    axes[0, 1].scatter(y_test, y_test_pred, alpha=0.6, edgecolors="k",
                       s=50, color="orange")
    axes[0, 1].plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "r--", lw=2
        )
    axes[0, 1].set_xlabel("Actual IF")
    axes[0, 1].set_ylabel("Predicted IF")
    axes[0, 1].set_title("Actual vs Predicted (Test)")
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Residuals (Train)
    residuals_train = y_train - y_train_pred
    axes[0, 2].scatter(y_train_pred, residuals_train,
                       alpha=0.6, edgecolors="k", s=50)
    axes[0, 2].axhline(0, color="r", linestyle="--", lw=2)
    axes[0, 2].set_xlabel("Predicted IF")
    axes[0, 2].set_ylabel("Residuals")
    axes[0, 2].set_title("Residuals (Train)")
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Residuals (Test)
    residuals_test = y_test - y_test_pred
    axes[1, 0].scatter(y_test_pred, residuals_test,
                       alpha=0.6, edgecolors="k",
                       s=50, color="orange")
    axes[1, 0].axhline(0, color="r", linestyle="--", lw=2)
    axes[1, 0].set_xlabel("Predicted IF")
    axes[1, 0].set_ylabel("Residuals")
    axes[1, 0].set_title("Residuals (Test)")
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Permutation Feature Importance
    axes[1, 1].barh(
        np.array(feature_names)[idx],
        perm.importances_mean[idx],
        color="forestgreen",
        edgecolor="black"
        )
    axes[1, 1].set_xlabel("Importance Score")
    axes[1, 1].set_title("Relative Importance of Input Parameters")
    axes[1, 1].invert_yaxis()
    axes[1, 1].grid(True, axis="x", alpha=0.3)

    # 6. Metrics Comparison
    metrics = ["R2", "RMSE", "MAE"]
    train_metrics = [train_r2, train_rmse, train_mae]
    test_metrics = [test_r2, test_rmse, test_mae]

    x = np.arange(len(metrics))
    width = 0.35

    axes[1, 2].bar(x - width/2, train_metrics, width,
                   label="Train", color="steelblue", edgecolor="black")
    axes[1, 2].bar(x + width/2, test_metrics, width,
                   label="Test", color="coral", edgecolor="black")

    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(metrics)
    axes[1, 2].set_ylabel("Value")
    axes[1, 2].set_title("Model Performance Metrics")
    axes[1, 2].legend()
    axes[1, 2].grid(True, axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    legacy_fig_path = os.path.join(base_path, "RF_Analysis.png")
    plt.savefig(legacy_fig_path, dpi=600, bbox_inches="tight")
    plt.show()
    plt.close()
    logging.info(f"Legacy comprehensive visualization saved at: {legacy_fig_path}")
    # ==========================================================
    # Design-Oriented Parametric Sensitivity Analysis (OAT)
    # ==========================================================
    logging.info("Running design-oriented parametric sensitivity analysis...")

    ref_sd = X["S/D"].median()
    ref_ld = X["L/D"].median()
    ref_pile = X["Pile_Group"].mode()[0]

    sd_range = np.linspace(X["S/D"].min(), X["S/D"].max(), 50)
    ld_range = np.linspace(X["L/D"].min(), X["L/D"].max(), 50)

    def parametric_prediction(ref_days, variable_name, values):
        df = pd.DataFrame({
            "Days": ref_days,
            "Pile_Group": ref_pile,
            "S/D": ref_sd,
            "L/D": ref_ld
            }, index=range(len(values)))
        df[variable_name] = values
        return best_model.predict(df)

    for ref_days in sorted(X["Days"].unique()):
        if_sd = parametric_prediction(ref_days, "S/D", sd_range)
        if_ld = parametric_prediction(ref_days, "L/D", ld_range)

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(sd_range, if_sd, lw=2)
        plt.xlabel("S/D")
        plt.ylabel("IF")
        plt.title(f"Parametric Sensitivity: IF vs S/D (Day {ref_days})")
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(ld_range, if_ld, lw=2, color="darkorange")
        plt.xlabel("L/D")
        plt.ylabel("IF")
        plt.title(f"Parametric Sensitivity: IF vs L/D (Day {ref_days})")
        plt.grid(True, alpha=0.3)

        param_fig_path = os.path.join(base_path, f"Design_Parametric_Sensitivity_Day{ref_days}.png")
        plt.tight_layout()
        plt.savefig(param_fig_path, dpi=600, bbox_inches="tight")
        plt.show()
        plt.close()
        logging.info(f"Design-oriented sensitivity plots saved at: {param_fig_path}")

    # ==========================================================
    # Global Sensitivity Analysis (Sobol Indices)
    # ==========================================================
    logging.info("Performing Sobol global sensitivity analysis...")

    for pile_group in sorted(X["Pile_Group"].unique()):
        logging.info(f"Sobol analysis for Pile_Group = {pile_group}")

        problem = {
            "num_vars": 3,
            "names": ["Days", "S/D", "L/D"],
            "bounds": [
                [X["Days"].min(), X["Days"].max()],
                [X["S/D"].min(), X["S/D"].max()],
                [X["L/D"].min(), X["L/D"].max()]
                ]
            }

        param_values = sobol.sample(problem, N=1024, calc_second_order=False)

        gsa_df = pd.DataFrame(param_values, columns=problem["names"])
        gsa_df["Pile_Group"] = pile_group
        gsa_df = gsa_df[X.columns]

        Y = best_model.predict(gsa_df)
        Si = sobol_analyze.analyze(problem, Y, calc_second_order=False, print_to_console=False)
        
        sobol_df = pd.DataFrame({
            "Parameter": problem["names"],
            "S1": Si["S1"],
            "ST": Si["ST"]
            })

        sobol_df.to_csv(
            os.path.join(base_path, f"Sobol_{pile_group}.csv"),
            index=False
        )
        
        plt.figure(figsize=(6, 4))
        plt.bar(problem["names"], Si["ST"], color="slateblue", edgecolor="black")
        plt.ylabel("Total Sobol Index")
        plt.title(f"Sobol Analysis ({pile_group})")
        plt.grid(True, axis="y", alpha=0.3)

        sobol_fig_path = os.path.join(base_path, f"Sobol_Global_Sensitivity_{pile_group}.png")
        plt.tight_layout()
        plt.savefig(sobol_fig_path, dpi=600, bbox_inches="tight")
        plt.show()
        plt.close()
        logging.info(f"Sobol sensitivity results saved at: {sobol_fig_path}")

    # ==========================================================
    # SHAP Analysis
    # ==========================================================
    logging.info("SHAP analysis...")

    X_test_transformed = best_model.named_steps["preprocess"].transform(X_test)
    rf_model = best_model.named_steps["model"]

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test_transformed)

    feature_names = best_model.named_steps["preprocess"].get_feature_names_out()
    feature_names = [name.split('__', 1)[-1] for name in feature_names]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean_SHAP": mean_abs_shap
    }).sort_values(by="Mean_SHAP", ascending=False)
    
    shap_df.to_csv(
        os.path.join(base_path, "SHAP_Values.csv"),
        index=False
    )
    
    shap_top = shap_df.head(10)

    plt.figure(figsize=(8, 5))
    plt.barh(shap_top["Feature"][::-1], shap_top["Mean_SHAP"][::-1],
             color="steelblue", edgecolor="black")
    plt.xlabel("Mean SHAP value")
    plt.title("Mean Absolute SHAP Values (Feature Importance)")
    plt.grid(True, alpha=0.3, axis="x")

    shap_fig_path = os.path.join(base_path, "SHAP_Feature_Importance.png")
    plt.tight_layout()
    plt.savefig(shap_fig_path, dpi=600, bbox_inches="tight")
    plt.show()
    plt.close()
    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names, show=False)
    summary_fig_path = os.path.join(base_path, "SHAP_Summary_Plot.png")
    plt.tight_layout()
    plt.savefig(summary_fig_path, dpi=600, bbox_inches="tight")
    plt.show()
    plt.close()
    logging.info(f"SHAP summary plot saved at: {summary_fig_path}")
    
    # ==========================================================
    # Response surface and contour plot for IF (S/D vs L/D)
    # ==========================================================
    logging.info("Generating response surface for IF as a function of S/D and L/D...")

    ref_pile = X["Pile_Group"].mode()[0]

    sd_vals = np.linspace(X["S/D"].min(), X["S/D"].max(), 80)
    ld_vals = np.linspace(X["L/D"].min(), X["L/D"].max(), 80)
    SD, LD = np.meshgrid(sd_vals, ld_vals)

    for ref_days in sorted(X["Days"].unique()):
        grid_df = pd.DataFrame({
            "Days": ref_days,
            "Pile_Group": ref_pile,
            "S/D": SD.ravel(),
            "L/D": LD.ravel()
            })

        IF_grid = best_model.predict(grid_df).reshape(SD.shape)

        fig, ax = plt.subplots(figsize=(7.5, 6))

        cf = ax.contourf(
            SD, LD, IF_grid,
            levels=25,
            cmap="viridis"
            )

        ctr = ax.contour(
            SD, LD, IF_grid,
            levels=10,
           colors="black",
           linewidths=0.8
           )
        ax.clabel(ctr, inline=True, fontsize=8, fmt="%.2f")

        ax.set_xlabel("S/D")
        ax.set_ylabel("L/D")
        ax.set_title(f"Response Surface of IF (Day {ref_days})")

        cbar = fig.colorbar(cf, ax=ax)
        cbar.set_label("IF")

        ax.grid(True, alpha=0.25)
        plt.tight_layout()

        response_surface_path = os.path.join(base_path, f"Response_Surface_IF_Day{ref_days}.png")
        plt.savefig(response_surface_path, dpi=600, bbox_inches="tight")
        plt.show()
        plt.close()
        logging.info(f"Response surface saved at: {response_surface_path}")


    logging.info("Analysis completed.")


if __name__ == "__main__":
    main()