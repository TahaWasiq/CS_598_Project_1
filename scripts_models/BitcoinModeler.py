from Utilities import *


from typing import List, Optional
from sklearn.feature_selection import mutual_info_regression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.base import clone


# === Model Wrappers ===========================================================
class ModelWrapper:
    """
    Generic training/eval wrapper around a scikit-learn regressor.
    Keeps interface uniform across Linear / Ridge / Lasso.
    """
    def __init__(self, name: str, model):
        self.name = name
        self.model = model
        self.lasso_nonzero=None
        self.ridge_nonzero=None

    def print_nonzero_coefs(self, feature_names: List[str], top_k: int = 20):
        """
        Prints the top |coef| features from a fitted model (e.g., Lasso, Ridge).
        """

        if not hasattr(self.model, "coef_"):
            print("Model has no coefficients.")
            return
        coefs = np.asarray(self.model.coef_)
        idx = np.argsort(np.abs(coefs))[::-1]  # sort by magnitude, descending
        take = idx[:min(top_k, len(idx))]
        rows = [(feature_names[i], coefs[i]) for i in take if coefs[i] != 0]
        df = pd.DataFrame(rows, columns=["feature", "coef"]).reset_index(drop=True)
        return df
    
    def get_lasso_nonzero_coef(self, epsilon=1e-6): # Feature selection info for Lasso
        """
        Retrieve nonzero coefficents from the Lasso model.
        """
        self.lasso_nonzero = np.sum(np.abs(self.model.coef_) > epsilon)
        return self.lasso_nonzero
    
    def get_ridge_nonzero_coef(self, epsilon=1e-6): # Feature selection info for Ridge
        """
        Retrieve nonzero coefficents from the Ridge model.
        """
        self.ridge_nonzero = np.sum(np.abs(self.model.coef_) > epsilon)
        return self.ridge_nonzero

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    @staticmethod
    def _eval(y_true, y_pred) -> Metrics:
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        rmse_nd = rmse/np.std(y_true)
        mae_nd = mae/np.std(y_true)
        r2 = r2_score(y_true, y_pred)
        return Metrics(mse=mse, rmse=rmse, mae=mae, rmse_nd=rmse_nd, mae_nd=mae_nd, r2=r2)

    def evaluate(self, bundle: DataBundle) -> Dict[str, Metrics]:
        """
        Returns metrics for Validation and Test sets.
        """
        pred_val = self.model.predict(bundle.X_val)
        pred_test = self.model.predict(bundle.X_test)
        return {
            "Validation": self._eval(bundle.y_val, pred_val),
            "Test": self._eval(bundle.y_test, pred_test),
        }
    
    def calculate_predictions(self, X):
        return self.model.predict(X)

    def calculate_residuals(self, X, y_true):
        return y_true - self.calculate_predictions(X)
    
    def plot_feature_importance(self, feature_names, title, plot_name, top_n=20):

        if hasattr(self.model, "coef_"):

            coef_df = pd.DataFrame({
            'feature': feature_names,
                'coefficient':np.asarray(self.model.coef_)})

            # Get top features by absolute coefficient value
            coef_df['abs_coef'] = np.abs(coef_df['coefficient'])
            top_features = coef_df.nlargest(top_n, 'abs_coef')

            plt.figure(figsize=(12, 8))
            colors = ['red' if x < 0 else 'blue' for x in top_features['coefficient']]
            plt.barh(range(len(top_features)), top_features['coefficient'], color=colors)
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Coefficient Value', fontsize=24)
            plt.title(f'{title} - Top {top_n} Features by Absolute Value', fontsize=24)
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            plt.tight_layout()
            plt.show()
            plt.savefig(plot_name, format='pdf', bbox_inches='tight')

            return top_features
        else:
            print(f"Model {title} does not have coefficients")
            return None
        
    @staticmethod
    def best_summary(results_df: pd.DataFrame, metric: str = "RMSE", prefer_scaled: str | None = None) -> pd.DataFrame:
        """
        - Pick the best per Horizon by lowest Validation <metric>.
        - Optionally restrict to Scaled? == 'Yes' or 'No' via prefer_scaled.
        - Return the winner's Test metrics in a tidy table.
        """
        df = results_df.copy()
        if prefer_scaled in {"Yes", "No"}:
            df = df[df["Scaled?"] == prefer_scaled]
        val = df[df["Dataset"] == "Validation"].copy()

        if (metric == "R2"):

            winners_idx = val.groupby("Horizon")[metric].idxmax()
        else:
            winners_idx = val.groupby("Horizon")[metric].idxmin()

        winners = val.loc[winners_idx, ["Horizon", "Model", "Scaled?", metric]].rename(
            columns={metric: f"Val_{metric}"})
        test = df[df["Dataset"] == "Test"].copy()
        merged = winners.merge(test, on=["Horizon", "Model", "Scaled?"], how="left", suffixes=("", "_Test"))
        cols = ["Horizon", "Model", "Scaled?", f"Val_{metric}", "MSE", "RMSE", "MAE", "RMSE_ND", "MAE_ND", "R2"]
        return merged[cols].sort_values(["Horizon"]).reset_index(drop=True)

def build_default_models() -> List[ModelWrapper]:
    """
    Sets up models.
    """
    return [
        ModelWrapper("Linear", LinearRegression()),
        ModelWrapper("Ridge", RidgeCV(alphas=np.logspace(-6, 6, 500))),
        ModelWrapper("Lasso", LassoCV(alphas=np.logspace(-6, 6, 500), max_iter=10000)),
    ]

# === Data & Scaling Managers ==================================================
class DataManager:
    """
    prepares data
    """
    @staticmethod
    def prepare(train_df, val_df, test_df, target_col: str, name: str) -> DataBundle:
        feature_cols = [c for c in train_df.columns if c not in ["date", target_col]]
        return DataBundle(
            X_train=train_df[feature_cols],
            y_train=train_df[target_col],
            X_val=val_df[feature_cols],
            y_val=val_df[target_col],
            X_test=test_df[feature_cols],
            y_test=test_df[target_col],
            feature_cols=feature_cols,
            name=name,
        )

class ScalerManager:
    """
    Fits a StandardScaler on train and transforms val/test.
    Note: your earlier cells already scale again; this class is optional,
    and provided for clean reuse when you want class-based scaling.
    """
    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, X_train):
        return self.scaler.fit_transform(X_train)

    def transform(self, X):
        return self.scaler.transform(X)

    def scale_bundle(self, bundle: DataBundle) -> DataBundle:
        Xtr = self.fit_transform(bundle.X_train)
        Xva = self.transform(bundle.X_val)
        Xte = self.transform(bundle.X_test)
        return DataBundle(
            X_train=Xtr, y_train=bundle.y_train,
            X_val=Xva, y_val=bundle.y_val,
            X_test=Xte, y_test=bundle.y_test,
            feature_cols=bundle.feature_cols, name=bundle.name
        )

#==== Spline ===============================
class Bspline:
    def __init__(self,bundle):
        self.bundle=bundle
        self.name = "B-Spline with Ridge"

    def customSpline(self):
        """
        Creates a basis spline model with Ridge.
        """
        Xtr, ytr = self.bundle.X_train.copy(), self.bundle.y_train.copy()
        Xva, yva = self.bundle.X_val.copy(), self.bundle.y_val.copy()
        Xte, yte = self.bundle.X_test.copy(), self.bundle.y_test.copy()

        # align columns (order matters for transforms)
        Xva = Xva[Xtr.columns]
        Xte = Xte[Xtr.columns]

        candidates = ['btc_close', 'btc_ema12', 'btc_ema26', 'btc_roll_mean_close_7', 'btc_rsi14', 'btc_macd',
                      'btc_atr14', 'btc_roll_std_close_7', 'btc_ret1']
        candidates = [c for c in candidates if c in Xtr.columns]

        def pick_top_k_spline_cols(X, y, pool, k=4):
            if not pool: return []
            mi = mutual_info_regression(X[pool].values, y.values, random_state=598)
            order = np.argsort(mi)[::-1]
            return [pool[i] for i in order[:min(k, len(pool))]]

        smooth_cols = pick_top_k_spline_cols(Xtr, ytr, candidates, k=4)
        if not smooth_cols and "btc_close" in Xtr.columns:
            smooth_cols = ["btc_close"]

        passthrough_cols = [c for c in Xtr.columns if c not in smooth_cols]

        print("Bundle: ", self.bundle.name, "Spline columns (selected):", smooth_cols)
        print("Bundle: ", self.bundle.name, "Passthrough features     :", len(passthrough_cols))

        #  Build transformer: few splines + passthrough --------
        pre = ColumnTransformer(
            transformers=[
                ("num", "passthrough", passthrough_cols),
                ("spl", SplineTransformer(
                    degree=3, n_knots=8, knots="quantile",
                    extrapolation="linear", include_bias=False
                ), smooth_cols),
            ],
            remainder="drop",
            sparse_threshold=0.0,  # force dense for reliable scaling
        )

        steps = [("pre", pre)]

        steps.append(("ridge", Ridge(random_state=598)))
        pipe = Pipeline(steps)

        # Small grid on VAL via PredefinedSplit --------
        X_trval = pd.concat([Xtr, Xva], axis=0)
        y_trval = np.r_[ytr.values, yva.values]
        test_fold = np.r_[np.full(len(Xtr), -1), np.zeros(len(Xva), dtype=int)]  # -1=TRAIN, 0=VAL
        ps = PredefinedSplit(test_fold)

        grid = {
            "pre__spl__n_knots": [6, 8, 10, 12],
            "ridge__alpha": np.logspace(-6, 6, 500),
        }

        gs = GridSearchCV(pipe, grid, scoring="neg_mean_squared_error", cv=ps, n_jobs=-1, verbose=0)

        gs.fit(X_trval, y_trval)
        print("Bundle: ", self.bundle.name, "Best params:", gs.best_params_, "| Val MSE:", -gs.best_score_)

        # Refit on TRAIN only (true VAL metrics), then TRAIN+VAL (TEST) --------
        best = clone(gs.best_estimator_)

        # VAL metrics (train-only refit)
        best.fit(Xtr, ytr)
        yhat_va = best.predict(Xva)

        # TEST metrics (train+val refit)
        self.best_tv = clone(gs.best_estimator_)
        self.best_tv.fit(X_trval, y_trval)
        yhat_te = self.best_tv.predict(Xte)

        stats = self.evaluate(yva, yhat_va, yte, yhat_te)
        return stats
    
    def predict(self, X):
        return self.best_tv.predict(X)

    def evaluate(self,yva,yhat_va,yte,yhat_te):
        return {
            "Validation": ModelWrapper._eval(yva, yhat_va),
            "Test": ModelWrapper._eval(yte, yhat_te),
        }
    
# === Experiment Runner & Reporter ============================================
class ExperimentRunner:
    """
    Runs multiple models on a given DataBundle (one horizon).
    """
    def __init__(self, bundle: DataBundle, scaled: bool = False):
        self.bundle = bundle
        self.scaled = scaled  # purely for labeling
        self.models: List[ModelWrapper] = build_default_models()
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

    def run(self) -> pd.DataFrame:
        """
        Trains each model on train, gets metrics on val/test, returns a tidy DataFrame.
        """
        rows = []
        
        for mw in self.models:
            mw.fit(self.bundle.X_train, self.bundle.y_train)
            metrics = mw.evaluate(self.bundle)
            for split_name, met in metrics.items():
                rows.append({
                    "Horizon": self.bundle.name,
                    "Scaled?": "Yes" if self.scaled else "No",
                    "Model": mw.name,
                    "Dataset": split_name,
                    **met.as_row(),
                })
        BasisSpliner = Bspline(self.bundle)
        customMetrics = BasisSpliner.customSpline()
        self.models.append(BasisSpliner)
        for split_name, met in customMetrics.items():
            rows.append({
                "Horizon": self.bundle.name,
                "Scaled?": "Yes" if self.scaled else "No",
                "Model": "B-Spline with Ridge",
                "Dataset": split_name,
                **met.as_row(),
            })
        return pd.DataFrame(rows)

class ResultsVisualizer:
    """
    Lightweight plotting helpers (matplotlib only; respects your seaborn import too).
    """
    @staticmethod
    def bar_by_model(df: pd.DataFrame, metric: str = "RMSE", horizon: Optional[str] = None):
        # Filter optional horizon
        plot_df = df.copy()
        if horizon is not None:
            plot_df = plot_df[plot_df["Horizon"] == horizon]

        # Pivot to have models on x, datasets as hue-like
        pivoted = plot_df.pivot_table(
            index=["Model"], columns=["Dataset"], values=metric, aggfunc="mean"
        ).sort_index()

        # Simple bar plot
        ax = pivoted.plot(kind="bar", figsize=(9, 5), title=f"{metric} by Model")
        ax.set_ylabel(metric)
        ax.set_xlabel("Model")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()
