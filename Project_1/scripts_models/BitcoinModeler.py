from Utilities import *


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
    def lasso_nonzero_coefs(self, feature_names: List[str], top_k: int = 20):
        """
        Prints the top |coef| features from a fitted Lasso model.
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
    def get_lasso_nonzero_coef(self,features,epsilon=1e-6):# Feature selection info for Lasso
        """
        Retrieve nonZero coef
        """


        self.lasso_nonzero = np.sum(np.abs(self.model.coef_) > epsilon)
        return self.lasso_nonzero


    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    @staticmethod
    def _eval(y_true, y_pred) -> Metrics:
        mse = mean_squared_error(y_true, y_pred)
        rmse = float(np.sqrt(mse))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return Metrics(mse=mse, rmse=rmse, mae=mae, r2=r2)

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

    def plot_feature_importance(self, feature_names, title, top_n=20):


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
            plt.xlabel('Coefficient Value')
            plt.title(f'{title} - Top {top_n} Features by Absolute Coefficient')
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            plt.tight_layout()
            plt.show()

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
        cols = ["Horizon", "Model", "Scaled?", f"Val_{metric}", "MSE", "RMSE", "MAE", "R2"]
        return merged[cols].sort_values(["Horizon"]).reset_index(drop=True)


def build_default_models(random_state: int = 42) -> List[ModelWrapper]:
    """
    Sets up models .
    """
    return [
        ModelWrapper("Linear", LinearRegression()),
        ModelWrapper("Ridge", Ridge(alpha=1.0, random_state=random_state)),
        ModelWrapper("Lasso", Lasso(alpha=0.01, random_state=random_state, max_iter=10000)),
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


# === Experiment Runner & Reporter ============================================
class ExperimentRunner:
    """
    Runs multiple models on a given DataBundle (one horizon).
    """
    def __init__(self, bundle: DataBundle, scaled: bool = False):
        self.bundle = bundle
        self.scaled = scaled  # purely for labeling
        self.models: List[ModelWrapper] = build_default_models()

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
# === Experiment Runner & Reporter ============================================
class ExperimentRunner:
    """
    Runs multiple models on a given DataBundle (one horizon).
    """
    def __init__(self, bundle: DataBundle, scaled: bool = False):
        self.bundle = bundle
        self.scaled = scaled  # purely for labeling
        self.models: List[ModelWrapper] = build_default_models()

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
        return pd.DataFrame(rows)

class ResultsVisualizer:
    """
    Lightweight plotting helpers
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
