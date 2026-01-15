import os
import joblib
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = "secret_key_for_session"

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

RAW_PATH = os.path.join(DATA_DIR, "sales_raw.csv")
MONTHLY_PATH = os.path.join(DATA_DIR, "monthly.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "monthly_linear_model.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

# ---- Payroll paths ----
PAYROLL_RAW_PATH = os.path.join(DATA_DIR, "payroll_raw.csv")

PAYROLL_ISO_PATH = os.path.join(MODEL_DIR, "payroll_iso_by_role.pkl")
PAYROLL_SCALER_PATH = os.path.join(MODEL_DIR, "payroll_scaler.pkl")
PAYROLL_KMEANS_PATH = os.path.join(MODEL_DIR, "payroll_kmeans.pkl")
PAYROLL_METRICS_PATH = os.path.join(MODEL_DIR, "payroll_metrics.json")

# resultados (para poder mostrarlos en la web sin recalcular cada vez)
PAYROLL_ROLE_IF_PATH = os.path.join(DATA_DIR, "payroll_role_if.csv")
PAYROLL_EMP_CLUSTER_PATH = os.path.join(DATA_DIR, "payroll_emp_clusters.csv")


os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# app ya definida arriba


# -------------------------
# Helpers de datos/features
# -------------------------

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans

PAYROLL_FEATURES = ["hours_biweekly", "gross_pay", "employee_benefits", "net_pay"]

def load_payroll_raw() -> pd.DataFrame:
    if not os.path.exists(PAYROLL_RAW_PATH):
        return pd.DataFrame()
    df = pd.read_csv(PAYROLL_RAW_PATH)
    return df

def save_payroll_raw(df: pd.DataFrame) -> None:
    df.to_csv(PAYROLL_RAW_PATH, index=False)

def load_payroll_models():
    iso_by_role = joblib.load(PAYROLL_ISO_PATH) if os.path.exists(PAYROLL_ISO_PATH) else None
    scaler = joblib.load(PAYROLL_SCALER_PATH) if os.path.exists(PAYROLL_SCALER_PATH) else None
    kmeans = joblib.load(PAYROLL_KMEANS_PATH) if os.path.exists(PAYROLL_KMEANS_PATH) else None
    return iso_by_role, scaler, kmeans

def load_payroll_metrics():
    if not os.path.exists(PAYROLL_METRICS_PATH):
        return None
    with open(PAYROLL_METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def train_payroll_models(payroll: pd.DataFrame, features: list[str], min_role_rows: int = 50,
                         contamination: float = 0.02, n_clusters: int = 3):
    # Validación mínima
    required = {"employee_id", "role"} | set(features)
    missing = required - set(payroll.columns)
    if missing:
        raise ValueError(f"Payroll: faltan columnas requeridas: {missing}")

    iso_by_role = {}
    results = []

    roles_total = int(payroll["role"].nunique())

    for role, df_role in payroll.groupby("role"):
        if len(df_role) < min_role_rows:
            continue

        X = df_role[features].astype(float)
        scaler_role = StandardScaler()
        Xs = scaler_role.fit_transform(X)

        iso = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42
        )
        preds = iso.fit_predict(Xs)

        iso_by_role[role] = {"scaler": scaler_role, "model": iso}

        tmp = df_role.copy()
        tmp["anomaly_role"] = preds  # -1 anomalía, 1 normal
        results.append(tmp)

    payroll_role_if = pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    # Clustering por empleado (media por empleado+rol)
    emp = (
        payroll
        .groupby(["employee_id", "role"], as_index=False)[features]
        .mean()
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(emp[features].astype(float))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    emp["cluster"] = kmeans.fit_predict(X_scaled)

    metrics = {
        "features": features,
        "roles_total": roles_total,
        "roles_modeled": int(len(iso_by_role)),
        "min_role_rows": int(min_role_rows),
        "contamination": float(contamination),
        "n_clusters": int(n_clusters),
        "rows_payroll": int(len(payroll)),
        "rows_role_if": int(len(payroll_role_if)),
        "rows_emp": int(len(emp))
    }

    return iso_by_role, scaler, kmeans, payroll_role_if, emp, metrics

def build_monthly_from_raw(raw_df: pd.DataFrame, min_date: str | None = "2020-01-01") -> pd.DataFrame:
    """
    CSV real (nivel día):
    date,year,month,store_id,channel,transactions,revenue,gst_collected,net_revenue,dataset
    """
    df = raw_df.copy()

    # Tipos
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise ValueError("Hay fechas inválidas en 'date'.")
    if min_date is not None:
        df = df[df["date"] >= min_date]
    required = {"store_id", "channel", "transactions", "net_revenue"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    # Normalizamos a mes fin de mes (ME), que es lo que luego uso en el forecast
    df["date_month"] = df["date"].dt.to_period("M").dt.to_timestamp("M")

    monthly = (
        df.groupby(["date_month", "store_id", "channel"], as_index=False)
          .agg(
              net_revenue=("net_revenue", "sum"),
              transactions=("transactions", "sum"),
              revenue=("revenue", "sum") if "revenue" in df.columns else ("net_revenue", "sum"),
              gst_collected=("gst_collected", "sum") if "gst_collected" in df.columns else ("net_revenue", "sum"),
          )
    )

    monthly = monthly.rename(columns={"date_month": "date"})
    monthly["year"] = monthly["date"].dt.year
    monthly["month"] = monthly["date"].dt.month
    monthly["avg_ticket"] = np.where(
        monthly["transactions"] > 0,
        monthly["net_revenue"] / monthly["transactions"],
        0.0
    )

    return monthly

def add_lag_roll_features(monthly: pd.DataFrame) -> pd.DataFrame:
    monthly = monthly.sort_values(["store_id", "channel", "date"]).copy()

    def add_features(g):
        g = g.copy()
        g["lag_1"] = g["net_revenue"].shift(1)
        g["lag_3"] = g["net_revenue"].shift(3)
        g["lag_6"] = g["net_revenue"].shift(6)
        g["lag_12"] = g["net_revenue"].shift(12)

        g["roll_3"] = g["net_revenue"].shift(1).rolling(3).mean()
        g["roll_6"] = g["net_revenue"].shift(1).rolling(6).mean()
        g["roll_12"] = g["net_revenue"].shift(1).rolling(12).mean()
        return g

    monthly_feat = (
        monthly.groupby(["store_id", "channel"], group_keys=False)
               .apply(add_features)
               .dropna()
               .reset_index(drop=True)
    )
    return monthly_feat


def build_model_pipeline() -> Pipeline:
    cat_cols = ["store_id", "channel", "month"]
    num_cols = ["lag_1", "lag_3", "lag_6", "lag_12", "roll_3", "roll_6", "roll_12", "avg_ticket"]

    preprocess = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    model = Pipeline(
        steps=[
            ("prep", preprocess),
            ("model", LinearRegression()),
        ]
    )
    return model


def load_monthly() -> pd.DataFrame:
    if not os.path.exists(MONTHLY_PATH):
        return pd.DataFrame()
    df = pd.read_csv(MONTHLY_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


def save_monthly(df: pd.DataFrame) -> None:
    out = df.copy()
    # guardar date en formato ISO
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(MONTHLY_PATH, index=False)


def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def save_model(model) -> None:
    joblib.dump(model, MODEL_PATH)

def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# -------------------------
# Forecast recursivo
# -------------------------

FEATURE_COLS = ["store_id","channel","month","lag_1","lag_3","lag_6","lag_12","roll_3","roll_6","roll_12", "avg_ticket"]

def forecast_for_store_channel(monthly: pd.DataFrame, model, store_id: str, channel: str, H: int = 12) -> pd.DataFrame:
    """
    monthly: agregado mensual real (date, store_id, channel, net_revenue)
    Devuelve un DF con H predicciones futuras para ese store+channel.
    """
    hist = monthly[(monthly["store_id"] == store_id) & (monthly["channel"] == channel)].copy()
    hist = hist.sort_values("date")

    if hist.shape[0] < 13:
        raise ValueError("No hay suficiente histórico para ese store+channel (mínimo ~13 meses para lags/rollings).")

    extended = hist[["date","store_id","channel","net_revenue","avg_ticket"]].copy()

    last_date = extended["date"].max()
    future_dates = pd.date_range(last_date + pd.offsets.MonthEnd(1), periods=H, freq="ME")

    preds = []
    for d in future_dates:
        y = extended["net_revenue"]

        # construir features “a mano” desde el extended actual
        lag_1  = y.iloc[-1]
        lag_3  = y.iloc[-3] if len(y) >= 3 else None
        lag_6  = y.iloc[-6] if len(y) >= 6 else None
        lag_12 = y.iloc[-12] if len(y) >= 12 else None

        roll_3  = y.iloc[-3:].mean() if len(y) >= 3 else None
        roll_6  = y.iloc[-6:].mean() if len(y) >= 6 else None
        roll_12 = y.iloc[-12:].mean() if len(y) >= 12 else None
        avg_ticket = float(extended["avg_ticket"].iloc[-1]) if len(extended) else 0.0

        X_step = pd.DataFrame([{
            "store_id": store_id,
            "channel": channel,
            "month": d.month,
            "lag_1": lag_1,
            "lag_3": lag_3,
            "lag_6": lag_6,
            "lag_12": lag_12,
            "roll_3": roll_3,
            "roll_6": roll_6,
            "roll_12": roll_12,
            "avg_ticket": avg_ticket
        }])[FEATURE_COLS]

        y_pred = float(model.predict(X_step)[0])
        preds.append({"date": d, "store_id": store_id, "channel": channel, "forecast_net_revenue": y_pred})

        # alimentar histórico
        extended = pd.concat(
            [extended, pd.DataFrame([{"date": d, "store_id": store_id, "channel": channel, "net_revenue": y_pred, "avg_ticket": avg_ticket}])],
            ignore_index=True
        )

    out = pd.DataFrame(preds)
    return out

def forecast_aggregated(monthly, model, store_ids, channels, H=12):
    """
    Genera forecast agregando predicciones de múltiples store/channel.
    store_ids: lista de tiendas
    channels: lista de canales
    """
    all_forecasts = []

    for store in store_ids:
        for channel in channels:
            try:
                fc = forecast_for_store_channel(
                    monthly=monthly,
                    model=model,
                    store_id=store,
                    channel=channel,
                    H=H
                )
                all_forecasts.append(fc)
            except Exception:
                # combinaciones sin suficiente histórico se ignoran
                continue

    if not all_forecasts:
        raise ValueError("No hay combinaciones válidas para generar forecast.")

    df = pd.concat(all_forecasts, ignore_index=True)

    # Agregación final
    agg = (
        df.groupby("date", as_index=False)["forecast_net_revenue"]
          .sum()
    )

    return agg


# -------------------------
# Rutas web
# -------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    monthly = load_monthly()
    model = load_model()

    stores = sorted(monthly["store_id"].dropna().unique().tolist()) if not monthly.empty else []
    channels = sorted(monthly["channel"].dropna().unique().tolist()) if not monthly.empty else []

    forecast_df = None
    error = None

    if request.method == "POST":
        store_id = request.form.get("store_id")    # puede ser "ALL"
        channel = request.form.get("channel")      # puede ser "ALL"
        horizon = int(request.form.get("horizon", "12"))

        if model is None:
            error = "No hay modelo guardado todavía. Ve a Admin y entrena."
        else:
            try:
                # Casos: agregado vs específico
                if store_id == "ALL" and channel == "ALL":
                    forecast_df = forecast_aggregated(
                        monthly, model,
                        store_ids=stores,
                        channels=channels,
                        H=horizon
                    )

                elif store_id == "ALL":
                    forecast_df = forecast_aggregated(
                        monthly, model,
                        store_ids=stores,
                        channels=[channel],
                        H=horizon
                    )

                elif channel == "ALL":
                    forecast_df = forecast_aggregated(
                        monthly, model,
                        store_ids=[store_id],
                        channels=channels,
                        H=horizon
                    )

                else:
                    forecast_df = forecast_for_store_channel(
                        monthly, model, store_id, channel, H=horizon
                    )

                # Formateo final para la tabla
                forecast_df["date"] = pd.to_datetime(forecast_df["date"]).dt.strftime("%Y-%m-%d")
                forecast_df["forecast_net_revenue"] = forecast_df["forecast_net_revenue"].round(2)

                # Si es forecast agregado, no tendrá store/channel: los añadimos para que tu tabla no “reviente”
                if "store_id" not in forecast_df.columns:
                    forecast_df["store_id"] = "ALL" if store_id == "ALL" else store_id
                if "channel" not in forecast_df.columns:
                    forecast_df["channel"] = "ALL" if channel == "ALL" else channel

            except Exception as e:
                error = str(e)

    historical_df = None
    if not monthly.empty:
        # histórico agregado para que cuadre con forecast agregado
        if request.method == "POST":
            store_id = request.form.get("store_id")
            channel  = request.form.get("channel")

            hist = monthly.copy()

            if store_id != "ALL":
                hist = hist[hist["store_id"] == store_id]
            if channel != "ALL":
                hist = hist[hist["channel"] == channel]

            # Agregamos por mes
            historical_df = (
                hist.groupby("date", as_index=False)["net_revenue"]
                    .sum()
                    .sort_values("date")
            )
        else:
            # por defecto: total
            historical_df = (
                monthly.groupby("date", as_index=False)["net_revenue"]
                    .sum()
                    .sort_values("date")
            )

    # formateo
    historical_records = None
    if historical_df is not None and not historical_df.empty:
        tmp = historical_df.copy()
        tmp["date"] = pd.to_datetime(tmp["date"]).dt.strftime("%Y-%m-%d")
        tmp["net_revenue"] = tmp["net_revenue"].round(2)
        historical_records = tmp.to_dict("records")


    metrics = load_metrics()
    mae_test = None
    if metrics and "mae_test" in metrics:
        mae_test = float(metrics["mae_test"])


    return render_template(
        "index.html",
        stores=stores,
        channels=channels,
        forecast=forecast_df.to_dict("records") if forecast_df is not None else None,
        historical=historical_records,
        mae_test=mae_test,
        error=error,
        model_exists=(model is not None)
    )

@app.route("/admin", methods=["GET"])
def admin():
    monthly = load_monthly()
    model = load_model()
    metrics = load_metrics()

    payroll_metrics = load_payroll_metrics()

    info = {
        "raw_exists": os.path.exists(RAW_PATH),
        "monthly_rows": int(monthly.shape[0]) if not monthly.empty else 0,
        "model_exists": model is not None,
        "metrics": metrics,

        "payroll_raw_exists": os.path.exists(PAYROLL_RAW_PATH),
        "payroll_iso_exists": os.path.exists(PAYROLL_ISO_PATH),
        "payroll_kmeans_exists": os.path.exists(PAYROLL_KMEANS_PATH),
        "payroll_metrics": payroll_metrics
    }
    return render_template("admin.html", info=info)


@app.route("/admin/upload", methods=["POST"])
def admin_upload():
    file = request.files.get("file")
    if not file:
        flash("No se ha subido ningún archivo.", "error")
        return redirect(url_for("admin"))

    try:
        # 1) Leer CSV nuevo
        raw_new = pd.read_csv(file)
        raw_new["date"] = pd.to_datetime(raw_new["date"], errors="coerce")

        if raw_new["date"].isna().any():
            raise ValueError("El CSV contiene fechas inválidas en la columna 'date'.")

        # 2) Si existe raw previo, hacer append
        if os.path.exists(RAW_PATH):
            raw_old = pd.read_csv(RAW_PATH)
            raw_old["date"] = pd.to_datetime(raw_old["date"], errors="coerce")

            raw_all = pd.concat([raw_old, raw_new], ignore_index=True)
        else:
            raw_all = raw_new

        # 3) Eliminar duplicados por clave natural
        # (en tu dataset hay 1 fila por día + tienda + canal)
        raw_all = raw_all.drop_duplicates(
            subset=["date", "store_id", "channel"],
            keep="last"
        )

        # 4) Guardar raw consolidado
        raw_all.to_csv(RAW_PATH, index=False)

        # 5) Recalcular agregado mensual
        monthly = build_monthly_from_raw(raw_all)
        save_monthly(monthly)

        flash("Datos subidos correctamente y monthly recalculado.", "ok")

    except Exception as e:
        flash(f"Error al procesar CSV: {e}", "error")

    return redirect(url_for("admin"))


@app.route("/admin/retrain", methods=["POST"])
def admin_retrain():
    try:
        include_pre_2020 = request.form.get("include_pre_2020") == "1"

        min_date = None if include_pre_2020 else "2020-01-01"
        training_mode = "crisis" if include_pre_2020 else "normal"

        if not os.path.exists(RAW_PATH):
            raise ValueError("No existe sales_raw.csv. Sube datos primero.")

        raw = pd.read_csv(RAW_PATH)
        monthly = build_monthly_from_raw(raw, min_date=min_date)
        save_monthly(monthly)

        monthly_feat = add_lag_roll_features(monthly)

        TEST_MONTHS = 12

        monthly_feat = monthly_feat.sort_values("date").copy()
        dates = sorted(monthly_feat["date"].dropna().unique())

        if len(dates) <= TEST_MONTHS:
            raise ValueError(f"No hay suficientes meses únicos para validar con {TEST_MONTHS} meses de test.")

        X_all = monthly_feat[FEATURE_COLS]
        y_all = monthly_feat["net_revenue"].astype(float)

        last_dates = dates[-TEST_MONTHS:]

        train_df = monthly_feat[~monthly_feat["date"].isin(last_dates)]
        test_df  = monthly_feat[monthly_feat["date"].isin(last_dates)]

        X_train = train_df[FEATURE_COLS]
        y_train = train_df["net_revenue"].astype(float)

        X_test  = test_df[FEATURE_COLS]
        y_test  = test_df["net_revenue"].astype(float)

        model = build_model_pipeline()
        model.fit(X_train, y_train)

        pred_train = model.predict(X_train)
        pred_test  = model.predict(X_test)

        rmse_train = float(np.sqrt(mean_squared_error(y_train, pred_train)))
        rmse_test  = float(np.sqrt(mean_squared_error(y_test, pred_test)))

        mae_train = float(mean_absolute_error(y_train, pred_train))
        mae_test  = float(mean_absolute_error(y_test, pred_test))

        metrics = {
            "training_mode": training_mode,
            "min_date": min_date,
            "test_months": TEST_MONTHS,

            "rmse_train": float(np.sqrt(mean_squared_error(y_train, pred_train))),
            "mae_train":  float(mean_absolute_error(y_train, pred_train)),

            "rmse_test":  float(np.sqrt(mean_squared_error(y_test, pred_test))),
            "mae_test":   float(mean_absolute_error(y_test, pred_test)),

            "train_rows": int(len(X_train)),
            "test_rows":  int(len(X_test)),

            "last_train_date": str(pd.to_datetime(train_df["date"].max()).date()),
            "last_test_date":  str(pd.to_datetime(test_df["date"].max()).date()),
        }

        # Reentrena final con TODO (opcional pero recomendable para producción)
        # Así el modelo final aprovecha todo el histórico.
        model.fit(X_all, y_all)

        save_model(model)

        # Guardar métricas
        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        flash(
            f"Modelo reentrenado y guardado. MAE test: {metrics['mae_test']:.2f} | RMSE test: {metrics['rmse_test']:.2f}",
            "ok"
        )

    except Exception as e:
        flash(f"Error reentrenando: {e}", "error")

    return redirect(url_for("admin"))

@app.route("/admin/payroll/upload", methods=["POST"])
def admin_payroll_upload():
    file = request.files.get("file")
    if not file:
        flash("No se ha subido ningún archivo de payroll.", "error")
        return redirect(url_for("admin"))

    try:
        df = pd.read_csv(file)

        # opcional: si existe uno anterior, puedes hacer append + drop_duplicates.
        # aquí lo dejo simple: sustituir
        save_payroll_raw(df)

        flash("Payroll subido correctamente.", "ok")
    except Exception as e:
        flash(f"Error al subir payroll: {e}", "error")

    return redirect(url_for("admin"))

@app.route("/admin/payroll/retrain", methods=["POST"])
def admin_payroll_retrain():
    try:
        payroll = load_payroll_raw()
        if payroll.empty:
            raise ValueError("No existe payroll_raw.csv. Sube payroll primero.")

        iso_by_role, scaler, kmeans, payroll_role_if, emp_clusters, metrics = train_payroll_models(
            payroll=payroll,
            features=PAYROLL_FEATURES,
            min_role_rows=50,
            contamination=0.02,
            n_clusters=3
        )

        # guardar modelos
        joblib.dump(iso_by_role, PAYROLL_ISO_PATH)
        joblib.dump(scaler, PAYROLL_SCALER_PATH)
        joblib.dump(kmeans, PAYROLL_KMEANS_PATH)

        # guardar resultados para visualización
        payroll_role_if.to_csv(PAYROLL_ROLE_IF_PATH, index=False)
        emp_clusters.to_csv(PAYROLL_EMP_CLUSTER_PATH, index=False)

        # guardar métricas
        with open(PAYROLL_METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        flash(
            f"Payroll models entrenados. Roles modelados: {metrics['roles_modeled']} | Empleados cluster: {metrics['rows_emp']}",
            "ok"
        )

    except Exception as e:
        flash(f"Error reentrenando payroll: {e}", "error")

    return redirect(url_for("admin"))

import math

@app.route("/payroll", methods=["GET"])
def payroll_view():
    # Cargar resultados ya calculados
    role_if = pd.read_csv(PAYROLL_ROLE_IF_PATH) if os.path.exists(PAYROLL_ROLE_IF_PATH) else pd.DataFrame()
    emp = pd.read_csv(PAYROLL_EMP_CLUSTER_PATH) if os.path.exists(PAYROLL_EMP_CLUSTER_PATH) else pd.DataFrame()

    if not role_if.empty and "pay_period_start" in role_if.columns:
        role_if["pay_period_start"] = pd.to_datetime(role_if["pay_period_start"], errors="coerce")

    # ---- Filtros por query params ----
    role = request.args.get("role", "").strip()
    location = request.args.get("location", "").strip()
    employee_id = request.args.get("employee_id", "").strip()
    anomaly_only = request.args.get("anomaly_only", "0") == "1"

    cluster = request.args.get("cluster", "").strip()

    df = role_if.copy()

    if not df.empty:
        if role:
            df = df[df["role"] == role]
        if location and "location" in df.columns:
            df = df[df["location"] == location]
        if employee_id:
            df = df[df["employee_id"].astype(str) == employee_id]
        if anomaly_only and "anomaly_role" in df.columns:
            df = df[df["anomaly_role"] == -1]

    empf = emp.copy()
    if not empf.empty:
        if role:
            empf = empf[empf["role"] == role]
        if employee_id:
            empf = empf[empf["employee_id"].astype(str) == employee_id]
        if cluster != "":
            try:
                cl = int(cluster)
                empf = empf[empf["cluster"] == cl]
            except:
                pass

    # ---- Listas para los selects ----
    roles = sorted(role_if["role"].dropna().unique().tolist()) if not role_if.empty else []
    locations = sorted(role_if["location"].dropna().unique().tolist()) if (not role_if.empty and "location" in role_if.columns) else []
    clusters_list = sorted(emp["cluster"].dropna().unique().tolist()) if not emp.empty else []

    # ---- Resumen / KPI ----
    total_rows = int(role_if.shape[0]) if not role_if.empty else 0
    total_anoms = int((role_if["anomaly_role"] == -1).sum()) if (not role_if.empty and "anomaly_role" in role_if.columns) else 0

    filtered_rows = int(df.shape[0]) if not df.empty else 0
    filtered_anoms = int((df["anomaly_role"] == -1).sum()) if (not df.empty and "anomaly_role" in df.columns) else 0

    # ---- Datos para gráficos ----
    # 1) Barras: anomalías por rol (top 12)
    role_anom_counts = []
    if not role_if.empty and "anomaly_role" in role_if.columns:
        tmp = (
            role_if.assign(is_anom=(role_if["anomaly_role"] == -1).astype(int))
                  .groupby("role", as_index=False)["is_anom"].sum()
                  .sort_values("is_anom", ascending=False)
                  .head(12)
        )
        role_anom_counts = [{"role": r["role"], "anoms": int(r["is_anom"])} for _, r in tmp.iterrows()]

    # 2) Línea: anomalías por mes (sobre el DF filtrado actual)
    anoms_by_month = []
    if not df.empty and "anomaly_role" in df.columns and "pay_period_start" in df.columns:
        tmp = df.dropna(subset=["pay_period_start"]).copy()
        tmp["month"] = tmp["pay_period_start"].dt.to_period("M").dt.to_timestamp("M")
        tmp = (
            tmp.assign(is_anom=(tmp["anomaly_role"] == -1).astype(int))
               .groupby("month", as_index=False)["is_anom"].sum()
               .sort_values("month")
        )
        anoms_by_month = [{"month": m["month"].strftime("%Y-%m"), "anoms": int(m["is_anom"])} for _, m in tmp.iterrows()]

    # 3) Scatter: hours vs gross (muestra para no petar)
    scatter = []
    if not df.empty and {"hours_biweekly","gross_pay","anomaly_role"}.issubset(df.columns):
        sample = df.sample(min(800, len(df)), random_state=42) if len(df) > 800 else df
        scatter = [
            {"x": float(r["hours_biweekly"]), "y": float(r["gross_pay"]), "anomaly": int(r["anomaly_role"])}
            for _, r in sample.iterrows()
        ]

    # ---- Tablas: limitar filas para que no sea infinito ----
    anomalies_rows = df[df["anomaly_role"] == -1].head(20).to_dict("records") if (not df.empty and "anomaly_role" in df.columns) else []
    clusters_rows = empf.head(20).to_dict("records") if not empf.empty else []

    return render_template(
        "payroll.html",
        roles=roles,
        locations=locations,
        clusters_list=clusters_list,

        selected_role=role,
        selected_location=location,
        selected_employee_id=employee_id,
        selected_anomaly_only=anomaly_only,
        selected_cluster=cluster,

        total_rows=total_rows,
        total_anoms=total_anoms,
        filtered_rows=filtered_rows,
        filtered_anoms=filtered_anoms,

        role_anom_counts=role_anom_counts,
        anoms_by_month=anoms_by_month,
        scatter=scatter,

        anomalies=anomalies_rows,
        clusters=clusters_rows
    )


if __name__ == "__main__":
    app.run(debug=True)
