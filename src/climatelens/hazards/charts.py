"""Interactive Plotly chart generation for climate hazards using nielytics.core.visualization."""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from nielytics import (
    plot_bar,
    plot_distribution,
    plot_geographic,
    plot_timeseries,
)

from climatelens.constants import (
    PROCESSED_DATA_DIR,
    STATES_MAP,
    normalize_state_id,
)

logger = logging.getLogger("climatelens.charts")


def _figure_to_spec(fig: Any) -> dict[str, Any]:
    """Convert Plotly figure to standard JSON-safe dict."""
    if hasattr(fig, "to_json"):
        return json.loads(fig.to_json())
    if isinstance(fig, dict):
        return fig
    return {"data": [], "layout": {}}


def generate_timeseries_chart(
    df: pd.DataFrame,
    date_col: str,
    val_col: str,
    title: str | None = None,
) -> dict[str, Any]:
    """Generate interactive timeseries chart using nielytics.plot_timeseries."""
    clean_df = df.dropna(subset=[date_col, val_col]).copy()
    clean_df[date_col] = pd.to_datetime(clean_df[date_col])
    clean_df[val_col] = pd.to_numeric(clean_df[val_col], errors="coerce")
    clean_df = clean_df.dropna(subset=[val_col]).sort_values(date_col)

    fig = plot_timeseries(clean_df, date_col, val_col, title=title)
    return _figure_to_spec(fig)


def generate_bar_chart(
    df: pd.DataFrame,
    cat_col: str,
    val_col: str,
    title: str | None = None,
) -> dict[str, Any]:
    """Generate categorical bar chart using nielytics.plot_bar."""
    clean_df = df.dropna(subset=[cat_col, val_col]).copy()
    clean_df[cat_col] = clean_df[cat_col].astype(str)
    clean_df[val_col] = pd.to_numeric(clean_df[val_col], errors="coerce")
    clean_df = clean_df.dropna(subset=[val_col])

    fig = plot_bar(clean_df, cat_col, val_col, title=title)
    return _figure_to_spec(fig)


def generate_distribution_chart(
    df: pd.DataFrame,
    val_col: str,
    bins: int = 20,
    title: str | None = None,
) -> dict[str, Any]:
    """Generate probability density distribution chart using nielytics.plot_distribution."""
    clean_df = df.dropna(subset=[val_col]).copy()
    clean_df[val_col] = pd.to_numeric(clean_df[val_col], errors="coerce")
    clean_df = clean_df.dropna(subset=[val_col])

    fig = plot_distribution(clean_df, val_col, bins=bins, title=title)
    return _figure_to_spec(fig)


def generate_geographic_chart(
    df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    val_col: str | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """Generate geographic map chart using nielytics.plot_geographic and Plotly map."""
    clean_df = df.dropna(subset=[lat_col, lon_col]).copy()
    clean_df[lat_col] = pd.to_numeric(clean_df[lat_col], errors="coerce")
    clean_df[lon_col] = pd.to_numeric(clean_df[lon_col], errors="coerce")
    clean_df = clean_df.dropna(subset=[lat_col, lon_col])

    # Run nielytics validation & Matplotlib generation
    meta: dict[str, Any] = {}
    try:
        mpl_fig = plot_geographic(
            clean_df[[lat_col, lon_col] + ([val_col] if val_col else [])],
            latitude=lat_col,
            longitude=lon_col,
            value=val_col,
            title=title,
        )
        meta = getattr(mpl_fig, "nielytics_metadata", {})
    except Exception as exc:
        logger.debug("nielytics.plot_geographic notice: %s", exc)

    # Build interactive Plotly Scattergeo spec for browser rendering
    marker_dict: dict[str, Any] = {"size": 6, "opacity": 0.8}
    if val_col and val_col in clean_df:
        marker_dict["color"] = clean_df[val_col]
        marker_dict["colorscale"] = "Viridis"
        marker_dict["showscale"] = True

    hover_text = clean_df.get("name", clean_df.get("location", clean_df.get("event_id", ""))).astype(str)

    fig = go.Figure(
        go.Scattergeo(
            lon=clean_df[lon_col],
            lat=clean_df[lat_col],
            text=hover_text,
            mode="markers",
            marker=marker_dict,
        )
    )
    fig.update_layout(
        title=title or "Geographic Distribution",
        geo=dict(
            scope="asia",
            center=dict(lat=22.0, lon=82.0),
            projection_scale=3.5,
            showland=True,
            landcolor="#f8fafc",
            showocean=True,
            oceancolor="#e0f2fe",
            showcountries=True,
        ),
        margin=dict(l=0, r=0, t=30, b=0),
    )

    spec = _figure_to_spec(fig)
    spec["nielytics_metadata"] = meta
    return spec


def generate_hazard_chart(
    hazard: str,
    chart_type: str,
    state_id: str | None = None,
    year: int | None = None,
) -> dict[str, Any]:
    """Dispatch hazard and chart_type to appropriate data source and Plotly chart builder."""
    haz = hazard.lower().strip()
    ctype = chart_type.lower().strip()
    sid = normalize_state_id(state_id) if state_id else None

    if haz in ("heatwave", "heat", "temperature"):
        from climatelens.hazards.heat_hazard import _load_state_heatwave_data
        df = _load_state_heatwave_data()
        if sid:
            df = df[df["state_id"] == sid]

        if ctype == "timeseries":
            df["date"] = pd.to_datetime(df["year"].astype(str) + "-06-01")
            sname = STATES_MAP[sid]["state_name"] if sid else "State"
            return generate_timeseries_chart(
                df,
                date_col="date",
                val_col="total_heat_wave_days",
                title=f"Annual Heatwave Days - {sname} (1967-2024)",
            )
        elif ctype == "bar":
            y = year or 2024
            y_df = df[df["year"] == y].sort_values("total_heat_wave_days", ascending=False).head(15)
            return generate_bar_chart(
                y_df,
                cat_col="state_name",
                val_col="total_heat_wave_days",
                title=f"Heatwave Days by State in {y}",
            )
        elif ctype == "distribution":
            valid = df.dropna(subset=["summer_max_temp_deg_c"])
            return generate_distribution_chart(
                valid,
                val_col="summer_max_temp_deg_c",
                bins=20,
                title="Summer Maximum Temperature Distribution (°C)",
            )
        else:
            raise ValueError(f"Unsupported chart_type '{chart_type}' for heatwave hazard")

    elif haz in ("rainfall", "rain", "precipitation"):
        from climatelens.hazards.rainfall_hazard import _load_rainfall_daily, _load_rainfall_state_year
        if ctype == "timeseries":
            df = _load_rainfall_daily()
            if sid:
                df = df[df["entity_id"] == sid]
            if year:
                df = df[df["date"].dt.year == year]
            sname = STATES_MAP[sid]["state_name"] if sid else "India"
            return generate_timeseries_chart(
                df,
                date_col="date",
                val_col="daily_actual_mm",
                title=f"Daily Rainfall (mm) - {sname}",
            )
        elif ctype == "bar":
            df = _load_rainfall_state_year()
            y = year or 2026
            df = df[df["year"] == y].sort_values("monsoon_departure_pct")
            return generate_bar_chart(
                df,
                cat_col="state_name",
                val_col="monsoon_departure_pct",
                title=f"Monsoon Rainfall Departure (%) by State - {y}",
            )
        elif ctype == "distribution":
            df = _load_rainfall_daily()
            if sid:
                df = df[df["entity_id"] == sid]
            return generate_distribution_chart(
                df,
                val_col="daily_actual_mm",
                bins=25,
                title="Daily Actual Rainfall Distribution (mm)",
            )
        else:
            raise ValueError(f"Unsupported chart_type '{chart_type}' for rainfall hazard")

    elif haz in ("landslide", "landslides"):
        from climatelens.hazards.landslide_hazard import _load_landslide_data
        df = _load_landslide_data()
        if sid:
            df = df[df["state_id"] == sid]
        if year:
            df = df[df["year"] == year]

        if ctype == "bar":
            if not sid:
                st_counts = df["state_name"].value_counts().reset_index()
                st_counts.columns = ["state_name", "events_count"]
                return generate_bar_chart(
                    st_counts.head(15),
                    cat_col="state_name",
                    val_col="events_count",
                    title="Landslide Events by State (NASA GLC)",
                )
            else:
                yr_counts = df["year"].value_counts().reset_index()
                yr_counts.columns = ["year", "events_count"]
                yr_counts = yr_counts.sort_values("year")
                return generate_bar_chart(
                    yr_counts,
                    cat_col="year",
                    val_col="events_count",
                    title=f"Annual Landslide Frequency - {STATES_MAP[sid]['state_name']}",
                )
        elif ctype in ("geographic", "map"):
            return generate_geographic_chart(
                df,
                lat_col="latitude",
                lon_col="longitude",
                val_col="reported_magnitude",
                title="NASA GLC Landslide Events in India",
            )
        elif ctype == "distribution":
            return generate_distribution_chart(
                df,
                val_col="reported_magnitude",
                bins=10,
                title="Landslide Magnitude / Severity Distribution",
            )
        else:
            raise ValueError(f"Unsupported chart_type '{chart_type}' for landslide hazard")

    elif haz in ("flood", "floods"):
        from climatelens.hazards.flood_hazard import _load_flood_data
        df = _load_flood_data()
        if sid:
            df = df[df["primary_state_id"] == sid]

        if ctype == "timeseries":
            valid_yr = df.dropna(subset=["year"]).copy()
            valid_yr["year"] = valid_yr["year"].astype(int)
            annual = valid_yr.groupby("year")["event_id"].count().reset_index()
            annual.columns = ["year", "event_count"]
            annual["date"] = pd.to_datetime(annual["year"].astype(str) + "-07-01", format="%Y-%m-%d")
            return generate_timeseries_chart(
                annual,
                date_col="date",
                val_col="event_count",
                title="Historical Flood Frequency per Year (1967-2023)",
            )
        elif ctype == "bar":
            st_counts = df["primary_state_name"].value_counts().reset_index().head(15)
            st_counts.columns = ["primary_state_name", "event_count"]
            return generate_bar_chart(
                st_counts,
                cat_col="primary_state_name",
                val_col="event_count",
                title="Historical Flood Events by Primary State",
            )
        elif ctype == "distribution":
            fatal = df.dropna(subset=["human_fatality"])
            return generate_distribution_chart(
                fatal,
                val_col="human_fatality",
                bins=20,
                title="Recorded Flood Casualties Distribution",
            )
        else:
            raise ValueError(f"Unsupported chart_type '{chart_type}' for flood hazard")

    elif haz in ("cyclone", "cyclones"):
        from climatelens.hazards.cyclone_hazard import _load_cyclone_data
        df = _load_cyclone_data()
        if year:
            df = df[df["season"] == year]

        if ctype == "timeseries":
            valid_sn = df.dropna(subset=["season"]).copy()
            valid_sn["season"] = valid_sn["season"].astype(int)
            storms = valid_sn.drop_duplicates(subset=["sid"])
            annual = storms.groupby("season")["sid"].count().reset_index()
            annual.columns = ["season", "storm_count"]
            annual["date"] = pd.to_datetime(annual["season"].astype(str) + "-01-01", format="%Y-%m-%d")
            return generate_timeseries_chart(
                annual,
                date_col="date",
                val_col="storm_count",
                title="Annual North Indian Ocean Cyclone Frequency (IBTrACS)",
            )
        elif ctype == "bar":
            y = year or 2020
            y_df = df[df["season"] == y].drop_duplicates(subset=["sid"])
            return generate_bar_chart(
                y_df,
                cat_col="name",
                val_col="wmo_wind_kmh",
                title=f"Peak Wind Speed (km/h) of Cyclones in {y}",
            )
        elif ctype in ("geographic", "map"):
            return generate_geographic_chart(
                df,
                lat_col="latitude",
                lon_col="longitude",
                val_col="wmo_wind_kmh",
                title=f"Cyclone Track Points (Season {year or 'Historical'})",
            )
        else:
            raise ValueError(f"Unsupported chart_type '{chart_type}' for cyclone hazard")

    elif haz in ("drought", "droughts"):
        from climatelens.hazards.drought_hazard import _load_drought_data
        df = _load_drought_data()

        if ctype == "timeseries":
            return generate_timeseries_chart(
                df,
                date_col="date",
                val_col="d0_d4_pct",
                title="National Drought Area Percentage (D0-D4) 2021-2026",
            )
        elif ctype == "distribution":
            return generate_distribution_chart(
                df,
                val_col="d0_d4_pct",
                bins=15,
                title="National Drought Impact Percentage Distribution",
            )
        else:
            raise ValueError(f"Unsupported chart_type '{chart_type}' for drought hazard")

    elif haz in ("forecasting", "temperature_forecast", "forecast"):
        from climatelens.engine.forecasting import forecast_summer_heatwaves
        target_state = sid or "MAHARASHTRA"
        fc = forecast_summer_heatwaves(target_state, horizon_days=7)
        if fc.get("status") == "no_data":
            fig = go.Figure()
            fig.update_layout(title=fc.get("message", f"No forecast data for {target_state}"))
            return _figure_to_spec(fig)

        if ctype in ("timeseries", "line"):
            preds = fc.get("predictions", [])
            dates = [p["target_date"] for p in preds]
            point_t = [p["predicted_tmax_c"] for p in preds]
            upper_95 = [p["uncertainty_intervals"]["95%"]["upper"] for p in preds]
            lower_95 = [p["uncertainty_intervals"]["95%"]["lower"] for p in preds]
            upper_80 = [p["uncertainty_intervals"]["80%"]["upper"] for p in preds]
            lower_80 = [p["uncertainty_intervals"]["80%"]["lower"] for p in preds]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=upper_95, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=dates, y=lower_95, mode="lines", fill="tonexty", fillcolor="rgba(255, 140, 0, 0.15)", line=dict(width=0), name="95% Confidence Interval"))
            fig.add_trace(go.Scatter(x=dates, y=upper_80, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=dates, y=lower_80, mode="lines", fill="tonexty", fillcolor="rgba(255, 140, 0, 0.3)", line=dict(width=0), name="80% Confidence Interval"))
            fig.add_trace(go.Scatter(x=dates, y=point_t, mode="lines+markers", line=dict(color="#d95f02", width=3), name="Predicted Tmax (°C)"))

            sname = fc.get("state_name", target_state)
            fig.update_layout(
                title=f"7-Day Operational Temperature Forecast - {sname}",
                xaxis_title="Date",
                yaxis_title="Summer Max Temperature (°C)",
                template="plotly_white",
            )
            return _figure_to_spec(fig)

        elif ctype in ("bar", "risk"):
            risk_list = fc.get("heatwave_risk", [])
            dates = [r["target_date"] for r in risk_list]
            probs = [r["exceedance_probability"] for r in risk_list]
            levels = [r["risk_level"] for r in risk_list]
            color_map = {"Low": "#2ca02c", "Moderate": "#ff7f0e", "High": "#d62728"}
            colors = [color_map.get(lvl, "#1f77b4") for lvl in levels]

            fig = go.Figure(data=[go.Bar(
                x=dates,
                y=probs,
                marker_color=colors,
                text=[f"{lvl} ({p:.1%})" for lvl, p in zip(levels, probs)],
                textposition="auto",
            )])
            sname = fc.get("state_name", target_state)
            fig.update_layout(
                title=f"Daily Heatwave Exceedance Probability (>= 40°C) - {sname}",
                xaxis_title="Date",
                yaxis_title="Exceedance Probability",
                yaxis=dict(range=[0, 1.05]),
                template="plotly_white",
            )
            return _figure_to_spec(fig)
        else:
            raise ValueError(f"Unsupported chart_type '{chart_type}' for forecasting hazard")

    elif haz in ("projections", "climate_projections", "cmip6"):
        from climatelens.engine.forecasting import get_state_cmip6_projections
        target_state = sid or "MAHARASHTRA"
        pr = get_state_cmip6_projections(target_state)
        if pr.get("status") == "no_data":
            fig = go.Figure()
            fig.update_layout(title=pr.get("message", f"No projection data for {target_state}"))
            return _figure_to_spec(fig)

        if ctype in ("timeseries", "line", "bar"):
            fig = go.Figure()
            palette = {"SSP1-2.6": "#2ca02c", "SSP2-4.5": "#ff7f0e", "SSP5-8.5": "#d62728"}
            for ssp_name, ssp_info in pr.get("scenarios", {}).items():
                years = [p["target_year"] for p in ssp_info.get("projections", [])]
                temps = [p["projected_absolute_temperature_c"] for p in ssp_info.get("projections", [])]
                c = palette.get(ssp_name, "#1f77b4")
                if ctype == "bar":
                    fig.add_trace(go.Bar(x=[str(y) for y in years], y=temps, name=ssp_name, marker_color=c))
                else:
                    fig.add_trace(go.Scatter(x=years, y=temps, mode="lines+markers", name=ssp_name, line=dict(color=c, width=3)))

            sname = pr.get("state_name", target_state)
            fig.update_layout(
                title=f"CMIP6 Summer Temperature Projections (2030-2050) - {sname}",
                xaxis_title="Year",
                yaxis_title="Projected Summer Tmax (°C)",
                barmode="group" if ctype == "bar" else None,
                template="plotly_white",
            )
            return _figure_to_spec(fig)
        else:
            raise ValueError(f"Unsupported chart_type '{chart_type}' for projections hazard")

    elif haz in ("multi_hazard", "composite_risk"):
        from climatelens.engine.multi_hazard import compute_state_composite_hazard_scores
        comp = compute_state_composite_hazard_scores()

        if ctype in ("bar", "ranking"):
            sorted_states = sorted(comp.values(), key=lambda x: x.get("composite_score_pct", 0.0), reverse=True)[:15]
            state_names = [s["state_name"] for s in sorted_states]
            scores = [s["composite_score_pct"] for s in sorted_states]

            fig = go.Figure(data=[go.Bar(
                x=state_names,
                y=scores,
                marker_color="#b2182b",
                text=[f"{sc:.1f}" for sc in scores],
                textposition="auto",
            )])
            fig.update_layout(
                title="Top 15 States by Multi-Hazard Composite Risk Index (0-100)",
                xaxis_title="State / UT",
                yaxis_title="Composite Risk Index",
                xaxis_tickangle=-45,
                template="plotly_white",
            )
            return _figure_to_spec(fig)

        elif ctype in ("distribution", "box"):
            scores = [s.get("composite_score_pct", 0.0) for s in comp.values()]
            fig = go.Figure(data=[go.Histogram(
                x=scores,
                nbinsx=12,
                marker_color="#2166ac",
                opacity=0.8,
            )])
            fig.update_layout(
                title="Distribution of Multi-Hazard Composite Risk Scores Across States/UTs",
                xaxis_title="Composite Risk Index (0-100)",
                yaxis_title="Number of States / UTs",
                template="plotly_white",
            )
            return _figure_to_spec(fig)
        else:
            raise ValueError(f"Unsupported chart_type '{chart_type}' for multi-hazard")

    else:
        raise ValueError(
            f"Unknown hazard '{hazard}'. Must be one of: heatwave, rainfall, landslide, flood, cyclone, drought, forecasting, projections, multi_hazard"
        )
