import truststore
truststore.inject_into_ssl()
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from pathlib import Path

st.set_page_config(page_title="StormWise | Decision Support Tool", page_icon="🌊", layout="wide")
ROOT = Path(__file__).parent
sites = pd.read_csv(ROOT / "data" / "synthetic_overflow_sites.csv")
options = pd.read_csv(ROOT / "data" / "synthetic_intervention_options.csv")

# Backward-compatible fallback for older copies of the sample sites CSV.
# This is a synthetic placeholder, not a measured community indicator.
community_col = "community_amenity_benefit_0_100"
if community_col not in sites.columns:
    rng = np.random.default_rng(20261104)
    sites[community_col] = rng.integers(35, 96, size=len(sites))


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_yorkshire_edm():
    """Fetch real 2025 Yorkshire Water EDM overflow locations and annual spill attributes."""
    url = (
        "https://services3.arcgis.com/Bb8lfThdhugyc4G3/arcgis/rest/services/"
        "Event_Duration_Monitoring_Storm_Overflows_2025/FeatureServer/0/query"
    )
    params = {
        "where": "waterCompanyName LIKE '%Yorkshire%'",
        "outFields": (
            "waterCompanyName,siteNameEA,siteNameWASC,permitReferenceEA,"
            "recievingWaterName,countedSpills,totalDurationAllSpillsHrs,"
            "Latitude,Longitude,cabaCatchment,localAuthority,edmOperationPercent"
        ),
        "returnGeometry": "false",
        "resultRecordCount": 2000,
        "outSR": 4326,
        "f": "json"
    }
    response = requests.get(url, params=params, timeout=45)
    response.raise_for_status()
    payload = response.json()
    if payload.get("error"):
        raise RuntimeError(payload["error"].get("message", "Public ArcGIS query failed"))
    records = [f.get("attributes", {}) for f in payload.get("features", [])]
    result = pd.DataFrame(records)
    if result.empty:
        return result

    result["latitude"] = pd.to_numeric(result["Latitude"], errors="coerce")
    result["longitude"] = pd.to_numeric(result["Longitude"], errors="coerce")
    # Approximate River Aire corridor extent only; this is not a delineated catchment polygon.
    result = result[
        result["latitude"].between(53.65, 54.20) &
        result["longitude"].between(-2.35, -1.15)
    ].copy()
    result = result.dropna(subset=["latitude", "longitude"])
    result["countedSpills"] = pd.to_numeric(result["countedSpills"], errors="coerce").fillna(0)
    result["totalDurationAllSpillsHrs"] = pd.to_numeric(
        result["totalDurationAllSpillsHrs"], errors="coerce"
    )
    result["asset_label"] = (
        result["siteNameWASC"].fillna(result["siteNameEA"]).fillna("Unnamed overflow")
    )
    return result

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_osm_rivers():
    """Display-only river/watercourse context for the Aire corridor."""
    query = """[out:json][timeout:25];(way[waterway=river](53.65,-2.35,54.20,-1.15);way[waterway=stream][name](53.65,-2.35,54.20,-1.15););out geom;"""
    endpoints = [
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass-api.de/api/interpreter",
    ]
    for url in endpoints:
        try:
            r = requests.post(url, data={"data": query}, timeout=35)
            r.raise_for_status()
            lines=[]
            for el in r.json().get("elements", []):
                geom=el.get("geometry") or []
                coords=[(g["lon"],g["lat"]) for g in geom if "lon" in g and "lat" in g]
                if len(coords) >= 2:
                    lines.append({"name":el.get("tags",{}).get("name") or "River / watercourse","coords":coords})
            if lines:
                return lines
        except Exception:
            continue
    return []

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_ea_aire_catchments():
    """Official EA Aire operational catchment boundaries from Catchment Data Explorer."""
    catchments = {
        "Aire Lower": 3265,
        "Aire Middle": 3307,
        "Aire Upper": 3484,
    }
    features=[]
    for name, cid in catchments.items():
        try:
            url=f"https://environment.data.gov.uk/catchment-planning/OperationalCatchment/{cid}.geojson"
            r=requests.get(url, timeout=25)
            r.raise_for_status()
            gj=r.json()
            for feat in gj.get("features", []):
                feat.setdefault("properties", {})["stormwise_name"] = name
                features.append(feat)
        except Exception:
            continue
    return {"type":"FeatureCollection","features":features}

# ---------- Styling ----------
st.markdown("""
<style>
:root{--navy:#062f4f;--blue:#0b78b5;--aqua:#18a6b8;--bg:#f3f7fa;--card:#ffffff;--line:#d7e5ee;--text:#17364d;--muted:#647c8d;--green:#2e9e59;--amber:#f2a629;--red:#d94a48}
.stApp{background:linear-gradient(180deg,#f7fbfd 0%,#f1f6f9 100%);color:var(--text)}
.block-container{padding-top:.65rem;padding-bottom:1.25rem;max-width:1780px}
header[data-testid="stHeader"]{background:transparent}
.hero{background:linear-gradient(120deg,#052f4f 0%,#07517a 63%,#0b82ae 100%);border-radius:18px;padding:18px 24px;color:white;margin-bottom:12px;box-shadow:0 10px 30px rgba(4,43,75,.18);display:flex;align-items:center;justify-content:space-between;gap:18px}
.hero-left{display:flex;align-items:center;gap:16px}.hero-mark{height:46px;width:46px;border-radius:14px;background:rgba(255,255,255,.14);display:grid;place-items:center;font-size:24px;border:1px solid rgba(255,255,255,.18)}
.hero .brand{font-size:29px;font-weight:850;letter-spacing:-.7px}.hero .subtitle{font-size:13px;font-weight:600;opacity:.9;margin-top:2px}.hero .tag{font-size:11px;background:rgba(255,255,255,.13);border:1px solid rgba(255,255,255,.2);padding:6px 10px;border-radius:999px;white-space:nowrap}
.panel{border:1px solid var(--line);border-radius:16px;padding:14px 16px;background:rgba(255,255,255,.96);box-shadow:0 4px 18px rgba(20,61,87,.06);margin-bottom:11px}
.panel-title{color:#073f69;font-size:16px;font-weight:800;margin-bottom:8px;letter-spacing:-.15px}
.info-note{background:linear-gradient(135deg,#edf8fc,#f4fbfd);border:1px solid #cfe6ef;border-left:4px solid var(--aqua);border-radius:10px;padding:10px 12px;color:#315c73;font-size:11.5px;line-height:1.45}
.small-muted{color:var(--muted);font-size:11px}
div[data-testid="stMetric"]{background:linear-gradient(180deg,#fff,#f8fbfd);border:1px solid var(--line);border-radius:13px;padding:10px 13px;box-shadow:0 2px 8px rgba(23,54,77,.04)}
div[data-testid="stMetricLabel"]{color:#5b7587}div[data-testid="stMetricValue"]{color:#073f69;font-weight:800}
.stTabs [data-baseweb="tab-list"]{gap:5px;background:#eaf2f6;padding:5px;border-radius:13px;margin-bottom:8px}.stTabs [data-baseweb="tab"]{border:0;border-radius:9px;padding:8px 13px;color:#4f6b7d;font-weight:650}.stTabs [aria-selected="true"]{background:white!important;color:#076b9f!important;box-shadow:0 2px 7px rgba(27,74,103,.10)}
div[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:11px;overflow:hidden}
div[data-testid="stSlider"] [data-baseweb="slider"]{margin-top:2px}
button[kind="primary"],.stDownloadButton button{border-radius:10px!important;font-weight:700!important}
div[data-testid="stPlotlyChart"]{border-radius:13px;overflow:hidden}
[data-testid="stExpander"]{border:1px solid var(--line)!important;border-radius:11px!important;background:white}
hr{border-color:#dfe9ef}.stCaption{color:#6b8190!important}
@media(max-width:900px){.hero{align-items:flex-start}.hero .tag{display:none}.block-container{padding-left:.7rem;padding-right:.7rem}}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="hero-left">
    <div class="hero-mark">≈</div>
    <div><div class="brand">StormWise</div><div class="subtitle">Spatial optioneering for smarter overflow investment decisions</div></div>
  </div>
  <div class="tag">LIVE MCDA PROTOTYPE</div>
</div>
""", unsafe_allow_html=True)

# ---------- Scoring ----------
def scale_high(s):
    if s.max() == s.min(): return pd.Series(70.0, index=s.index)
    return (s - s.min()) / (s.max() - s.min()) * 100

def scale_low(s):
    if s.max() == s.min(): return pd.Series(70.0, index=s.index)
    return (s.max() - s) / (s.max() - s.min()) * 100

def assess(budget, weights, intervention_types):
    df = options[options.intervention_type.isin(intervention_types)].merge(sites, on="asset_id")
    if df.empty: return df, df
    df["whole_life_cost_gbp"] = df.capex_gbp + df.annual_opex_gbp * df.design_life_years
    df["benefit_per_cost"] = (
        df.spill_volume_reduction_pct * df.annual_spill_volume_m3 *
        df.receiving_water_sensitivity_0_100 / 100
    ) / (df.whole_life_cost_gbp / 1e6).clip(lower=.01)
    df["cost_score"] = pd.concat([
        scale_low(df.capex_gbp), scale_low(df.annual_opex_gbp),
        scale_low(df.whole_life_cost_gbp), scale_high(df.benefit_per_cost)
    ], axis=1).mean(axis=1)
    df["performance_score"] = df[
        ["spill_frequency_reduction_pct", "spill_duration_reduction_pct",
         "spill_volume_reduction_pct"]
    ].mean(axis=1)
    df["environment_score"] = df.environmental_benefit_0_100
    df["buildability_score"] = df[
        ["land_availability_0_100", "site_accessibility_0_100",
         "ground_suitability_0_100", "planning_utility_suitability_0_100"]
    ].mean(axis=1)
    df["community_score"] = df.community_amenity_benefit_0_100
    score_cols = ["environment_score", "performance_score", "cost_score",
                  "buildability_score", "community_score"]
    total_w = sum(weights.values())
    df["overall_score"] = sum(df[c] * weights[k] for c, k in zip(
        score_cols, ["environment", "performance", "cost", "buildability", "community"]
    )) / total_w
    df["priority"] = pd.cut(df.overall_score, [-1, 50, 65, 80, 101],
                            labels=["Low", "Medium", "High", "Very high"])
    df["annual_volume_reduction_m3"] = df.annual_spill_volume_m3 * df.spill_volume_reduction_pct / 100
    df["annual_events_reduction"] = df.annual_spill_events * df.spill_frequency_reduction_pct / 100
    df["value_per_million"] = df.overall_score / (df.capex_gbp / 1e6).clip(lower=.001)

    # Transparent greedy portfolio heuristic: at most one intervention per asset.
    chosen, spent, used = [], 0.0, set()
    for _, r in df.sort_values("value_per_million", ascending=False).iterrows():
        if r.asset_id not in used and spent + r.capex_gbp <= budget * 1e6:
            chosen.append(r)
            used.add(r.asset_id)
            spent += r.capex_gbp
    portfolio = pd.DataFrame(chosen)
    return df, portfolio


def build_dynamic_priority_map(real_map_df, assessed_df, portfolio_df):
    """Attach the synthetic MCDA result to a stable subset of real EDM points for the POC map.

    The current repository does not contain a verified asset-level key between the public EDM
    layer and the synthetic option table. For demonstration, the highest-spilling real EDM points
    are paired deterministically with synthetic asset IDs. Replace this crosswalk with a verified
    permit/asset join before production use.
    """
    if real_map_df.empty or assessed_df.empty:
        return pd.DataFrame()

    # One recommended intervention per synthetic asset under the current slider weights.
    recommended = (
        assessed_df.sort_values(["asset_id", "overall_score"], ascending=[True, False])
        .groupby("asset_id", as_index=False)
        .first()
    )

    # Stable prototype crosswalk: highest-spilling real EDM points mapped to sorted synthetic IDs.
    real_ranked = (
        real_map_df.sort_values(["countedSpills", "asset_label"], ascending=[False, True])
        .head(len(recommended))
        .reset_index(drop=True)
        .copy()
    )
    synthetic_ids = sorted(recommended["asset_id"].astype(str).unique())
    real_ranked = real_ranked.head(len(synthetic_ids)).copy()
    real_ranked["asset_id"] = synthetic_ids[:len(real_ranked)]

    mapped = real_ranked.merge(
        recommended[[
            "asset_id", "place_name", "intervention_type", "capex_gbp", "overall_score",
            "environment_score", "performance_score", "cost_score",
            "buildability_score", "community_score"
        ]],
        on="asset_id",
        how="left",
    )

    funded_ids = set(portfolio_df["asset_id"].astype(str)) if not portfolio_df.empty else set()
    mapped["selected_in_budget"] = mapped["asset_id"].astype(str).isin(funded_ids)

    # Base recommendation strength calculated from the live MCDA score.
    # Dynamic priority is derived from the live MCDA score. Tertiles are used so the
    # current candidate set visibly contains High, Medium and Low priorities.
    # Budget selection remains a separate portfolio status, avoiding hidden downgrades.
    if mapped["overall_score"].nunique() >= 3:
        mapped["priority"] = pd.qcut(
            mapped["overall_score"].rank(method="first"),
            q=3, labels=["Low", "Medium", "High"]
        ).astype(str)
    else:
        mapped["priority"] = np.select(
            [mapped["overall_score"] >= 75, mapped["overall_score"] >= 55],
            ["High", "Medium"], default="Low"
        )
    mapped["budget_status"] = np.where(
        mapped["selected_in_budget"], "Selected within budget", "Outside current portfolio"
    )
    mapped["indicative_cost_m"] = mapped["capex_gbp"] / 1e6
    mapped["marker_score"] = mapped["overall_score"].clip(lower=1)
    return mapped

# ---------- Navigation ----------
tabs = st.tabs(["🗺️  Map & Assets", "📊  Intervention Priorities", "♟️  Portfolio Selection",
                "▥  Scenario Comparison", "ⓘ  About"])

# Controls live in a right-hand column on the map dashboard, like the reference.
with tabs[0]:
    left, right = st.columns([1.8, 1.0], gap="small")
    with right:
        st.markdown('<div class="panel"><div class="panel-title">💷 Investment scenario</div>', unsafe_allow_html=True)
        budget = st.slider("Total investment budget (£m)", 2.0, 60.0, 25.0, 1.0, key="budget_main")
        st.markdown(f"**£{budget:.0f}M** available for this scenario")
        st.markdown('<div class="info-note">Adjust the total investment budget to recalculate the intervention portfolio. The prototype selects a high-scoring combination within budget, subject to one intervention per asset.</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="panel"><div class="panel-title">⚖️ Decision priorities</div>', unsafe_allow_html=True)
        w_env = st.slider("🌿 Environmental impact", 0, 100, 30, 5, key="w_env")
        w_perf = st.slider("⚙️ Performance improvement", 0, 100, 25, 5, key="w_perf")
        w_cost = st.slider("🪙 Cost and value for money", 0, 100, 20, 5, key="w_cost")
        w_build = st.slider("🔧 Buildability and constraints", 0, 100, 15, 5, key="w_build")
        w_comm = st.slider("👥 Community and amenity", 0, 100, 10, 5, key="w_comm")
        types = st.multiselect("Intervention types", options.intervention_type.unique().tolist(),
                               default=options.intervention_type.unique().tolist(), key="types_main")
        st.markdown('</div>', unsafe_allow_html=True)

    weights = {"environment":w_env, "performance":w_perf, "cost":w_cost,
               "buildability":w_build, "community":w_comm}
    if sum(weights.values()) == 0:
        st.error("Set at least one assessment weight above zero.")
        st.stop()
    df, portfolio = assess(budget, weights, types)
    if df.empty:
        st.warning("Select at least one intervention type.")
        st.stop()
    spent = portfolio.capex_gbp.sum() if len(portfolio) else 0
    with left:
        st.markdown('<div class="panel"><div class="panel-title">🗺️ Live intervention priority map</div>', unsafe_allow_html=True)
        try:
            with st.spinner("Loading real Yorkshire Water EDM overflow locations for the Aire study area…"):
                real_map_df = fetch_yorkshire_edm()
            if real_map_df.empty:
                st.warning("No Yorkshire Water EDM points were returned within the current River Aire study extent.")
            else:
                priority_map_df = build_dynamic_priority_map(real_map_df, df, portfolio)
                if priority_map_df.empty:
                    st.warning("No dynamic intervention recommendations are available for the map.")
                else:
                    fig = px.scatter_map(
                        priority_map_df,
                        lat="latitude",
                        lon="longitude",
                        color="priority",
                        size="marker_score",
                        hover_name="asset_label",
                        hover_data={
                            "priority": True,
                            "intervention_type": True,
                            "overall_score": ":.1f",
                            "indicative_cost_m": ":.1f",
                            "budget_status": True,
                            "environment_score": ":.1f",
                            "performance_score": ":.1f",
                            "cost_score": ":.1f",
                            "buildability_score": ":.1f",
                            "community_score": ":.1f",
                            "countedSpills": True,
                            "totalDurationAllSpillsHrs": ":.1f",
                            "latitude": False,
                            "longitude": False,
                            "marker_score": False,
                        },
                        color_discrete_map={
                            "High": "#2E9E59",
                            "Medium": "#F2A629",
                            "Low": "#D94A48",
                        },
                        category_orders={"priority": ["High", "Medium", "Low"]},
                        map_style="open-street-map",
                        center={"lat": 53.8, "lon": -1.5},
                        zoom=8.0,
                        height=520,
                        size_max=14,
                    )
                    # Official EA operational catchments: explicit polygons for reliable MapLibre display.
                    catchment_geojson = fetch_ea_aire_catchments()
                    catchment_features = catchment_geojson.get("features", [])
                    catchment_legend_added = False
                    for feat in catchment_features:
                        name = feat.get("properties", {}).get("stormwise_name", "Aire operational catchment")
                        geom = feat.get("geometry") or {}
                        gtype = geom.get("type")
                        coords = geom.get("coordinates") or []
                        polygons = coords if gtype == "MultiPolygon" else [coords] if gtype == "Polygon" else []
                        for polygon in polygons:
                            for ring_i, ring in enumerate(polygon):
                                if not ring:
                                    continue
                                lons = [pt[0] for pt in ring]
                                lats = [pt[1] for pt in ring]
                                # Boundary only: keep the catchment as quiet map context.
                                fig.add_trace(go.Scattermap(
                                    lat=lats, lon=lons, mode="lines",
                                    fill="none",
                                    line=dict(width=1.5, color="rgba(20,126,168,0.58)"),
                                    name="Aire operational catchments",
                                    legendgroup="catchments",
                                    showlegend=(not catchment_legend_added and ring_i == 0),
                                    hovertemplate=name + "<extra>EA operational catchment</extra>"
                                ))
                                if ring_i == 0:
                                    catchment_legend_added = True

                    rivers = fetch_osm_rivers()
                    for river in rivers:
                        fig.add_trace(go.Scattermap(
                            lat=[c[1] for c in river["coords"]], lon=[c[0] for c in river["coords"]],
                            mode="lines", line=dict(width=1.25, color="rgba(43,131,198,0.55)"),
                            name="Rivers / watercourses", legendgroup="rivers", showlegend=False,
                            hovertemplate=river["name"] + "<extra>Watercourse</extra>"
                        ))
                    if rivers:
                        fig.add_trace(go.Scattermap(
                            lat=[None], lon=[None], mode="lines",
                            line=dict(width=2, color="rgba(43,131,198,0.65)"), name="Rivers / watercourses",
                            legendgroup="rivers", showlegend=True, hoverinfo="skip"
                        ))
                    fig.update_traces(marker={"opacity": 0.88}, selector=dict(mode="markers"))
                    # Add visual section titles inside the single floating legend.
                    priority_traces=[tr for tr in fig.data if getattr(tr, "mode", None)=="markers" and tr.name in ("High","Medium","Low")]
                    if priority_traces:
                        priority_traces[0].legendgrouptitle = dict(text="Overflow Priority")
                        for tr in priority_traces:
                            tr.legendgroup="priority"
                    context_traces=[tr for tr in fig.data if tr.name in ("Aire operational catchments","Rivers / watercourses") and getattr(tr, "showlegend", False)]
                    if context_traces:
                        context_traces[0].legendgrouptitle = dict(text="Context Layers")
                    fig.update_layout(
                        margin=dict(l=0, r=0, t=0, b=0),
                        uirevision="stormwise-map-view-v1",
                        map=dict(uirevision="stormwise-map-view-v1"),
                        legend=dict(
                            title=dict(text="StormWise Map", font=dict(size=12, color="#17364d")),
                            orientation="v", x=0.015, y=0.98,
                            xanchor="left", yanchor="top",
                            bgcolor="rgba(255,255,255,0.92)",
                            bordercolor="#d7e5ee", borderwidth=1,
                            font=dict(size=11, color="#17364d"), itemsizing="constant"
                        )
                    )
                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        key="stormwise_priority_map"
                    )
                    st.caption(
                        "Map priority recalculates from the current assessment weights and budget. "
                        "Green = High, amber = Medium, red = Low. Marker size represents the live "
                        "MCDA recommendation score. Hover to see the recommended intervention, score, "
                        "indicative cost and budget-selection status. The current POC uses a deterministic "
                        "crosswalk between the highest-spilling real EDM points and synthetic option assets; "
                        "replace this with a verified permit/asset join before production use. "
                        "Blue river lines are OpenStreetMap context loaded through Overpass when available. "
                        "Thin blue outlines are official Environment Agency Aire Upper, Aire Middle and Aire Lower operational catchment boundaries."
                    )
                with st.expander("Map data source and limitations"):
                    st.markdown(
                        "The points and EDM attributes are fetched from the Rivers Trust's public "
                        "2025 Event Duration Monitoring ArcGIS layer, which compiles annual-return "
                        "data from water companies and regulators. Yorkshire Water's own website "
                        "publishes annual EDM returns and notes that records may contain missing or "
                        "incorrect monitor data. The River Aire filter is a geographic bounding window, "
                        "not a hydrologically delineated catchment polygon."
                    )
                    st.markdown(
                        "[Rivers Trust 2025 EDM feature layer](https://services3.arcgis.com/Bb8lfThdhugyc4G3/arcgis/rest/services/Event_Duration_Monitoring_Storm_Overflows_2025/FeatureServer/0) · "
                        "[Yorkshire Water EDM data](https://www.yorkshirewater.com/environment/river-health/storm-overflow-investment/event-duration-monitoring/) · "
                        "[OpenStreetMap attribution](https://www.openstreetmap.org/copyright)"
                    )
        except Exception as e:
            st.error(f"Unable to load the live EDM layer: {e}")
            st.info(
                "This map requires internet access from the computer running Streamlit. "
                "If you're on a corporate network, the ArcGIS service or OSM tiles may be blocked."
            )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-title">★ Recommended portfolio</div>', unsafe_allow_html=True)
    if len(portfolio):
        show = portfolio.sort_values("overall_score", ascending=False).head(8)[
            ["asset_id","place_name","intervention_type","capex_gbp","overall_score","priority"]
        ].copy()
        show.insert(0, "Rank", range(1, len(show)+1))
        show["capex_gbp"] = show.capex_gbp.map(lambda v:f"£{v/1e6:.1f}M")
        show["overall_score"] = show.overall_score.round(0).astype(int)
        show = show.rename(columns={"asset_id":"Asset ID","place_name":"Location",
                                    "intervention_type":"Intervention type",
                                    "capex_gbp":"Estimated cost","overall_score":"Score",
                                    "priority":"Priority"})
        st.dataframe(show, use_container_width=True, hide_index=True)
    else:
        st.info("No interventions fit within this budget.")
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Portfolio investment", f"£{spent/1e6:.1f}M", f"of £{budget:.0f}M budget")
    k2.metric("Interventions selected", len(portfolio))
    k3.metric("Indicative spill volume reduction", f"{portfolio.annual_volume_reduction_m3.sum():,.0f} m³/yr" if len(portfolio) else "0 m³/yr")
    k4.metric("Assets covered", portfolio.asset_id.nunique() if len(portfolio) else 0)
    st.markdown('</div>', unsafe_allow_html=True)

with tabs[1]:
    st.subheader("Intervention priorities")
    st.caption("Rankings recalculate using the current assessment weights and selected intervention types.")
    weights = {"environment":w_env, "performance":w_perf, "cost":w_cost,
               "buildability":w_build, "community":w_comm}
    ranked, _ = assess(budget, weights, types)
    ranked = ranked.sort_values("overall_score", ascending=False)
    view = ranked[["asset_id","place_name","intervention_type","capex_gbp","environment_score",
                   "performance_score","cost_score","buildability_score","community_score","overall_score","priority"]].copy()
    view["capex_gbp"] = view.capex_gbp.map(lambda v:f"£{v:,.0f}")
    st.dataframe(view.round(1), use_container_width=True, hide_index=True)
    st.download_button("Download prioritisation CSV", ranked.to_csv(index=False).encode(),
                       "stormwise_intervention_priorities.csv", "text/csv")

with tabs[2]:
    st.subheader("Portfolio selection")
    st.write("The current prototype uses a transparent greedy selection heuristic, ranking options by weighted score per £m and selecting at most one intervention per asset. It is not a mathematically guaranteed global optimum.")
    if len(portfolio):
        st.dataframe(portfolio[["asset_id","place_name","intervention_type","capex_gbp","overall_score",
                                "annual_events_reduction","annual_volume_reduction_m3"]].round(1),
                     use_container_width=True, hide_index=True)
        st.download_button("Export selected portfolio", portfolio.to_csv(index=False).encode(),
                           "stormwise_selected_portfolio.csv", "text/csv")
    else:
        st.info("No interventions fit the selected budget.")
    st.caption("Benefits are summed synthetic estimates. Hydraulic interactions and uncertainty are not modelled.")

with tabs[3]:
    st.subheader("Scenario comparison")
    st.caption("Compare how the selected weights influence the portfolio at three indicative budget levels.")
    rows = []
    for bgt in [10.0, 25.0, 50.0]:
        dfx, pfx = assess(bgt, weights, types)
        rows.append({"Budget (£m)":bgt, "Selected interventions":len(pfx),
                     "Investment (£m)":pfx.capex_gbp.sum()/1e6 if len(pfx) else 0,
                     "Volume reduction (m³/yr)":pfx.annual_volume_reduction_m3.sum() if len(pfx) else 0,
                     "Events reduced (/yr)":pfx.annual_events_reduction.sum() if len(pfx) else 0})
    comp = pd.DataFrame(rows)
    st.dataframe(comp.round(1), use_container_width=True, hide_index=True)
    chart = px.bar(comp, x="Budget (£m)", y="Volume reduction (m³/yr)",
                   title="Indicative annual spill-volume reduction by budget")
    st.plotly_chart(chart, use_container_width=True)

with tabs[4]:
    st.subheader("About this prototype")
    st.warning("DEMONSTRATION ONLY — mapped EDM locations and annual spill fields are public 2025 records; intervention options, costs, benefits, constraints and the temporary asset crosswalk are synthetic. Do not use for engineering, investment or regulatory decisions.")
    st.markdown("""
**Prototype workflow**
1. View public 2025 Yorkshire Water EDM overflow points in the River Aire study window.
2. Adjust budget and assessment criteria.
3. Compare intervention options and their relative scores.
4. Select a budget-constrained portfolio and compare scenarios.

**Assessment criteria**
- **Environmental impact:** synthetic environmental benefit score.
- **Performance improvement:** assumed spill frequency, duration and volume reductions.
- **Cost and value for money:** relative CAPEX, OPEX, whole-life cost and benefit-per-cost indicator.
- **Buildability and constraints:** land availability, access, ground suitability and planning/utility suitability.
- **Community and amenity:** synthetic contextual score added for demonstration; it is not derived from population or real community datasets.

**Limitations**
- Scores are relative to the options currently selected and normalised against the current dataset.
- The budget selector is a greedy heuristic, not an exact optimiser.
- No hydraulic modelling, engineering design, regulatory assessment, spatial constraint overlays, uncertainty propagation, discounting or catchment-wide interactions are modelled.
- The map uses a rectangular River Aire study window and a temporary deterministic crosswalk from real EDM points to synthetic intervention options; it is not a verified asset-level join or an official catchment boundary.
- Check licensing, provenance and quality before adding real open data, and record sources in the Data & Process Governance Log.
    """)
    st.download_button("Download overflow asset data", sites.to_csv(index=False).encode(),
                       "synthetic_overflow_sites.csv", "text/csv")
    st.download_button("Download intervention options", options.to_csv(index=False).encode(),
                       "synthetic_intervention_options.csv", "text/csv")
    st.download_button("Download data dictionary", pd.DataFrame([
        ["community_amenity_benefit_0_100","Synthetic demonstration score for community/amenity context; not real demographic data."],
        ["environmental_benefit_0_100","Synthetic environmental benefit estimate."],
        ["spill_frequency_reduction_pct","Assumed percentage reduction in annual spill event frequency."],
        ["spill_duration_reduction_pct","Assumed percentage reduction in annual spill duration."],
        ["spill_volume_reduction_pct","Assumed percentage reduction in annual spill volume."]
    ], columns=["Field","Description"]).to_csv(index=False).encode(),
                       "stormwise_data_dictionary_additions.csv", "text/csv")

st.caption("StormWise hackathon prototype • Public EDM context + synthetic intervention model • For demonstration and discussion")
