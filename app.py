#Chnage Requirement as per your desired

import json, io
import pandas as pd
from flask import Flask, render_template_string, request, Response
from collections import Counter

app       = Flask(__name__)
FILE_PATH = "2.73 Total.csv"

# RAG thresholds
RAG_GREEN_PCT  = 30   # Resolved % >= 30 -> GREEN
RAG_AMBER_PCT  = 10   # Resolved % 10-29 -> AMBER, <10 -> RED
RAG_GREEN_RSRP = -100 # RSRP >= -100 -> GREEN
RAG_AMBER_RSRP = -105 # RSRP -105 to -100 -> AMBER, < -105 -> RED

def _rag_pct(pct):
    if pct >= RAG_GREEN_PCT:  return "green"
    if pct >= RAG_AMBER_PCT:  return "amber"
    return "red"

def _rag_rsrp(rsrp):
    if rsrp >= RAG_GREEN_RSRP: return "green"
    if rsrp >= RAG_AMBER_RSRP: return "amber"
    return "red"

def _overall_rag(pct, rsrp):
    scores = {"green": 0, "amber": 1, "red": 2}
    worst  = max(scores[_rag_pct(pct)], scores[_rag_rsrp(rsrp)])
    return ["green", "amber", "red"][worst]

# ============================================================================
# DATA LAYER
# ============================================================================
def load_and_process_data():
    if not __import__("os").path.exists(FILE_PATH):
        return None
    for enc in ("cp1252", "latin-1", "utf-8"):
        try:
            df = pd.read_csv(FILE_PATH, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        return None 

    COL_SHIFT    = next((c for c in df.columns if 'ownership shift' in c.lower() or 'ownershift' in c.lower()), None)
    if COL_SHIFT is None:
        COL_SHIFT = next((c for c in df.columns if 'ownership' in c.lower()), "Ownership")
        
    COL_REMARK   = next((c for c in [
        "Categogrization of Remark By Concern Team IMP",
        "Categorization of Remark By Concern Team IMP",
        "Categorisation of Remark By Concern Team IMP",
    ] if c in df.columns), "Categogrization of Remark By Concern Team IMP")
    COL_DETAILED = next((c for c in df.columns if "Detailed Remark" in c), "Detailed Remark if required")
    COL_GRID     = next((c for c in df.columns if c.startswith("Grid ID")), "Grid ID - 100Meter Grid if 2 consicutive grids are attached it will merge with single grid Id")
    COL_RSRP     = "RSRP"
    COL_BUCKET   = next((c for c in df.columns if "Grid" in c and "Bucket" in c and "Meter" in c), "Grid Samrt Meter Count Bucket for focus target area")
    COL_LAT      = "Device_Lat"
    COL_LNG      = "Device_long"
    
    DISTRICT_CANDIDATES = [
        "District", "district", "DISTRICT",
        "District Name", "District_Name", "DistrictName",
        "Dist", "dist", "DIST",
        "Distt", "distt",
        "City", "city",
        "Taluka", "taluka",
        "Zone", "zone",
    ]
    COL_DISTRICT = next((c for c in DISTRICT_CANDIDATES if c in df.columns), None)
    if COL_DISTRICT is None:
        COL_DISTRICT = next((c for c in df.columns if 'dist' in c.lower()), None)
    if COL_DISTRICT is None:
        COL_DISTRICT = "__NO_DISTRICT__"
        df["__NO_DISTRICT__"] = "Unknown"

    COL_STATUS   = "Status"

    for col in ["Circle","Region","Ownership",COL_SHIFT,COL_REMARK,COL_DETAILED,
                COL_STATUS,"Resolved",COL_GRID,COL_RSRP,COL_BUCKET,COL_LAT,COL_LNG]:
        if col not in df.columns:
            df[col] = 0 if col == COL_RSRP else "N/A"

    df = df.fillna("N/A")
    df["Final Ownership"] = df[COL_SHIFT].astype(str).str.strip()
    df["Task Category"]   = df[COL_REMARK].astype(str).str.strip().replace({"N/A":"Still Working","":"Still Working"})
    df["Detailed Remark"] = df[COL_DETAILED].astype(str).str.strip().replace({"N/A":""})
    df["Resolved Status"] = df["Resolved"].astype(str).str.strip()
    df["RSRP_Num"]        = pd.to_numeric(df[COL_RSRP], errors="coerce").fillna(0)
    df["Is_Resolved"]     = (df["Resolved Status"].str.lower() == "resolved").astype(int)
    
    import sys as _sys
    print(f"[District] Auto-detected column: '{COL_DISTRICT}'", file=_sys.stderr)
    df["District"] = df[COL_DISTRICT].astype(str).str.strip()
    df["Kanban_Status"]   = df[COL_STATUS].astype(str).str.strip()

    def _count(kw): return int(df["Final Ownership"].str.contains(kw, case=False, na=False).sum())
    total  = len(df)
    b2b_t  = _count("B2B")
    exp_t  = _count("Experience")
    plan_t = _count("Planning")
    kpis = {
        "total": f"{total:,}", "b2b": f"{b2b_t:,}", "exp": f"{exp_t:,}", "plan": f"{plan_t:,}",
        "b2b_pct":  round(b2b_t/total*100,1)  if total else 0,
        "exp_pct":  round(exp_t/total*100,1)   if total else 0,
        "plan_pct": round(plan_t/total*100,1)  if total else 0,
    }

    owner_counts = df["Final Ownership"].value_counts()
    circle_owner = pd.crosstab(df["Circle"], df["Final Ownership"])
    def _gc(col):
        if col in circle_owner.columns: return circle_owner[col].tolist()
        m = [c for c in circle_owner.columns if col.lower() in str(c).lower()]
        return circle_owner[m].sum(axis=1).tolist() if m else [0]*len(circle_owner)

    mat = df.groupby(["Final Ownership","Task Category","Circle"]).agg(
        Meter_Count=("Final Ownership","count"),
        Resolved_Count=("Resolved Status", lambda x: int((x.astype(str).str.strip().str.lower()=="resolved").sum()))
    ).reset_index()

    grid_df = df[df[COL_GRID]!="N/A"].groupby([COL_GRID,"Circle"]).agg(
        Meter_Count=(COL_GRID,"count"), Avg_RSRP=("RSRP_Num","mean"),
        Resolved_Count=("Is_Resolved","sum"), Bucket=(COL_BUCKET,"first"),
        Lat=(COL_LAT,"first"), Lng=(COL_LNG,"first")
    ).reset_index().rename(columns={COL_GRID:"Grid_ID"})
    grid_df["Avg_RSRP"] = grid_df["Avg_RSRP"].fillna(0).round(1)
    grid_df = grid_df.sort_values("Meter_Count", ascending=False)
    grid_data = grid_df[["Grid_ID","Meter_Count","Avg_RSRP","Resolved_Count","Circle","Bucket","Lat","Lng"]].values.tolist()

    dist_df = df.groupby(["Circle","District"]).agg(
        Total=("Is_Resolved","count"), Resolved=("Is_Resolved","sum"),
        Avg_RSRP=("RSRP_Num","mean"),
        B2B=("Final Ownership", lambda x: int(x.str.contains("B2B",case=False,na=False).sum())),
        Experience=("Final Ownership", lambda x: int(x.str.contains("Experience",case=False,na=False).sum())),
        Planning=("Final Ownership", lambda x: int(x.str.contains("Planning",case=False,na=False).sum())),
    ).reset_index()
    dist_df["Resolved_Pct"] = (dist_df["Resolved"]/dist_df["Total"]*100).round(1)
    dist_df["Avg_RSRP"]     = dist_df["Avg_RSRP"].round(1)
    dist_df["RAG_Resolved"] = dist_df["Resolved_Pct"].apply(_rag_pct)
    dist_df["RAG_RSRP"]     = dist_df["Avg_RSRP"].apply(_rag_rsrp)
    dist_df["RAG_Overall"]  = dist_df.apply(lambda r: _overall_rag(r["Resolved_Pct"], r["Avg_RSRP"]), axis=1)
    dist_df = dist_df.sort_values(["Circle","RAG_Overall","Resolved_Pct"], ascending=[True,True,True])
    scorecard_data = dist_df.to_dict(orient="records")
    circles_list   = sorted([c for c in df["Circle"].dropna().unique() if c != "N/A"])

    COL_TASK_CAT  = next((c for c in ["Categorization of Remark By Concern Team IMP",
                                       "Categogrization of Remark By Concern Team IMP",
                                       "Categorisation of Remark By Concern Team IMP"] if c in df.columns),
                          "Categorization of Remark By Concern Team IMP")
    COL_NEXT_STEP = next((c for c in ["Latest Updated 5/6/2026","Latest Updated","Latest_Updated"] if c in df.columns), None)
    COL_PLAN_SITE = next((c for c in ["Plan Site ID if Any","Plan_Site_ID","Plan Site ID"] if c in df.columns), None)
    COL_SITE_STS  = next((c for c in ["Site Status","Site_Status"] if c in df.columns), None)
    COL_ACTIVITY  = next((c for c in ["Activity Done","Activity_Done"] if c in df.columns), None)
    COL_CUST_ACT  = next((c for c in ["Customer Action","Customer_Action"] if c in df.columns), None)
    COL_SUBCAT    = next((c for c in ["Subcategory","Sub Category","SubCategory"] if c in df.columns), None)

    def _safe_col(col_name):
        if col_name and col_name in df.columns:
            return df[col_name].astype(str).str.strip().replace({"N/A":"","nan":"","<NA>":""})
        return pd.Series([""] * len(df), index=df.index)

    df["Next_Step"]    = _safe_col(COL_NEXT_STEP)
    df["Plan_Site"]    = _safe_col(COL_PLAN_SITE)
    df["Site_Status"]  = _safe_col(COL_SITE_STS)
    df["Activity_Done"]= _safe_col(COL_ACTIVITY)
    df["Cust_Action"]  = _safe_col(COL_CUST_ACT)
    df["Subcategory"]  = _safe_col(COL_SUBCAT)
    df["Bucket"]       = _safe_col(COL_BUCKET)

    def _site_status_breakdown(s_df):
        site_map = {}
        for _, row in s_df[s_df["Plan_Site"].str.len() > 0].iterrows():
            for s in str(row["Plan_Site"]).split(","):
                s = s.strip()
                if s and s.lower() not in ("n/a","nan","") and s not in site_map:
                    site_map[s] = str(row["Site_Status"]).strip()
        counts = {}
        for sts in site_map.values():
            if sts and sts.lower() not in ("n/a","nan","","<na>"):
                counts[sts] = counts.get(sts, 0) + 1
        return counts, len(site_map)

    def _build_stage_data(s_df, color, total_scope):
        cnt = len(s_df)
        if cnt == 0:
            return {"count":0,"noncomm_total":0,"color":color,"b2b":0,"exp":0,"plan":0,
                    "circles":{},"resolved_total":0,"resolved_circles":{},
                    "noncomm_circles":{},"noncomm_buckets":{},"activity_breakdown":{},
                    "cust_action_breakdown":{},"action_plan":{},"site_plan_breakdown":{},
                    "total_sites":0,"site_statuses":{},"next_steps":[],"pct":0}

        circles = s_df["Circle"].value_counts().head(5).to_dict()

        raw_resolved = s_df["Resolved Status"].astype(str).str.strip().str.lower()
        is_empty_resolved = raw_resolved.isin(["", "nan", "n/a", "<na>", "none", "0", "null", "-", "false", "pending", "unresolved", "still working"])

        resolved_mask    = ~is_empty_resolved
        resolved_total   = int(resolved_mask.sum())
        resolved_circles = s_df[resolved_mask]["Circle"].value_counts().to_dict()

        unresolved_mask  = is_empty_resolved
        noncomm_total    = int(unresolved_mask.sum())
        noncomm_circles  = s_df[unresolved_mask]["Circle"].value_counts().to_dict()

        act_series = s_df["Activity_Done"]
        activity_breakdown = {}
        for v in act_series:
            if v:
                activity_breakdown[v] = activity_breakdown.get(v, 0) + 1
        activity_breakdown = dict(sorted(activity_breakdown.items(), key=lambda x: -x[1])[:8])

        cust_series = s_df["Cust_Action"]
        cust_action_breakdown = {}
        for v in cust_series:
            if v:
                cust_action_breakdown[v] = cust_action_breakdown.get(v, 0) + 1
        cust_action_breakdown = dict(sorted(cust_action_breakdown.items(), key=lambda x: -x[1])[:6])

        unresolved = s_df[unresolved_mask]
        next_steps_raw = [v for v in unresolved["Next_Step"] if v]
        step_counts = Counter(next_steps_raw)
        action_plan = dict(step_counts.most_common(6))

        bucket_counts_raw = unresolved["Bucket"].value_counts().to_dict()
        noncomm_buckets = {k if str(k).strip() else "Unknown": v for k, v in bucket_counts_raw.items()}

        site_statuses, total_sites = _site_status_breakdown(s_df)
        task_counts = s_df["Task Category"].value_counts().head(8).to_dict()
        next_steps = list(dict(Counter(next_steps_raw).most_common(4)).keys())

        return {
            "count":                 cnt,
            "color":                 color,
            "b2b":                   int(s_df["Final Ownership"].str.contains("B2B",        case=False, na=False).sum()),
            "exp":                   int(s_df["Final Ownership"].str.contains("Experience", case=False, na=False).sum()),
            "plan":                  int(s_df["Final Ownership"].str.contains("Planning",   case=False, na=False).sum()),
            "circles":               circles,
            "resolved_total":        resolved_total,
            "resolved_circles":      resolved_circles,
            "noncomm_total":         noncomm_total,
            "noncomm_circles":       noncomm_circles,
            "noncomm_buckets":       noncomm_buckets,
            "activity_breakdown":    activity_breakdown,
            "cust_action_breakdown": cust_action_breakdown,
            "action_plan":           action_plan,
            "task_counts":           task_counts,
            "total_sites":           total_sites,
            "site_statuses":         site_statuses,
            "next_steps":            next_steps,
            "pct":                   round(cnt / total_scope * 100, 1) if total_scope else 0,
        }

    all_kanban_data = {}
    
    def get_kanban_rows(target_df):
        total_scope = len(target_df)
        k_b2b  = target_df["Final Ownership"].str.contains("B2B", case=False, na=False)
        k_exp  = target_df["Final Ownership"].str.contains("Experience", case=False, na=False)
        k_plan = target_df["Final Ownership"].str.contains("Planning", case=False, na=False)
        
        s_defs = [
            ("B2B Action", k_b2b,  "#06b6d4"),
            ("Experience", k_exp,  "#f59e0b"),
            ("Planning",   k_plan, "#f43f5e"),
        ]
        
        k_rows = []
        for stage_name, mask, color in s_defs:
            data = _build_stage_data(target_df[mask].copy(), color, total_scope)
            data["stage"] = stage_name
            k_rows.append(data)
        return k_rows

    all_kanban_data["All"] = get_kanban_rows(df)
    for circ in circles_list:
        sub_df = df[df["Circle"].astype(str).str.strip().str.upper() == circ.upper()]
        all_kanban_data[circ] = get_kanban_rows(sub_df)

    chart_data = {
        "owner_labels":   json.dumps(owner_counts.index.tolist()),
        "owner_data":     json.dumps(owner_counts.values.tolist()),
        "circles":        json.dumps(circle_owner.index.tolist()),
        "b2b":            json.dumps(_gc("B2B")),
        "exp":            json.dumps(_gc("Experience")),
        "plan":           json.dumps(_gc("Planning")),
        "kpis":           kpis,
        "matrix_data":    json.dumps(mat.to_dict(orient="records")),
        "grid_data":      json.dumps(grid_data),
        "scorecard_data": json.dumps(scorecard_data),
        "circles_list":   json.dumps(circles_list),
        "kanban_data_all":json.dumps(all_kanban_data, default=str),
        "total_raw": total, "b2b_raw": b2b_t, "exp_raw": exp_t, "plan_raw": plan_t,
    }
    tables = {
        "tab1": _build_team_summary(df, "B2B",        "#06b6d4"),
        "tab2": _build_team_summary(df, "Experience",  "#f59e0b"),
        "tab3": _build_team_summary(df, "Planning",    "#f43f5e"),
    }
    return chart_data, tables

def _build_team_summary(df, kw, accent):
    t = df[df["Final Ownership"].str.contains(kw, case=False, na=False)].copy()
    if t.empty: return []
    s = (t.groupby("Task Category").agg(
        count=("Task Category","count"),
        remarks=("Detailed Remark", lambda x: " | ".join(sorted(set(v for v in x if v)))),
        resolved_count=("Resolved Status", lambda x: int((x.astype(str).str.strip().str.lower()=="resolved").sum())),
    ).reset_index().sort_values("count", ascending=False))
    mx = int(s["count"].max()) if not s.empty else 1
    tot = int(s["count"].sum()); tr = int(s["resolved_count"].sum())
    rows = []
    for _, r in s.iterrows():
        c = int(r["count"]); rc = int(r["resolved_count"])
        rows.append({"category":str(r["Task Category"]),"count":f"{c:,}","pct_bar":round(c/mx*100),
                     "pct_of_total":round(c/tot*100,1),"remarks":str(r["remarks"]),
                     "resolved_count":f"{rc:,}","resolved_pct":round(rc/c*100,1) if c else 0,
                     "has_resolved":rc>0,"accent":accent})
    rows.append({"category":"GRAND TOTAL","count":f"{tot:,}","pct_bar":100,"pct_of_total":100,
                 "remarks":"","resolved_count":f"{tr:,}","resolved_pct":round(tr/tot*100,1) if tot else 0,
                 "has_resolved":tr>0,"accent":accent,"is_total":True})
    return rows

@app.route("/export_grid_details")
def export_grid_details():
    try:
        df = pd.read_csv(FILE_PATH, encoding="cp1252")
    except UnicodeDecodeError:
        df = pd.read_csv(FILE_PATH, encoding="latin-1")
    except FileNotFoundError:
        return "File not found.", 404
    cf = request.args.get("circle","All"); bf = request.args.get("bucket","All")
    cg = "Grid ID - 100Meter Grid if 2 consicutive grids are attached it will merge with single grid Id"
    cb = "Grid Samrt Meter Count Bucket for focus target area"
    if cg in df.columns: df = df[df[cg].notna()]
    if cf != "All" and "Circle" in df.columns: df = df[df["Circle"].astype(str).str.upper()==cf.upper()]
    if bf != "All" and cb in df.columns:
        b = bf.replace(" ","").replace("Devices","").lower()
        df = df[df[cb].astype(str).str.lower().str.replace(" ","").str.replace("devices","").str.contains(b,na=False)]
    out = io.StringIO(); df.to_csv(out, index=False); out.seek(0)
    return Response(out.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment;filename=Grid_{cf}.csv"})

@app.route("/download_kml")
def download_kml():
    lat = request.args.get("lat"); lng = request.args.get("lng")
    gid = request.args.get("grid_id","Grid")
    if not lat or not lng: return "Coords missing.",400
    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Placemark><name>{gid}</name>
  <LookAt><longitude>{lng}</longitude><latitude>{lat}</latitude>
    <altitude>0</altitude><heading>0</heading><tilt>45</tilt>
    <range>800</range><altitudeMode>relativeToGround</altitudeMode></LookAt>
  <Point><coordinates>{lng},{lat},0</coordinates></Point>
</Placemark></kml>"""
    return Response(kml, mimetype="application/vnd.google-earth.kml+xml",
                    headers={"Content-Disposition": f"attachment;filename={gid}.kml"})

@app.route("/save_edit", methods=["POST"])
def save_edit():
    import json as _json
    from flask import jsonify
    EDIT_PASSWORD = "adani2024"          
    EDITABLE_FIELDS = {
        "Resolved",
        "Status",
        "Detailed Remark if required",
        "Categorization of Remark By Concern Team IMP",
    }
    try:
        payload   = request.get_json(force=True)
        password  = payload.get("password", "")
        meter_id  = str(payload.get("meter_id", "")).strip()
        field     = str(payload.get("field", "")).strip()
        new_value = str(payload.get("value", "")).strip()

        if password != EDIT_PASSWORD:
            return jsonify({"ok": False, "msg": "Wrong password"}), 403
        if field not in EDITABLE_FIELDS:
            return jsonify({"ok": False, "msg": f"Field '{field}' is not editable"}), 400

        try:
            df = pd.read_csv(FILE_PATH, encoding="cp1252")
        except UnicodeDecodeError:
            df = pd.read_csv(FILE_PATH, encoding="latin-1")

        id_col = None
        for candidate in ["Meter ID", "Meter_ID", "MeterID", "meter_id", "Device_ID", "device_id"]:
            if candidate in df.columns:
                id_col = candidate
                break
        if id_col is None:
            id_col = df.columns[0]          

        mask = df[id_col].astype(str).str.strip() == meter_id
        if not mask.any():
            return jsonify({"ok": False, "msg": f"Meter ID '{meter_id}' not found"}), 404

        df.loc[mask, field] = new_value
        df.to_csv(FILE_PATH, index=False)
        return jsonify({"ok": True, "msg": f"Saved: {field} = {new_value} for {meter_id}"})

    except Exception as exc:
        return jsonify({"ok": False, "msg": str(exc)}), 500

@app.route("/health")
def health():
    import os
    csv_ok = os.path.exists(FILE_PATH)
    return {"status": "ok", "csv_found": csv_ok, "file": FILE_PATH}, 200


TEMPLATE = r"""
{# Macros must be defined before first use #}
{% macro _team_table(rows, team_name, accent) %}
{% if not rows %}
  <p style="padding:32px;color:var(--text-m)">No data for this team.</p>
{% else %}
<div class="table-wrap"><table class="summary">
  <thead><tr>
    <th style="width:35%">{{ team_name }} Action Item</th>
    <th style="width:28%">Device Count</th><th>Detailed Remarks</th>
    <th style="width:130px;text-align:center">Status</th>
  </tr></thead>
  <tbody>
  {% for row in rows %}
  <tr {% if row.get('is_total') %}class="total-row"{% endif %}>
    <td>{% if row.get('is_total') %}GRAND TOTAL{% else %}<span class="task-name">{{ row.category }}</span>{% endif %}</td>
    <td>{% if row.get('is_total') %}{{ row.count }}
        {% else %}<div class="count-cell">
          <span class="count-num">{{ row.count }}</span>
          <div class="bar-track"><div class="bar-fill" style="width:{{ row.pct_bar }}%;background:{{ accent }}"></div></div>
          <span class="pct-label">{{ row.pct_of_total }}%</span>
        </div>{% endif %}</td>
    <td><span class="remarks-text">{{ row.remarks }}</span></td>
    <td style="text-align:center;min-width:130px">
      {% if row.get('is_total') %}
        {% if row.has_resolved %}<div class="resolved-stack">
          <span class="status-badge badge-resolved"><span class="badge-dot dot-green"></span>Resolved</span>
          <span class="resolved-sub">{{ row.resolved_count }} &middot; {{ row.resolved_pct }}%</span>
        </div>{% else %}<span class="status-badge badge-empty">&#8212;</span>{% endif %}
      {% elif row.has_resolved %}<div class="resolved-stack">
        <span class="status-badge badge-resolved"><span class="badge-dot dot-green"></span>Resolved</span>
        <span class="resolved-sub">{{ row.resolved_count }} of {{ row.count }}</span>
      </div>
      {% else %}<span class="status-badge badge-pending"><span class="badge-dot dot-amber"></span>Pending</span>{% endif %}
    </td>
  </tr>{% endfor %}
  </tbody>
</table></div>{% endif %}{% endmacro %}
<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
<meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>Smart Meter Operations | CXO Dashboard v3</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700&family=DM+Mono:wght@400;500;700&display=swap" rel="stylesheet"/>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/exceljs/4.3.0/exceljs.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/FileSaver.js/2.0.5/FileSaver.min.js"></script>
<!-- IMPORT html2canvas for Image Export -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
<style>
:root{
  --bg:#f0f4f8;--surface:#fff;--surface2:#f8fafc;--border:#e2e8f0;
  --text-h:#0f172a;--text-b:#334155;--text-m:#64748b;--text-s:#94a3b8;
  --cyan:#06b6d4;--cyan-soft:#ecfeff;--amber:#f59e0b;--amber-soft:#fffbeb;
  --rose:#f43f5e;--rose-soft:#fff1f2;--indigo:#6366f1;--indigo-soft:#eef2ff;
  --green:#10b981;--green-soft:#ecfdf5;
  --shadow-sm:0 1px 3px rgba(15,23,42,.06),0 1px 2px rgba(15,23,42,.04);
  --shadow-md:0 4px 16px rgba(15,23,42,.08),0 2px 6px rgba(15,23,42,.04);
  --r:14px;
  --rag-green:#16a34a;--rag-green-bg:#f0fdf4;--rag-green-border:#bbf7d0;
  --rag-amber:#d97706;--rag-amber-bg:#fffbeb;--rag-amber-border:#fde68a;
  --rag-red:#dc2626;--rag-red-bg:#fef2f2;--rag-red-border:#fecaca;
}
[data-theme="dark"]{
  --bg:#0a0f1e;--surface:#111827;--surface2:#1f2937;--border:#374151;
  --text-h:#f1f5f9;--text-b:#cbd5e1;--text-m:#94a3b8;--text-s:#64748b;
  --cyan-soft:#0c4a6e;--amber-soft:#451a03;--rose-soft:#4c0519;--indigo-soft:#1e1b4b;--green-soft:#052e16;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:'DM Sans',system-ui,sans-serif;background:var(--bg);color:var(--text-b);font-size:14px;line-height:1.6;-webkit-font-smoothing:antialiased;transition:background .3s,color .3s}
.topbar{background:var(--surface);border-bottom:1px solid var(--border);padding:0 28px;height:60px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:200;box-shadow:var(--shadow-sm)}
.topbar-brand{display:flex;align-items:center;gap:10px}
.topbar-logo{width:34px;height:34px;border-radius:9px;background:linear-gradient(135deg,#0ea5e9,#6366f1);display:flex;align-items:center;justify-content:center;font-size:17px}
.topbar-title{font-size:14px;font-weight:700;color:var(--text-h)}.topbar-sub{font-size:11px;color:var(--text-m)}
.topbar-right{display:flex;align-items:center;gap:10px}
.nav-pills{display:flex;gap:3px;background:var(--surface2);border:1px solid var(--border);border-radius:10px;padding:3px}
.nav-pill{padding:6px 13px;border:none;background:transparent;border-radius:7px;font-family:inherit;font-size:12px;font-weight:600;color:var(--text-m);cursor:pointer;transition:all .18s;white-space:nowrap}
.nav-pill:hover{color:var(--text-h)}.nav-pill.active{background:var(--text-h);color:#fff;box-shadow:0 2px 6px rgba(0,0,0,.2)}
.live-badge{display:inline-flex;align-items:center;gap:5px;background:var(--green-soft);color:var(--rag-green);border:1px solid var(--rag-green-border);border-radius:20px;padding:4px 10px;font-size:11px;font-weight:700}
.live-dot{width:6px;height:6px;border-radius:50%;background:var(--rag-green);animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
.topbar-ts{font-size:11px;color:var(--text-m)}
.dm-btn{width:34px;height:34px;border:1px solid var(--border);border-radius:8px;background:var(--surface2);cursor:pointer;font-size:15px;display:flex;align-items:center;justify-content:center;transition:all .2s}
.dm-btn:hover{border-color:var(--indigo)}
.main{padding:24px 28px 48px;max-width:1440px;margin:0 auto;width:100%}
.section-label{font-size:10px;font-weight:800;letter-spacing:.9px;text-transform:uppercase;color:var(--text-s);margin-bottom:12px;display:flex;align-items:center;gap:8px}
.section-label::after{content:'';flex:1;height:1px;background:var(--border)}
.kpi-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:24px}
.kpi-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:20px 22px 18px;box-shadow:var(--shadow-sm);position:relative;overflow:hidden;transition:transform .2s,box-shadow .2s}
.kpi-card:hover{transform:translateY(-2px);box-shadow:var(--shadow-md)}
.kpi-card::after{content:'';position:absolute;top:0;left:0;right:0;height:3px;border-radius:var(--r) var(--r) 0 0}
.kpi-total::after{background:linear-gradient(90deg,#6366f1,#8b5cf6)}
.kpi-cyan::after{background:linear-gradient(90deg,#06b6d4,#0ea5e9)}
.kpi-amber::after{background:linear-gradient(90deg,#f59e0b,#fbbf24)}
.kpi-rose::after{background:linear-gradient(90deg,#f43f5e,#fb7185)}
.kpi-label{font-size:10px;font-weight:800;letter-spacing:.7px;text-transform:uppercase;margin-bottom:8px}
.kpi-total .kpi-label{color:var(--indigo)}.kpi-cyan .kpi-label{color:var(--cyan)}
.kpi-amber .kpi-label{color:var(--amber)}.kpi-rose .kpi-label{color:var(--rose)}
.kpi-value{font-size:2rem;font-weight:700;color:var(--text-h);font-family:'DM Mono',monospace;line-height:1.1}
.kpi-meta{margin-top:8px;display:flex;align-items:center;gap:6px}
.kpi-pct{font-size:10.5px;font-weight:700;padding:2px 7px;border-radius:20px}
.kpi-total .kpi-pct{background:var(--indigo-soft);color:var(--indigo)}.kpi-cyan .kpi-pct{background:var(--cyan-soft);color:var(--cyan)}
.kpi-amber .kpi-pct{background:var(--amber-soft);color:var(--amber)}.kpi-rose .kpi-pct{background:var(--rose-soft);color:var(--rose)}
.kpi-meta-label{font-size:11px;color:var(--text-m)}
.chart-grid{display:grid;grid-template-columns:320px 1fr;gap:16px;margin-bottom:24px}
.chart-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:22px;box-shadow:var(--shadow-sm)}
.chart-title{font-size:13px;font-weight:700;color:var(--text-h);margin-bottom:3px}
.chart-subtitle{font-size:11px;color:var(--text-m);margin-bottom:16px}
.donut-wrap{display:flex;flex-direction:column;align-items:center}
.donut-center{position:absolute;top:50%;left:50%;transform:translate(-50%,-60%);text-align:center;pointer-events:none}
.donut-center-val{font-size:1.4rem;font-weight:700;color:var(--text-h);font-family:'DM Mono',monospace}
.donut-center-lbl{font-size:9.5px;color:var(--text-m);font-weight:600}
.legend-pills{display:flex;flex-wrap:wrap;gap:6px;margin-top:14px;justify-content:center}
.legend-pill{display:flex;align-items:center;gap:5px;background:var(--surface2);border:1px solid var(--border);border-radius:20px;padding:3px 9px;font-size:10.5px;font-weight:500;color:var(--text-b)}
.legend-dot{width:7px;height:7px;border-radius:50%;flex-shrink:0}
.tabs-container{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);box-shadow:var(--shadow-sm);overflow:hidden}
.tabs-nav{display:flex;border-bottom:1px solid var(--border);background:var(--surface2);padding:0 6px;overflow-x:auto;scrollbar-width:none}
.tab-btn{padding:12px 16px;border:none;background:transparent;font-family:inherit;font-size:12.5px;font-weight:500;color:var(--text-m);cursor:pointer;position:relative;transition:color .15s;white-space:nowrap}
.tab-btn::after{content:'';position:absolute;bottom:0;left:0;right:0;height:2px;background:var(--indigo);border-radius:2px 2px 0 0;transform:scaleX(0);transition:transform .2s}
.tab-btn.active{color:var(--text-h);font-weight:700}.tab-btn.active::after{transform:scaleX(1)}
.tab-count{background:var(--border);color:var(--text-m);border-radius:20px;padding:1px 6px;font-size:10px;margin-left:5px;font-weight:700}
.tab-btn.active .tab-count{background:var(--indigo-soft);color:var(--indigo)}
.tab-pane{display:none}.tab-pane.active{display:block}
.table-wrap{overflow-x:auto}
table.matrix,table.summary{width:100%;border-collapse:collapse;font-size:12.5px}
table.matrix thead tr,table.summary thead tr{background:var(--surface2)}
table.matrix th,table.summary th{padding:11px 14px;font-size:10px;font-weight:800;letter-spacing:.6px;text-transform:uppercase;color:var(--text-m);border-bottom:2px solid var(--border);white-space:nowrap;text-align:left}
.num{text-align:right}
table.matrix td,table.summary td{padding:10px 14px;border-bottom:1px solid var(--border);color:var(--text-b);vertical-align:middle}
table.matrix tbody tr:hover,table.summary tbody tr:hover:not(.total-row){background:var(--surface2)}
table.matrix tbody tr:last-child td,table.summary tr.total-row td{font-weight:700;color:var(--text-h);background:var(--surface2);border-top:2px solid var(--border);border-bottom:none}
.owner-tag{padding:2px 9px;border-radius:20px;font-size:11px;font-weight:700}
.tag-b2b{background:var(--cyan-soft);color:var(--cyan)}.tag-exp{background:var(--amber-soft);color:#b45309}
.tag-plan{background:var(--rose-soft);color:var(--rose)}.tag-default{background:var(--surface2);color:var(--text-m)}
.task-name{font-weight:600;color:var(--text-h);font-size:13px}
.count-cell{display:flex;align-items:center;gap:10px;min-width:190px}
.count-num{font-family:'DM Mono',monospace;font-weight:700;font-size:13px;color:var(--text-h);min-width:65px}
.bar-track{flex:1;height:6px;background:var(--border);border-radius:10px}
.bar-fill{height:100%;border-radius:10px}
.pct-label{font-size:10.5px;color:var(--text-s);min-width:36px;text-align:right}
.remarks-text{font-size:11.5px;color:var(--text-m);max-width:380px;word-break:break-word;line-height:1.5}
.status-badge{display:inline-flex;align-items:center;gap:4px;padding:3px 10px;border-radius:20px;font-size:10.5px;font-weight:700}
.badge-resolved{background:var(--green-soft);color:var(--rag-green);border:1px solid var(--rag-green-border)}
.badge-pending{background:#fef9c3;color:#a16207;border:1px solid #fde68a}
.badge-empty{background:transparent;color:transparent;border:none}
.resolved-stack{display:flex;flex-direction:column;align-items:center;gap:3px}
.resolved-sub{font-size:10px;color:var(--rag-green);font-weight:700}
.badge-dot{width:5px;height:5px;border-radius:50%}.dot-green{background:var(--rag-green)}.dot-amber{background:var(--amber)}
.filter-bar{display:flex;justify-content:space-between;align-items:center;padding:14px 18px;background:var(--surface);border-bottom:1px solid var(--border)}
.filter-controls-card{display:flex;gap:20px;align-items:center;background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:14px 18px;margin-bottom:20px;box-shadow:var(--shadow-sm);flex-wrap:wrap}
.filter-label{font-size:11px;font-weight:700;color:var(--text-m);margin-right:6px;text-transform:uppercase;letter-spacing:.4px}
.filter-select{padding:7px 28px 7px 10px;border:1px solid var(--border);border-radius:7px;font-family:inherit;font-size:12.5px;font-weight:600;color:var(--text-h);background-color:var(--surface2);outline:none;cursor:pointer;appearance:none;background-image:url("data:image/svg+xml;charset=US-ASCII,%3Csvg xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22 width%3D%22292.4%22 height%3D%22292.4%22%3E%3Cpath fill%3D%22%2364748b%22 d%3D%22M287 69.4a17.6 17.6 0 0 0-13-5.4H18.4c-5 0-9.3 1.8-12.9 5.4A17.6 17.6 0 0 0 0 82.2c0 5 1.8 9.3 5.4 12.9l128 127.9c3.6 3.6 7.8 5.4 12.8 5.4s9.2-1.8 12.8-5.4L287 95c3.5-3.5 5.4-7.8 5.4-12.8 0-5-1.9-9.2-5.5-12.8z%22%2F%3E%3C%2Fsvg%3E");background-repeat:no-repeat;background-position:right 10px top 50%;background-size:9px auto;transition:border-color .15s}
.filter-select:focus{border-color:var(--indigo)}
.btn-action{display:flex;align-items:center;gap:5px;padding:7px 14px;border:1px solid var(--border);border-radius:7px;font-family:inherit;font-size:12px;font-weight:700;cursor:pointer;background:var(--surface);transition:all .18s;color:var(--text-b)}
.btn-action.excel{color:#10b981}.btn-action.excel:hover{background:var(--green-soft);border-color:#10b981}
.btn-action.raw:hover{background:var(--surface2)}
.grid-kpi-banner{background:linear-gradient(135deg,#0f2942,#175E7B);color:#fff;text-align:center;padding:22px;border-radius:var(--r);box-shadow:var(--shadow-sm)}
.grid-kpi-banner .title{font-size:11px;font-weight:700;letter-spacing:.5px;text-transform:uppercase;color:#94a3b8;margin-bottom:6px}
.grid-kpi-banner .val{font-size:2rem;font-weight:700;font-family:'DM Mono',monospace}
.earth-link{color:var(--cyan);margin-left:6px;display:inline-flex;align-items:center;transition:transform .2s}
.earth-link:hover{color:#0284c7;transform:scale(1.15)}
.rag-legend{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:16px}
.rag-legend-item{display:flex;align-items:center;gap:7px;font-size:12px;font-weight:600;color:var(--text-b)}
.rag-dot{width:12px;height:12px;border-radius:50%}
.rag-dot.green{background:var(--rag-green)}.rag-dot.amber{background:var(--rag-amber)}.rag-dot.red{background:var(--rag-red)}
.rag-summary-row{display:flex;gap:12px;margin-bottom:16px;flex-wrap:wrap}
.rag-summary-pill{display:flex;align-items:center;gap:8px;padding:8px 16px;border-radius:30px;font-size:12.5px;font-weight:700;border:1.5px solid}
.rag-summary-pill.green{background:var(--rag-green-bg);color:var(--rag-green);border-color:var(--rag-green-border)}
.rag-summary-pill.amber{background:var(--rag-amber-bg);color:var(--rag-amber);border-color:var(--rag-amber-border)}
.rag-summary-pill.red{background:var(--rag-red-bg);color:var(--rag-red);border-color:var(--rag-red-border)}
.rag-summary-num{font-size:1.2rem;font-family:'DM Mono',monospace}
.scorecard-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px;padding:4px 0 8px}
.sc-card{background:var(--surface);border:1.5px solid var(--border);border-radius:12px;padding:16px 18px;box-shadow:var(--shadow-sm);position:relative;overflow:hidden;transition:transform .2s,box-shadow .2s}
.sc-card:hover{transform:translateY(-2px);box-shadow:var(--shadow-md)}
.sc-card::before{content:'';position:absolute;top:0;left:0;bottom:0;width:4px;border-radius:12px 0 0 12px}
.sc-card.green::before{background:var(--rag-green)}.sc-card.amber::before{background:var(--rag-amber)}.sc-card.red::before{background:var(--rag-red)}
.sc-card-header{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px}
.sc-district{font-size:14px;font-weight:700;color:var(--text-h)}
.sc-circle-tag{font-size:10px;font-weight:700;padding:2px 8px;border-radius:20px;background:var(--indigo-soft);color:var(--indigo)}
.sc-rag-badge{display:inline-flex;align-items:center;gap:4px;padding:4px 10px;border-radius:20px;font-size:11px;font-weight:800;text-transform:uppercase;letter-spacing:.4px}
.sc-rag-badge.green{background:var(--rag-green-bg);color:var(--rag-green);border:1px solid var(--rag-green-border)}
.sc-rag-badge.amber{background:var(--rag-amber-bg);color:var(--rag-amber);border:1px solid var(--rag-amber-border)}
.sc-rag-badge.red{background:var(--rag-red-bg);color:var(--rag-red);border:1px solid var(--rag-red-border)}
.sc-metrics{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:12px}
.sc-metric{background:var(--surface2);border-radius:8px;padding:8px 10px}
.sc-metric-label{font-size:9.5px;font-weight:700;letter-spacing:.5px;text-transform:uppercase;color:var(--text-s);margin-bottom:2px}
.sc-metric-value{font-size:1.1rem;font-weight:700;font-family:'DM Mono',monospace;color:var(--text-h)}
.sc-metric-value.green{color:var(--rag-green)}.sc-metric-value.amber{color:var(--rag-amber)}.sc-metric-value.red{color:var(--rag-red)}
.sc-team-bars{display:flex;flex-direction:column;gap:5px}
.sc-team-row{display:flex;align-items:center;gap:8px;font-size:10.5px}
.sc-team-label{width:64px;color:var(--text-m);font-weight:600;flex-shrink:0}
.sc-team-track{flex:1;height:5px;background:var(--border);border-radius:6px;overflow:hidden}
.sc-team-fill{height:100%;border-radius:6px}
.sc-team-count{width:44px;text-align:right;font-family:'DM Mono',monospace;font-size:10.5px;font-weight:700;color:var(--text-b)}

.sprint-stage-headers{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:0}
.sprint-col-header{border-radius:12px 12px 0 0;padding:14px 20px;color:#fff;display:flex;justify-content:space-between;align-items:center}
.sprint-col-header-title{font-size:12px;font-weight:800;text-transform:uppercase;letter-spacing:.8px}
.sprint-col-header-badge{font-family:'DM Mono',monospace;font-size:13px;font-weight:700;background:rgba(255,255,255,.25);padding:3px 10px;border-radius:20px}

.sprint-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
.sprint-cell{background:var(--surface);border:1px solid var(--border);padding:0;border-top:none}
.sprint-cell:first-child{border-radius:0 0 0 12px}
.sprint-cell:last-child{border-radius:0 0 12px 0}

.sc-section-hdr{background:var(--surface2);border-bottom:1px solid var(--border);border-top:1px solid var(--border);padding:8px 16px;font-size:9.5px;font-weight:800;letter-spacing:.7px;text-transform:uppercase;color:var(--text-s)}
.sc-section-body{padding:12px 16px}

.sprint-kpi-block{padding:14px 16px;border-bottom:1px solid var(--border)}
.sprint-big-num{font-size:2.2rem;font-weight:700;font-family:'DM Mono',monospace;color:var(--text-h);line-height:1}
.sprint-scope-pct{font-size:11px;color:var(--text-m);font-weight:600;margin:4px 0 10px}
.sprint-circle-chips{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:10px}
.sprint-circle-chip{font-size:9.5px;font-weight:700;padding:2px 7px;border-radius:20px;background:var(--indigo-soft);color:var(--indigo)}
.sprint-team-bars{display:flex;flex-direction:column;gap:4px}
.sprint-team-row{display:flex;align-items:center;gap:8px;font-size:10.5px}
.sprint-team-lbl{width:60px;color:var(--text-m);font-weight:600;flex-shrink:0}
.sprint-team-track{flex:1;height:5px;background:var(--border);border-radius:6px;overflow:hidden}
.sprint-team-fill{height:100%;border-radius:6px}
.sprint-team-fill.b2b{background:var(--cyan)}.sprint-team-fill.exp{background:var(--amber)}.sprint-team-fill.plan{background:var(--rose)}
.sprint-team-count{width:52px;text-align:right;font-family:'DM Mono',monospace;font-size:10.5px;font-weight:700}

.sprint-summary-row{display:flex;justify-content:space-between;align-items:baseline;padding:5px 0;border-bottom:1px solid var(--border);font-size:11.5px}
.sprint-summary-row:last-child{border-bottom:none}
.sprint-summary-label{color:var(--text-m);font-weight:600}
.sprint-summary-val{font-family:'DM Mono',monospace;font-weight:700;color:var(--text-h)}
.sprint-summary-val.green{color:var(--rag-green)}
.sprint-summary-sub{font-size:10px;color:var(--text-s);margin-top:4px}
.sprint-circle-breakdown{display:flex;flex-wrap:wrap;gap:3px;margin-top:5px}
.sprint-circle-val{font-size:9.5px;background:var(--surface2);border:1px solid var(--border);border-radius:4px;padding:1px 5px;color:var(--text-m)}

.sprint-activity-row{display:flex;justify-content:space-between;align-items:center;padding:5px 0;border-bottom:1px dashed var(--border);font-size:11.5px;gap:8px}
.sprint-activity-row:last-child{border-bottom:none}
.sprint-activity-name{flex:1;color:var(--text-b);font-weight:500;line-height:1.4}
.sprint-activity-count{font-family:'DM Mono',monospace;font-weight:800;font-size:12px;flex-shrink:0}
.sprint-activity-bar{height:3px;background:var(--border);border-radius:3px;margin-top:3px;overflow:hidden}
.sprint-activity-fill{height:100%;border-radius:3px}
.sprint-total-row{display:flex;justify-content:space-between;align-items:center;padding:7px 0;font-size:12px;font-weight:800;color:var(--text-h);border-top:2px solid var(--border);margin-top:4px}
.sprint-dl-btn{display:inline-flex;align-items:center;gap:4px;padding:3px 10px;background:var(--indigo-soft);color:var(--indigo);border:1px solid #c7d2fe;border-radius:20px;font-size:10px;font-weight:700;cursor:pointer;text-decoration:none;transition:all .15s}
.sprint-dl-btn:hover{background:var(--indigo);color:#fff}

.site-status-chip{display:inline-flex;align-items:center;gap:3px;padding:2px 8px;border-radius:20px;font-size:10px;font-weight:700;margin:2px}
.chip-live{background:#dcfce7;color:#16a34a}.chip-planned{background:#dbeafe;color:#1d4ed8}
.chip-onair{background:#fef9c3;color:#a16207}.chip-other{background:var(--surface2);color:var(--text-m)}
.sprint-sites-banner{display:flex;align-items:center;gap:8px;background:var(--indigo-soft);border-radius:8px;padding:8px 12px;margin-bottom:8px}
.sprint-sites-count{font-size:1.1rem;font-weight:800;font-family:'DM Mono',monospace;color:var(--indigo)}

.action-plan-row{padding:6px 0;border-bottom:1px dashed var(--border);font-size:11.5px}
.action-plan-row:last-child{border-bottom:none}
.action-plan-text{color:var(--text-b);line-height:1.5}
.action-plan-count{font-family:'DM Mono',monospace;font-weight:700;color:var(--text-h);float:right;margin-left:8px}

@media(max-width:1024px){.kpi-grid{grid-template-columns:repeat(2,1fr)}.chart-grid{grid-template-columns:1fr}.kanban-board{grid-template-columns:repeat(2,1fr)}}
@media(max-width:640px){.kpi-grid{grid-template-columns:1fr}.kanban-board{grid-template-columns:1fr}.main{padding:16px}}

.edit-fab{position:fixed;bottom:28px;right:28px;width:52px;height:52px;border-radius:50%;background:linear-gradient(135deg,#6366f1,#8b5cf6);color:#fff;border:none;font-size:22px;cursor:pointer;box-shadow:0 4px 16px rgba(99,102,241,.5);z-index:300;transition:transform .2s}
.edit-fab:hover{transform:scale(1.1)}
.edit-drawer{position:fixed;bottom:92px;right:28px;width:360px;background:var(--surface);border:1px solid var(--border);border-radius:16px;box-shadow:0 20px 48px rgba(15,23,42,.18);z-index:300;padding:22px;display:none;animation:slideUp .25s ease}
.edit-drawer.open{display:block}
@keyframes slideUp{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}
.edit-drawer-title{font-size:13px;font-weight:800;color:var(--text-h);margin-bottom:14px;display:flex;justify-content:space-between;align-items:center}
.edit-drawer-close{background:none;border:none;font-size:18px;cursor:pointer;color:var(--text-m)}
.edit-field-group{margin-bottom:12px}
.edit-field-label{font-size:10.5px;font-weight:700;letter-spacing:.4px;text-transform:uppercase;color:var(--text-m);margin-bottom:4px}
.edit-input{width:100%;padding:8px 10px;border:1px solid var(--border);border-radius:8px;font-family:inherit;font-size:13px;color:var(--text-h);background:var(--surface2);outline:none;transition:border-color .15s}
.edit-input:focus{border-color:var(--indigo);box-shadow:0 0 0 2px var(--indigo-soft)}
.edit-select{width:100%;padding:8px 10px;border:1px solid var(--border);border-radius:8px;font-family:inherit;font-size:13px;color:var(--text-h);background:var(--surface2);outline:none;appearance:none;cursor:pointer}
.edit-save-btn{width:100%;padding:10px;background:linear-gradient(135deg,#6366f1,#8b5cf6);color:#fff;border:none;border-radius:8px;font-family:inherit;font-size:13px;font-weight:700;cursor:pointer;margin-top:4px;transition:opacity .2s}
.edit-save-btn:hover{opacity:.88}.edit-save-btn:disabled{opacity:.5;cursor:not-allowed}
.edit-status{font-size:12px;margin-top:8px;text-align:center;min-height:18px;font-weight:600}
.edit-status.ok{color:var(--rag-green)}.edit-status.err{color:var(--rag-red)}
.edit-warning{font-size:10.5px;color:var(--rag-amber);background:var(--rag-amber-bg);border:1px solid var(--rag-amber-border);border-radius:7px;padding:6px 10px;margin-bottom:12px;line-height:1.5}
</style>
</head>
<body>
<div class="shell">
<header class="topbar">
  <div class="topbar-brand">
    <div class="topbar-logo">&#128225;</div>
    <div><div class="topbar-title">Smart Meter Operations</div><div class="topbar-sub">CXO Executive Dashboard v3</div></div>
  </div>
  <div class="topbar-right">
    <nav class="nav-pills">
      <button class="nav-pill active"  onclick="switchPage(0,this)">Summary</button>
      <button class="nav-pill"         onclick="switchPage(1,this)">Grid Analytics</button>
      <button class="nav-pill"         onclick="switchPage(2,this)">District Health</button>
      <button class="nav-pill"         onclick="switchPage(3,this)">Sprint Board</button>
    </nav>
    <span class="live-badge"><span class="live-dot"></span>LIVE</span>
    <span class="topbar-ts" id="ts-label"></span>
    <button class="dm-btn" id="dm-btn" title="Toggle dark mode" onclick="toggleDark()">&#127769;</button>
  </div>
</header>

<main class="main" id="page-summary">
  <div class="section-label">Operations KPIs</div>
  <div class="kpi-grid">
    <div class="kpi-card kpi-total"><div class="kpi-label">Total Non-Communicating</div>
      <div class="kpi-value" id="kpi-total">{{ c.kpis.total }}</div>
      <div class="kpi-meta"><span class="kpi-pct">100%</span><span class="kpi-meta-label">Full recovery scope</span></div></div>
    <div class="kpi-card kpi-cyan"><div class="kpi-label">B2B Actionable</div>
      <div class="kpi-value" id="kpi-b2b">{{ c.kpis.b2b }}</div>
      <div class="kpi-meta"><span class="kpi-pct">{{ c.kpis.b2b_pct }}%</span><span class="kpi-meta-label">Field team</span></div></div>
    <div class="kpi-card kpi-amber"><div class="kpi-label">Experience Targets</div>
      <div class="kpi-value" id="kpi-exp">{{ c.kpis.exp }}</div>
      <div class="kpi-meta"><span class="kpi-pct">{{ c.kpis.exp_pct }}%</span><span class="kpi-meta-label">RF optimisation</span></div></div>
    <div class="kpi-card kpi-rose"><div class="kpi-label">Planning &#8212; New Sites</div>
      <div class="kpi-value" id="kpi-plan">{{ c.kpis.plan }}</div>
      <div class="kpi-meta"><span class="kpi-pct">{{ c.kpis.plan_pct }}%</span><span class="kpi-meta-label">Site pipeline</span></div></div>
  </div>
  <div class="section-label">Distribution Analysis</div>
  <div class="chart-grid">
    <div class="chart-card">
      <div class="chart-title">Ownership Split</div><div class="chart-subtitle">Non-communicating meters by team</div>
      <div class="donut-wrap" style="height:230px;position:relative">
        <canvas id="donutChart"></canvas>
        <div class="donut-center"><div class="donut-center-val" id="donut-center-val">&#8212;</div><div class="donut-center-lbl">Total</div></div>
      </div>
      <div class="legend-pills" id="donut-legend"></div>
    </div>
    <div class="chart-card">
      <div class="chart-title">Circle-wise Breakdown</div><div class="chart-subtitle">B2B &middot; Experience &middot; Planning per circle</div>
      <div style="height:250px;position:relative"><canvas id="barChart"></canvas></div>
    </div>
  </div>
  <div class="section-label">Detailed Tracking</div>
  <div class="tabs-container">
    <div class="tabs-nav">
      <button class="tab-btn active" onclick="showTab(0,this)">Overview Matrix</button>
      <button class="tab-btn" onclick="showTab(1,this)">B2B <span class="tab-count">{{ c.kpis.b2b }}</span></button>
      <button class="tab-btn" onclick="showTab(2,this)">Experience <span class="tab-count">{{ c.kpis.exp }}</span></button>
      <button class="tab-btn" onclick="showTab(3,this)">Planning <span class="tab-count">{{ c.kpis.plan }}</span></button>
    </div>
    <div class="tab-pane active" id="pane-0">
      <div class="filter-bar">
        <div style="display:flex;gap:14px;flex-wrap:wrap">
          <div style="display:flex;align-items:center"><span class="filter-label">Team:</span>
            <select id="filter-team" class="filter-select" onchange="renderMatrix()">
              <option value="All">All Teams</option><option>B2B</option><option>Experience</option><option>Planning</option>
            </select></div>
          <div style="display:flex;align-items:center"><span class="filter-label">Circle:</span>
            <select id="filter-circle" class="filter-select" onchange="renderMatrix()">
              <option value="All">All Circles</option><option>AP</option><option>BIH</option><option>MAH</option><option>MUM</option><option>NESA</option>
            </select></div>
        </div>
        <button class="btn-action excel" onclick="exportMatrixExcel()">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline></svg>
          Export Excel
        </button>
      </div>
      <div class="table-wrap">
        <table class="matrix"><thead><tr>
          <th style="width:18%">Final Ownership</th><th style="width:52%">Task Category</th>
          <th class="num" style="width:15%">Meter Count</th><th class="num" style="width:15%">Resolved</th>
        </tr></thead><tbody id="matrix-tbody"></tbody></table>
      </div>
    </div>
    <div class="tab-pane" id="pane-1">{{ _team_table(t.tab1, 'B2B', '#06b6d4') }}</div>
    <div class="tab-pane" id="pane-2">{{ _team_table(t.tab2, 'Experience', '#f59e0b') }}</div>
    <div class="tab-pane" id="pane-3">{{ _team_table(t.tab3, 'Planning', '#f43f5e') }}</div>
  </div>
</main>

<main class="main" id="page-grid-analytics" style="display:none">
  <div class="section-label">Grid Performance Analytics</div>
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:20px">
    <div class="grid-kpi-banner"><div class="title">Total Grids</div><div class="val" id="grid-kpi-total">0</div></div>
    <div class="grid-kpi-banner"><div class="title">Total Meters</div><div class="val" id="grid-kpi-meters">0</div></div>
    <div class="grid-kpi-banner"><div class="title">Total Resolved</div><div class="val" id="grid-kpi-resolved">0</div></div>
  </div>
  <div class="filter-controls-card">
    <span class="filter-label" style="font-size:13px;color:var(--text-h)">Filters:</span>
    <div style="display:flex;align-items:center"><span class="filter-label">Circle:</span>
      <select id="filter-grid-circle" class="filter-select" onchange="renderGridAnalytics()">
        <option value="All">All</option><option>AP</option><option>BIH</option><option>MAH</option><option>MUM</option><option>NESA</option>
      </select></div>
    <div style="display:flex;align-items:center"><span class="filter-label">Bucket:</span>
      <select id="filter-grid-bucket" class="filter-select" onchange="renderGridAnalytics()">
        <option value="All">All</option><option>&gt;250 Devices</option><option>100-250 Devices</option>
        <option>50-100 Devices</option><option>25-50 Devices</option><option>10-25 Devices</option><option>&lt;10 Devices</option>
      </select></div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:20px">
    <div class="chart-card"><div class="chart-title">Top 10 Grids</div><div class="chart-subtitle">Highest density grids</div><div style="height:240px;position:relative"><canvas id="gridBarChart"></canvas></div></div>
    <div class="chart-card"><div class="chart-title">RSRP Benchmark</div><div class="chart-subtitle">Avg signal vs &#8722;105 dBm threshold</div><div style="height:240px;position:relative"><canvas id="rsrpLineChart"></canvas></div></div>
  </div>
  <div class="tabs-container">
    <div class="filter-bar">
      <span style="font-size:13px;font-weight:700;color:var(--text-h)">Grid Performance Data</span>
      <button class="btn-action raw" onclick="exportGridRawData()">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
        Export Raw CSV
      </button>
    </div>
    <div class="table-wrap" style="max-height:480px;overflow-y:auto">
      <table class="matrix"><thead style="position:sticky;top:0;z-index:10"><tr>
        <th>Grid ID</th><th class="num">Meters</th><th class="num">Avg RSRP</th><th class="num">Resolved</th>
      </tr></thead><tbody id="grid-tbody"></tbody></table>
    </div>
  </div>
</main>

<main class="main" id="page-scorecard" style="display:none">
  <div class="section-label">District Health Scorecard</div>
  <p style="font-size:12.5px;color:var(--text-m);margin-bottom:16px;line-height:1.7">
    RAG status per district based on two signals:
    <strong>Resolved %</strong> (&#127881; &#8805;30% &middot; &#128993; 10&#8211;29% &middot; &#128308; &lt;10%) and
    <strong>Avg RSRP</strong> (&#127881; &#8805;&#8722;100 dBm &middot; &#128993; &#8722;105 to &#8722;100 &middot; &#128308; &lt;&#8722;105 dBm).
    Worst signal determines overall badge. Filter by circle to drill down.
  </p>
  <div class="rag-legend">
    <div class="rag-legend-item"><div class="rag-dot green"></div>Green &#8212; On Track</div>
    <div class="rag-legend-item"><div class="rag-dot amber"></div>Amber &#8212; Needs Attention</div>
    <div class="rag-legend-item"><div class="rag-dot red"></div>Red &#8212; Critical</div>
  </div>
  <div class="rag-summary-row">
    <div class="rag-summary-pill green"><span class="rag-summary-num" id="sc-green-count">0</span>&nbsp;Green</div>
    <div class="rag-summary-pill amber"><span class="rag-summary-num" id="sc-amber-count">0</span>&nbsp;Amber</div>
    <div class="rag-summary-pill red">  <span class="rag-summary-num" id="sc-red-count">0</span>&nbsp;Red</div>
  </div>
  <div class="filter-controls-card" style="margin-top:14px">
    <span class="filter-label" style="font-size:13px;color:var(--text-h)">Filters:</span>
    <div style="display:flex;align-items:center"><span class="filter-label">Circle:</span>
      <select id="filter-sc-circle" class="filter-select" onchange="renderScorecard()"><option value="All">All Circles</option></select></div>
    <div style="display:flex;align-items:center"><span class="filter-label">RAG:</span>
      <select id="filter-sc-rag" class="filter-select" onchange="renderScorecard()">
        <option value="All">All</option><option value="red">Red Only</option>
        <option value="amber">Amber Only</option><option value="green">Green Only</option>
      </select></div>
    <span style="margin-left:auto;font-size:11.5px;color:var(--text-m)" id="sc-showing-label"></span>
  </div>
  <div class="scorecard-grid" id="scorecard-grid"></div>
</main>

<main class="main" id="page-kanban" style="display:none">
  <div class="section-label">Agile Sprint Board</div>
  
  <div class="filter-controls-card" style="padding:10px 18px; margin-bottom:16px; justify-content:space-between;">
    <div style="display:flex; gap:20px; align-items:center;">
      <span class="filter-label" style="font-size:13px;color:var(--text-h)">Filters:</span>
      <div style="display:flex;align-items:center"><span class="filter-label">Circle:</span>
        <select id="filter-kanban-circle" class="filter-select" onchange="renderKanban()">
          <option value="All">All Circles</option>
        </select>
      </div>
    </div>
    <!-- NEW EXPORT IMAGE BUTTON -->
    <button class="btn-action" style="color:#0284c7;border-color:#e0f2fe;background:#f0f9ff;" onclick="downloadSummaryImage(event)">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>
      Download Image for Email
    </button>
  </div>

  <!-- WRAPPER FOR EXPORT -->
  <div id="kanban-export-area" style="padding: 10px; margin: -10px; border-radius: 10px;">
    <div class="sprint-stage-headers" id="sprint-stage-headers"></div>
    <div class="sprint-grid" id="sprint-grid"></div>
  </div>

  <div class="section-label" style="margin-top:28px">Stage Distribution</div>
  <div class="chart-card" style="margin-bottom:24px">
    <div class="chart-title">Meters per Sprint Stage — Team Split</div>
    <div class="chart-subtitle">B2B / Experience / Planning breakdown by stage</div>
    <div style="height:260px;position:relative"><canvas id="kanbanChart"></canvas></div>
  </div>
</main>
</div>

<script>
"use strict";
window.onerror=function(msg,src,line,col,err){
  console.error("JS Error:",msg,"line",line,err);
  return false; 
};
try{document.getElementById("ts-label").textContent="Updated: "+new Date().toLocaleString("en-IN",{dateStyle:"medium",timeStyle:"short"});}catch(e){}

function toggleDark(){
  var html=document.documentElement;
  var dark=html.getAttribute("data-theme")==="dark";
  html.setAttribute("data-theme",dark?"light":"dark");
  document.getElementById("dm-btn").textContent=dark?"🌙":"☀️";
}

var PAGE_IDS=["page-summary","page-grid-analytics","page-scorecard","page-kanban"];
function switchPage(idx,btn){
  PAGE_IDS.forEach(function(id){var el=document.getElementById(id);if(el)el.style.display="none";});
  var t=document.getElementById(PAGE_IDS[idx]);if(t)t.style.display="block";
  document.querySelectorAll(".nav-pill").forEach(function(b){b.classList.remove("active");});
  if(btn)btn.classList.add("active");
  if(idx===1)renderGridAnalytics();
  if(idx===2)renderScorecard();
  if(idx===3)renderKanban();
}
function showTab(idx,btn){
  document.querySelectorAll(".tab-pane").forEach(function(p,i){p.classList.toggle("active",i===idx);});
  document.querySelectorAll(".tab-btn").forEach(function(b){b.classList.remove("active");});
  if(btn)btn.classList.add("active");
}

function animateCounter(el,target,dur){
  dur=dur||1400;
  var start=performance.now();
  function fmt(n){return Math.round(n).toLocaleString("en-IN");}
  function tick(now){
    var p=Math.min((now-start)/dur,1);
    var ease=1-Math.pow(1-p,4);
    el.textContent=fmt(ease*target);
    if(p<1)requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}
window.addEventListener("DOMContentLoaded",function(){
  animateCounter(document.getElementById("kpi-total"), {{ c.total_raw }});
  animateCounter(document.getElementById("kpi-b2b"),   {{ c.b2b_raw }});
  animateCounter(document.getElementById("kpi-exp"),   {{ c.exp_raw }});
  animateCounter(document.getElementById("kpi-plan"),  {{ c.plan_raw }});
});

Chart.defaults.font.family="'DM Sans',system-ui,sans-serif";
Chart.defaults.color="#64748b";
var PAL={cyan:"#06b6d4",amber:"#f59e0b",rose:"#f43f5e",indigo:"#6366f1"};

document.addEventListener("DOMContentLoaded",function(){
  try{
    var labels={{ c.owner_labels|safe }};
    var data={{ c.owner_data|safe }};
    var colors=[PAL.cyan,PAL.amber,PAL.rose,PAL.indigo,"#8b5cf6","#10b981","#ec4899"];
    var total=data.reduce(function(a,b){return a+b;},0);
    document.getElementById("donut-center-val").textContent=total>=1000?(total/1000).toFixed(0)+"K":total;
    var leg=document.getElementById("donut-legend");
    labels.forEach(function(l,i){
      var p=document.createElement("div");p.className="legend-pill";
      p.innerHTML='<span class="legend-dot" style="background:'+colors[i%colors.length]+'"></span>'+l;
      leg.appendChild(p);
    });
    new Chart(document.getElementById("donutChart"),{
      type:"doughnut",
      data:{labels:labels,datasets:[{data:data,backgroundColor:colors.slice(0,labels.length),borderWidth:2,borderColor:"#fff",hoverOffset:6}]},
      options:{cutout:"68%",plugins:{legend:{display:false},tooltip:{callbacks:{
        label:function(ctx){return " "+ctx.label+": "+ctx.parsed.toLocaleString("en-IN")+" ("+((ctx.parsed/total)*100).toFixed(1)+"%)";}}}}
      }
    });
  }catch(e){console.error("Donut error:",e);}

  try{
    new Chart(document.getElementById("barChart"),{
      type:"bar",
      data:{labels:{{ c.circles|safe }},datasets:[
        {label:"B2B",data:{{ c.b2b|safe }},backgroundColor:PAL.cyan,borderRadius:4},
        {label:"Experience",data:{{ c.exp|safe }},backgroundColor:PAL.amber,borderRadius:4},
        {label:"Planning",data:{{ c.plan|safe }},backgroundColor:PAL.rose,borderRadius:4}
      ]},
      options:{indexAxis:"y",responsive:true,maintainAspectRatio:false,
        scales:{x:{stacked:true,grid:{color:"#f1f5f9"},ticks:{callback:function(v){return v>=1000?(v/1000).toFixed(0)+"K":v;}}},y:{stacked:true,grid:{display:false}}},
        plugins:{legend:{position:"top",labels:{usePointStyle:true,pointStyle:"circle",padding:16}},
          tooltip:{callbacks:{label:function(ctx){return " "+ctx.dataset.label+": "+ctx.parsed.x.toLocaleString("en-IN");}}}}}
    });
  }catch(e){console.error("Bar chart error:",e);}
});

var rawMatrixData={{ c.matrix_data|safe }};
function renderMatrix(){
  var tf=document.getElementById("filter-team").value;
  var cf=document.getElementById("filter-circle").value;
  var filtered=rawMatrixData.filter(function(d){
    var mt=tf==="All"||String(d["Final Ownership"]).toUpperCase().indexOf(tf.toUpperCase())>=0;
    var mc=cf==="All"||String(d["Circle"]).toUpperCase()===cf.toUpperCase();
    return mt&&mc;
  });
  var agg={};
  filtered.forEach(function(d){
    var k=d["Final Ownership"]+"|||"+d["Task Category"];
    if(!agg[k])agg[k]={owner:d["Final Ownership"],task:d["Task Category"],count:0,resolved:0};
    agg[k].count+=d["Meter_Count"];agg[k].resolved+=d["Resolved_Count"];
  });
  var arr=Object.values(agg).sort(function(a,b){return b.count-a.count;});
  var tbody=document.getElementById("matrix-tbody");tbody.innerHTML="";
  if(!arr.length){tbody.innerHTML='<tr><td colspan="4" style="text-align:center;padding:24px;color:var(--text-m)">No data.</td></tr>';return;}
  var tc=0,tr2=0;
  arr.forEach(function(row){
    tc+=row.count;tr2+=row.resolved;
    var tag=row.owner.toUpperCase().indexOf("B2B")>=0?"tag-b2b":row.owner.toUpperCase().indexOf("EXPERIENCE")>=0?"tag-exp":row.owner.toUpperCase().indexOf("PLANNING")>=0?"tag-plan":"tag-default";
    var t=document.createElement("tr");
    t.innerHTML='<td><span class="owner-tag '+tag+'">'+row.owner+'</span></td><td class="task-name">'+row.task+'</td>'
      +'<td class="num" style="font-family:\'DM Mono\',monospace">'+row.count.toLocaleString("en-IN")+'</td>'
      +'<td class="num" style="font-family:\'DM Mono\',monospace;color:var(--rag-green);font-weight:700">'+row.resolved.toLocaleString("en-IN")+'</td>';
    tbody.appendChild(t);
  });
  var tot=document.createElement("tr");
  tot.innerHTML='<td colspan="2" style="font-weight:700;color:var(--text-h)">GRAND TOTAL</td>'
    +'<td class="num" style="font-family:\'DM Mono\',monospace;font-weight:700">'+tc.toLocaleString("en-IN")+'</td>'
    +'<td class="num" style="font-family:\'DM Mono\',monospace;font-weight:700;color:var(--rag-green)">'+tr2.toLocaleString("en-IN")+'</td>';
  tbody.appendChild(tot);
}
document.addEventListener("DOMContentLoaded",renderMatrix);

var rawGridList={{ c.grid_data|safe }};
var rawGridData=rawGridList.map(function(r){return{Grid_ID:r[0],Meter_Count:r[1],Avg_RSRP:r[2],Resolved_Count:r[3],Circle:r[4],Bucket:r[5],Lat:r[6],Lng:r[7]};});
var gridBarInst=null,rsrpInst=null;
function renderGridAnalytics(){
  var bf=document.getElementById("filter-grid-bucket").value;
  var cf=document.getElementById("filter-grid-circle").value;
  var data=rawGridData.filter(function(d){
    var mb=bf==="All"||(String(d.Bucket).toLowerCase().replace(/\s|devices/g,""))===bf.toLowerCase().replace(/\s|devices/g,"");
    var mc=cf==="All"||String(d.Circle).toUpperCase()===cf.toUpperCase();
    return mb&&mc;
  }).sort(function(a,b){return b.Meter_Count-a.Meter_Count;});
  document.getElementById("grid-kpi-total").innerText=data.length.toLocaleString("en-IN");
  document.getElementById("grid-kpi-meters").innerText=data.reduce(function(s,d){return s+d.Meter_Count;},0).toLocaleString("en-IN");
  document.getElementById("grid-kpi-resolved").innerText=data.reduce(function(s,d){return s+d.Resolved_Count;},0).toLocaleString("en-IN");
  var tbody=document.getElementById("grid-tbody");tbody.innerHTML="";
  data.slice(0,100).forEach(function(row){
    var el=(row.Lat&&row.Lat!=="N/A")
      ?'<a href="/download_kml?lat='+row.Lat+'&lng='+row.Lng+'&grid_id='+row.Grid_ID+'" class="earth-link" title="Google Earth KML"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="2" y1="12" x2="22" y2="12"></line><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path></svg></a>':"";
    var tr=document.createElement("tr");
    tr.innerHTML='<td style="font-weight:700;display:flex;align-items:center">'+row.Grid_ID+el+"</td>"
      +'<td class="num" style="font-family:\'DM Mono\'">'+row.Meter_Count.toLocaleString("en-IN")+"</td>"
      +'<td class="num" style="font-family:\'DM Mono\'">'+row.Avg_RSRP+" dBm</td>"
      +'<td class="num" style="font-family:\'DM Mono\';color:var(--rag-green);font-weight:700">'+row.Resolved_Count.toLocaleString("en-IN")+"</td>";
    tbody.appendChild(tr);
  });
  if(data.length>100){var info=document.createElement("tr");info.innerHTML='<td colspan="4" style="text-align:center;padding:12px;color:var(--text-m);font-weight:600">Top 100 shown. Export CSV for full data.</td>';tbody.appendChild(info);}
  var top10=data.slice(0,10);
  if(gridBarInst)gridBarInst.destroy();
  gridBarInst=new Chart(document.getElementById("gridBarChart"),{
    type:"bar",data:{labels:top10.map(function(d){return d.Grid_ID;}),datasets:[{label:"Meters",data:top10.map(function(d){return d.Meter_Count;}),backgroundColor:"#0f172a",borderRadius:4}]},
    options:{maintainAspectRatio:false,scales:{y:{beginAtZero:true,grid:{color:"#f1f5f9"}},x:{grid:{display:false}}},plugins:{legend:{display:false}}}
  });
  if(rsrpInst)rsrpInst.destroy();
  rsrpInst=new Chart(document.getElementById("rsrpLineChart"),{
    type:"line",data:{labels:top10.map(function(d){return d.Grid_ID;}),datasets:[
      {label:"Avg RSRP",data:top10.map(function(d){return d.Avg_RSRP;}),borderColor:"#0ea5e9",fill:false,tension:.3,pointBackgroundColor:"#0ea5e9",pointBorderColor:"#fff",pointBorderWidth:2},
      {label:"Threshold (-105)",data:top10.map(function(){return -105;}),borderColor:"#f97316",borderWidth:2,pointRadius:0,fill:false,borderDash:[6,3]}
    ]},
    options:{maintainAspectRatio:false,scales:{y:{reverse:true,grid:{color:"#f1f5f9"}},x:{grid:{display:false}}},plugins:{legend:{position:"top",labels:{usePointStyle:true,padding:14}}}}
  });
}
function exportGridRawData(){
  var c=document.getElementById("filter-grid-circle").value;
  var b=document.getElementById("filter-grid-bucket").value;
  window.location.href="/export_grid_details?circle="+encodeURIComponent(c)+"&bucket="+encodeURIComponent(b);
}

var rawScorecard={{ c.scorecard_data|safe }};
var circlesList={{ c.circles_list|safe }};
(function(){
  var sel=document.getElementById("filter-sc-circle");
  circlesList.forEach(function(c){var o=document.createElement("option");o.value=c;o.textContent=c;sel.appendChild(o);});
})();
function renderScorecard(){
  var cf=document.getElementById("filter-sc-circle").value;
  var rf=document.getElementById("filter-sc-rag").value;
  var circData=rawScorecard.filter(function(d){return cf==="All"||String(d.Circle).toUpperCase()===cf.toUpperCase();});
  document.getElementById("sc-green-count").textContent=circData.filter(function(d){return d.RAG_Overall==="green";}).length;
  document.getElementById("sc-amber-count").textContent=circData.filter(function(d){return d.RAG_Overall==="amber";}).length;
  document.getElementById("sc-red-count").textContent=circData.filter(function(d){return d.RAG_Overall==="red";}).length;
  var data=circData.filter(function(d){return rf==="All"||d.RAG_Overall===rf;});
  document.getElementById("sc-showing-label").textContent="Showing "+data.length+" of "+circData.length+" districts";
  var grid=document.getElementById("scorecard-grid");grid.innerHTML="";
  if(!data.length){grid.innerHTML='<p style="padding:32px;color:var(--text-m);grid-column:1/-1">No districts match filters.</p>';return;}
  var ragLabel={green:"&#127881; On Track",amber:"&#128993; Attention",red:"&#128308; Critical"};
  data.forEach(function(d){
    var b2bW=d.Total?Math.round(d.B2B/d.Total*100):0;
    var expW=d.Total?Math.round(d.Experience/d.Total*100):0;
    var planW=d.Total?Math.round(d.Planning/d.Total*100):0;
    var card=document.createElement("div");card.className="sc-card "+d.RAG_Overall;
    card.innerHTML=
      '<div class="sc-card-header">'
        +'<div><div class="sc-district">'+d.District+'</div><span class="sc-circle-tag">'+d.Circle+'</span></div>'
        +'<span class="sc-rag-badge '+d.RAG_Overall+'">'+ragLabel[d.RAG_Overall]+'</span>'
      +'</div>'
      +'<div class="sc-metrics">'
        +'<div class="sc-metric"><div class="sc-metric-label">Total Meters</div><div class="sc-metric-value">'+d.Total.toLocaleString("en-IN")+'</div></div>'
        +'<div class="sc-metric"><div class="sc-metric-label">Resolved %</div><div class="sc-metric-value '+d.RAG_Resolved+'">'+d.Resolved_Pct+'%</div></div>'
        +'<div class="sc-metric"><div class="sc-metric-label">Avg RSRP</div><div class="sc-metric-value '+d.RAG_RSRP+'">'+d.Avg_RSRP+' dBm</div></div>'
        +'<div class="sc-metric"><div class="sc-metric-label">Resolved Count</div><div class="sc-metric-value green">'+d.Resolved.toLocaleString("en-IN")+'</div></div>'
      +'</div>'
      +'<div class="sc-team-bars">'
        +'<div class="sc-team-row"><span class="sc-team-label">B2B</span><div class="sc-team-track"><div class="sc-team-fill" style="width:'+b2bW+'%;background:var(--cyan)"></div></div><span class="sc-team-count">'+d.B2B.toLocaleString("en-IN")+'</span></div>'
        +'<div class="sc-team-row"><span class="sc-team-label">Experience</span><div class="sc-team-track"><div class="sc-team-fill" style="width:'+expW+'%;background:var(--amber)"></div></div><span class="sc-team-count">'+d.Experience.toLocaleString("en-IN")+'</span></div>'
        +'<div class="sc-team-row"><span class="sc-team-label">Planning</span><div class="sc-team-track"><div class="sc-team-fill" style="width:'+planW+'%;background:var(--rose)"></div></div><span class="sc-team-count">'+d.Planning.toLocaleString("en-IN")+'</span></div>'
      +'</div>';
    grid.appendChild(card);
  });
}

var rawKanbanAll = {{ c.kanban_data_all|safe }};

document.addEventListener("DOMContentLoaded", function() {
  var kbSel = document.getElementById("filter-kanban-circle");
  if(kbSel && circlesList) {
    circlesList.forEach(function(c) {
      var o = document.createElement("option"); o.value = c; o.textContent = c; kbSel.appendChild(o);
    });
  }
});

function _chipCls(s){
  s=s.toLowerCase();
  if(s.includes("live"))return "chip-live";
  if(s.includes("on air")||s.includes("onair"))return "chip-onair";
  if(s.includes("plan"))return "chip-planned";
  return "chip-other";
}

function _fmt(n){return Number(n).toLocaleString("en-IN");}

function _circleBreakdown(obj){
  return Object.entries(obj||{}).map(function(e){
    return "<span class=\"sprint-circle-val\">"+e[0]+": "+_fmt(e[1])+"</span>";
  }).join("");
}

function _bucketTable(obj) {
  if (!obj || Object.keys(obj).length === 0) return "";
  var rows = Object.entries(obj).map(function(e) {
    var name = e[0] === "Unknown" ? "No Bucket Data" : e[0];
    return "<div style=\"display:flex;justify-content:space-between;align-items:center;padding:4px 0;border-bottom:1px dashed var(--border);font-size:10.5px;\">"
         + "<span style=\"color:var(--text-m)\">" + name + "</span>"
         + "<span style=\"font-family:'DM Mono',monospace;font-weight:700;color:var(--text-h)\">" + _fmt(e[1]) + "</span>"
         + "</div>";
  }).join("");
  return "<div style=\"margin-top:12px; border-top:2px solid var(--border); padding-top:8px;\">"
       + "<div style=\"font-size:9.5px;font-weight:800;letter-spacing:.5px;text-transform:uppercase;color:var(--text-s);margin-bottom:6px;\">Grid Bucket Breakdown</div>"
       + rows + "</div>";
}

function _activityRows(obj, color, totalCount){
  if(!obj||!Object.keys(obj).length) return "<p style=\"font-size:11px;color:var(--text-m);padding:6px 0\">No data</p>";
  var maxVal=Math.max.apply(null,Object.values(obj));
  var total=Object.values(obj).reduce(function(a,b){return a+b;},0);
  var rows=Object.entries(obj).map(function(e){
    var w=maxVal?Math.round(e[1]/maxVal*100):0;
    return "<div class=\"sprint-activity-row\">"
      +"<div class=\"sprint-activity-name\">"+e[0]
        +"<div class=\"sprint-activity-bar\"><div class=\"sprint-activity-fill\" style=\"width:"+w+"%;background:"+color+"\"></div></div>"
      +"</div>"
      +"<span class=\"sprint-activity-count\" style=\"color:"+color+"\">"+_fmt(e[1])+"</span>"
    +"</div>";
  }).join("");
  rows+="<div class=\"sprint-total-row\"><span>Total</span><span>"+_fmt(total)+"</span></div>";
  return rows;
}

/* Download specific metric data (Total, Resolved, Non-Comm) */
function downloadMetricCSV(stage, metric, circ) {
  var url = "/export_stage_data?circle=" + encodeURIComponent(circ) + 
            "&stage=" + encodeURIComponent(stage) + 
            "&metric=" + encodeURIComponent(metric);
  window.location.href = url;
}

/* Helper to render the download icon next to numbers */
function _dlIcon(stage, metric, circ) {
  return `<a title="Download ${metric} data" style="cursor:pointer; font-size:16px; margin-left:8px; text-decoration:none;" onclick="downloadMetricCSV('${stage}', '${metric}', '${circ}')">&#11015;</a>`;
}

// =========================================================================
// NEW: DOWNLOAD IMAGE FUNCTION
// =========================================================================
function downloadSummaryImage(e){
  var btn = e.currentTarget;
  var originalText = btn.innerHTML;
  btn.innerHTML = "⏳ Generating...";
  btn.disabled = true;
  
  var target = document.getElementById("kanban-export-area");
  var gridChildren = document.getElementById("sprint-grid").children;
  var hiddenElements = [];
  
  // The top summary is the first 9 cells (KPI, Resolved, Non-Comm)
  // Hide everything below the 9th cell
  for(var i=9; i<gridChildren.length; i++){
      hiddenElements.push({el: gridChildren[i], display: gridChildren[i].style.display});
      gridChildren[i].style.display = "none";
  }
  
  // Grab current background color to make sure screenshot matches dark/light mode
  var bgColor = window.getComputedStyle(document.body).backgroundColor;
  target.style.backgroundColor = bgColor;

  setTimeout(function() {
      html2canvas(target, { scale: 2, useCORS: true, backgroundColor: bgColor }).then(function(canvas) {
          var link = document.createElement('a');
          var circ = document.getElementById("filter-kanban-circle").value;
          link.download = 'Adani_Sprint_Summary_' + circ + '.png';
          link.href = canvas.toDataURL('image/png');
          link.click();
          
          // Restore hidden elements
          hiddenElements.forEach(function(item) { item.el.style.display = item.display; });
          target.style.backgroundColor = ""; 
          btn.innerHTML = originalText;
          btn.disabled = false;
      }).catch(function(err) {
          console.error("Screenshot Error: ", err);
          hiddenElements.forEach(function(item) { item.el.style.display = item.display; });
          target.style.backgroundColor = "";
          btn.innerHTML = originalText;
          btn.disabled = false;
          alert("Failed to capture image. Check console for details.");
      });
  }, 150);
}
// =========================================================================

function renderKanban(){
  var sel = document.getElementById("filter-kanban-circle");
  var circ = sel ? sel.value : "All";
  var rawKanban = rawKanbanAll[circ] || rawKanbanAll["All"];

  var hdrs = document.getElementById("sprint-stage-headers");
  var grid = document.getElementById("sprint-grid");
  if (!hdrs || !grid) return;
  hdrs.innerHTML = "";
  grid.innerHTML = "";

  /* Stage header row */
  rawKanban.forEach(function(k){
    var h = document.createElement("div");
    h.className = "sprint-col-header";
    h.style.background = k.color;
    h.innerHTML = "<span class=\"sprint-col-header-title\">" + k.stage + "</span>"
      + "<span class=\"sprint-col-header-badge\">" + _fmt(k.count) + "</span>";
    hdrs.appendChild(h);
  });

  /* SECTION 1: KPI block */
  rawKanban.forEach(function(k){
    var b2bW = k.count ? Math.round(k.b2b / k.count * 100) : 0;
    var expW = k.count ? Math.round(k.exp / k.count * 100) : 0;
    var planW= k.count ? Math.round(k.plan / k.count * 100) : 0;
    var bigNum = k.count >= 1000 ? (k.count / 1000).toFixed(1) + "K" : _fmt(k.count);
    var chips = Object.entries(k.circles || {}).slice(0,5).map(function(e){
      return "<span class=\"sprint-circle-chip\">" + e[0] + ": " + _fmt(e[1]) + "</span>";
    }).join("");
    
    var cell = document.createElement("div");
    cell.className = "sprint-cell sprint-kpi-block";
    cell.innerHTML = 
      "<div class=\"sprint-big-num\">" + bigNum + _dlIcon(k.stage, 'total', circ) + "</div>"
      + "<div class=\"sprint-scope-pct\">" + k.pct + "% of total scope</div>"
      + (chips ? "<div class=\"sprint-circle-chips\">" + chips + "</div>" : "")
      + "<div class=\"sprint-team-bars\">"
        + "<div class=\"sprint-team-row\"><span class=\"sprint-team-lbl\">B2B</span>"
          + "<div class=\"sprint-team-track\"><div class=\"sprint-team-fill b2b\" style=\"width:" + b2bW + "%\"></div></div>"
          + "<span class=\"sprint-team-count\">" + _fmt(k.b2b) + "</span></div>"
        + "<div class=\"sprint-team-row\"><span class=\"sprint-team-lbl\">Experience</span>"
          + "<div class=\"sprint-team-track\"><div class=\"sprint-team-fill exp\" style=\"width:" + expW + "%\"></div></div>"
          + "<span class=\"sprint-team-count\">" + _fmt(k.exp) + "</span></div>"
        + "<div class=\"sprint-team-row\"><span class=\"sprint-team-lbl\">Planning</span>"
          + "<div class=\"sprint-team-track\"><div class=\"sprint-team-fill plan\" style=\"width:" + planW + "%\"></div></div>"
          + "<span class=\"sprint-team-count\">" + _fmt(k.plan) + "</span></div>"
      + "</div>";
    grid.appendChild(cell);
  });

  /* SECTION 2: Resolved Count */
  rawKanban.forEach(function(k){
    var cell = document.createElement("div");
    cell.className = "sprint-cell";
    cell.innerHTML = 
      "<div class=\"sc-section-hdr\">&#9989; Resolved Count</div>"
      + "<div class=\"sc-section-body\">"
        + "<div class=\"sprint-summary-row\">"
          + "<span class=\"sprint-summary-label\">Total Resolved</span>"
          + "<span class=\"sprint-summary-val green\">" + _fmt(k.resolved_total) + _dlIcon(k.stage, 'resolved', circ) + "</span>"
        + "</div>"
        + "<div class=\"sprint-summary-sub\">Circle-wise:</div>"
        + "<div class=\"sprint-circle-breakdown\">" + _circleBreakdown(k.resolved_circles) + "</div>"
      + "</div>";
    grid.appendChild(cell);
  });

  /* SECTION 3: Non-Comm Total + bucket */
  rawKanban.forEach(function(k){
    var cell = document.createElement("div");
    cell.className = "sprint-cell";
    cell.innerHTML = 
      "<div class=\"sc-section-hdr\">&#128308; Non-Comm Total</div>"
      + "<div class=\"sc-section-body\">"
        + "<div class=\"sprint-summary-row\">"
          + "<span class=\"sprint-summary-label\">Total Non-Comm</span>"
          + "<span class=\"sprint-summary-val\">" + _fmt(k.noncomm_total) + _dlIcon(k.stage, 'noncomm', circ) + "</span>"
        + "</div>"
        + "<div class=\"sprint-summary-sub\">Circle-wise:</div>"
        + "<div class=\"sprint-circle-breakdown\">" + _circleBreakdown(k.noncomm_circles) + "</div>"
        + _bucketTable(k.noncomm_buckets)
      + "</div>";
    grid.appendChild(cell);
  });

  /* SECTION DIVIDER */
  var divider = document.createElement("div");
  divider.style.cssText = "grid-column:1/-1;background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:10px 16px;font-size:10px;font-weight:800;letter-spacing:.8px;text-transform:uppercase;color:var(--text-s);margin-top:8px";
  divider.textContent = "Category Breakdown";
  grid.appendChild(divider);

  /* SECTION 4: Activity to Resolved Meter */
  rawKanban.forEach(function(k){
    var cell=document.createElement("div");
    cell.className="sprint-cell";
    var total=Object.values(k.activity_breakdown||{}).reduce(function(a,b){return a+b;},0);
    cell.innerHTML=
      "<div class=\"sc-section-hdr\">&#128296; Activity to Resolved Meter <span style=\"font-size:9px;opacity:.7\">(col: Activity Done)</span></div>"
      +"<div class=\"sc-section-body\">"
        +_activityRows(k.activity_breakdown, k.color, k.count)
        +(total?"<a class=\"sprint-dl-btn\" onclick=\"downloadStageCSV('"+k.stage+"','activity', '"+circ+"')\">&#11015; Download</a>":"")
      +"</div>";
    grid.appendChild(cell);
  });

  /* SECTION 5: Customer Action */
  rawKanban.forEach(function(k){
    var cell=document.createElement("div");
    cell.className="sprint-cell";
    cell.innerHTML=
      "<div class=\"sc-section-hdr\">&#128101; Customer Action <span style=\"font-size:9px;opacity:.7\">(col: Customer Action)</span></div>"
      +"<div class=\"sc-section-body\">"
        +_activityRows(k.cust_action_breakdown, "#6366f1", k.count)
      +"</div>";
    grid.appendChild(cell);
  });

  /* SECTION 6: Action Plan */
  rawKanban.forEach(function(k){
    var cell=document.createElement("div");
    cell.className="sprint-cell";
    var planRows=Object.entries(k.action_plan||{}).map(function(e){
      return "<div class=\"action-plan-row\">"
        +"<span class=\"action-plan-count\">"+_fmt(e[1])+"</span>"
        +"<span class=\"action-plan-text\">"+e[0]+"</span>"
      +"</div>";
    }).join("");
    cell.innerHTML=
      "<div class=\"sc-section-hdr\">&#128204; Action Plan <span style=\"font-size:9px;opacity:.7\">(only blank Resolved → Latest Updated 5/6)</span></div>"
      +"<div class=\"sc-section-body\">"
        +(planRows||"<p style=\"font-size:11px;color:var(--text-m)\">No pending action plans</p>")
      +"</div>";
    grid.appendChild(cell);
  });

  /* SECTION 7: New Site Plan */
  rawKanban.forEach(function(k){
    var cell=document.createElement("div");
    cell.className="sprint-cell";

    var siteStatusChips=Object.entries(k.site_statuses||{}).map(function(e){
      return "<span class=\"site-status-chip "+_chipCls(e[0])+"\">"+e[0]+": "+e[1]+"</span>";
    }).join("");

    var taskRows=Object.entries(k.task_counts||{}).map(function(e){
      return "<div class=\"sprint-activity-row\">"
        +"<span class=\"sprint-activity-name\">"+e[0]+"</span>"
        +"<span class=\"sprint-activity-count\" style=\"color:"+k.color+"\">"+_fmt(e[1])+"</span>"
      +"</div>";
    }).join("");

    cell.innerHTML=
      "<div class=\"sc-section-hdr\">&#128205; New Site Plan (Plan Site ID if Any)</div>"
      +"<div class=\"sc-section-body\">"
        +(k.total_sites>0
          ? "<div class=\"sprint-sites-banner\"><span class=\"sprint-sites-count\">"+_fmt(k.total_sites)+"</span><span style=\"font-size:11px;color:var(--indigo);font-weight:600\">Unique Planned Sites (deduped)</span></div>"
          : "")
        +(siteStatusChips?"<div style=\"margin-bottom:8px\">"+siteStatusChips+"</div>":"")
        +(taskRows||"<p style=\"font-size:11px;color:var(--text-m)\">No site plan data</p>")
        +(k.total_sites>0?"<a class=\"sprint-dl-btn\" style=\"margin-top:6px\" onclick=\"downloadStageCSV('"+k.stage+"','sites', '"+circ+"')\">&#11015; Download Site List</a>":"")
      +"</div>";
    grid.appendChild(cell);
  });

    if(window._kbChart)window._kbChart.destroy();
  window._kbChart=new Chart(document.getElementById("kanbanChart"),{
    type:"bar",
    data:{labels:rawKanban.map(function(k){return k.stage;}),datasets:[
      {label:"B2B",data:rawKanban.map(function(k){return k.b2b;}),backgroundColor:PAL.cyan,borderRadius:4},
      {label:"Experience",data:rawKanban.map(function(k){return k.exp;}),backgroundColor:PAL.amber,borderRadius:4},
      {label:"Planning",data:rawKanban.map(function(k){return k.plan;}),backgroundColor:PAL.rose,borderRadius:4}
    ]},
    options:{responsive:true,maintainAspectRatio:false,
      scales:{x:{stacked:true,grid:{display:false}},y:{stacked:true,grid:{color:"#f1f5f9"},ticks:{callback:function(v){return v>=1000?(v/1000).toFixed(0)+"K":v;}}}},
      plugins:{legend:{position:"top",labels:{usePointStyle:true,padding:16}},
        tooltip:{callbacks:{label:function(ctx){return " "+ctx.dataset.label+": "+ctx.parsed.y.toLocaleString("en-IN");}}}}}
  });
}

function downloadStageCSV(stage, type, circ){
  circ = circ || "All";
  var rKanban = rawKanbanAll[circ] || rawKanbanAll["All"];
  var k=rKanban.find(function(r){return r.stage===stage;});
  if(!k) return;
  var rows, filename;
  if(type==="activity"){
    rows=[["Activity","Count"]].concat(Object.entries(k.activity_breakdown||{}));
    filename=stage+"_"+circ+"_Activity_Done.csv";
  } else {
    rows=[["Site Plan Category","Count"]].concat(Object.entries(k.task_counts||{}));
    filename=stage+"_"+circ+"_Site_Plan.csv";
  }
  var csv=rows.map(function(r){return r.join(",");}).join("\n");
  var blob=new Blob([csv],{type:"text/csv"});
  var a=document.createElement("a");
  a.href=URL.createObjectURL(blob);
  a.download=filename;
  a.click();
}

async function exportMatrixExcel(){
  if(typeof ExcelJS==="undefined"){alert("ExcelJS failed.");return;}
  var wb=new ExcelJS.Workbook();var ws=wb.addWorksheet("Matrix");
  ws.columns=[{header:"OWNERSHIP",key:"owner",width:22},{header:"TASK CATEGORY",key:"task",width:60},{header:"METER COUNT",key:"count",width:16},{header:"RESOLVED",key:"resolved",width:16}];
  var hdr=ws.getRow(1);hdr.font={bold:true,color:{argb:"FFFFFFFF"}};hdr.fill={type:"pattern",pattern:"solid",fgColor:{argb:"FF1E293B"}};
  document.querySelectorAll("#matrix-tbody tr").forEach(function(tr){
    var cols=tr.querySelectorAll("td");if(cols.length<3)return;
    var isTot=cols.length===3;
    var r=ws.addRow({owner:cols[0].innerText.trim(),task:isTot?"":cols[1].innerText.trim(),
      count:parseInt((isTot?cols[1]:cols[2]).innerText.replace(/,/g,""),10)||0,
      resolved:parseInt((isTot?cols[2]:cols[3]).innerText.replace(/,/g,""),10)||0});
    r.getCell("count").numFmt="#,##0";r.getCell("resolved").numFmt="#,##0";
    if(isTot){r.font={bold:true};r.fill={type:"pattern",pattern:"solid",fgColor:{argb:"FFF1F5F9"}};}
  });
  var buf=await wb.xlsx.writeBuffer();
  saveAs(new Blob([buf],{type:"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}),
    "Adani_Matrix_"+document.getElementById("filter-team").value+"_"+document.getElementById("filter-circle").value+".xlsx");
}

function toggleEditDrawer(){
  var d=document.getElementById("edit-drawer");
  d.classList.toggle("open");
}

async function submitEdit(){
  var meterId  = document.getElementById("edit-meter-id").value.trim();
  var field    = document.getElementById("edit-field-select").value;
  var newVal   = document.getElementById("edit-new-value").value.trim();
  var password = document.getElementById("edit-password").value;
  var statusEl = document.getElementById("edit-status");
  var btn      = document.getElementById("edit-save-btn");

  if(!meterId||!newVal||!password){
    statusEl.textContent="Fill all fields."; statusEl.className="edit-status err"; return;
  }
  btn.disabled=true; statusEl.textContent="Saving..."; statusEl.className="edit-status";

  try{
    var resp = await fetch("/save_edit",{
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({meter_id:meterId, field:field, value:newVal, password:password})
    });
    var data = await resp.json();
    if(data.ok){
      statusEl.textContent="✅ "+data.msg; statusEl.className="edit-status ok";
      document.getElementById("edit-meter-id").value="";
      document.getElementById("edit-new-value").value="";
      document.getElementById("edit-password").value="";
    } else {
      statusEl.textContent="❌ "+data.msg; statusEl.className="edit-status err";
    }
  }catch(e){
    statusEl.textContent="❌ Network error: "+e.message; statusEl.className="edit-status err";
  }
  btn.disabled=false;
}
</script>
</body>
</html>
"""

# ============================================================================
# ROUTES
# ============================================================================
@app.route("/")
def index():
    result = load_and_process_data()
    if not result:
        return ("<div style='font-family:system-ui;padding:40px'><h2>Data file not found</h2>"
                "<p>Place <code>2.73 Total.csv</code> in same folder.</p></div>"), 404
    chart_data, tables = result
    return render_template_string(TEMPLATE, c=chart_data, t=tables)

@app.route("/export_stage_data")
def export_stage_data():
    try:
        df = pd.read_csv(FILE_PATH, encoding="cp1252")
    except UnicodeDecodeError:
        df = pd.read_csv(FILE_PATH, encoding="latin-1")
    except FileNotFoundError:
        return "File not found.", 404

    circ = request.args.get("circle", "All")
    stage = request.args.get("stage", "")
    metric = request.args.get("metric", "total")

    COL_SHIFT = next((c for c in df.columns if 'ownership shift' in c.lower() or 'ownershift' in c.lower()), None)
    if COL_SHIFT is None:
        COL_SHIFT = next((c for c in df.columns if 'ownership' in c.lower()), "Ownership")
    
    df["Final Ownership"] = df[COL_SHIFT].astype(str).str.strip()
    df["Resolved Status"] = df.get("Resolved", pd.Series(["N/A"]*len(df))).astype(str).str.strip()

    if circ != "All" and "Circle" in df.columns:
        df = df[df["Circle"].astype(str).str.strip().str.upper() == circ.upper()]

    if stage == "B2B Action":
        df = df[df["Final Ownership"].str.contains("B2B", case=False, na=False)]
    elif stage == "Experience":
        df = df[df["Final Ownership"].str.contains("Experience", case=False, na=False)]
    elif stage == "Planning":
        df = df[df["Final Ownership"].str.contains("Planning", case=False, na=False)]

    raw_resolved = df["Resolved Status"].str.lower()
    is_empty_resolved = raw_resolved.isin(["", "nan", "n/a", "<na>", "none", "0", "null", "-", "false", "pending", "unresolved", "still working"])

    if metric == "resolved":
        df = df[~is_empty_resolved]
    elif metric == "noncomm":
        df = df[is_empty_resolved]

    out = io.StringIO()
    df.to_csv(out, index=False)
    out.seek(0)
    
    safe_circ = circ.replace(" ", "_")
    safe_stage = stage.replace(" ", "_")
    return Response(out.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment;filename={safe_stage}_{metric}_{safe_circ}.csv"})

if __name__ == "__main__":
    app.run(debug=True, port=3001)
