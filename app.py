from flask import Flask, render_template, jsonify, send_file, request
import os
import io
import json
import glob
import tempfile
import uuid
from datetime import date, timedelta

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize, shapes
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds
from affine import Affine
from PIL import Image
from shapely.geometry import mapping, shape
from shapely.ops import unary_union
from rasterio.windows import Window, transform as window_transform, bounds as window_bounds
from shapely.geometry import box

app = Flask(__name__)

# =====================================================
# CONFIGURATION
# Keep the same file names as your original app.py.
# Put app.py, the data files, and the templates folder together.
# Shapefile sidecar files (.dbf, .shx, .prj, etc.) must also be present.
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEM_FILE = os.path.join(BASE_DIR, "nsn_dem30m.tif")
SHAPEFILE = os.path.join(BASE_DIR, "Fields2.shp")
WATERWAY_FILE = os.path.join(BASE_DIR, "nsn_wtwpoly.shp")
ROADS_FILE = os.path.join(BASE_DIR, "nsn_road2.shp")
BOUNDARY_FILE = os.path.join(BASE_DIR, "nsn_prov.shp")
FLOOD_PRONE_FILE = os.path.join(BASE_DIR, "Flood_area12.shp")

# Performance switches: False means the shapefile is not read into memory at startup.
# Turn these on if you specifically need these reference layers on the map.
LOAD_ROADS_LAYER = False
LOAD_FLOOD_PRONE_LAYER = False
EXCLUDE_OTHER_CROPS = True
DISPLAY_SIMPLIFY_TOLERANCE = 0.0001  # WGS84 degrees; display only, analysis uses original geometry.
MAX_FLOOD_DISPLAY_DIMENSION = 900   # simplify flood outline generation for browser display.

UTM_CRS = "EPSG:32647"
WGS_CRS = "EPSG:4326"

CROP_TYPES = ["Rice", "Cassava", "Corn", "Sugarcane", "Fruits", "Other Crops"]
COLORS = {
    "Rice": "#4CAF50", "Cassava": "#914624", "Corn": "#FFEB3B",
    "Sugarcane": "#934AC3", "Fruits": "#F762B9", "Other Crops": "#E22525"
}

# Illustrative defaults only. Edit in the web interface for your own local values.
# yield_kg_ha = harvest in kilograms per hectare; price_baht_kg = baht per kilogram.
# Yield defaults below are converted from the previous illustrative kg/rai values
# using 1 hectare = 6.25 rai. Replace with locally validated harvest data.
CROP_DEFAULTS = {
    "Rice": {"yield_kg_ha": 3750, "price_baht_kg": 12, "duration_days": 120, "damage_factor": 0.75},
    "Cassava": {"yield_kg_ha": 18750, "price_baht_kg": 2.5, "duration_days": 270, "damage_factor": 0.65},
    "Corn": {"yield_kg_ha": 4375, "price_baht_kg": 9, "duration_days": 90, "damage_factor": 0.75},
    "Sugarcane": {"yield_kg_ha": 62500, "price_baht_kg": 1.2, "duration_days": 365, "damage_factor": 0.55},
    "Fruits": {"yield_kg_ha": 9375, "price_baht_kg": 20, "duration_days": 365, "damage_factor": 0.60},
    "Other Crops": {"yield_kg_ha": 3125, "price_baht_kg": 5, "duration_days": 100, "damage_factor": 0.70},
}

# Data loaded at startup
fields_utm = None
fields_web = None
datasets = {}
dem_data = None
dem_transform = None
dem_crs = None
dem_png = None
dem_bounds = None
dem_display_elevation = None
dem_palette_cache = {}
startup_warnings = []
geojson_cache = {}
excluded_crop_count = 0
# Only the latest scenario is retained. Per-field scalar results are written as
# JSON Lines to temporary storage so the simulation response does not include a
# huge duplicate set of polygon geometries.
scenario_registry = {}
SCENARIO_CACHE_PREFIX = os.path.join(tempfile.gettempdir(), "flood_webgis_scenario_")


def clear_scenario_cache():
    """Remove older ephemeral scenario records from this worker."""
    scenario_registry.clear()
    for path in glob.glob(SCENARIO_CACHE_PREFIX + "*.jsonl"):
        try:
            os.remove(path)
        except OSError as exc:
            print(f"WARNING: could not remove old scenario cache {path}: {exc}")


def clean_geometries(gdf):
    """Drop empty geometries and repair invalid polygons where possible."""
    if gdf is None or gdf.empty:
        return gdf
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        # buffer(0) is broadly compatible with common GeoPandas/Shapely versions.
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].buffer(0)
    return gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty & gdf.geometry.is_valid].copy()


def load_vector(filepath, label):
    if not os.path.exists(filepath):
        startup_warnings.append(f"{label}: file not found ({os.path.basename(filepath)})")
        print(f"WARNING: {label} not found: {filepath}")
        return None
    try:
        gdf = gpd.read_file(filepath)
        if gdf.empty:
            startup_warnings.append(f"{label}: file contains no features")
            return None
        if gdf.crs is None:
            startup_warnings.append(f"{label}: CRS is missing")
            print(f"ERROR: {label} has no CRS information")
            return None
        gdf = clean_geometries(gdf)
        if gdf.empty:
            startup_warnings.append(f"{label}: no valid geometries")
            return None
        print(f"OK: {label}: {len(gdf)} features; source CRS={gdf.crs}")
        return gdf
    except Exception as exc:
        startup_warnings.append(f"{label}: failed to load ({exc})")
        print(f"ERROR loading {label}: {exc}")
        return None


def _display_text(value):
    """Return a safe, non-empty string for popup metadata, or None."""
    if value is None:
        return None
    try:
        if bool(np.asarray(value).ndim == 0 and np.issubdtype(type(value), np.number) and np.isnan(value)):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text or text.casefold() in {"nan", "none", "<na>", "null"}:
        return None
    return text


def first_attribute(row, candidates):
    """Find the first non-empty source attribute, ignoring column-name case."""
    keys = {str(key).casefold(): key for key in row.index}
    for candidate in candidates:
        actual_key = keys.get(candidate.casefold())
        if actual_key is not None:
            value = _display_text(row.get(actual_key))
            if value:
                return value
    return None


def geojson_features(gdf, tolerance=DISPLAY_SIMPLIFY_TOLERANCE, fields_layer=False, layer_kind=None):
    """Serialize one vector page to GeoJSON with stable popup-friendly attributes."""
    if gdf is None or gdf.empty:
        return []
    web = gdf.to_crs(WGS_CRS)
    output = []
    for idx, row in web.iterrows():
        try:
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            if tolerance:
                geom = geom.simplify(tolerance, preserve_topology=True)
            if geom.is_empty:
                continue

            if fields_layer:
                raw_crop = first_attribute(row, ["Crop_type", "crop_type", "CROP_TYPE"])
                crop_lookup = {name.casefold(): name for name in CROP_TYPES}
                crop = crop_lookup.get((raw_crop or "Other Crops").casefold(), "Other Crops")
                raw_field_id = first_attribute(row, ["OBJECTID_1", "OBJECTID", "FID", "field_id", "id"])
                field_id = raw_field_id or str(idx)
                display_name = first_attribute(row, ["field_name_en", "FIELD_NAME", "NAME_EN", "NAME", "name"])
                if not display_name:
                    display_name = f"Agricultural field {field_id}"
                area_m2 = float(row.get("area_m2", 0.0) or 0.0)
                land_use_en = first_attribute(row, ["LU_DES_EN", "LANDUSE_EN", "LAND_USE_EN", "DES_EN", "TYPE_EN"])
                land_use_th = first_attribute(row, ["LU_DES_TH", "LANDUSE_TH", "LAND_USE_TH", "DES_TH"])
                land_use_code = first_attribute(row, ["LU_CODE", "LULC_CODE", "LANDUSE_CODE"])
                props = {
                    "name": display_name,
                    "display_name": display_name,
                    "field_id": field_id,
                    "Crop_type": crop,
                    "crop_type": crop,
                    "crop_type_en": crop,
                    "land_use_en": land_use_en,
                    "land_use_th": land_use_th,
                    "land_use_code": land_use_code,
                    "area_m2": area_m2,
                    "area_ha": area_m2 / 10000.0,
                    "area_rai": area_m2 / 1600.0,
                    "color": COLORS.get(crop, "#999999")
                }
            else:
                if layer_kind == "boundary":
                    display_name = first_attribute(row, [
                        "PROVINCE_EN", "PROV_EN", "NAME_EN", "NAME_ENG", "ADM1_EN", "NAME_1"
                    ]) or "Nakhon Sawan"
                    props = {"name": display_name, "display_name": display_name, "feature_type": "province"}
                    props["province_name"] = display_name
                elif layer_kind == "waterways":
                    # Prefer explicitly English-labelled name fields. Generic name fields
                    # are only used if they do not appear to contain Thai text.
                    display_name = first_attribute(row, [
                        "WATERWAY_NAME_EN", "WATER_NAME_EN", "RIVER_NAME_EN", "CANAL_NAME_EN",
                        "NAME_EN", "NAME_ENG", "NAME_E", "ENG_NAME", "ENGNAME", "WTR_NAME_EN"
                    ])
                    if not display_name:
                        generic_name = first_attribute(row, [
                            "WATERWAY_NAME", "WTR_NAME", "WATER_NAME", "RIVER_NAME",
                            "CANAL_NAME", "WATERBODY", "NAME", "name"
                        ])
                        if generic_name and not any("\u0e00" <= char <= "\u0e7f" for char in generic_name):
                            display_name = generic_name
                    description_en = first_attribute(row, [
                        "LU_DES_EN", "DESCRIPTION_EN", "DES_EN", "TYPE_EN", "CLASS_EN",
                        "WATERWAY_TYPE_EN", "WATERWAY_TYPE", "TYPE", "LU_DES"
                    ])
                    if not display_name:
                        display_name = description_en or f"Waterway polygon {idx}"
                    props = {
                        "name": display_name,
                        "display_name": display_name,
                        "feature_type": "waterway",
                        "description_en": description_en
                    }
                elif layer_kind == "flood_prone":
                    display_name = first_attribute(row, [
                        "NAME_EN", "NAME_ENG", "ENG_NAME", "NAME", "name", "LU_DES_EN", "TYPE_EN"
                    ]) or f"Flood-prone area {idx}"
                    props = {"name": display_name, "display_name": display_name, "feature_type": "flood-prone area"}
                elif layer_kind == "roads":
                    display_name = first_attribute(row, [
                        "ROAD_NAME_EN", "ROAD_NAME", "NAME_EN", "NAME_ENG", "NAME", "name", "ROAD_TYPE_EN", "TYPE_EN"
                    ]) or f"Road feature {idx}"
                    props = {"name": display_name, "display_name": display_name, "feature_type": "road"}
                else:
                    display_name = first_attribute(row, ["NAME_EN", "NAME_ENG", "NAME", "name"]) or f"Feature {idx}"
                    props = {"name": display_name, "display_name": display_name}

            # Include selected English/Thai classification attributes for water features too.
            if not fields_layer and layer_kind in {"waterways", "flood_prone", "roads"}:
                description_th = first_attribute(row, ["LU_DES_TH", "DESCRIPTION_TH", "DES_TH", "TYPE_TH"])
                code_value = first_attribute(row, ["LU_CODE", "TYPE_CODE", "CLASS_CODE"])
                if description_th:
                    props["description_th"] = description_th
                if code_value:
                    props["feature_code"] = code_value

            feature = {"type": "Feature", "geometry": mapping(geom), "properties": props}
            json.dumps(feature, allow_nan=False)
            output.append(feature)
        except (ValueError, TypeError, OverflowError) as exc:
            print(f"Skipping feature {idx}: {exc}")
    print(f"Converted {len(output)} {layer_kind or ('field' if fields_layer else 'vector')} features to GeoJSON.")
    return output


def load_dem():
    global dem_data, dem_transform, dem_crs, dem_png, dem_bounds, dem_display_elevation, dem_palette_cache
    if not os.path.exists(DEM_FILE):
        startup_warnings.append(f"DEM: file not found ({os.path.basename(DEM_FILE)})")
        print(f"WARNING: DEM not found: {DEM_FILE}")
        return
    try:
        with rasterio.open(DEM_FILE) as src:
            if src.crs is None:
                startup_warnings.append("DEM: CRS is missing")
                return
            masked_dem = src.read(1, masked=True).astype(np.float32)
            dem_data = masked_dem.filled(np.nan)
            del masked_dem  # avoid keeping a second full-size DEM array during startup
            dem_transform = src.transform
            dem_crs = src.crs

            # Build a lightweight PNG solely for visual display in Leaflet.
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src.crs, WGS_CRS, src.width, src.height, *src.bounds
            )
            max_dimension = 1600
            scale = max(1.0, dst_width / max_dimension, dst_height / max_dimension)
            out_width = max(1, int(dst_width / scale))
            out_height = max(1, int(dst_height / scale))
            dst_transform = dst_transform * Affine.scale(dst_width / out_width, dst_height / out_height)
            elevation = np.full((out_height, out_width), np.nan, dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1), destination=elevation,
                src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata,
                dst_transform=dst_transform, dst_crs=WGS_CRS, dst_nodata=np.nan,
                resampling=Resampling.bilinear
            )
            left = dst_transform.c
            top = dst_transform.f
            right = left + dst_transform.a * out_width
            bottom = top + dst_transform.e * out_height
            dem_bounds = (left, bottom, right, top)

            # Keep the downsampled georeferenced elevation grid so the browser can choose
            # another display colour ramp without reprojecting the original DEM each time.
            dem_display_elevation = elevation
            dem_palette_cache = {}
            valid = np.isfinite(elevation)
            if valid.any():
                low, high = np.percentile(elevation[valid], [2, 98])
                if high <= low:
                    high = low + 1
                # Important: elevation contains NaN in NoData/blank areas.
                # Casting NaN to int produces the minimum int64 value, which then
                # causes the reported "index -9223372036854775808 is out of bounds".
                # Normalize valid pixels only; invalid pixels are transparent in the PNG.
                normalized = np.zeros(elevation.shape, dtype=np.float32)
                normalized[valid] = np.clip(
                    (elevation[valid] - low) / (high - low), 0.0, 1.0
                )
                stops = np.array([
                    [25, 70, 180], [35, 180, 220], [55, 180, 90],
                    [240, 220, 65], [140, 85, 45]
                ], dtype=np.float32)
                pos = normalized * (len(stops) - 1)
                ix = np.minimum(pos.astype(np.int16), len(stops) - 2)
                frac = (pos - ix)[..., None]
                rgb = (stops[ix] * (1.0 - frac) + stops[ix + 1] * frac).astype(np.uint8)
                alpha = np.where(valid, 175, 0).astype(np.uint8)
                image = Image.fromarray(np.dstack((rgb, alpha)), mode="RGBA")
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                dem_png = buffer.getvalue()
        print(f"OK: DEM loaded: shape={dem_data.shape}; CRS={dem_crs}")
    except Exception as exc:
        startup_warnings.append(f"DEM: failed to load ({exc})")
        print(f"ERROR preparing DEM: {exc}")
        dem_data = None


def load_all_data():
    global fields_utm, fields_web, excluded_crop_count
    load_dem()
    fields = load_vector(SHAPEFILE, "Fields")
    if fields is not None:
        if "Crop_type" not in fields.columns:
            # Do not drop every field blindly when the expected attribute is absent.
            # In that case, the actual crop-name column must be identified in Fields.shp.
            fields["Crop_type"] = "Other Crops"
            startup_warnings.append(
                "Fields.shp has no 'Crop_type' column. Could not safely filter Other Crops; "
                "all fields are temporarily labelled Other Crops. Check the source attribute names."
            )
        else:
            # Normalize known crop labels and map unknown/null values to Other Crops.
            crop_lookup = {name.casefold(): name for name in CROP_TYPES}
            source_crop = fields["Crop_type"].fillna("Other Crops").astype(str).str.strip()
            fields["Crop_type"] = source_crop.map(lambda value: crop_lookup.get(value.casefold(), "Other Crops"))
            if EXCLUDE_OTHER_CROPS:
                before = len(fields)
                fields = fields[fields["Crop_type"].isin(["Rice", "Cassava", "Corn", "Sugarcane", "Fruits"])].copy()
                excluded_crop_count = before - len(fields)
                print(f"Crop filter: kept {len(fields)} fields; excluded {excluded_crop_count} Other/unknown crop fields.")
                startup_warnings.append(
                    f"Crop filter active: excluded {excluded_crop_count} Other/unknown crop fields."
                )
        # Area in square metres using a projected CRS suitable for the study area.
        fields["area_m2"] = fields.to_crs(UTM_CRS).geometry.area.values
        fields_utm = fields.to_crs(UTM_CRS)
        fields_utm["Crop_type"] = fields["Crop_type"].values
        fields_utm["area_m2"] = fields["area_m2"].values
        fields_web = fields_utm.to_crs(WGS_CRS)
    else:
        fields_utm = None
        fields_web = None
    datasets["waterways"] = load_vector(WATERWAY_FILE, "Waterways")
    datasets["roads"] = load_vector(ROADS_FILE, "Roads") if LOAD_ROADS_LAYER else None
    if not LOAD_ROADS_LAYER:
        startup_warnings.append("Roads layer disabled at startup for faster loading (set LOAD_ROADS_LAYER=True to enable).")
    datasets["boundary"] = load_vector(BOUNDARY_FILE, "Boundary")
    datasets["flood_prone"] = load_vector(FLOOD_PRONE_FILE, "Flood-prone areas") if LOAD_FLOOD_PRONE_LAYER else None
    if not LOAD_FLOOD_PRONE_LAYER:
        startup_warnings.append("Flood-prone shapefile layer disabled at startup for faster loading (set LOAD_FLOOD_PRONE_LAYER=True to enable).")


load_all_data()


# Colour ramps used for DEM display only; analysis always uses the original DEM values.
DEM_PALETTES = {
    "terrain": [[25, 70, 180], [35, 180, 220], [55, 180, 90], [240, 220, 65], [140, 85, 45]],
    "grayscale": [[20, 20, 20], [75, 75, 75], [140, 140, 140], [205, 205, 205], [248, 248, 248]],
    "ocean": [[10, 45, 130], [20, 140, 190], [50, 185, 120], [220, 210, 105], [130, 80, 45]],
    "warm": [[40, 25, 90], [95, 45, 130], [190, 55, 80], [245, 135, 40], [255, 235, 120]],
}

def render_dem_png(palette_name="terrain"):
    """Render an already-downsampled DEM using a user-selected display palette."""
    if dem_display_elevation is None:
        return None
    palette_name = palette_name if palette_name in DEM_PALETTES else "terrain"
    if palette_name in dem_palette_cache:
        return dem_palette_cache[palette_name]
    elevation = dem_display_elevation
    valid = np.isfinite(elevation)
    if not valid.any():
        return None
    low, high = np.percentile(elevation[valid], [2, 98])
    if high <= low:
        high = low + 1
    normalized = np.zeros(elevation.shape, dtype=np.float32)
    normalized[valid] = np.clip((elevation[valid] - low) / (high - low), 0.0, 1.0)
    stops = np.asarray(DEM_PALETTES[palette_name], dtype=np.float32)
    pos = normalized * (len(stops) - 1)
    ix = np.minimum(pos.astype(np.int16), len(stops) - 2)
    frac = (pos - ix)[..., None]
    rgb = (stops[ix] * (1.0 - frac) + stops[ix + 1] * frac).astype(np.uint8)
    alpha = np.where(valid, 175, 0).astype(np.uint8)
    image = Image.fromarray(np.dstack((rgb, alpha)), mode="RGBA")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    png = buffer.getvalue()
    # Cache at most the small fixed set of available ramps to avoid recomputing on every toggle.
    dem_palette_cache[palette_name] = png
    return png

# =====================================================
# FLASK ROUTES
# =====================================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({
        "ok": True,
        "dem_available": dem_data is not None,
        "fields_count": 0 if fields_utm is None else int(len(fields_utm)),
        "excluded_crop_fields": int(excluded_crop_count),
        "warnings": startup_warnings
    })


@app.route("/api/layers")
def get_layers():
    # Cache serialized features in memory so repeat page loads do not re-convert shapefiles.
    if "layers" not in geojson_cache:
        geojson_cache["layers"] = {
            "boundary": geojson_features(datasets.get("boundary"), layer_kind="boundary"),
            "flood_prone": geojson_features(datasets.get("flood_prone"), layer_kind="flood_prone"),
            "waterways": geojson_features(datasets.get("waterways"), layer_kind="waterways"),
            "roads": geojson_features(datasets.get("roads"), layer_kind="roads")
        }
    return jsonify(geojson_cache["layers"])


@app.route("/api/fields")
def get_fields():
    """Return agricultural field polygons in small pages to avoid oversized JSON responses."""
    if fields_utm is None:
        return jsonify({"error": f"Crop fields are unavailable. Check {os.path.basename(SHAPEFILE)} and its sidecar files."}), 503
    try:
        offset = max(0, int(request.args.get("offset", 0)))
        limit = int(request.args.get("limit", 250))
    except (TypeError, ValueError):
        return jsonify({"error": "offset and limit must be integers"}), 400
    limit = min(max(limit, 50), 500)
    total = int(len(fields_utm))
    end = min(offset + limit, total)
    if offset >= total:
        return jsonify({"features": [], "offset": offset, "limit": limit, "total": total, "done": True})
    page = fields_utm.iloc[offset:end]
    features = geojson_features(page, fields_layer=True)
    return jsonify({
        "features": features,
        "offset": offset,
        "limit": limit,
        "total": total,
        "done": end >= total
    })


@app.route("/api/dem")
def get_dem():
    palette = request.args.get("palette", "terrain").strip().lower()
    if palette not in DEM_PALETTES:
        return jsonify({"error": f"Unknown DEM palette: {palette}", "available_palettes": list(DEM_PALETTES)}), 400
    png = render_dem_png(palette)
    if png is None:
        return jsonify({"error": "DEM is unavailable"}), 404
    response = send_file(io.BytesIO(png), mimetype="image/png", download_name=f"dem-{palette}.png", max_age=3600)
    return response


@app.route("/api/dem-info")
def get_dem_info():
    if dem_data is None or dem_transform is None or dem_crs is None:
        return jsonify({"available": False, "error": "DEM unavailable"}), 404
    height, width = dem_data.shape
    left, bottom, right, top = rasterio.transform.array_bounds(height, width, dem_transform)
    bounds = transform_bounds(dem_crs, WGS_CRS, left, bottom, right, top, densify_pts=21)
    # Sample at most about one million cells for display slider statistics.
    # This avoids making a full-sized boolean mask and copying every valid DEM cell.
    row_step = max(1, int(np.ceil(height / 1000)))
    col_step = max(1, int(np.ceil(width / 1000)))
    sample = dem_data[::row_step, ::col_step]
    valid_sample = sample[np.isfinite(sample)]
    if not valid_sample.size:
        return jsonify({"available": False, "error": "DEM has no valid pixels"}), 404
    min_elevation, max_elevation = np.percentile(valid_sample, [2, 98])
    return jsonify({
        "available": True,
        "bounds": [[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
        "min_elevation": float(min_elevation),
        "max_elevation": float(max_elevation),
        "elevation_statistics_sampled": True,
        "vertical_units": "Use the same vertical units/datum as the DEM; normally metres."
    })


# =====================================================
# FLOOD SCENARIO MODEL
# This is a simple elevation-threshold ('bathtub') scenario, not a hydraulic model.
# =====================================================
def create_flood_mask(water_level, connected_only=False):
    # Build the threshold mask in-place to reduce temporary arrays on low-RAM hosts.
    valid = np.isfinite(dem_data)
    low_cells = np.empty(dem_data.shape, dtype=bool)
    np.less_equal(dem_data, water_level, out=low_cells)
    np.logical_and(low_cells, valid, out=low_cells)
    del valid

    if not connected_only or datasets.get("waterways") is None:
        return low_cells
    try:
        from scipy import ndimage
        water = datasets["waterways"].to_crs(dem_crs)
        seed_raster = rasterize(
            [(geom, 1) for geom in water.geometry if geom is not None and not geom.is_empty],
            out_shape=dem_data.shape, transform=dem_transform, fill=0,
            all_touched=True, dtype="uint8"
        )
        # Rasterize only writes 0/1, so view as bool without allocating a second raster.
        seeds = seed_raster.view(np.bool_)
        seeds &= low_cells
        if not seeds.any():
            print("WARNING: no low waterway seed pixels; using elevation-threshold mask.")
            del seed_raster, seeds
            return low_cells
        connected = ndimage.binary_propagation(
            seeds, structure=np.ones((3, 3), dtype=bool), mask=low_cells
        )
        del seed_raster, seeds, low_cells
        return connected
    except ImportError:
        print("WARNING: scipy not installed; using elevation-threshold mask.")
        return low_cells
    except Exception as exc:
        print(f"WARNING: connectivity mask failed ({exc}); using elevation-threshold mask.")
        return low_cells


def flood_mask_geojson(mask):
    """Build a coarser display outline; the original full-resolution mask is still used for analysis."""
    if not mask.any():
        return []
    try:
        # Polygonizing every DEM cell can be extremely slow. Downsample only the display outline.
        # Analysis of fields and losses still uses the original-resolution flood mask.
        from math import ceil
        factor = max(1, int(ceil(max(mask.shape) / MAX_FLOOD_DISPLAY_DIMENSION)))
        preview = mask[::factor, ::factor]
        preview_transform = dem_transform * Affine.scale(factor, factor)
        preview = preview.astype("uint8")
        polygons = [shape(geom) for geom, value in shapes(
            preview, mask=preview.astype(bool), transform=preview_transform
        ) if value == 1]
        if not polygons:
            return []
        merged = unary_union(polygons)
        gdf = gpd.GeoDataFrame(geometry=[merged], crs=dem_crs).to_crs(WGS_CRS)
        geom = gdf.geometry.iloc[0].simplify(0.0001, preserve_topology=True)
        if geom.is_empty:
            return []
        return [{"type": "Feature", "geometry": mapping(geom),
                 "properties": {"name": "Simulated flood extent (display approximation)"}}]
    except Exception as exc:
        print(f"WARNING: flood polygonization failed: {exc}")
        return []


def get_crop_parameters(payload, crop):
    """Read crop settings. Yield is expressed as kg/ha to match the damage equation.

    For compatibility with older clients, yield_kg_rai is accepted and converted
    using 6.25 rai per hectare.
    """
    defaults = CROP_DEFAULTS[crop]
    custom = (payload.get("crop_parameters") or {}).get(crop, {})
    if "yield_kg_ha" in custom:
        yield_kg_ha = float(custom["yield_kg_ha"])
    elif "yield_kg_rai" in custom:
        yield_kg_ha = float(custom["yield_kg_rai"]) * 6.25
    elif "yield_kg_ha" in defaults:
        yield_kg_ha = float(defaults["yield_kg_ha"])
    else:
        yield_kg_ha = float(defaults.get("yield_kg_rai", 0)) * 6.25

    params = {
        "yield_kg_ha": yield_kg_ha,
        "price_baht_kg": float(custom.get("price_baht_kg", defaults["price_baht_kg"])),
        "duration_days": int(custom.get("duration_days", defaults["duration_days"])),
        # Interpreted as the maximum fragility multiplier for this crop, 0–1.
        "damage_factor": float(custom.get("damage_factor", defaults["damage_factor"]))
    }
    if not np.isfinite(params["yield_kg_ha"]) or params["yield_kg_ha"] < 0:
        raise ValueError(f"Yield for {crop} must be finite and zero or greater")
    if not np.isfinite(params["price_baht_kg"]) or params["price_baht_kg"] < 0:
        raise ValueError(f"Price for {crop} must be finite and zero or greater")
    if params["duration_days"] <= 0:
        raise ValueError(f"Crop duration for {crop} must be greater than zero")
    if not np.isfinite(params["damage_factor"]) or not 0 <= params["damage_factor"] <= 1:
        raise ValueError(f"Fragility multiplier for {crop} must be between 0 and 1")
    return params


def simulate_flood(payload):
    if dem_data is None:
        raise ValueError(f"DEM is unavailable. Check {os.path.basename(DEM_FILE)} and its CRS.")
    if fields_utm is None or fields_web is None:
        raise ValueError("Crop fields are unavailable. Check Fields.shp and its sidecar files.")

    water_level = float(payload.get("river_level", 30.0))
    # Optional user-defined range for water level. Older API clients that do not
    # send these keys remain supported; the current UI sends both values.
    water_level_min = payload.get("water_level_min")
    water_level_max = payload.get("water_level_max")
    if water_level_min is not None or water_level_max is not None:
        if water_level_min is None or water_level_max is None:
            raise ValueError("Enter both the minimum and maximum water levels")
        water_level_min = float(water_level_min)
        water_level_max = float(water_level_max)
        if not np.isfinite(water_level_min) or not np.isfinite(water_level_max):
            raise ValueError("Water-level limits must be finite numbers")
        if water_level_min >= water_level_max:
            raise ValueError("Minimum water level must be lower than maximum water level")
        if not water_level_min <= water_level <= water_level_max:
            raise ValueError("Selected water level must be between the minimum and maximum limits")
    days_until_flood = int(payload.get("days_until_flood", 60))
    if days_until_flood < 0 or days_until_flood > 730:
        raise ValueError("Days until flood must be between 0 and 730")
    try:
        scenario_date = date.fromisoformat(payload.get("scenario_date", date.today().isoformat()))
    except (TypeError, ValueError):
        raise ValueError("Scenario date must be a valid date")
    flood_date = scenario_date + timedelta(days=days_until_flood)
    planting_shifts = payload.get("planting_shifts", {}) or {}
    connected_only = bool(payload.get("river_connected_only", False))
    flood_duration_days = int(payload.get("flood_duration_days", 3))
    if flood_duration_days < 0 or flood_duration_days > 365:
        raise ValueError("Flood duration must be between 0 and 365 days")
    fragility_reference_depth_m = float(payload.get("fragility_reference_depth_m", 1.0))
    fragility_reference_duration_days = float(payload.get("fragility_reference_duration_days", 7.0))
    if (not np.isfinite(fragility_reference_depth_m) or fragility_reference_depth_m <= 0
            or not np.isfinite(fragility_reference_duration_days) or fragility_reference_duration_days <= 0):
        raise ValueError("Fragility reference depth and duration must be finite values greater than zero")

    # Calculate using small raster windows to reduce peak memory.
    flood_mask = create_flood_mask(
        water_level, connected_only=connected_only
    )
    flooded_pixel_count = int(flood_mask.sum())

    fields_dem = (
        fields_utm
        if fields_utm.crs == dem_crs
        else fields_utm.to_crs(dem_crs)
    )

    n_fields = len(fields_dem)

    if n_fields == 0:
        raise ValueError("No valid crop fields available")

    # uint16 is sufficient for 25,521 field IDs.
    if n_fields > np.iinfo(np.uint16).max:
        raise ValueError("Too many fields for uint16 raster labels")

    n_codes = n_fields + 1
    row_by_code = {
        pos + 1: idx
        for pos, idx in enumerate(fields_dem.index)
    }

    total_counts = np.zeros(n_codes, dtype=np.int64)
    flooded_counts = np.zeros(n_codes, dtype=np.int64)
    depth_sum = np.zeros(n_codes, dtype=np.float64)
    fragility_basis_sum = np.zeros(n_codes, dtype=np.float64)

    duration_response = min(
        1.0,
        flood_duration_days / fragility_reference_duration_days
    )

    height, width = dem_data.shape
    fields_sindex = fields_dem.sindex
    block_rows = 128

    for y0 in range(0, height, block_rows):
        y1 = min(y0 + block_rows, height)

        window = Window(0, y0, width, y1 - y0)
        block_dem = dem_data[y0:y1, :]
        valid_block = np.isfinite(block_dem)

        # Only rasterize fields intersecting this block.
        block_geom = box(*window_bounds(window, dem_transform))
        positions = fields_sindex.query(
            block_geom, predicate="intersects"
        )
        positions = np.sort(
            np.asarray(positions, dtype=np.int64)
        )

        shapes_for_block = []
        for pos in positions:
            geom = fields_dem.geometry.iloc[int(pos)]
            if geom is not None and not geom.is_empty:
                shapes_for_block.append((geom, int(pos) + 1))

        if shapes_for_block:
            labels = rasterize(
                shapes_for_block,
                out_shape=(y1 - y0, width),
                transform=window_transform(window, dem_transform),
                fill=0,
                all_touched=False,
                dtype="uint16"
            )
        else:
            labels = np.zeros(
                (y1 - y0, width), dtype=np.uint16
            )

        valid_codes = labels[valid_block]
        if valid_codes.size:
            total_counts += np.bincount(
                valid_codes, minlength=n_codes
            )
        del valid_codes

        flooded_block = (
            flood_mask[y0:y1, :]
            & valid_block
            & (labels > 0)
        )

        if flooded_block.any():
            flood_codes = labels[flooded_block]
            flooded_counts += np.bincount(
                flood_codes, minlength=n_codes
            )

            depths = np.maximum(
                0.0,
                water_level - block_dem[flooded_block]
            ).astype(np.float32, copy=False)

            depth_sum += np.bincount(
                flood_codes,
                weights=depths,
                minlength=n_codes
            )

            depth_response = np.clip(
                depths / fragility_reference_depth_m,
                0.0,
                1.0
            )
            fragility_basis = (
                depth_response * duration_response
            )

            fragility_basis_sum += np.bincount(
                flood_codes,
                weights=fragility_basis,
                minlength=n_codes
            )

            del flood_codes, depths
            del depth_response, fragility_basis

        del labels, block_dem, valid_block, flooded_block
    # The flood mask is not needed after the raster-block pass. Release it before
    # generating response geometries to lower peak memory on Render Free.
    del flood_mask

    stats = {crop: {"area_total_rai": 0.0, "area_affected_rai": 0.0,
                    "value_at_risk_baht": 0.0, "loss_baht": 0.0,
                    "fields_affected": 0} for crop in CROP_TYPES}
    connected_note = ""

    # Keep detailed per-field results on temporary disk. The browser already has
    # agricultural polygon geometries from paginated /api/fields, so the simulation
    # returns attributes only and never duplicates up to hundreds of polygons.
    clear_scenario_cache()
    scenario_id = uuid.uuid4().hex
    metrics_path = SCENARIO_CACHE_PREFIX + scenario_id + ".jsonl"
    metrics_offsets = []
    metrics_fp = open(metrics_path, "w", encoding="utf-8")
    try:
        for code, idx in row_by_code.items():
            total_pixels = int(total_counts[code])
            if total_pixels <= 0:
                continue
            field_row = fields_utm.loc[idx]
            crop = str(field_row.get("Crop_type", "Other Crops"))
            if crop not in CROP_TYPES:
                crop = "Other Crops"
            total_area_m2 = float(field_row.get("area_m2", 0.0))
            total_area_rai = total_area_m2 / 1600.0
            stats[crop]["area_total_rai"] += total_area_rai

            flood_pixels = int(flooded_counts[code])
            if flood_pixels <= 0:
                continue
            fraction = min(1.0, flood_pixels / total_pixels)
            affected_area_rai = total_area_rai * fraction
            mean_depth = float(depth_sum[code] / flood_pixels) if flood_pixels else 0.0
            params = get_crop_parameters(payload, crop)
            shift = int(planting_shifts.get(crop, 0))
            if shift < -730 or shift > 730:
                raise ValueError("Planting shift must be between -730 and 730 days")
            planting_date = scenario_date + timedelta(days=shift)
            crop_age = (flood_date - planting_date).days
            crop_present = 0 <= crop_age <= params["duration_days"]

            # Damage = H(kg/ha) * P(THB/kg) * sum_i[a_i(ha) * f_i(h_i,d_i)].
            # Pixel area is scaled so valid raster cells sum to the field's measured area.
            pixel_area_ha = (total_area_m2 / 10000.0) / total_pixels
            exposed_area_ha = pixel_area_ha * flood_pixels
            if crop_present:
                weighted_fragility_area_ha = (
                    pixel_area_ha * params["damage_factor"] * fragility_basis_sum[code]
                )
                value_at_risk = exposed_area_ha * params["yield_kg_ha"] * params["price_baht_kg"]
                estimated_loss = (
                    params["yield_kg_ha"] * params["price_baht_kg"] * weighted_fragility_area_ha
                )
                vulnerability = estimated_loss / value_at_risk if value_at_risk > 0 else 0.0
            else:
                weighted_fragility_area_ha = 0.0
                value_at_risk = 0.0
                estimated_loss = 0.0
                vulnerability = 0.0

            stats[crop]["area_affected_rai"] += affected_area_rai
            stats[crop]["value_at_risk_baht"] += value_at_risk
            stats[crop]["loss_baht"] += estimated_loss
            stats[crop]["fields_affected"] += 1

            source_row = fields_utm.loc[idx]
            field_id = first_attribute(source_row, ["OBJECTID_1", "OBJECTID", "FID", "field_id", "id"]) or str(idx)
            display_name = first_attribute(source_row, ["field_name_en", "FIELD_NAME", "NAME_EN", "NAME", "name"]) or f"Agricultural field {field_id}"
            feature_properties = {
                "name": display_name,
                "display_name": display_name,
                "field_id": field_id,
                "Crop_type": crop,
                "crop_type": crop,
                "crop_type_en": crop,
                "land_use_en": first_attribute(source_row, ["LU_DES_EN", "LANDUSE_EN", "LAND_USE_EN", "DES_EN", "TYPE_EN"]),
                "land_use_th": first_attribute(source_row, ["LU_DES_TH", "LANDUSE_TH", "LAND_USE_TH", "DES_TH"]),
                "land_use_code": first_attribute(source_row, ["LU_CODE", "LULC_CODE", "LANDUSE_CODE"]),
                "area_m2": round(total_area_m2, 3),
                "area_rai": round(total_area_rai, 3),
                "area_ha": round(total_area_m2 / 10000.0, 5),
                "affected_area_rai": round(affected_area_rai, 3),
                "flood_fraction": round(fraction, 4),
                "mean_flood_depth_m": round(mean_depth, 2),
                "flood_duration_days": flood_duration_days,
                "crop_age_at_flood_days": crop_age,
                "crop_present_at_flood": bool(crop_present),
                "weighted_fragility_area_ha": round(weighted_fragility_area_ha, 5),
                "vulnerability": round(vulnerability, 3),
                "value_at_risk_baht": round(value_at_risk, 2),
                "loss_baht": round(estimated_loss, 2)
            }
            record = {"field_id": str(field_id), "properties": feature_properties}
            metrics_offsets.append(metrics_fp.tell())
            metrics_fp.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    finally:
        metrics_fp.close()

    scenario_registry[scenario_id] = {
        "path": metrics_path,
        "offsets": metrics_offsets,
        "total": len(metrics_offsets)
    }

    for crop_stats in stats.values():
        for key in ("area_total_rai", "area_affected_rai", "value_at_risk_baht", "loss_baht"):
            crop_stats[key] = round(crop_stats[key], 2)

    if connected_only and datasets.get("waterways") is not None:
        connected_note = "Connectivity was requested; if no suitable waterway seed cells existed, the model fell back to the elevation-threshold mask."
    return {
        # All crop polygon geometries already exist in the paginated Crop fields layer.
        # Keep this key for compatibility, but avoid resending duplicate geometry.
        "fields": [],
        "scenario_id": scenario_id,
        "stats": stats,
        "summary": {
            "river_level": water_level,
            "water_level_min": water_level_min,
            "water_level_max": water_level_max,
            "scenario_date": scenario_date.isoformat(),
            "flood_date": flood_date.isoformat(),
            "flood_duration_days": flood_duration_days,
            "fragility_reference_depth_m": fragility_reference_depth_m,
            "fragility_reference_duration_days": fragility_reference_duration_days,
            "total_affected_area_rai": round(sum(s["area_affected_rai"] for s in stats.values()), 2),
            "total_loss_baht": round(sum(s["loss_baht"] for s in stats.values()), 2),
            "total_value_at_risk_baht": round(sum(s["value_at_risk_baht"] for s in stats.values()), 2),
            "flooded_pixel_count": flooded_pixel_count,
            "affected_fields_total": int(sum(v["fields_affected"] for v in stats.values())),
            "affected_fields_metrics_available": len(metrics_offsets),
            "model_note": "Simple DEM elevation-threshold screening model; damage uses a provisional depth-duration fragility function and is not a calibrated hydraulic/agricultural forecast. Statistics include all affected fields; the browser highlights affected polygons in the preloaded agricultural field layer.",
            "connectivity_note": connected_note
        }
    }


@app.route("/api/scenario/<scenario_id>/fields")
def get_scenario_fields(scenario_id):
    """Return affected-field popup attributes in pages without polygon geometry."""
    scenario = scenario_registry.get(scenario_id)
    if scenario is None or not os.path.exists(scenario.get("path", "")):
        return jsonify({"error": "Scenario results expired or are unavailable. Run the scenario again."}), 404
    try:
        offset = max(0, int(request.args.get("offset", 0)))
        limit = int(request.args.get("limit", 250))
    except (TypeError, ValueError):
        return jsonify({"error": "offset and limit must be integers"}), 400
    limit = min(max(limit, 50), 500)
    total = int(scenario["total"])
    end = min(offset + limit, total)
    records = []
    try:
        with open(scenario["path"], "r", encoding="utf-8") as handle:
            for position in range(offset, end):
                handle.seek(scenario["offsets"][position])
                line = handle.readline()
                if line:
                    records.append(json.loads(line))
    except (OSError, ValueError, IndexError) as exc:
        app.logger.exception("Could not read paginated scenario field metrics")
        return jsonify({"error": f"Could not read scenario field results: {exc}"}), 500
    return jsonify({
        "features": records,
        "offset": offset,
        "limit": limit,
        "total": total,
        "done": end >= total
    })


@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    try:
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "Request body must be valid JSON"}), 400
        return jsonify(simulate_flood(payload))
    except (ValueError, TypeError, OverflowError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("Flood simulation failed")
        return jsonify({"error": f"Simulation failed: {exc}"}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
