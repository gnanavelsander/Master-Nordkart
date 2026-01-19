import requests
import json

# def lookup_kommune(name: str):
#     url = "https://api.kartverket.no/kommuneinfo/v1/sok"
#     params = {"knavn": name}

#     r = requests.get(url, params=params, timeout=10)
#     r.raise_for_status()
#     data = r.json()

#     avgrensningsboks = data["kommuner"][0]["avgrensningsboks"]
#     coords = avgrensningsboks["coordinates"][0]

#     lons = [pt[0] for pt in coords]
#     lats = [pt[1] for pt in coords]

#     return min(lons), min(lats), max(lons), max(lats)


def city_bbox_where_clause(min_lon, min_lat, max_lon, max_lat):
    return f"""
a.geom && ST_Transform(
    ST_MakeEnvelope(
        {min_lon}, {min_lat},
        {max_lon}, {max_lat},
        4326
    ),
    25833
)
""".strip()


def lookup_kommuneNr(name: str):
    url = "https://api.kartverket.no/kommuneinfo/v1/sok"
    params = {"knavn": name}

    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()

    fylkeNr = data["kommuner"][0]["kommunenummer"]
 

    return fylkeNr

def lookup_kommuneGeo(nr: str):
    # url = f"https://api.test.kartverket.no/kommuneinfo/v1/fylker/{nr}/omrade"

    url = f"https://api.test.kartverket.no/kommuneinfo/v1/kommuner/{nr}/omrade"

    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()

    fylkeNr = data["omrade"]["coordinates"]
 

    return fylkeNr

def multipolygon_to_bbox(coords):
    minx = miny = float("inf")
    maxx = maxy = float("-inf")

    for polygon in coords:     
        for ring in polygon:   
            for lon, lat in ring:
                minx = min(minx, lon)
                miny = min(miny, lat)
                maxx = max(maxx, lon)
                maxy = max(maxy, lat)

    return minx, miny, maxx, maxy



def city_polygon_where_clause(coords_4326, target_srid=25833, geom_col="a.geom"):
    geojson = json.dumps({"type": "MultiPolygon", "coordinates": coords_4326})
    return f"""
ST_Intersects(
  {geom_col},
  ST_Transform(ST_GeomFromGeoJSON('{geojson}'), {target_srid})
)
""".strip()






# if __name__ == "__main__":
#     print(lookup_kommuneNr("Oslo"))
    
    

#     # print(lookup_kommuneGeo("4202"))

#     # bbox = lookup_kommuneGeo("42")
#     # print(multipolygon_to_bbox(bbox))