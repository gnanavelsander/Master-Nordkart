import requests
from langchain.tools import tool

@tool
def kommune_where_clause(name: str) -> str:
    """
    Return a PostGIS WHERE clause that restricts results to a Norwegian municipality
    using a bounding box derived from Kartverket APIs.
    """

    # Look up municipality number by name
    r = requests.get(
        "https://api.kartverket.no/kommuneinfo/v1/sok",
        params={"knavn": name},
        timeout=10,
    )
    r.raise_for_status()
    kommune_nr = r.json()["kommuner"][0]["kommunenummer"]

    # Fetch municipality geometry (MultiPolygon)
    r = requests.get(
        f"https://api.test.kartverket.no/kommuneinfo/v1/kommuner/{kommune_nr}/omrade",
        timeout=10,
    )
    r.raise_for_status()
    coords = r.json()["omrade"]["coordinates"]

    # Compute bounding box from MultiPolygon
    minx = miny = float("inf")
    maxx = maxy = float("-inf")

    for polygon in coords:
        for ring in polygon:
            for lon, lat in ring:
                minx = min(minx, lon)
                miny = min(miny, lat)
                maxx = max(maxx, lon)
                maxy = max(maxy, lat)

    # Return PostGIS WHERE clause using transformed bounding box
    return f"""
a.geom && ST_Transform(
  ST_MakeEnvelope({minx}, {miny}, {maxx}, {maxy}, 4326),
  25833
)
""".strip()




