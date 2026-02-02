

from shapely import wkt
import folium
from shapely.geometry import Polygon, MultiPolygon


def normalize_geom(g):
    if isinstance(g, Polygon):
        return [g]
    if isinstance(g, MultiPolygon):
        return list(g.geoms)
    return []

def showMap(df):
    first_geom = wkt.loads(df["wkt_geom"].iloc[0])
    m = folium.Map(location=[first_geom.centroid.y, first_geom.centroid.x], zoom_start=16)    
    
    for w in df["wkt_geom"]:
        geom = wkt.loads(w)
        for poly in normalize_geom(geom):
            folium.GeoJson(
                poly.__geo_interface__,
                style_function=lambda x: {
                    "color": "red",
                    "weight": 2,
                    "fillColor": "yellow",
                    "fillOpacity": 0.3,
                },
            ).add_to(m)
    
    return m