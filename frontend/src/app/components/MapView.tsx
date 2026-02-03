"use client";

import { MapContainer, TileLayer, GeoJSON } from "react-leaflet";
import { FeatureGroup } from "react-leaflet";
import { EditControl } from "react-leaflet-draw";

type Props = {
  geojson?: GeoJSON.FeatureCollection;
  onDraw?: (geom: GeoJSON.Polygon | GeoJSON.MultiPolygon) => void;
};

export default function MapView({ geojson, onDraw }: Props) {
  return (
    <MapContainer
      center={[59.9139, 10.7522]} // Oslo (example)
      zoom={12}
      style={{ height: 320, width: "100%" }}
    >
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution="© OpenStreetMap contributors"
      />

      <FeatureGroup>
        <EditControl
          position="topright"
          draw={{
            rectangle: true,
            polygon: true,
            polyline: false,
            circle: false,
            marker: false,
            circlemarker: false,
          }}
          onCreated={(e: any) => {
            const gj = e.layer.toGeoJSON();
            onDraw?.(gj.geometry);
          }}
        />
      </FeatureGroup>

      {geojson && <GeoJSON data={geojson} />}
    </MapContainer>
  );
}
