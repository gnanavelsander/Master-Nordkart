"use client";

import { MapContainer, TileLayer, GeoJSON as RLGeoJSON, FeatureGroup } from "react-leaflet";
import { EditControl } from "react-leaflet-draw";
import type { FeatureCollection, Polygon, MultiPolygon } from "geojson";

type Props = {
  geojson?: FeatureCollection;
  onDraw?: (geom: Polygon | MultiPolygon) => void;
};

export default function MapView({ geojson, onDraw }: Props) {
  return (
    <MapContainer
      center={[59.9139, 10.7522]}
      zoom={12}
      className="h-[400px] w-full rounded-lg"
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
          onCreated={(e: unknown) => {
            const layer = (e as { layer: { toGeoJSON: () => any } }).layer;
            const gj = layer.toGeoJSON();
            onDraw?.(gj.geometry);
          }}
        />
      </FeatureGroup>

      {geojson && <RLGeoJSON data={geojson} />}
    </MapContainer>
  );
}
