"use client";

import { useState } from "react";
import MapView from "./components/MapView";

export default function Page() {
  const [aoi, setAoi] = useState<GeoJSON.Polygon | GeoJSON.MultiPolygon | undefined>();
  const [geojson, setGeojson] = useState<GeoJSON.FeatureCollection | undefined>();

  return (
    <div style={{ padding: 16 }}>
      <h2>Map + Draw</h2>
      <MapView
        geojson={geojson}
        onDraw={(geom) => {
          setAoi(geom);
          console.log("AOI geometry:", geom);
        }}
      />
    </div>
  );
}
