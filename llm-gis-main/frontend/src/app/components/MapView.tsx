"use client";

import { useEffect, useState } from "react";
import { MapContainer, TileLayer, GeoJSON as RLGeoJSON, FeatureGroup, useMap } from "react-leaflet";
import { EditControl } from "react-leaflet-draw";
import type { Feature, FeatureCollection, Polygon, MultiPolygon } from "geojson";
import { circleMarker, type CircleMarkerOptions, type PathOptions, type StyleFunction } from "leaflet";

type Props = {
  geojson?: FeatureCollection;
  onDraw?: (geom: Polygon | MultiPolygon | null) => void;
  onFeatureSelect?: (feature: Feature | null) => void;
};

type BasemapMode = "light" | "positron" | "dark" | "satellite" | "topographic";

const BASEMAPS: Record<BasemapMode, { label: string; url: string; attribution: string }> = {
  light: {
    label: "Light",
    url: "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    attribution: "© OpenStreetMap contributors",
  },
  positron: {
    label: "Positron",
    url: "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
    attribution: "&copy; OpenStreetMap contributors &copy; CARTO",
  },
  dark: {
    label: "Dark",
    url: "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
    attribution: "&copy; OpenStreetMap contributors &copy; CARTO",
  },
  satellite: {
    label: "Satellite",
    url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attribution: "Tiles © Esri",
  },
  topographic: {
    label: "Topographic",
    url: "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
    attribution: "Map data © OpenStreetMap contributors, SRTM | Map style © OpenTopoMap",
  },
};

const OVERLAY_STYLES: Record<
  BasemapMode,
  {
    color: string;
    fillColor: string;
    fillOpacity: number;
    weight: number;
  }
> = {
  light: { color: "#1d4ed8", fillColor: "#3b82f6", fillOpacity: 0.2, weight: 3 },
  positron: { color: "#0f766e", fillColor: "#14b8a6", fillOpacity: 0.22, weight: 3 },
  dark: { color: "#f8fafc", fillColor: "#38bdf8", fillOpacity: 0.28, weight: 3 },
  satellite: { color: "#fde047", fillColor: "#facc15", fillOpacity: 0.22, weight: 3 },
  topographic: { color: "#7c3aed", fillColor: "#a855f7", fillOpacity: 0.2, weight: 3 },
};

const POINT_STYLES: Record<BasemapMode, CircleMarkerOptions> = {
  light: {
    radius: 8,
    color: "#0f172a",
    weight: 2,
    fillColor: "#2563eb",
    fillOpacity: 0.95,
  },
  positron: {
    radius: 8,
    color: "#134e4a",
    weight: 2,
    fillColor: "#14b8a6",
    fillOpacity: 0.95,
  },
  dark: {
    radius: 8,
    color: "#e2e8f0",
    weight: 2,
    fillColor: "#38bdf8",
    fillOpacity: 0.95,
  },
  satellite: {
    radius: 8,
    color: "#422006",
    weight: 2,
    fillColor: "#facc15",
    fillOpacity: 0.98,
  },
  topographic: {
    radius: 8,
    color: "#2e1065",
    weight: 2,
    fillColor: "#c084fc",
    fillOpacity: 0.95,
  },
};

function MapResizeHandler() {
  const map = useMap();

  useEffect(() => {
    let disposed = false;
    const container = map.getContainer();

    const refreshMapSize = () => {
      window.requestAnimationFrame(() => {
        const leafletMap = map as typeof map & { _loaded?: boolean };
        if (disposed) return;
        if (!container?.isConnected) return;
        if (!leafletMap._loaded) return;
        if (!map.getPanes()?.mapPane) return;
        map.invalidateSize({ pan: false });
      });
    };

    map.whenReady(() => {
      if (!disposed) refreshMapSize();
    });

    if (typeof ResizeObserver === "undefined") {
      window.addEventListener("resize", refreshMapSize);
      return () => {
        disposed = true;
        window.removeEventListener("resize", refreshMapSize);
      };
    }

    const observer = new ResizeObserver(() => {
      refreshMapSize();
    });

    observer.observe(container);

    return () => {
      disposed = true;
      observer.disconnect();
    };
  }, [map]);

  return null;
}

export default function MapView({ geojson, onDraw, onFeatureSelect }: Props) {
  const [basemapMode, setBasemapMode] = useState<BasemapMode>("light");
  const [selectedFeatureId, setSelectedFeatureId] = useState<string | null>(null);
  const basemap = BASEMAPS[basemapMode];
  const overlayStyle = OVERLAY_STYLES[basemapMode];
  const pointStyle = POINT_STYLES[basemapMode];
  const featureId = (feature?: Feature | null) => {
    if (!feature) return null;
    const props = (feature.properties ?? {}) as Record<string, unknown>;
    return String(
      props.id ?? props.objid ?? props.gid ?? props.name ?? JSON.stringify(props).slice(0, 120)
    );
  };
  const geojsonStyle: StyleFunction = (feature) => {
    const isSelected = featureId(feature as Feature) === selectedFeatureId;
    return {
      color: isSelected ? "#f59e0b" : overlayStyle.color,
      fillColor: isSelected ? "#f59e0b" : overlayStyle.fillColor,
      fillOpacity: isSelected ? 0.32 : overlayStyle.fillOpacity,
      weight: isSelected ? overlayStyle.weight + 2 : overlayStyle.weight,
    } satisfies PathOptions;
  };

  return (
    <div className="relative h-full min-h-[530px] w-full">
      <div className="absolute right-3 top-40 z-[700] flex flex-col gap-1">
        {(Object.entries(BASEMAPS) as [BasemapMode, (typeof BASEMAPS)[BasemapMode]][]).map(
          ([key, value], index) => (
            <button
              key={key}
              type="button"
              onClick={() => setBasemapMode(key)}
              title={`Switch to ${value.label.toLowerCase()} basemap`}
              aria-label={`Switch to ${value.label} basemap`}
              className={[
                "flex h-[26px] w-[26px] items-center justify-center rounded border border-[#ccc] text-[12px] font-medium leading-none shadow-[0_1px_5px_rgba(0,0,0,0.4)]",
                index === 2 ? "mt-1" : "",
                key === basemapMode
                  ? "bg-[#f4f4f4] text-[#222]"
                  : "bg-white text-[#222] hover:bg-[#f4f4f4]",
              ].join(" ")}
              aria-pressed={key === basemapMode}
            >
              {value.label.charAt(0)}
            </button>
          ),
        )}
      </div>

      <MapContainer
        center={[59.9139, 10.7522]}
        zoom={12}
        className="h-full w-full rounded-lg"
      >
        <MapResizeHandler />
        <TileLayer
          url={basemap.url}
          attribution={basemap.attribution}
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
            onDeleted={() => {
              onDraw?.(null);
            }}
          />
        </FeatureGroup>

        {geojson && (
          <RLGeoJSON
            data={geojson}
            style={geojsonStyle}
            pointToLayer={(feature, latlng) => {
              const isSelected = featureId(feature as Feature) === selectedFeatureId;
              return circleMarker(latlng, {
                ...pointStyle,
                radius: isSelected ? (pointStyle.radius ?? 8) + 2 : pointStyle.radius,
                color: isSelected ? "#f59e0b" : pointStyle.color,
                fillColor: isSelected ? "#fbbf24" : pointStyle.fillColor,
              });
            }}
            onEachFeature={(feature, layer) => {
              layer.on("click", () => {
                const id = featureId(feature as Feature);
                setSelectedFeatureId(id);
                onFeatureSelect?.(feature as Feature);
              });
            }}
          />
        )}
      </MapContainer>
    </div>
  );
}
