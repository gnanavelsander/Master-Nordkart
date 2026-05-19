"use client";

import dynamic from "next/dynamic";
import type { ComponentProps } from "react";
import type MapView from "./MapView";

const MapViewNoSSR = dynamic(() => import("./MapView"), { ssr: false });

type Props = ComponentProps<typeof MapViewNoSSR>;

export default function MapClient(props: Props) {
  return <MapViewNoSSR {...props} />;
}
