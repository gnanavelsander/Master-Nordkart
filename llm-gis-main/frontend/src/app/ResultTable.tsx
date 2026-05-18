"use client";

type Props = {
  modelId: string;
  rows: any[];
  onClose: () => void;
};

function truncate(val: any, col: string): string {
  const str = String(val);

  if (col === "wkt_geom" || col === "geom" || col === "geojson_geom") {
    return str.slice(0, 10) + "...";
  }

  return str;
}

export default function ResultTable({ modelId, rows, onClose }: Props) {
  if (!rows || rows.length === 0) return null;

  return (
    <div className="h-full w-full overflow-hidden rounded-[28px] border border-[#034E31]/10 bg-white/95 shadow-[0_24px_70px_rgba(3,78,49,0.18)] backdrop-blur">
      <div className="flex items-center justify-between border-b border-[#034E31]/10 bg-[#F4FBF7] px-5 py-4">
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[#24BD76]">
            Result table
          </p>
          <p className="mt-1 text-sm font-semibold text-[#034E31]">
            {modelId}
          </p>
        </div>

        <button
          onClick={onClose}
          className="rounded-full border border-[#034E31]/10 bg-white px-4 py-2 text-xs font-medium text-[#034E31] shadow-sm transition hover:bg-[#C8EBDC]/45"
        >
          Close
        </button>
      </div>

      <div className="h-[calc(100%-73px)] overflow-auto custom-scroll">
        <table className="w-full border-collapse text-xs text-[#034E31]">
          <thead className="sticky top-0 z-10 bg-white">
            <tr>
              {Object.keys(rows[0]).map((col) => (
                <th
                  key={col}
                  className="border-b border-[#034E31]/10 px-4 py-3 text-left font-semibold text-[#034E31]/70"
                >
                  {col}
                </th>
              ))}
            </tr>
          </thead>

          <tbody>
            {rows.map((row, i) => (
              <tr
                key={i}
                className="border-t border-[#034E31]/5 transition hover:bg-[#C8EBDC]/25"
              >
                {Object.entries(row).map(([col, val], j) => (
                  <td
                    key={j}
                    className="max-w-[240px] truncate px-4 py-3 align-top"
                    title={String(val)}
                  >
                    {truncate(val, col)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}