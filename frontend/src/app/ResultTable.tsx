"use client";

type Props = {
    modelId: string;
    rows: any[];
    onClose: () => void;
};

function truncate(val: any, col: string): string {
    const str = String(val);
    // truncate geometry columns
    if (col === "wkt_geom" || col === "geom" || col === "geojson_geom") {
        return str.slice(0, 10) + "...";
    }
    return str;
}

export default function ResultTable({ modelId, rows, onClose }: Props) {
    if (!rows || rows.length === 0) return null;

    return (
        <div className="w-full border border-gray-700 rounded-xl overflow-auto max-h-[300px] custom-scroll">
            <div className="flex items-center justify-between px-3 py-2 bg-zinc-800">
                <span className="text-xs text-green-400 font-mono">{modelId}</span>
                <button onClick={onClose} className="text-xs text-gray-400 hover:text-white">
                    Close
                </button>
            </div>
            <table className="w-full text-xs text-white">
                <thead className="bg-zinc-900 sticky top-0">
                    <tr>
                        {Object.keys(rows[0]).map((col) => (
                            <th key={col} className="px-3 py-2 text-left text-gray-400">
                                {col}
                            </th>
                        ))}
                    </tr>
                </thead>
                <tbody>
                    {rows.map((row, i) => (
                        <tr key={i} className="border-t border-gray-800">
                            {Object.entries(row).map(([col, val], j) => (
                                <td key={j} className="px-3 py-2 max-w-[200px] truncate">
                                    {truncate(val, col)}
                                </td>
                            ))}
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}