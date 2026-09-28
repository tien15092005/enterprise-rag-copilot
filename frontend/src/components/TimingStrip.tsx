export function TimingStrip({ values }: { values: Record<string, number> }) {
  return (
    <div className="timing-strip">
      {Object.entries(values)
        .filter(([, value]) => value > 0)
        .map(([label, value]) => (
          <span key={label}><b>{label.replaceAll("_", " ")}</b> {value.toFixed(1)} ms</span>
        ))}
    </div>
  );
}
