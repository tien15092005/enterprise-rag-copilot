interface StatusBadgeProps {
  value: string;
  tone?: "neutral" | "success" | "warning" | "danger" | "info";
}

const inferredTone: Record<string, StatusBadgeProps["tone"]> = {
  active: "success",
  ready: "success",
  indexed: "success",
  succeeded: "success",
  failed: "danger",
  deleting: "danger",
  processing: "warning",
  indexing: "warning",
  pending: "warning",
  current: "info",
};

export function StatusBadge({ value, tone }: StatusBadgeProps) {
  return (
    <span className={`badge badge--${tone ?? inferredTone[value] ?? "neutral"}`}>
      {value.replaceAll("_", " ")}
    </span>
  );
}
