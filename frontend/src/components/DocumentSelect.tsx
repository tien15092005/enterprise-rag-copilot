import type { LogicalDocument } from "../types/api";

export function DocumentSelect({ documents, value, onChange, optional = false, id = "document" }: {
  documents: LogicalDocument[];
  value: string;
  onChange: (value: string) => void;
  optional?: boolean;
  id?: string;
}) {
  return (
    <select id={id} value={value} onChange={(event) => onChange(event.target.value)} required={!optional}>
      <option value="">{optional ? "All / let agent resolve" : "Select a document"}</option>
      {documents.map((document) => (
        <option value={document.logical_document_id} key={document.logical_document_id}>
          {document.title}
        </option>
      ))}
    </select>
  );
}
