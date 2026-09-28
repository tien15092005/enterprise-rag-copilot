import { useCallback, useEffect, useState } from "react";

import { api } from "../api/client";
import type { DocumentVersion, LogicalDocument } from "../types/api";

async function loadDocumentCatalog() {
  const documents = await api.listDocuments();
  const pairs = await Promise.all(
    documents.map(async (document) => [
      document.logical_document_id,
      await api.listVersions(document.logical_document_id),
    ] as const),
  );
  return { documents, versions: Object.fromEntries(pairs) };
}

export function useDocuments() {
  const [documents, setDocuments] = useState<LogicalDocument[]>([]);
  const [versions, setVersions] = useState<Record<string, DocumentVersion[]>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const catalog = await loadDocumentCatalog();
      setDocuments(catalog.documents);
      setVersions(catalog.versions);
    } catch (cause) {
      setError(cause instanceof Error ? cause : new Error("Unable to load documents."));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function loadInitialCatalog() {
      try {
        const catalog = await loadDocumentCatalog();
        if (cancelled) return;
        setDocuments(catalog.documents);
        setVersions(catalog.versions);
      } catch (cause) {
        if (!cancelled) {
          setError(cause instanceof Error ? cause : new Error("Unable to load documents."));
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    void loadInitialCatalog();
    return () => { cancelled = true; };
  }, []);

  return { documents, versions, loading, error, refresh };
}
