import { useEffect, useState } from "react";

import { AgentPage } from "./pages/AgentPage";
import { AskPage } from "./pages/AskPage";
import { ComparePage } from "./pages/ComparePage";
import { DocumentsPage } from "./pages/DocumentsPage";
import { OverviewPage } from "./pages/OverviewPage";
import { SystemPage } from "./pages/SystemPage";

const routes = [
  { id: "overview", label: "Overview" },
  { id: "documents", label: "Documents" },
  { id: "ask", label: "Ask / RAG" },
  { id: "compare", label: "Compare versions" },
  { id: "agent", label: "Agent" },
  { id: "system", label: "Evaluation" },
] as const;
type Route = typeof routes[number]["id"];

function currentRoute(): Route {
  const hash = window.location.hash.slice(1);
  return routes.some((route) => route.id === hash) ? hash as Route : "overview";
}

export default function App() {
  const [route, setRoute] = useState<Route>(currentRoute);
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    const update = () => { setRoute(currentRoute()); setMenuOpen(false); window.scrollTo({ top: 0 }); };
    window.addEventListener("hashchange", update);
    return () => window.removeEventListener("hashchange", update);
  }, []);

  function navigate(target: string) { window.location.hash = target; }
  const page = route === "documents" ? <DocumentsPage /> : route === "ask" ? <AskPage /> : route === "compare" ? <ComparePage /> : route === "agent" ? <AgentPage /> : route === "system" ? <SystemPage /> : <OverviewPage navigate={navigate} />;

  return (
    <div className="app-shell">
      <header className="topbar">
        <button className="brand" onClick={() => navigate("overview")}><span className="brand__mark">EK</span><span><strong>Enterprise Knowledge</strong><small>Copilot</small></span></button>
        <button className="mobile-menu" aria-label="Toggle navigation" aria-expanded={menuOpen} onClick={() => setMenuOpen((value) => !value)}>☰</button>
        <nav className={menuOpen ? "is-open" : ""} aria-label="Primary navigation">{routes.map((item) => <a href={`#${item.id}`} className={route === item.id ? "is-active" : ""} key={item.id}>{item.label}</a>)}</nav>
        <div className="system-status"><span /> Backend API</div>
      </header>
      <main className={`main-content main-content--${route}`}>{page}</main>
      <footer><span>Enterprise Knowledge Copilot</span><span>Version-aware · bilingual · grounded</span></footer>
    </div>
  );
}
