import { useEffect, useMemo, useState } from "react";

const API_URL =
    import.meta.env.VITE_API_URL ||
    (typeof window !== "undefined" ? window.location.origin : "http://localhost:8000");

type Stats = {
    sources: number;
    chunks: number;
};

type Citation = {
    source: string;
    content: string;
    score?: number | null;
};

type Timings = {
    embed_ms?: number | null;
    search_ms?: number | null;
    rerank_ms?: number | null;
    llm_ms?: number | null;
    total_ms?: number | null;
};

type QueryResponse = {
    answer: string;
    citations: Citation[];
    blocked?: boolean;
    reason?: string | null;
    timings?: Timings | null;
};

const gradeMs = (ms: number | null | undefined): string => {
    if (ms === null || ms === undefined) return "t-unknown";
    if (ms < 200) return "t-teal";
    if (ms < 500) return "t-green";
    if (ms < 2000) return "t-yellow";
    if (ms < 5000) return "t-orange";
    return "t-red";
};

const fmtMs = (ms: number | null | undefined): string => {
    if (ms === null || ms === undefined) return "—";
    if (ms < 1000) return `${Math.round(ms)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
};

type IngestKind = "ok" | "err" | "info" | "";

const defaultStats: Stats = { sources: 0, chunks: 0 };

export default function App() {
    const apiBase = useMemo(() => API_URL.replace(/\/$/, ""), []);
    const [health, setHealth] = useState<"checking" | "online" | "offline">("checking");
    const [stats, setStats] = useState<Stats>(defaultStats);
    const [files, setFiles] = useState<File[]>([]);
    const [noteSource, setNoteSource] = useState("note");
    const [noteText, setNoteText] = useState("");
    const [ingestStatus, setIngestStatus] = useState("");
    const [ingestKind, setIngestKind] = useState<IngestKind>("");
    const [question, setQuestion] = useState("");
    const [answer, setAnswer] = useState("");
    const [citations, setCitations] = useState<Citation[]>([]);
    const [timings, setTimings] = useState<Timings | null>(null);
    const [blocked, setBlocked] = useState(false);
    const [blockedReason, setBlockedReason] = useState<string | null>(null);
    const [busy, setBusy] = useState(false);
    const [dragActive, setDragActive] = useState(false);

    useEffect(() => {
        void refreshHealth();
        void refreshStats();
    }, []);

    const refreshHealth = async () => {
        try {
            const response = await fetch(`${apiBase}/health`);
            setHealth(response.ok ? "online" : "offline");
        } catch {
            setHealth("offline");
        }
    };

    const refreshStats = async () => {
        try {
            const response = await fetch(`${apiBase}/stats`);
            if (!response.ok) {
                return;
            }
            const data = (await response.json()) as Stats;
            setStats(data);
        } catch {
            setStats(defaultStats);
        }
    };

    const handleFileChange = (incoming: FileList | null) => {
        if (!incoming) {
            return;
        }
        setFiles((current) => [...current, ...Array.from(incoming)]);
    };

    const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
        event.preventDefault();
        setDragActive(false);
        if (event.dataTransfer.files.length) {
            handleFileChange(event.dataTransfer.files);
        }
    };

    const handleDrag = (event: React.DragEvent<HTMLDivElement>) => {
        event.preventDefault();
        setDragActive(event.type === "dragover");
    };

    const setStatus = (text: string, kind: IngestKind) => {
        setIngestStatus(text);
        setIngestKind(kind);
    };

    const uploadFiles = async () => {
        if (!files.length) {
            setStatus("Select at least one file first.", "err");
            return;
        }
        setBusy(true);
        setStatus("Uploading and ingesting...", "info");
        const formData = new FormData();
        files.forEach((file) => formData.append("files", file));

        try {
            const response = await fetch(`${apiBase}/ingest/files`, {
                method: "POST",
                body: formData
            });
            if (!response.ok) {
                const errorText = await response.text();
                setStatus(`Upload failed: ${errorText}`, "err");
                return;
            }
            const result = await response.json();
            setStatus(
                `Ingested ${result.documents} document(s) → ${result.chunks} chunks.`,
                "ok"
            );
            setFiles([]);
            await refreshStats();
        } catch {
            setStatus("Upload failed. Check the API and try again.", "err");
        } finally {
            setBusy(false);
        }
    };

    const ingestNote = async () => {
        if (!noteText.trim()) {
            setStatus("Paste or type content before ingesting.", "err");
            return;
        }
        setBusy(true);
        setStatus("Ingesting note...", "info");
        const payload = {
            documents: [
                {
                    source: noteSource || "note",
                    content: noteText
                }
            ]
        };

        try {
            const response = await fetch(`${apiBase}/ingest`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(payload)
            });
            if (!response.ok) {
                const errorText = await response.text();
                setStatus(`Ingest failed: ${errorText}`, "err");
                return;
            }
            const result = await response.json();
            setStatus(
                `Ingested ${result.documents} document(s) → ${result.chunks} chunks.`,
                "ok"
            );
            setNoteText("");
            await refreshStats();
        } catch {
            setStatus("Ingest failed. Check the API and try again.", "err");
        } finally {
            setBusy(false);
        }
    };

    const runQuery = async () => {
        if (!question.trim()) {
            return;
        }
        setBusy(true);
        setAnswer("");
        setCitations([]);
        setTimings(null);
        setBlocked(false);
        setBlockedReason(null);

        try {
            const response = await fetch(`${apiBase}/query/stream`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question })
            });
            if (!response.ok || !response.body) {
                const errorText = await response.text();
                setAnswer(`Query failed: ${errorText}`);
                return;
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = "";
            let answerText = "";

            const handleEvent = (event: any) => {
                if (event.type === "meta") {
                    setCitations(event.citations || []);
                } else if (event.type === "token") {
                    answerText += event.value || "";
                    setAnswer(answerText);
                } else if (event.type === "done") {
                    setTimings(event.timings || null);
                } else if (event.type === "blocked") {
                    setBlocked(true);
                    setBlockedReason(event.reason ?? null);
                    setTimings(event.timings || null);
                } else if (event.type === "error") {
                    setAnswer(`Query failed: ${event.detail}`);
                }
            };

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });
                let sep;
                while ((sep = buffer.indexOf("\n\n")) !== -1) {
                    const frame = buffer.slice(0, sep);
                    buffer = buffer.slice(sep + 2);
                    const line = frame.trim();
                    if (!line.startsWith("data:")) continue;
                    const payload = line.slice(5).trim();
                    if (!payload) continue;
                    try {
                        handleEvent(JSON.parse(payload));
                    } catch {
                        // ignore malformed frames
                    }
                }
            }
        } catch {
            setAnswer("Query failed. Check the API and try again.");
        } finally {
            setBusy(false);
        }
    };

    const healthLabel =
        health === "online" ? "operational" : health === "offline" ? "unreachable" : "checking";

    return (
        <div className="app">
            <nav className="nav">
                <div className="brand">
                    <span className="brand-mark">A</span>
                    <span className="brand-name">aegis-rag-lab</span>
                    <span className="brand-tag">/ v0.1</span>
                </div>
                <div className="nav-meta">
                    <span className="endpoint">{apiBase}</span>
                    <span className={`pill ${health}`}>
                        <span className="dot" />
                        {healthLabel}
                    </span>
                </div>
            </nav>

            <header className="hero">
                <div className="hero-left">
                    <p className="eyebrow">Agentic RAG · LangGraph · pgvector</p>
                    <h1>
                        Security-first retrieval, <em>with guardrails and evals.</em>
                    </h1>
                    <p className="subtitle">
                        Upload documents, run queries through a guarded LangGraph pipeline, and
                        inspect citations. Built for reproducible evaluation and zero-config local
                        runs.
                    </p>
                </div>

                <aside className="status-panel">
                    <div className="status-panel-header">
                        <span>// runtime</span>
                        <span className={`pill ${health}`}>
                            <span className="dot" />
                            {healthLabel}
                        </span>
                    </div>
                    <div className="status-rows">
                        <div className="status-row">
                            <span className="label">sources</span>
                            <span className="value">{stats.sources}</span>
                        </div>
                        <div className="status-row">
                            <span className="label">chunks</span>
                            <span className="value">{stats.chunks}</span>
                        </div>
                        <div className="status-row">
                            <span className="label">endpoint</span>
                            <span className="value">{apiBase.replace(/^https?:\/\//, "")}</span>
                        </div>
                    </div>
                </aside>
            </header>

            <main className="grid">
                <section className="card">
                    <div className="card-header">
                        <span className="card-title">Ingest</span>
                        <span className="card-meta">POST /ingest · /ingest/files</span>
                    </div>
                    <div className="card-body">
                        <div
                            className={`dropzone ${dragActive ? "active" : ""}`}
                            onDrop={handleDrop}
                            onDragOver={handleDrag}
                            onDragLeave={handleDrag}
                        >
                            <input
                                id="file-input"
                                type="file"
                                multiple
                                accept=".md,.txt,.jsonl,.pdf,.docx,.html,.htm"
                                onChange={(event) => handleFileChange(event.target.files)}
                            />
                            <label htmlFor="file-input">
                                <span className="drop-title">Drop files or click to upload</span>
                                <span className="drop-subtitle">.md · .txt · .jsonl · .pdf · .docx · .html</span>
                            </label>
                        </div>

                        <div className="file-list">
                            {files.length ? (
                                files.map((file) => (
                                    <div key={`${file.name}-${file.size}`} className="file-item">
                                        <span>{file.name}</span>
                                        <span className="mono">
                                            {Math.round(file.size / 1024)} KB
                                        </span>
                                    </div>
                                ))
                            ) : (
                                <p className="muted">No files queued.</p>
                            )}
                        </div>

                        <div className="actions">
                            <button onClick={uploadFiles} disabled={busy}>
                                Upload &amp; ingest
                            </button>
                            <button
                                className="ghost"
                                onClick={() => setFiles([])}
                                disabled={busy || !files.length}
                            >
                                Clear
                            </button>
                        </div>

                        <div className="divider" />

                        <h3 className="sub-head">// quick note</h3>
                        <div className="stack">
                            <input
                                value={noteSource}
                                onChange={(event) => setNoteSource(event.target.value)}
                                placeholder="source label (e.g. policy.md)"
                            />
                            <textarea
                                value={noteText}
                                onChange={(event) => setNoteText(event.target.value)}
                                placeholder="Paste a paragraph or a security note..."
                                rows={6}
                            />
                            <div className="actions">
                                <button onClick={ingestNote} disabled={busy}>
                                    Ingest note
                                </button>
                            </div>
                        </div>

                        {ingestStatus ? (
                            <p className={`status-text ${ingestKind}`}>{ingestStatus}</p>
                        ) : null}
                    </div>
                </section>

                <section className="card">
                    <div className="card-header">
                        <span className="card-title">Query</span>
                        <span className="card-meta">POST /query</span>
                    </div>
                    <div className="card-body">
                        <div className="stack">
                            <textarea
                                value={question}
                                onChange={(event) => setQuestion(event.target.value)}
                                placeholder="Ask a question about your indexed documents..."
                                rows={4}
                            />
                            <div className="actions">
                                <button onClick={runQuery} disabled={busy}>
                                    Run query
                                </button>
                                <button
                                    className="ghost"
                                    onClick={() => {
                                        setQuestion("");
                                        setAnswer("");
                                        setCitations([]);
                                        setTimings(null);
                                        setBlocked(false);
                                        setBlockedReason(null);
                                    }}
                                    disabled={busy}
                                >
                                    Reset
                                </button>
                            </div>
                        </div>

                        {blocked ? (
                            <div className="alert">
                                <strong>Blocked by guardrails.</strong>
                                <p>Reason: {blockedReason || "policy"}</p>
                            </div>
                        ) : null}

                        <h3 className="sub-head">// answer</h3>
                        <div className="answer">
                            {answer ? <p>{answer}</p> : <p className="muted">No answer yet.</p>}
                        </div>

                        {timings ? (
                            <div className="timings">
                                <span className={`chip ${gradeMs(timings.embed_ms)}`}>
                                    embed <span className="chip-val">{fmtMs(timings.embed_ms)}</span>
                                </span>
                                <span className="chip-sep">+</span>
                                <span className={`chip ${gradeMs(timings.search_ms)}`}>
                                    search <span className="chip-val">{fmtMs(timings.search_ms)}</span>
                                </span>
                                {typeof timings.rerank_ms === "number" ? (
                                    <>
                                        <span className="chip-sep">+</span>
                                        <span className={`chip ${gradeMs(timings.rerank_ms)}`}>
                                            rerank <span className="chip-val">{fmtMs(timings.rerank_ms)}</span>
                                        </span>
                                    </>
                                ) : null}
                                <span className="chip-sep">+</span>
                                <span className={`chip ${gradeMs(timings.llm_ms)}`}>
                                    llm <span className="chip-val">{fmtMs(timings.llm_ms)}</span>
                                </span>
                                <span className="chip-sep">=</span>
                                <span className={`chip ${gradeMs(timings.total_ms)} chip-total`}>
                                    total <span className="chip-val">{fmtMs(timings.total_ms)}</span>
                                </span>
                            </div>
                        ) : null}

                        <h3 className="sub-head">// citations</h3>
                        <div className="citations">
                            {citations.length ? (
                                <ul>
                                    {citations.map((citation, index) => (
                                        <li
                                            key={`${citation.source}-${index}`}
                                            className="citation"
                                        >
                                            <div className="citation-head">
                                                <span className="mono">
                                                    {citation.source}
                                                </span>
                                                {typeof citation.score === "number" ? (
                                                    <span className="citation-score">
                                                        {citation.score.toFixed(3)}
                                                    </span>
                                                ) : null}
                                            </div>
                                            <div className="citation-body">
                                                {citation.content}
                                            </div>
                                        </li>
                                    ))}
                                </ul>
                            ) : (
                                <p className="muted">No citations yet.</p>
                            )}
                        </div>
                    </div>
                </section>
            </main>
        </div>
    );
}
