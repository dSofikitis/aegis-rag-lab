import { useEffect, useMemo, useState } from "react";

const API_URL =
    import.meta.env.VITE_API_URL ||
    (typeof window !== "undefined" ? window.location.origin : "http://localhost:8000");

type Stats = {
    sources: number;
    chunks: number;
};

type QueryResponse = {
    answer: string;
    citations: string[];
    blocked?: boolean;
    reason?: string | null;
};

const defaultStats: Stats = { sources: 0, chunks: 0 };

export default function App() {
    const apiBase = useMemo(() => API_URL.replace(/\/$/, ""), []);
    const [health, setHealth] = useState("checking");
    const [stats, setStats] = useState<Stats>(defaultStats);
    const [files, setFiles] = useState<File[]>([]);
    const [noteSource, setNoteSource] = useState("note");
    const [noteText, setNoteText] = useState("");
    const [ingestStatus, setIngestStatus] = useState("");
    const [question, setQuestion] = useState("");
    const [answer, setAnswer] = useState("");
    const [citations, setCitations] = useState<string[]>([]);
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

    const uploadFiles = async () => {
        if (!files.length) {
            setIngestStatus("Select at least one file first.");
            return;
        }
        setBusy(true);
        setIngestStatus("Uploading and ingesting...");
        const formData = new FormData();
        files.forEach((file) => formData.append("files", file));

        try {
            const response = await fetch(`${apiBase}/ingest/files`, {
                method: "POST",
                body: formData
            });
            if (!response.ok) {
                const errorText = await response.text();
                setIngestStatus(`Upload failed: ${errorText}`);
                return;
            }
            const result = await response.json();
            setIngestStatus(
                `Ingested ${result.documents} documents (${result.chunks} chunks).`
            );
            setFiles([]);
            await refreshStats();
        } catch (error) {
            setIngestStatus("Upload failed. Check the API and try again.");
        } finally {
            setBusy(false);
        }
    };

    const ingestNote = async () => {
        if (!noteText.trim()) {
            setIngestStatus("Paste or type content before ingesting.");
            return;
        }
        setBusy(true);
        setIngestStatus("Ingesting note...");
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
                setIngestStatus(`Ingest failed: ${errorText}`);
                return;
            }
            const result = await response.json();
            setIngestStatus(
                `Ingested ${result.documents} documents (${result.chunks} chunks).`
            );
            setNoteText("");
            await refreshStats();
        } catch {
            setIngestStatus("Ingest failed. Check the API and try again.");
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
        setBlocked(false);
        setBlockedReason(null);

        try {
            const response = await fetch(`${apiBase}/query`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ question })
            });
            if (!response.ok) {
                const errorText = await response.text();
                setAnswer(`Query failed: ${errorText}`);
                return;
            }
            const result = (await response.json()) as QueryResponse;
            setAnswer(result.answer || "");
            setCitations(result.citations || []);
            setBlocked(Boolean(result.blocked));
            setBlockedReason(result.reason ?? null);
        } catch {
            setAnswer("Query failed. Check the API and try again.");
        } finally {
            setBusy(false);
        }
    };

    return (
        <div className="app">
            <header className="hero">
                <div>
                    <p className="eyebrow">Aegis RAG Lab</p>
                    <h1>Security-first RAG, with guardrails and evals.</h1>
                    <p className="subtitle">
                        Upload documents, ask questions, and inspect citations instantly.
                    </p>
                </div>
                <div className="status-card">
                    <div className="status-row">
                        <span className={`status-dot ${health}`}></span>
                        <span className="status-label">API {health}</span>
                    </div>
                    <div className="status-row">
                        <span className="stat">
                            <strong>{stats.sources}</strong> sources
                        </span>
                        <span className="stat">
                            <strong>{stats.chunks}</strong> chunks
                        </span>
                    </div>
                    <div className="status-row mono">{apiBase}</div>
                </div>
            </header>

            <main className="grid">
                <section className="card">
                    <h2>Ingest documents</h2>
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
                            accept=".md,.txt,.jsonl"
                            onChange={(event) => handleFileChange(event.target.files)}
                        />
                        <label htmlFor="file-input">
                            <span className="drop-title">Drag files or click to upload</span>
                            <span className="drop-subtitle">MD, TXT, JSONL supported</span>
                        </label>
                    </div>
                    <div className="file-list">
                        {files.length ? (
                            files.map((file) => (
                                <div key={`${file.name}-${file.size}`} className="file-item">
                                    <span>{file.name}</span>
                                    <span className="mono">{Math.round(file.size / 1024)} KB</span>
                                </div>
                            ))
                        ) : (
                            <p className="muted">No files selected yet.</p>
                        )}
                    </div>
                    <div className="actions">
                        <button onClick={uploadFiles} disabled={busy}>
                            Upload and ingest
                        </button>
                        <button
                            className="ghost"
                            onClick={() => setFiles([])}
                            disabled={busy}
                        >
                            Clear files
                        </button>
                    </div>

                    <div className="divider" />

                    <h3>Quick note ingest</h3>
                    <div className="stack">
                        <input
                            value={noteSource}
                            onChange={(event) => setNoteSource(event.target.value)}
                            placeholder="Source label"
                        />
                        <textarea
                            value={noteText}
                            onChange={(event) => setNoteText(event.target.value)}
                            placeholder="Paste a paragraph or a security note..."
                            rows={6}
                        />
                        <button onClick={ingestNote} disabled={busy}>
                            Ingest note
                        </button>
                    </div>
                    {ingestStatus ? <p className="status-text">{ingestStatus}</p> : null}
                </section>

                <section className="card">
                    <h2>Ask a question</h2>
                    <div className="stack">
                        <textarea
                            value={question}
                            onChange={(event) => setQuestion(event.target.value)}
                            placeholder="Ask about your uploaded documents..."
                            rows={4}
                        />
                        <button onClick={runQuery} disabled={busy}>
                            Run query
                        </button>
                    </div>

                    {blocked ? (
                        <div className="alert">
                            <strong>Blocked by guardrails.</strong>
                            <p className="muted">Reason: {blockedReason || "policy"}</p>
                        </div>
                    ) : null}

                    <div className="answer">
                        <h3>Answer</h3>
                        {answer ? <p>{answer}</p> : <p className="muted">No answer yet.</p>}
                    </div>

                    <div className="citations">
                        <h3>Citations</h3>
                        {citations.length ? (
                            <ul>
                                {citations.map((citation, index) => (
                                    <li key={`${citation}-${index}`} className="mono">
                                        {citation}
                                    </li>
                                ))}
                            </ul>
                        ) : (
                            <p className="muted">No citations yet.</p>
                        )}
                    </div>
                </section>
            </main>
        </div>
    );
}
