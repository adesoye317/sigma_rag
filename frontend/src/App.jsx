import { useState, useRef, useEffect, useCallback } from "react";

const API       = import.meta.env.VITE_API_URL   || "http://localhost:8000";
const TENANT_ID = import.meta.env.VITE_TENANT_ID || "founder-demo";
const SIGMA_BLUE = "#1a3c5e";
const ACCENT     = "#2563eb";

// ── API helpers ───────────────────────────────────────────────────────────────
const hdrs = (extra = {}) => ({ "X-Tenant-Id": TENANT_ID, ...extra });

const api = {
  files:   () => fetch(`${API}/files`,   { headers: hdrs() }).then(r => r.json()),
  prompts: () => fetch(`${API}/prompts`, { headers: hdrs() }).then(r => r.json()),
  deleteFile: (id) => fetch(`${API}/files/${id}`, { method: "DELETE", headers: hdrs() }),
  upload: (file) => {
    const fd = new FormData();
    fd.append("file", file);
    return fetch(`${API}/upload`, { method: "POST", headers: hdrs(), body: fd }).then(r => r.json());
  },
  listConversations: () =>
    fetch(`${API}/conversations`, { headers: hdrs() }).then(r => r.json()),
  getEvalLog: () =>
    fetch(`${API}/conversations/eval`, { headers: hdrs() }).then(r => r.json()),
  createConversation: (title = "New Chat") =>
    fetch(`${API}/conversations`, {
      method: "POST",
      headers: hdrs({ "Content-Type": "application/json" }),
      body: JSON.stringify({ title }),
    }).then(r => r.json()),
  getMessages: (convId) =>
    fetch(`${API}/conversations/${convId}/messages`, { headers: hdrs() }).then(r => r.json()),
  saveMessage: (convId, msg) =>
    fetch(`${API}/conversations/${convId}/messages`, {
      method: "POST",
      headers: hdrs({ "Content-Type": "application/json" }),
      body: JSON.stringify(msg),
    }),
  deleteConversation: (convId) =>
    fetch(`${API}/conversations/${convId}`, { method: "DELETE", headers: hdrs() }),
};

// ── Streaming chat ────────────────────────────────────────────────────────────
async function streamChat(question, history, onChunk, onDone) {
  const res = await fetch(`${API}/chat`, {
    method: "POST",
    headers: hdrs({ "Content-Type": "application/json" }),
    body: JSON.stringify({ question, history }),
  });
  if (!res.ok) throw new Error(await res.text());

  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) {
    const data = await res.json();
    onChunk(data.answer || "I don't have this in your uploaded documents.");
    onDone({
      sources: data.sources || [], confidence: data.confidence || 0,
      missing: data.missing || false, suggestion: data.suggestion || "",
      benchmark: data.benchmark || null,
    });
    return;
  }

  const reader  = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const lines = buf.split("\n");
    buf = lines.pop();
    for (const line of lines) {
      if (line.startsWith("event: done")) continue;
      if (line.startsWith("data: ")) {
        const payload = line.slice(6);
        try {
          const meta = JSON.parse(payload);
          if (meta.sources !== undefined) { onDone(meta); continue; }
        } catch {}
        onChunk(payload);
      }
    }
  }
}

// ── Utilities ─────────────────────────────────────────────────────────────────
function now() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function stripInlineCitations(text) {
  return text
    .replace(/\s*\([^)]*\.pdf[^)]*\)/gi, "")
    .replace(/\s*\(p\.\d+\)/gi, "")
    .trim();
}

function dedupeSources(sources) {
  const map = new Map();
  for (const s of (sources || [])) {
    const key = `${s.doc}::${s.page}`;
    if (!map.has(key) || s.similarity > map.get(key).similarity)
      map.set(key, s);
  }
  return Array.from(map.values()).sort((a, b) => b.similarity - a.similarity);
}

// ── FIX: disambiguate identical conversation titles ───────────────────────────
// Appends " (2)", " (3)" etc. to duplicate titles so the sidebar is readable
function disambiguateTitles(sessions) {
  const counts = {};
  const seen   = {};
  for (const s of sessions) counts[s.title] = (counts[s.title] || 0) + 1;
  return sessions.map(s => {
    if (counts[s.title] <= 1) return s;
    seen[s.title] = (seen[s.title] || 0) + 1;
    return { ...s, displayTitle: seen[s.title] === 1 ? s.title : `${s.title} (${seen[s.title]})` };
  });
}

// ── BENCHMARK DATA (gap #3 from review) ──────────────────────────────────────
// Keyed by the same topic keywords used in _refusal_hint on the backend.
// The backend can optionally return a `benchmark` object; if it does we show it.
// If not (older backend), we derive it client-side from the question.
const BENCHMARKS = {
  cac: {
    label: "Customer Acquisition Cost (CAC)",
    note: "Typical SaaS CAC ranges $200–$1,500 depending on segment.",
    stats: [
      { tier: "SMB SaaS",        value: "$205" },
      { tier: "Mid-Market SaaS", value: "$535" },
      { tier: "Enterprise SaaS", value: "$1,450" },
    ],
  },
  ltv: {
    label: "Lifetime Value (LTV)",
    note: "Healthy SaaS targets LTV:CAC ≥ 3×.",
    stats: [
      { tier: "Median SaaS",        value: "3.0× CAC" },
      { tier: "Top-quartile SaaS",  value: "5.0× CAC" },
    ],
  },
  mrr: {
    label: "Monthly Recurring Revenue (MRR)",
    note: "Early-stage SaaS benchmarks by ARR milestone.",
    stats: [
      { tier: "$1M ARR milestone",  value: "~18–24 months avg" },
      { tier: "Growth rate (seed)", value: "15–20% MoM" },
    ],
  },
  "burn rate": {
    label: "Burn Rate",
    note: "Typical monthly burn for seed/Series A startups.",
    stats: [
      { tier: "Seed stage",     value: "$50K–$150K/mo" },
      { tier: "Series A",       value: "$200K–$500K/mo" },
    ],
  },
  runway: {
    label: "Runway",
    note: "Best practice runway targets.",
    stats: [
      { tier: "Minimum",        value: "12 months" },
      { tier: "Recommended",    value: "18–24 months" },
    ],
  },
  "loan size": {
    label: "Loan Size — First-time Borrower Benchmarks",
    note: "Typical microloan / SMB lending ranges.",
    stats: [
      { tier: "Microloan (first-time)", value: "$5K–$50K" },
      { tier: "SBA 7(a) small loan",   value: "Up to $500K" },
    ],
  },
  "interest rate": {
    label: "SMB Loan Interest Rates",
    note: "2024 benchmarks — rates vary by credit score and tenure.",
    stats: [
      { tier: "Bank / credit union",  value: "6%–13% APR" },
      { tier: "Online lender",        value: "10%–30% APR" },
    ],
  },
  onboarding: {
    label: "Onboarding Completion Benchmarks",
    note: "Industry benchmarks for SaaS / program onboarding.",
    stats: [
      { tier: "Avg completion rate",  value: "60–70%" },
      { tier: "Best-in-class",        value: "> 85%" },
    ],
  },
};

function detectBenchmark(question) {
  const q = question.toLowerCase();
  for (const [keyword, data] of Object.entries(BENCHMARKS)) {
    if (q.includes(keyword)) return data;
  }
  return null;
}

const WELCOME = "Hi! I'm Horo. Ask me anything — I'll answer only from your uploaded documents and cite my sources.";

// ── Sub-components ────────────────────────────────────────────────────────────

// Group sources by document, list all pages in a single badge per doc
function SourceBadges({ sources }) {
  const deduped = dedupeSources(sources);
  if (!deduped.length) return null;

  // Aggregate: { docName → { pages: number[], maxSim: float } }
  const byDoc = new Map();
  for (const s of deduped) {
    const name = s.doc.split("/").pop().replace(/\.pdf$/i, "");
    if (!byDoc.has(name)) byDoc.set(name, { pages: [], maxSim: 0 });
    const entry = byDoc.get(name);
    entry.pages.push(s.page);
    if (s.similarity > entry.maxSim) entry.maxSim = s.similarity;
  }

  return (
    <div style={{ marginTop: 8, display: "flex", flexWrap: "wrap", gap: 6 }}>
      {[...byDoc.entries()].map(([name, { pages, maxSim }], i) => (
        <span key={i} style={{
          background: "#eff6ff", color: "#1d4ed8", border: "1px solid #bfdbfe",
          borderRadius: 20, padding: "3px 12px", fontSize: 11,
          display: "inline-flex", alignItems: "center", gap: 4,
        }}>
          📄 {name}, pp.{pages.sort((a,b) => a-b).join(", ")}
          <span style={{ opacity: 0.55, fontSize: 10 }}>({Math.round(maxSim * 100)}%)</span>
        </span>
      ))}
    </div>
  );
}

// NEW: Benchmark card — shown in refusal when we have benchmark data
function BenchmarkCard({ benchmark }) {
  if (!benchmark) return null;
  return (
    <div style={{
      marginTop: 10, background: "#f0fdf4", border: "1px solid #86efac",
      borderRadius: 12, padding: "12px 14px", fontSize: 13,
    }}>
      <div style={{ fontWeight: 700, color: "#166534", marginBottom: 4 }}>
        📊 Industry Benchmark — {benchmark.label}
      </div>
      <div style={{ color: "#14532d", marginBottom: 8, fontSize: 12 }}>{benchmark.note}</div>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        {benchmark.stats.map((s, i) => (
          <div key={i} style={{
            background: "#fff", border: "1px solid #bbf7d0", borderRadius: 8,
            padding: "6px 12px", textAlign: "center",
          }}>
            <div style={{ fontWeight: 700, color: "#166534", fontSize: 14 }}>{s.value}</div>
            <div style={{ fontSize: 10, color: "#6b7280", marginTop: 2 }}>{s.tier}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function renderBold(text) {
  return text.split(/\*\*(.*?)\*\*/g).map((part, i) =>
    i % 2 === 1 ? <strong key={i}>{part}</strong> : part
  );
}

function FormattedText({ text }) {
  const normalised = text
    .replace(/([^\n])(\s*)(\d{1,2})\.\s+(?=[A-Z*])/g, (_, prev, ws, num) => `${prev}\n${num}. `)
    .replace(/([^\n])(\s*)([-•])\s+/g, (_, prev) => `${prev}\n- `);

  const lines = normalised.split("\n").map(l => l.trim()).filter((l, i, arr) => l || arr[i - 1]);

  return (
    <div>
      {lines.map((line, i) => {
        if (!line) return <div key={i} style={{ height: 6 }} />;
        const num    = line.match(/^(\d{1,2})\.\s+([\s\S]*)/);
        const bullet = line.match(/^[-•]\s+([\s\S]*)/);

        if (num) return (
          <div key={i} style={{ display: "flex", gap: 10, marginBottom: 8, alignItems: "flex-start" }}>
            <span style={{
              fontWeight: 700, minWidth: 24, height: 24, borderRadius: "50%",
              background: "#eff6ff", color: "#2563eb", fontSize: 12,
              display: "flex", alignItems: "center", justifyContent: "center",
              flexShrink: 0, marginTop: 1,
            }}>{num[1]}</span>
            <span style={{ flex: 1, lineHeight: 1.6 }}>{renderBold(num[2])}</span>
          </div>
        );
        if (bullet) return (
          <div key={i} style={{ display: "flex", gap: 10, marginBottom: 6, alignItems: "flex-start" }}>
            <span style={{ color: "#2563eb", minWidth: 20, flexShrink: 0, paddingTop: 3, fontSize: 16, lineHeight: 1 }}>·</span>
            <span style={{ flex: 1, lineHeight: 1.6 }}>{renderBold(bullet[1])}</span>
          </div>
        );
        return <p key={i} style={{ margin: "0 0 6px", lineHeight: 1.7 }}>{renderBold(line)}</p>;
      })}
    </div>
  );
}

function Message({ msg }) {
  const isUser      = msg.role === "user";
  const displayText = isUser ? msg.text : stripInlineCitations(msg.text || "");
  // FIX: detect benchmark client-side if backend didn't send one
  const benchmark   = !isUser && msg.missing
    ? (msg.benchmark || detectBenchmark(msg.question || ""))
    : null;

  return (
    <div style={{ display: "flex", justifyContent: isUser ? "flex-end" : "flex-start", marginBottom: 20, gap: 10, alignItems: "flex-start" }}>
      {!isUser && (
        <div style={{ width: 34, height: 34, borderRadius: "50%", background: SIGMA_BLUE, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, color: "#fff", fontSize: 15 }}>✦</div>
      )}
      <div style={{ maxWidth: "74%", minWidth: 60 }}>
        <div style={{
          background: isUser ? ACCENT : "#fff",
          color: isUser ? "#fff" : "#1f2937",
          borderRadius: isUser ? "18px 18px 4px 18px" : "18px 18px 18px 4px",
          padding: "13px 16px",
          boxShadow: "0 1px 4px rgba(0,0,0,0.07)",
          fontSize: 14, lineHeight: 1.7,
        }}>
          {displayText
            ? (isUser ? displayText : <FormattedText text={displayText} />)
            : <span style={{ opacity: 0.35 }}>▌</span>}
        </div>

        {/* FIX: show all source badges, not just top-1 */}
        {!isUser && !msg.missing && <SourceBadges sources={msg.sources} />}

        {/* Citation mismatch warning */}
        {!isUser && !msg.missing && msg.citationMismatch && (
          <div style={{ marginTop: 6, background: "#fef3c7", border: "1px solid #fbbf24", borderRadius: 8, padding: "6px 10px", fontSize: 11, color: "#92400e" }}>
            ⚠ Inline citation may differ from retrieved source — verify page number.
          </div>
        )}

        {/* Refusal card with contextual hint + benchmark */}
        {msg.missing && (
          <div style={{ marginTop: 8 }}>
            <div style={{ background: "#fef3c7", border: "1px solid #fbbf24", borderRadius: 10, padding: "10px 14px", fontSize: 13, color: "#92400e", lineHeight: 1.5 }}>
              💡 {msg.suggestion || "Try uploading the relevant document."}
            </div>
            <BenchmarkCard benchmark={benchmark} />
          </div>
        )}

        <div style={{ fontSize: 11, color: "#9ca3af", marginTop: 5, textAlign: isUser ? "right" : "left" }}>{msg.time}</div>
      </div>
      {isUser && (
        <div style={{ width: 34, height: 34, borderRadius: "50%", background: "#e0e7ff", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, fontSize: 15 }}>👤</div>
      )}
    </div>
  );
}

// ── Eval tab ──────────────────────────────────────────────────────────────────
function EvalMetrics({ evalLog }) {
  const avg = arr => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
  const scores       = evalLog.map(e => e.confidence);
  const faithScores  = evalLog.filter(e => e.faithfulness  != null).map(e => e.faithfulness);
  const groundScores = evalLog.filter(e => e.groundingScore != null).map(e => e.groundingScore);
  const answered     = evalLog.filter(e => !e.missing).length;
  const refused      = evalLog.filter(e =>  e.missing).length;

  const cards = [
    { label: "Queries",          value: evalLog.length,                                                          color: "#2563eb" },
    { label: "Answered",         value: answered,                                                                color: "#22c55e" },
    { label: "Refused",          value: refused,                                                                 color: "#f59e0b" },
    { label: "Avg Confidence",   value: scores.length       ? `${Math.round(avg(scores)       * 100)}%` : "—", color: "#8b5cf6" },
    { label: "Avg Faithfulness", value: faithScores.length  ? `${Math.round(avg(faithScores)  * 100)}%` : "—", color: "#0891b2" },
    { label: "Avg Grounding",    value: groundScores.length ? `${Math.round(avg(groundScores) * 100)}%` : "—", color: "#0d9488" },
  ];

  const colScore = (val, hi, mid) => val >= hi ? "#22c55e" : val >= mid ? "#f59e0b" : "#ef4444";

  return (
    <div style={{ flex: 1, overflowY: "auto", padding: 24 }}>
      <div style={{ maxWidth: 900, margin: "0 auto" }}>
        <h2 style={{ margin: "0 0 4px", fontSize: 18, color: "#1f2937" }}>Evaluation Metrics</h2>
        <p style={{ margin: "0 0 20px", fontSize: 13, color: "#6b7280" }}>Live session stats — resets on page refresh</p>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(6,1fr)", gap: 12, marginBottom: 24 }}>
          {cards.map((c, i) => (
            <div key={i} style={{ background: "#fff", borderRadius: 12, padding: "16px 8px", boxShadow: "0 1px 4px rgba(0,0,0,0.06)", textAlign: "center" }}>
              <div style={{ fontSize: 22, fontWeight: 700, color: c.color }}>{c.value}</div>
              <div style={{ fontSize: 11, color: "#6b7280", marginTop: 4, lineHeight: 1.3 }}>{c.label}</div>
            </div>
          ))}
        </div>

        {evalLog.length === 0 ? (
          <div style={{ background: "#fff", borderRadius: 12, padding: 32, textAlign: "center", color: "#9ca3af", fontSize: 14 }}>
            No queries yet — ask Horo something in the Chat tab.
          </div>
        ) : (
          <div style={{ background: "#fff", borderRadius: 12, border: "1px solid #e2e8f0", overflow: "hidden" }}>
            <div style={{ display: "grid", gridTemplateColumns: "minmax(0,1fr) 100px 100px 100px 70px 100px", background: "#f8fafc", borderBottom: "1px solid #e2e8f0" }}>
              {["Question", "Confidence", "Faithful", "Grounding", "Sources", "Status"].map((h, i) => (
                <div key={i} style={{ padding: "10px 14px", fontSize: 11, fontWeight: 700, color: "#6b7280", textTransform: "uppercase", letterSpacing: 0.5 }}>{h}</div>
              ))}
            </div>
            {evalLog.map((e, i) => (
              <div key={i} style={{ display: "grid", gridTemplateColumns: "minmax(0,1fr) 100px 100px 100px 70px 100px", borderTop: i > 0 ? "1px solid #f1f5f9" : "none", alignItems: "start" }}>
                <div style={{ padding: "14px 14px" }}>
                  <div style={{ color: "#1f2937", fontSize: 13, lineHeight: 1.4, wordBreak: "break-word" }}>{e.question}</div>
                  {/* Citation mismatch flag in eval */}
                  {e.citationMismatch && (
                    <div style={{ fontSize: 11, color: "#b45309", background: "#fef3c7", borderRadius: 6, padding: "2px 6px", marginTop: 4, display: "inline-block" }}>
                      ⚠ Citation page mismatch
                    </div>
                  )}
                  {e.unsupportedClaims?.map((c, j) => (
                    <div key={j} style={{ fontSize: 11, color: "#b45309", background: "#fef3c7", borderRadius: 6, padding: "2px 6px", marginTop: 4, display: "inline-block" }}>⚠ {c}</div>
                  ))}
                </div>
                <div style={{ padding: "14px 14px", display: "flex", alignItems: "center" }}>
                  <span style={{ fontWeight: 700, fontSize: 14, color: colScore(e.confidence, 0.65, 0.5) }}>
                    {e.confidence ? `${Math.round(e.confidence * 100)}%` : "—"}
                  </span>
                </div>
                <div style={{ padding: "14px 14px", display: "flex", alignItems: "center" }}>
                  <span style={{ fontWeight: 700, fontSize: 14, color: colScore(e.faithfulness ?? 0, 0.85, 0.6) }}>
                    {e.faithfulness != null ? `${Math.round(e.faithfulness * 100)}%` : "—"}
                  </span>
                </div>
                <div style={{ padding: "14px 14px", display: "flex", alignItems: "center" }}>
                  <span style={{ fontWeight: 700, fontSize: 14, color: colScore(e.groundingScore ?? 0, 0.7, 0.5) }}>
                    {e.groundingScore != null ? `${Math.round(e.groundingScore * 100)}%` : "—"}
                  </span>
                </div>
                <div style={{ padding: "14px 14px", display: "flex", alignItems: "center" }}>
                  <span style={{ color: "#6b7280", fontSize: 13 }}>{e.sourceCount}</span>
                </div>
                <div style={{ padding: "14px 14px", display: "flex", alignItems: "center" }}>
                  <span style={{
                    background: e.missing ? "#fef3c7" : "#dcfce7",
                    color:      e.missing ? "#92400e" : "#166534",
                    borderRadius: 99, padding: "3px 12px", fontSize: 11, fontWeight: 600, whiteSpace: "nowrap",
                  }}>
                    {e.missing ? "Refused" : "Answered"}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════════════════
// Main App
// ══════════════════════════════════════════════════════════════════════════════
export default function App() {
  const [tab, setTab]               = useState("chat");
  const [sessions, setSessions]     = useState([]);
  const [activeConvId, setActiveConvId] = useState(null);
  const [messages, setMessages]     = useState([{ role: "assistant", text: WELCOME, time: now(), sources: [] }]);
  const [input, setInput]           = useState("");
  const [loading, setLoading]       = useState(false);
  const [files, setFiles]           = useState([]);
  const [prompts, setPrompts]       = useState([]);
  const [dragging, setDragging]     = useState(false);
  const [uploading, setUploading]   = useState(false);
  const [error, setError]           = useState(null);
  const [evalLog, setEvalLog]       = useState([]);

  const bottomRef  = useRef();
  const fileInput  = useRef();
  const historyRef = useRef([]);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  const loadConversations = useCallback(async () => {
    try { setSessions(await api.listConversations()); } catch {}
  }, []);

  const loadEvalLog = useCallback(async () => {
    try {
      const rows = await api.getEvalLog();
      setEvalLog(rows.map(r => ({
        question:          r.question,
        confidence:        r.confidence,
        faithfulness:      r.faithfulness,
        groundingScore:    r.grounding_score,
        unsupportedClaims: r.unsupported_claims || [],
        sourceCount:       r.source_count || 0,
        missing:           r.missing,
        // FIX: surface citation mismatch from backend if available
        citationMismatch:  r.citation_mismatch || false,
      })));
    } catch {}
  }, []);

  const loadFiles   = useCallback(async () => { try { setFiles(await api.files()); }   catch {} }, []);
  const loadPrompts = useCallback(async () => { try { setPrompts(await api.prompts()); } catch {} }, []);

  useEffect(() => { loadFiles(); loadPrompts(); loadConversations(); loadEvalLog(); }, []);

  // ── Citation cross-validation ─────────────────────────────────────────────
  // FIX: parse inline citations from answer text and compare with metadata sources
  function detectCitationMismatch(answerText, sources) {
    const inlinePages = [...answerText.matchAll(/p\.(\d+)/gi)].map(m => parseInt(m[1]));
    if (!inlinePages.length) return false;
    const sourcePages = new Set((sources || []).map(s => s.page));
    return inlinePages.some(p => !sourcePages.has(p));
  }

  // ── New chat ────────────────────────────────────────────────────────────────
  const startNewChat = async () => {
    const conv = await api.createConversation("New Chat");
    setActiveConvId(conv.id);
    setMessages([{ role: "assistant", text: WELCOME, time: now(), sources: [] }]);
    historyRef.current = [];
    await loadConversations();
  };

  // ── Load session ────────────────────────────────────────────────────────────
  const loadSession = async (conv) => {
    setActiveConvId(conv.id);
    setTab("chat");
    try {
      const msgs = await api.getMessages(conv.id);
      const uiMsgs = [
        { role: "assistant", text: WELCOME, time: "", sources: [] },
        ...msgs.map(m => ({
          role:              m.role,
          text:              m.content,
          time:              new Date(m.created_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
          sources:           m.sources            || [],
          faithfulness:      m.faithfulness,
          groundingScore:    m.grounding_score,
          unsupportedClaims: m.unsupported_claims || [],
          confidence:        m.confidence         || 0,
          missing:           m.confidence > 0 && m.confidence < 0.45,
          citationMismatch:  detectCitationMismatch(m.content, m.sources),
        })),
      ];
      setMessages(uiMsgs);
      historyRef.current = msgs
        .filter(m => m.role === "user" || m.role === "assistant")
        .map(m => ({ role: m.role, content: m.content }))
        .slice(-20);
    } catch {
      setError("Failed to load conversation.");
    }
  };

  // ── Delete session ──────────────────────────────────────────────────────────
  const deleteSession = async (e, convId) => {
    e.stopPropagation();
    await api.deleteConversation(convId);
    if (activeConvId === convId) {
      setActiveConvId(null);
      setMessages([{ role: "assistant", text: WELCOME, time: now(), sources: [] }]);
      historyRef.current = [];
    }
    await loadConversations();
  };

  // ── Send ────────────────────────────────────────────────────────────────────
  const send = async (text) => {
    const q = (text || input).trim();
    if (!q || loading) return;
    setInput("");
    setError(null);

    let convId = activeConvId;
    if (!convId) {
      const conv = await api.createConversation("New Chat");
      convId = conv.id;
      setActiveConvId(convId);
    }

    await api.saveMessage(convId, { role: "user", content: q });
    setMessages(m => [...m, { role: "user", text: q, time: now() }]);
    const pid = Date.now();
    setMessages(m => [...m, { id: pid, role: "assistant", text: "", time: now(), sources: [] }]);
    setLoading(true);

    try {
      let fullText = "";
      await streamChat(
        q,
        historyRef.current,
        (chunk) => {
          fullText += chunk;
          setMessages(m => m.map(msg => msg.id === pid ? { ...msg, text: fullText } : msg));
        },
        async (meta) => {
          const missing          = meta.missing || (meta.confidence > 0 && meta.confidence < 0.45);
          // FIX: cross-validate inline citations vs retrieved sources
          const citationMismatch = !missing && detectCitationMismatch(fullText, meta.sources);
          // FIX: attach question to message so Message component can derive benchmark
          const benchmark        = missing ? (meta.benchmark || detectBenchmark(q)) : null;

          setMessages(m => m.map(msg => msg.id === pid ? {
            ...msg,
            text:              fullText,
            sources:           meta.sources            || [],
            confidence:        meta.confidence         || 0,
            faithfulness:      meta.faithfulness,
            groundingScore:    meta.grounding_score,
            unsupportedClaims: meta.unsupported_claims || [],
            missing,
            suggestion:        meta.suggestion         || "",
            citationMismatch,
            benchmark,
            question:          q,   // needed by Message for benchmark fallback
          } : msg));

          await api.saveMessage(convId, {
            role:               "assistant",
            content:            fullText,
            sources:            meta.sources            || [],
            confidence:         meta.confidence         || 0,
            faithfulness:       meta.faithfulness,
            grounding_score:    meta.grounding_score,
            unsupported_claims: meta.unsupported_claims || [],
          });

          // Don't log greetings — they aren't RAG queries
          if (!meta.greeting) {
            setEvalLog(l => [...l, {
              question:          q,
              confidence:        meta.confidence        || 0,
              faithfulness:      meta.faithfulness,
              groundingScore:    meta.grounding_score,
              unsupportedClaims: meta.unsupported_claims || [],
              sourceCount:       dedupeSources(meta.sources || []).length,
              missing,
              citationMismatch,
            }]);
          }

          historyRef.current = [
            ...historyRef.current,
            { role: "user",      content: q        },
            { role: "assistant", content: fullText },
          ].slice(-20);

          await loadConversations();
          await loadEvalLog();
        },
      );
    } catch (e) {
      setError("Failed to reach backend: " + e.message);
      setMessages(m => m.filter(msg => msg.id !== pid));
    } finally {
      setLoading(false);
    }
  };

  // ── File handling ───────────────────────────────────────────────────────────
  const handleFiles = async (fileList) => {
    setUploading(true);
    for (const f of Array.from(fileList)) {
      try { await api.upload(f); } catch { setError(`Upload failed: ${f.name}`); }
    }
    await loadFiles(); await loadPrompts();
    setUploading(false);
  };
  const handleDrop = (e) => { e.preventDefault(); setDragging(false); handleFiles(e.dataTransfer.files); };

  // FIX: disambiguated session titles for sidebar
  const displaySessions = disambiguateTitles(sessions);

  const TABS = [["chat", "💬 Chat"], ["files", "📁 Knowledge Base"], ["eval", "📊 Evaluation"]];

  return (
    <div style={{ display: "flex", height: "100vh", fontFamily: "'Inter', sans-serif", background: "#f1f5f9", overflow: "hidden" }}>
      <style>{`
        * { box-sizing: border-box; }
        .pill { background:#fff; border:1px solid #e2e8f0; border-radius:12px; padding:10px 14px; cursor:pointer; transition:all .15s; font-size:13px; text-align:left; width:100%; }
        .pill:hover { border-color:${ACCENT}; background:#eff6ff; transform:translateY(-1px); box-shadow:0 2px 8px rgba(37,99,235,.1); }
        .tab { padding:7px 16px; border-radius:8px; border:none; cursor:pointer; font-size:13px; font-weight:500; transition:all .15s; }
        .btn { background:${ACCENT}; color:#fff; border:none; border-radius:12px; padding:10px 18px; cursor:pointer; font-size:14px; }
        .btn:disabled { opacity:.5; cursor:not-allowed; }
        .file-row:hover { background:#f8fafc; }
        ::-webkit-scrollbar { width:4px; }
        ::-webkit-scrollbar-thumb { background:#cbd5e1; border-radius:4px; }
        @keyframes spin { to { transform:rotate(360deg); } }
      `}</style>

      {/* ── Sidebar ── */}
      <div style={{ width: 240, background: SIGMA_BLUE, color: "#fff", display: "flex", flexDirection: "column" }}>
        <div style={{ padding: "20px 16px 16px", borderBottom: "1px solid rgba(255,255,255,.1)" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
            <span style={{ fontSize: 22 }}>✦</span>
            <span style={{ fontWeight: 700, fontSize: 18 }}>sigma</span>
          </div>
          <div style={{ fontSize: 11, opacity: .6 }}>Horo · Knowledge Co-Pilot</div>
        </div>

        <div style={{ padding: "12px 10px" }}>
          <button onClick={startNewChat}
            style={{ width: "100%", background: "rgba(255,255,255,.12)", color: "#fff", border: "1px dashed rgba(255,255,255,.25)", borderRadius: 10, padding: "8px 0", cursor: "pointer", fontSize: 13 }}>
            + New Chat
          </button>
        </div>

        <div style={{ flex: 1, overflowY: "auto", padding: "0 10px" }}>
          {displaySessions.length > 0 && (
            <div style={{ fontSize: 10, opacity: .45, padding: "8px 6px 4px", letterSpacing: 1, textTransform: "uppercase" }}>Recent</div>
          )}
          {/* FIX: use disambiguated displayTitle */}
          {displaySessions.map(s => (
            <div key={s.id} onClick={() => loadSession(s)}
              style={{
                display: "flex", alignItems: "center", gap: 6, borderRadius: 8,
                padding: "8px 10px", cursor: "pointer", marginBottom: 2,
                background: activeConvId === s.id ? "rgba(255,255,255,.18)" : "transparent",
                transition: "background .15s",
              }}
              onMouseEnter={e => e.currentTarget.style.background = activeConvId === s.id ? "rgba(255,255,255,.18)" : "rgba(255,255,255,.1)"}
              onMouseLeave={e => e.currentTarget.style.background = activeConvId === s.id ? "rgba(255,255,255,.18)" : "transparent"}
            >
              <span style={{ fontSize: 13, flexShrink: 0 }}>💬</span>
              <span style={{ flex: 1, fontSize: 12, color: "#fff", opacity: .8, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {s.displayTitle || s.title}
              </span>
              <button onClick={e => deleteSession(e, s.id)}
                style={{ background: "none", border: "none", color: "rgba(255,255,255,.35)", cursor: "pointer", fontSize: 13, padding: "0 2px", flexShrink: 0 }}
                onMouseEnter={e => e.currentTarget.style.color = "#ef4444"}
                onMouseLeave={e => e.currentTarget.style.color = "rgba(255,255,255,.35)"}>🗑</button>
            </div>
          ))}
        </div>

        <div style={{ padding: "12px 10px", borderTop: "1px solid rgba(255,255,255,.1)", fontSize: 11, opacity: .45 }}>
          🔒 {TENANT_ID} · {files.length} file{files.length !== 1 ? "s" : ""}
        </div>
      </div>

      {/* ── Main ── */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
        <div style={{ background: "#fff", padding: "12px 24px", display: "flex", alignItems: "center", justifyContent: "space-between", borderBottom: "1px solid #e2e8f0" }}>
          <div style={{ display: "flex", gap: 6 }}>
            {TABS.map(([t, label]) => (
              <button key={t} className="tab" onClick={() => setTab(t)}
                style={{ background: tab === t ? "#eff6ff" : "transparent", color: tab === t ? ACCENT : "#6b7280" }}>
                {label}
              </button>
            ))}
          </div>
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <span style={{ fontSize: 12, color: "#6b7280" }}>RAG · Anti-hallucination · pgvector</span>
            <span style={{ background: "#dcfce7", color: "#166534", fontSize: 11, padding: "2px 8px", borderRadius: 99, fontWeight: 600 }}>● Live</span>
          </div>
        </div>

        {error && (
          <div style={{ background: "#fee2e2", color: "#991b1b", padding: "8px 24px", fontSize: 13, display: "flex", justifyContent: "space-between" }}>
            ⚠ {error}
            <button onClick={() => setError(null)} style={{ background: "none", border: "none", cursor: "pointer", color: "#991b1b" }}>🗑</button>
          </div>
        )}

        {/* ── Chat ── */}
        {tab === "chat" && (
          <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>
            <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
              <div style={{ flex: 1, overflowY: "auto", padding: "24px 28px" }}>
                {messages.map((m, i) => <Message key={m.id || i} msg={m} />)}
                {loading && messages[messages.length - 1]?.text === "" && (
                  <div style={{ paddingLeft: 44, fontSize: 12, color: "#9ca3af", display: "flex", alignItems: "center", gap: 6 }}>
                    <div style={{ width: 14, height: 14, border: "2px solid #e2e8f0", borderTopColor: ACCENT, borderRadius: "50%", animation: "spin .7s linear infinite" }} />
                    Searching knowledge base…
                  </div>
                )}
                <div ref={bottomRef} />
              </div>

              {messages.length <= 1 && prompts.length > 0 && (
                <div style={{ padding: "0 24px 12px" }}>
                  <div style={{ fontSize: 12, color: "#6b7280", marginBottom: 8, fontWeight: 500 }}>💡 From your knowledge base</div>
                  <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                    {prompts.slice(0, 4).map((p, i) => (
                      <button key={i} onClick={() => send(p.prompt)}
                        style={{ background: "#fff", border: "1px solid #e2e8f0", borderRadius: 12, padding: "8px 14px", cursor: "pointer", fontSize: 13, transition: "all .15s" }}
                        onMouseEnter={e => { e.currentTarget.style.borderColor = ACCENT; e.currentTarget.style.background = "#eff6ff"; }}
                        onMouseLeave={e => { e.currentTarget.style.borderColor = "#e2e8f0"; e.currentTarget.style.background = "#fff"; }}>
                        {p.prompt}
                        <span style={{ marginLeft: 6, background: "#f1f5f9", color: "#64748b", borderRadius: 99, padding: "1px 7px", fontSize: 10 }}>{p.tag}</span>
                      </button>
                    ))}
                  </div>
                </div>
              )}
              {messages.length <= 1 && files.length === 0 && (
                <div style={{ padding: "0 24px 12px" }}>
                  <div style={{ background: "#fef3c7", border: "1px solid #fbbf24", borderRadius: 10, padding: "10px 14px", fontSize: 13, color: "#92400e" }}>
                    📂 No files yet — go to <strong>Knowledge Base</strong> to upload your first document.
                  </div>
                </div>
              )}

              <div style={{ padding: "12px 24px 20px", background: "#fff", borderTop: "1px solid #e2e8f0" }}>
                <div style={{ display: "flex", gap: 10, background: "#f8fafc", borderRadius: 16, padding: "6px 6px 6px 16px", border: "1.5px solid #e2e8f0" }}>
                  <input value={input} onChange={e => setInput(e.target.value)}
                    onKeyDown={e => e.key === "Enter" && !e.shiftKey && send()}
                    placeholder="Ask anything from your documents…"
                    style={{ flex: 1, border: "none", background: "transparent", outline: "none", fontSize: 14, color: "#1f2937" }} />
                  <button className="btn" onClick={() => send()} disabled={!input.trim() || loading}>➤</button>
                </div>
                <div style={{ fontSize: 11, color: "#9ca3af", marginTop: 6, textAlign: "center" }}>
                  Grounded answers only · PII masked · Private tenant · No cross-tenant access
                </div>
              </div>
            </div>

            {prompts.length > 0 && (
              <div style={{ width: 220, background: "#fff", borderLeft: "1px solid #e2e8f0", padding: "16px 12px", overflowY: "auto" }}>
                <div style={{ fontSize: 12, fontWeight: 600, color: "#374151", marginBottom: 12 }}>📚 Prompt Library</div>
                {prompts.map((p, i) => (
                  <button key={i} className="pill" onClick={() => send(p.prompt)} style={{ marginBottom: 8, display: "flex", flexDirection: "column", gap: 4 }}>
                    <span style={{ lineHeight: 1.4, fontSize: 12 }}>{p.prompt}</span>
                    <span style={{ background: "#f1f5f9", color: "#64748b", borderRadius: 99, padding: "1px 7px", fontSize: 10, alignSelf: "flex-start" }}>{p.tag}</span>
                  </button>
                ))}
              </div>
            )}
          </div>
        )}

        {/* ── Knowledge Base ── */}
        {tab === "files" && (
          <div style={{ flex: 1, overflowY: "auto", padding: 24 }}>
            <div style={{ maxWidth: 700, margin: "0 auto" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
                <div>
                  <h2 style={{ margin: 0, fontSize: 18, color: "#1f2937" }}>Knowledge Base</h2>
                  <p style={{ margin: "4px 0 0", fontSize: 13, color: "#6b7280" }}>{files.length} file{files.length !== 1 ? "s" : ""} · private tenant · pgvector</p>
                </div>
                <button className="btn" onClick={() => fileInput.current.click()} disabled={uploading}>
                  {uploading ? "Processing…" : "+ Upload File"}
                </button>
                <input ref={fileInput} type="file" multiple accept=".pdf,.docx,.xlsx,.txt" style={{ display: "none" }}
                  onChange={e => handleFiles(e.target.files)} />
              </div>

              <div onDragOver={e => { e.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={handleDrop}
                onClick={() => fileInput.current.click()}
                style={{ border: `2px dashed ${dragging ? "#2563eb" : "#cbd5e1"}`, borderRadius: 16, padding: 32, textAlign: "center", cursor: "pointer", background: dragging ? "#eff6ff" : "#f8fafc", marginBottom: 20, transition: "all .2s" }}>
                {uploading
                  ? <><div style={{ fontSize: 28 }}>⏳</div><div style={{ fontWeight: 600, color: "#374151" }}>Processing & embedding…</div></>
                  : <><div style={{ fontSize: 28 }}>📂</div><div style={{ fontWeight: 600, color: "#374151" }}>Drop files or click to upload</div>
                    <div style={{ fontSize: 12, color: "#9ca3af", marginTop: 4 }}>PDF, DOCX, XLSX, TXT · Auto-chunked · Azure OpenAI embeddings</div></>
                }
              </div>

              {files.length > 0 && (
                <div style={{ background: "#fff", borderRadius: 16, border: "1px solid #e2e8f0", overflow: "hidden" }}>
                  <div style={{ padding: "12px 16px", background: "#f8fafc", display: "grid", gridTemplateColumns: "1fr 90px 70px 70px 40px", gap: 8, fontSize: 11, color: "#6b7280", fontWeight: 600, textTransform: "uppercase" }}>
                    <span>File</span><span>Tag</span><span>Pages</span><span>Chunks</span><span></span>
                  </div>
                  {files.map(f => {
                    const icon = f.filename?.endsWith(".pdf") ? "📄" : f.filename?.endsWith(".xlsx") ? "📊" : "📝";
                    const tag  = f.tag || "Document";
                    // Shorten long filenames: keep first 32 chars + "…" + extension
                    const ext      = f.filename?.split(".").pop() || "";
                    const base     = f.filename?.replace(/\.[^.]+$/, "") || "";
                    const shortName = base.length > 32
                      ? `${base.slice(0, 32)}….${ext}`
                      : f.filename;
                    return (
                      <div key={f.id} className="file-row" style={{ padding: "14px 16px", display: "grid", gridTemplateColumns: "1fr 90px 70px 70px 40px", gap: 8, fontSize: 13, borderTop: "1px solid #f1f5f9", alignItems: "center" }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 0 }}>
                          <span style={{ fontSize: 18, flexShrink: 0 }}>{icon}</span>
                          <span
                            title={f.filename}
                            style={{ fontWeight: 500, color: "#1f2937", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                            {shortName}
                          </span>
                        </div>
                        <span style={{ background: "#eff6ff", color: "#2563eb", borderRadius: 99, padding: "2px 8px", fontSize: 11, textAlign: "center", whiteSpace: "nowrap" }}>{tag}</span>
                        <span style={{ color: "#6b7280" }}>{f.page_count ?? "—"}</span>
                        <span style={{ color: "#6b7280" }}>{f.chunk_count ?? "—"}</span>
                        <button onClick={() => { api.deleteFile(f.id); setFiles(fl => fl.filter(x => x.id !== f.id)); }}
                          style={{ background: "none", border: "none", cursor: "pointer", color: "#ef4444", fontSize: 16 }}>🗑</button>
                      </div>
                    );
                  })}
                </div>
              )}

              <div style={{ marginTop: 16, background: "#fef3c7", borderRadius: 12, padding: "12px 16px", fontSize: 13, color: "#92400e", display: "flex", gap: 8 }}>
                <span>🔒</span>
                <span>Files stored in your private tenant. Horo only searches <strong>your</strong> documents — zero cross-tenant access enforced at query level.</span>
              </div>
            </div>
          </div>
        )}

        {tab === "eval" && <EvalMetrics evalLog={evalLog} />}
      </div>
    </div>
  );
}