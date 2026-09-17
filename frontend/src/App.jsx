import { useState, useEffect, useCallback } from 'react'
import './index.css'
import { getHealth, getStats, getTransactions, analyzeRisk } from './api'

/* ================================================================
   Overview stat cards
   ================================================================ */
function OverviewCards({ stats }) {
  return (
    <div className="overview">
      <div className="stat-card">
        <div className="stat-label">Transactions Analyzed</div>
        <div className="stat-value blue">{stats.total?.toLocaleString() ?? '—'}</div>
      </div>
      <div className="stat-card">
        <div className="stat-label">High-Risk Detected</div>
        <div className="stat-value high">{stats.fraud_count ?? '—'}</div>
      </div>
      <div className="stat-card">
        <div className="stat-label">Customers</div>
        <div className="stat-value">{stats.customers?.toLocaleString() ?? '—'}</div>
      </div>
      <div className="stat-card">
        <div className="stat-label">Fraud Rate</div>
        <div className="stat-value warn">{stats.fraud_rate != null ? (stats.fraud_rate * 100).toFixed(2) + '%' : '—'}</div>
      </div>
    </div>
  )
}

/* ================================================================
   Model metrics bar
   ================================================================ */
function MetricsBar({ metrics }) {
  if (!metrics) return null
  return (
    <div className="metrics-bar">
      <div className="metric-chip">Precision <strong>{metrics.precision}</strong></div>
      <div className="metric-chip">Recall <strong>{metrics.recall}</strong></div>
      <div className="metric-chip">F1 <strong>{metrics.f1}</strong></div>
      <div className="metric-chip">FPR <strong>{metrics.fpr}</strong></div>
    </div>
  )
}

/* ================================================================
   Transaction table
   ================================================================ */
function TransactionTable({ transactions, total, selected, onSelect, analyses }) {
  return (
    <>
      <div className="table-header">
        <h2>Transaction Risk Monitor</h2>
        <span className="table-count">{total} transactions</span>
      </div>
      <table className="tx-table">
        <thead>
          <tr>
            <th>ID</th>
            <th>Time</th>
            <th>Amount</th>
            <th>Customer</th>
            <th>Country</th>
            <th>Risk Level</th>
            <th>Decision</th>
          </tr>
        </thead>
        <tbody>
          {transactions.map(tx => {
            const a = analyses[tx.transaction_id]
            return (
              <tr
                key={tx.transaction_id}
                className={selected?.transaction_id === tx.transaction_id ? 'selected' : ''}
                onClick={() => onSelect(tx)}
              >
                <td><span className="tx-id">{tx.transaction_id?.slice(0, 8)}…</span></td>
                <td>{tx.timestamp?.slice(0, 16)?.replace('T', ' ')}</td>
                <td><span className="tx-amount">${tx.amount?.toFixed(2)}</span></td>
                <td>{tx.customer_id?.slice(0, 8)}…</td>
                <td>{tx.country}</td>
                <td>{a ? <span className={`risk-badge ${a.risk_level}`}>{a.risk_level}</span> : <span className="risk-badge">—</span>}</td>
                <td>{a ? <span className={`decision-badge ${a.decision}`}>{a.decision?.replace('_', ' ')}</span> : '—'}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </>
  )
}

/* ================================================================
   Investigation panel
   ================================================================ */
function InvestigationPanel({ data }) {
  if (!data) {
    return (
      <div className="panel">
        <div className="panel-empty">
          <div className="panel-empty-icon">🔍</div>
          <div>Select a transaction to investigate</div>
          <div style={{ fontSize: 12 }}>Click any row in the table</div>
        </div>
      </div>
    )
  }

  const inv = data.investigation || {}
  const findings = inv.findings || []

  return (
    <div className="panel">
      <div className="panel-title">Investigation Report</div>

      {/* Score ring */}
      <div className="score-display">
        <div className={`score-ring ${data.risk_level}`}>
          <div className="score-number" style={{ color: riskColor(data.risk_level) }}>{data.risk_score?.toFixed(0)}</div>
          <div className="score-label">/ 100</div>
        </div>
        <span className={`risk-badge ${data.risk_level}`} style={{ fontSize: 13 }}>{data.risk_level} RISK</span>
      </div>

      {/* Decision */}
      <div style={{ textAlign: 'center', marginBottom: 20 }}>
        <span className={`decision-badge ${data.decision}`}>{data.decision?.replace('_', ' ')}</span>
      </div>

      {/* Transaction details */}
      <div className="panel-section">
        <h3>Transaction</h3>
        <div className="tx-detail-grid">
          <div className="tx-detail-item"><span className="tx-detail-label">ID</span><br /><span className="tx-detail-value">{data.transaction_id?.slice(0, 16)}</span></div>
          <div className="tx-detail-item"><span className="tx-detail-label">Amount</span><br /><span className="tx-detail-value">${data._tx?.amount?.toFixed(2)}</span></div>
          <div className="tx-detail-item"><span className="tx-detail-label">Method</span><br /><span className="tx-detail-value">{data._tx?.payment_method}</span></div>
          <div className="tx-detail-item"><span className="tx-detail-label">Category</span><br /><span className="tx-detail-value">{data._tx?.merchant_category}</span></div>
        </div>
      </div>

      {/* ML probability */}
      <div className="panel-section">
        <h3>ML Model</h3>
        <div className="behavioral-box">
          Fraud probability: <strong>{(data.ml_probability * 100).toFixed(1)}%</strong>
        </div>
      </div>

      {/* Findings */}
      {findings.length > 0 && (
        <div className="panel-section">
          <h3>Risk Findings ({findings.length})</h3>
          {findings.map((f, i) => (
            <div className="finding-card" key={i}>
              <div className="finding-header">
                <span className="finding-title">{f.title}</span>
                <span className={`risk-badge ${f.severity}`}>{f.severity}</span>
              </div>
              <div className="finding-evidence">{f.evidence}</div>
              <div className="finding-source">Source: {f.source_feature}</div>
            </div>
          ))}
        </div>
      )}

      {/* Behavioral assessment */}
      {inv.behavioral_assessment && (
        <div className="panel-section">
          <h3>Behavioral Assessment</h3>
          <div className="behavioral-box">{inv.behavioral_assessment}</div>
        </div>
      )}

      {/* Summary */}
      {inv.summary && (
        <div className="panel-section">
          <h3>Summary</h3>
          <div className="behavioral-box">{inv.summary}</div>
        </div>
      )}
    </div>
  )
}

function riskColor(level) {
  if (level === 'HIGH') return 'var(--risk-high)'
  if (level === 'MEDIUM') return 'var(--risk-medium)'
  return 'var(--risk-low)'
}

/* ================================================================
   App
   ================================================================ */
function App() {
  const [health, setHealth] = useState(null)
  const [stats, setStats] = useState({})
  const [transactions, setTransactions] = useState([])
  const [total, setTotal] = useState(0)
  const [analyses, setAnalyses] = useState({})   // txId -> analysis result
  const [selected, setSelected] = useState(null)  // currently selected analysis
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [analyzing, setAnalyzing] = useState(false)

  // Initial load
  useEffect(() => {
    async function init() {
      try {
        const [h, s, t] = await Promise.all([getHealth(), getStats(), getTransactions(50, 0)])
        setHealth(h)
        setStats(s)
        setTransactions(t.transactions || [])
        setTotal(t.total || 0)
      } catch (e) {
        setError(e.message)
      } finally {
        setLoading(false)
      }
    }
    init()
  }, [])

  // Auto-analyze first 50 transactions for risk badges
  useEffect(() => {
    if (transactions.length === 0) return
    let cancelled = false
    async function bulkAnalyze() {
      const results = {}
      for (const tx of transactions) {
        if (cancelled) break
        try {
          const payload = {
            transaction_id: tx.transaction_id,
            customer_id: tx.customer_id,
            timestamp: tx.timestamp,
            amount: tx.amount,
            currency: tx.currency || 'USD',
            payment_method: tx.payment_method || 'Credit Card',
            merchant_category: tx.merchant_category || 'Retail',
            country: tx.country || 'US',
            device_changed_recently: tx.device_changed_recently || 0,
            location_changed_recently: tx.location_changed_recently || 0,
          }
          const res = await analyzeRisk(payload)
          results[tx.transaction_id] = res
        } catch (_) { /* skip */ }
      }
      if (!cancelled) setAnalyses(prev => ({ ...prev, ...results }))
    }
    bulkAnalyze()
    return () => { cancelled = true }
  }, [transactions])

  // Select a transaction → run full analysis
  const handleSelect = useCallback(async (tx) => {
    if (analyses[tx.transaction_id]) {
      setSelected({ ...analyses[tx.transaction_id], _tx: tx })
      return
    }
    setAnalyzing(true)
    try {
      const payload = {
        transaction_id: tx.transaction_id,
        customer_id: tx.customer_id,
        timestamp: tx.timestamp,
        amount: tx.amount,
        currency: tx.currency || 'USD',
        payment_method: tx.payment_method || 'Credit Card',
        merchant_category: tx.merchant_category || 'Retail',
        country: tx.country || 'US',
        device_changed_recently: tx.device_changed_recently || 0,
        location_changed_recently: tx.location_changed_recently || 0,
      }
      const res = await analyzeRisk(payload)
      setAnalyses(prev => ({ ...prev, [tx.transaction_id]: res }))
      setSelected({ ...res, _tx: tx })
    } catch (e) {
      setError(e.message)
    } finally {
      setAnalyzing(false)
    }
  }, [analyses])

  if (loading) return <div className="loading"><div className="spinner" />Loading PayGuard…</div>
  if (error) return <div className="error">⚠ {error}</div>

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-brand">
          <div className="header-logo">PG</div>
          <div>
            <div className="header-title">PAYGUARD</div>
            <div className="header-subtitle">AI Payment Risk & Fraud Agent</div>
          </div>
        </div>
        <div className="header-status">
          <div className={`status-dot ${health?.model_loaded ? '' : 'offline'}`} />
          {health?.model_loaded ? 'Model Active' : 'Model Offline'}
        </div>
      </header>

      {/* Metrics bar */}
      <MetricsBar metrics={stats.model_metrics} />

      {/* Overview cards */}
      <OverviewCards stats={stats} />

      {/* Main grid: table + investigation panel */}
      <div className={`main ${selected ? '' : 'no-panel'}`}>
        <div className="left-pane">
          <TransactionTable
            transactions={transactions}
            total={total}
            selected={selected}
            onSelect={handleSelect}
            analyses={analyses}
          />
        </div>
        {selected && <InvestigationPanel data={selected} />}
      </div>
    </div>
  )
}

export default App
