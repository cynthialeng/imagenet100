import { useEffect, useState } from 'react'
import PredictionPanel from './PredictionPanel'

const percent = (value) => `${(value * 100).toFixed(1)}%`

async function readResponse(response) {
  const data = await response.json()
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Check the training settings and try again.')
  return data
}

function MetricChart({ history, epochs, metric, title }) {
  const accuracy = metric === 'accuracy'
  const maximum = accuracy ? 1 : Math.max(0.1, ...history.flatMap((row) => [row.train_loss, row.val_loss])) * 1.1
  const x = (epoch) => 55 + ((epoch - 1) / Math.max(1, epochs - 1)) * 505
  const y = (value) => 175 - (value / maximum) * 150
  const tick = (value) => accuracy ? percent(value) : value.toFixed(2)

  return <figure className="live-chart">
    <figcaption>{title}</figcaption>
    <svg viewBox="0 0 590 215" role="img" aria-label={`${title}: training and validation over ${history.length} completed epochs`}>
      {[0, maximum / 2, maximum].map((value) => <g key={value}>
        <line x1="55" x2="560" y1={y(value)} y2={y(value)} className="chart-grid" />
        <text x="47" y={y(value) + 4} textAnchor="end">{tick(value)}</text>
      </g>)}
      {[1, epochs].filter((value, index, array) => array.indexOf(value) === index).map((value) =>
        <text key={value} x={x(value)} y="194" textAnchor="middle">{value}</text>)}
      <text x="307" y="210" textAnchor="middle">Epoch</text>
      {['train', 'val'].map((split) => <g key={split} className={`chart-${split}`}>
        <polyline fill="none" strokeWidth="2" points={history.map((row) => `${x(row.epoch)},${y(row[`${split}_${metric}`])}`).join(' ')} />
        {history.length > 0 && <circle cx={x(history.at(-1).epoch)} cy={y(history.at(-1)[`${split}_${metric}`])} r="3" />}
      </g>)}
    </svg>
  </figure>
}

export default function TrainingPanel() {
  const [learningRate, setLearningRate] = useState('0.01')
  const [epochs, setEpochs] = useState('50')
  const [run, setRun] = useState(null)
  const [loading, setLoading] = useState(true)
  const [pending, setPending] = useState('')
  const [error, setError] = useState('')
  const [pollError, setPollError] = useState('')
  const running = run?.status === 'running'
  const completed = run?.status === 'completed'
  const latest = run?.history.at(-1)
  const metrics = completed ? run.best : latest

  useEffect(() => {
    const controller = new AbortController()
    fetch('/api/training/runs/latest', { signal: controller.signal })
      .then(readResponse)
      .then((data) => {
        setRun(data)
        if (data) {
          setLearningRate(String(data.config.learning_rate))
          setEpochs(String(data.config.epochs))
        }
      })
      .catch((err) => { if (err.name !== 'AbortError') setError('Could not load training. Check the backend connection.') })
      .finally(() => { if (!controller.signal.aborted) setLoading(false) })
    return () => controller.abort()
  }, [])

  useEffect(() => {
    if (!running) return undefined
    const controller = new AbortController()
    let timer
    async function poll() {
      try {
        const data = await fetch(`/api/training/runs/${run.id}`, { signal: controller.signal }).then(readResponse)
        setRun(data)
        setPollError('')
        if (data.status !== 'running') return
      } catch {
        if (controller.signal.aborted) return
        setPollError('Connection lost. Retrying…')
      }
      timer = window.setTimeout(poll, 750)
    }
    timer = window.setTimeout(poll, 300)
    return () => { controller.abort(); window.clearTimeout(timer) }
  }, [run?.id, running])

  async function action(name, event) {
    event?.preventDefault()
    setPending(name)
    setError('')
    try {
      const starting = name === 'start'
      const response = await fetch(starting ? '/api/training/runs' : `/api/training/runs/${run.id}/${name}`, {
        method: 'POST',
        ...(starting ? {
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ learning_rate: Number(learningRate), epochs: Number(epochs) }),
        } : {}),
      })
      const data = await readResponse(response)
      setRun(data)
      setPollError('')
    } catch (err) {
      setError(err.message)
    } finally {
      setPending('')
    }
  }

  return <section className="training-panel" aria-labelledby="training-heading">
    <h2 id="training-heading">Train a model</h2>
    <p>Same two-hidden-layer MLP, fresh weights each run.</p>
    <form onSubmit={(event) => action('start', event)} className="training-form">
      <label>Learning rate
        <input className="learning-rate-input" type="number" min="0.0001" max="0.1" step="any" required value={learningRate}
          disabled={running || Boolean(pending)} onChange={(event) => setLearningRate(event.target.value)} />
      </label>
      <label>Epochs
        <input type="number" min="1" max="200" step="1" required value={epochs}
          disabled={running || Boolean(pending)} onChange={(event) => setEpochs(event.target.value)} />
      </label>
      <button className="predict-button" disabled={loading || running || Boolean(pending)} type="submit">
        {pending === 'start' ? 'Starting…' : 'Start training'}
      </button>
      {running && <button type="button" className="secondary-button" disabled={Boolean(pending)} onClick={() => action('stop')}>
        {pending === 'stop' ? 'Stopping…' : 'Stop'}
      </button>}
    </form>
    {error && <p className="error-message" role="alert">{error}</p>}
    {pollError && <p className="error-message" role="status">{pollError}</p>}
    {run && <>
      <div className="training-status" role="status">
        <strong>{({ running: 'Training', completed: 'Complete', stopped: 'Stopped', failed: 'Failed', interrupted: 'Interrupted' })[run.status]}</strong>
        <span>Epoch {latest?.epoch || 0} / {run.config.epochs}</span>
      </div>
      <progress className="training-progress" aria-label="Training progress" value={latest?.epoch || 0} max={run.config.epochs} />
      {run.error && <p className="error-message">{run.error}</p>}
      <div className="chart-legend"><span className="legend-train">Training</span><span className="legend-val">Validation</span></div>
      <MetricChart history={run.history} epochs={run.config.epochs} metric="loss" title="Loss vs Epoch" />
      <MetricChart history={run.history} epochs={run.config.epochs} metric="accuracy" title="Accuracy vs Epoch" />
      {metrics && <div className="results-table-wrap">
        <table className="results-table">
          <caption>{completed ? 'Best checkpoint' : 'Latest Epoch'} · Epoch {metrics.epoch}</caption>
          <thead><tr><th scope="col">Split</th><th scope="col">Loss</th><th scope="col">Accuracy</th></tr></thead>
          <tbody>
            <tr><th scope="row">Training</th><td>{metrics.train_loss.toFixed(4)}</td><td>{percent(metrics.train_accuracy)}</td></tr>
            <tr><th scope="row">Validation</th><td>{metrics.val_loss.toFixed(4)}</td><td>{percent(metrics.val_accuracy)}</td></tr>
            <tr><th scope="row">Test</th><td>{run.test ? run.test.loss.toFixed(4) : '—'}</td><td>{run.test ? percent(run.test.accuracy) : '—'}</td></tr>
          </tbody>
        </table>
      </div>}
      {completed && <>
        {run.test && <p className="hint">Test error: {percent(run.test.error_rate)} on {run.test.images} images.</p>}
        <PredictionPanel key={run.id} endpoint={`/training/runs/${run.id}/predict`} title="Try this run" />
      </>}
    </>}
  </section>
}
