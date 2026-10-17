import { useEffect, useState } from 'react'

const percent = (value) => `${(value * 100).toFixed(1)}%`

export default function ResultsPanel() {
  const [results, setResults] = useState(null)
  const [error, setError] = useState('')
  const [chartError, setChartError] = useState(false)

  useEffect(() => {
    const controller = new AbortController()
    fetch('/api/model-results', { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error('Could not load saved results. Check that the backend is running and test evaluation is saved.')
        return response.json()
      })
      .then(setResults)
      .catch((err) => { if (err.name !== 'AbortError') setError(err.message) })
    return () => controller.abort()
  }, [])

  return (
    <section className="results-panel" aria-labelledby="results-heading">
      <h2 id="results-heading">Model results</h2>
      {error && <p className="error-message" role="alert">{error}</p>}
      {!results && !error && <p role="status">Loading saved results…</p>}
      {results && <>
        <p className="results-intro">ImageNet100: {results.split_counts.train} train / {results.split_counts.val} validation / {results.split_counts.test} test images.</p>
        <div className="model-facts">
          <div><span>MLP architecture</span><strong>{results.architecture.map((size) => size.toLocaleString()).join(' → ')}</strong></div>
          <div><span>Lowest validation loss</span><strong>Epoch {results.selected_epoch} of {results.epochs}</strong></div>
        </div>
        <div className="results-table-wrap">
          <table className="results-table">
            <caption>Epoch {results.selected_epoch}</caption>
            <thead><tr><th scope="col">Split</th><th scope="col">Loss</th><th scope="col">Accuracy</th></tr></thead>
            <tbody>
              <tr><th scope="row">Training</th><td>{results.selected_metrics.train_loss.toFixed(4)}</td><td>{percent(results.selected_metrics.train_accuracy)}</td></tr>
              <tr><th scope="row">Validation</th><td>{results.selected_metrics.val_loss.toFixed(4)}</td><td>{percent(results.selected_metrics.val_accuracy)}</td></tr>
            </tbody>
          </table>
        </div>
        <figure className="training-figure">
          {!chartError ? <a href={`/api${results.curves_url}`} target="_blank" rel="noreferrer">
            <img src={`/api${results.curves_url}`} onError={() => setChartError(true)}
              alt="Training and validation loss and accuracy plotted against epoch for the two-hidden-layer MLP" />
          </a> : <p className="error-message">The saved training plot is unavailable.</p>}
        </figure>
        <div className="results-table-wrap">
          <table className="results-table">
            <caption>Test Results</caption>
            <thead><tr><th scope="col">Model</th><th scope="col">Loss</th><th scope="col">Accuracy</th><th scope="col">Error rate</th></tr></thead>
            <tbody>{[['linear_baseline', 'Linear benchmark'], ['two_hidden', 'Two-hidden-layer MLP']].map(([key, name]) => {
              const model = results.models[key]
              return <tr key={key}><th scope="row">{name}</th><td>{model.loss.toFixed(4)}</td><td>{percent(model.accuracy)}</td><td>{percent(model.error_rate)}</td></tr>
            })}</tbody>
          </table>
        </div>
        <p className="hint">Note: {results.models.two_hidden.correct - results.models.linear_baseline.correct} more correct predictions than the linear benchmark.</p>
      </>}
    </section>
  )
}
