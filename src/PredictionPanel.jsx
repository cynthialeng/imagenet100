import { useEffect, useRef, useState } from 'react'

const API_URL = '/api'

export default function PredictionPanel({ endpoint = '/predict', title = 'Try the classifier' }) {
  const [samples, setSamples] = useState([])
  const [sampleId, setSampleId] = useState('')
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState('')
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [sampleError, setSampleError] = useState('')
  const [busy, setBusy] = useState(false)
  const requestId = useRef(0)
  const fileInput = useRef(null)
  const sample = samples.find((item) => String(item.id) === sampleId)

  useEffect(() => {
    const controller = new AbortController()
    fetch(`${API_URL}/prediction-samples`, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error('Sample images are unavailable. You can still upload an image.')
        return response.json()
      })
      .then(setSamples)
      .catch((err) => { if (err.name !== 'AbortError') setSampleError(err.message) })
    return () => { controller.abort(); requestId.current += 1 }
  }, [])

  useEffect(() => {
    if (!preview) return undefined
    return () => URL.revokeObjectURL(preview)
  }, [preview])

  function clearPrediction() {
    requestId.current += 1
    setResult(null)
    setError('')
    setBusy(false)
  }

  function chooseFile(event) {
    clearPrediction()
    setSampleId('')
    setPreview('')
    const next = event.target.files[0] || null
    if (next && next.size > 10 * 1024 * 1024) {
      setFile(null)
      setError('Choose an image that is 10 MB or smaller.')
      event.target.value = ''
      return
    }
    setFile(next)
    if (next) setPreview(URL.createObjectURL(next))
  }

  async function predict() {
    const current = ++requestId.current
    setBusy(true)
    setError('')
    setResult(null)
    try {
      let image = file
      if (!image && sample) {
        const response = await fetch(`${API_URL}${sample.url}`)
        if (!response.ok) throw new Error('Could not load the selected image.')
        image = await response.blob()
      }
      const response = await fetch(`${API_URL}${endpoint}`, {
        method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: image,
      })
      const data = await response.json()
      if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Prediction failed.')
      if (current === requestId.current) setResult(data)
    } catch (err) {
      if (current === requestId.current) setError(`${err.message} Check that the backend is running on port 8000.`)
    } finally {
      if (current === requestId.current) setBusy(false)
    }
  }

  return (
    <section className="prediction-panel" aria-label={title}>
      <h2>{title}</h2>
      <p className="prediction-intro">Select a test sample or upload an image of a tabby cat, cauliflower, or computer keyboard.</p>
      <div className="prediction-inputs">
        <label>Test sample
          <select value={sampleId} onChange={(event) => {
            clearPrediction()
            setSampleId(event.target.value)
            setFile(null)
            setPreview('')
            if (fileInput.current) fileInput.current.value = ''
          }}>
            <option value="">Choose an image…</option>
            {samples.map((item) => <option key={item.id} value={String(item.id)}>{item.ground_truth_label} — {item.filename}</option>)}
          </select>
        </label>
        <label>Or upload an image
          <input ref={fileInput} type="file" accept="image/*" onChange={chooseFile} />
        </label>
      </div>
      {sampleError && <p className="hint">{sampleError}</p>}
      {(sample || file) && <div className="prediction-preview">
        <img src={file ? preview : `${API_URL}${sample.url}`} alt={file ? 'Uploaded image for classification' : sample.ground_truth_label} />
        <p>{file ? file.name : `Actual label: ${sample.ground_truth_label}`}</p>
      </div>}
      <button className="predict-button" type="button" disabled={busy || (!file && !sample)} onClick={predict}>
        {busy ? 'Predicting…' : 'Predict image'}
      </button>
      <div aria-live="polite" aria-busy={busy}>
        {error && <p className="error-message" role="alert">{error}</p>}
        {result && <div className="prediction-result">
          <h3>Prediction: {result.predicted_class}</h3>
          {sample && <p>{result.predicted_class === sample.ground_truth_label ? 'Matches the actual label.' : `Actual label: ${sample.ground_truth_label}`}</p>}
          {result.probabilities.map((item) => <div className="probability-row" key={item.class_name}>
            <div><span>{item.class_name}</span><strong>{(item.probability * 100).toFixed(1)}%</strong></div>
            <progress aria-label={`${item.class_name} probability`} value={item.probability} max="1" />
          </div>)}
        </div>}
      </div>
    </section>
  )
}
