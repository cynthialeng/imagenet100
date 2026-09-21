import { useEffect, useState } from 'react'
import './App.css'

const API_URL = 'http://localhost:8000'

function App() {
  const [images, setImages] = useState([])
  const [selectedIndex, setSelectedIndex] = useState(0)
  const [annotations, setAnnotations] = useState([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState('')
  const [annotationError, setAnnotationError] = useState('')
  const [isLoadingAnnotations, setIsLoadingAnnotations] = useState(false)
  const [isSaving, setIsSaving] = useState(false)
  const [isAnnotating, setIsAnnotating] = useState(false)
  const [isDrawing, setIsDrawing] = useState(false)
  const [draftBox, setDraftBox] = useState(null)
  const [draftLabel, setDraftLabel] = useState('')
  const [selectedAnnotationId, setSelectedAnnotationId] = useState(null)
  const [zoom, setZoom] = useState(1)
  const [position, setPosition] = useState({ x: 0, y: 0 })
  const [isDragging, setIsDragging] = useState(false)
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 })

  useEffect(() => {
    fetch(`${API_URL}/images`)
      .then((response) => {
        if (!response.ok) throw new Error('Could not load image metadata.')
        return response.json()
      })
      .then(setImages)
      .catch((loadError) => setError(`${loadError.message} Is the backend running on port 8000?`))
      .finally(() => setIsLoading(false))
  }, [])

  const selectedImage = images[selectedIndex]

  useEffect(() => {
    if (!selectedImage) return

    fetch(`${API_URL}/images/${selectedImage.id}/annotations`)
      .then((response) => {
        if (!response.ok) throw new Error('Could not load annotations.')
        return response.json()
      })
      .then(setAnnotations)
      .catch((loadError) => setAnnotationError(loadError.message))
      .finally(() => setIsLoadingAnnotations(false))
  }, [selectedImage])

  useEffect(() => {
    if (!isAnnotating) return undefined

    const finishDrawingOutsideImage = () => setIsDrawing(false)
    window.addEventListener('pointerup', finishDrawingOutsideImage)
    window.addEventListener('pointercancel', finishDrawingOutsideImage)

    return () => {
      window.removeEventListener('pointerup', finishDrawingOutsideImage)
      window.removeEventListener('pointercancel', finishDrawingOutsideImage)
    }
  }, [isAnnotating])

  function selectImage(index) {
    setSelectedIndex(index)
    setIsLoadingAnnotations(true)
    setAnnotationError('')
    setSelectedAnnotationId(null)
    setZoom(1)
    setPosition({ x: 0, y: 0 })
    setIsAnnotating(false)
    setIsDrawing(false)
    setDraftBox(null)
    setDraftLabel('')
  }

  function changeZoom(amount) {
    setZoom((currentZoom) =>
      Math.min(3, Math.max(0.5, Number((currentZoom + amount).toFixed(2)))),
    )
  }

  function handlePointerDown(event) {
    event.currentTarget.setPointerCapture(event.pointerId)
    setIsDragging(true)
    setDragStart({
      x: event.clientX - position.x,
      y: event.clientY - position.y,
    })
  }

  function handlePointerMove(event) {
    if (!isDragging) return

    setPosition({
      x: event.clientX - dragStart.x,
      y: event.clientY - dragStart.y,
    })
  }

  function resetView() {
    setZoom(1)
    setPosition({ x: 0, y: 0 })
  }

  function toggleAnnotationMode() {
    setIsAnnotating((current) => !current)
    setIsDrawing(false)
    setDraftBox(null)
    setDraftLabel('')
    setSelectedAnnotationId(null)
    resetView()
  }

  function getNormalizedPoint(event, element) {
    const bounds = element.getBoundingClientRect()
    return {
      x: Math.min(1, Math.max(0, (event.clientX - bounds.left) / bounds.width)),
      y: Math.min(1, Math.max(0, (event.clientY - bounds.top) / bounds.height)),
    }
  }

  function startDrawing(event) {
    if (!isAnnotating || zoom !== 1) return

    event.currentTarget.setPointerCapture(event.pointerId)
    const point = getNormalizedPoint(event, event.currentTarget)
    setIsDrawing(true)
    setDraftBox({ x: point.x, y: point.y, width: 0, height: 0 })
    setDraftLabel('')
    setSelectedAnnotationId(null)
  }

  function drawBox(event) {
    if (!isAnnotating || !isDrawing || !draftBox) return

    const point = getNormalizedPoint(event, event.currentTarget)
    setDraftBox({
      x: Math.min(draftBox.x, point.x),
      y: Math.min(draftBox.y, point.y),
      width: Math.abs(point.x - draftBox.x),
      height: Math.abs(point.y - draftBox.y),
    })
  }

  function finishDrawing(event) {
    setIsDrawing(false)
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId)
    }
    if (draftBox && (draftBox.width < 0.01 || draftBox.height < 0.01)) {
      setDraftBox(null)
    }
  }

  async function saveAnnotation() {
    if (!selectedImage || !draftBox || !draftLabel.trim()) return

    setIsSaving(true)
    setAnnotationError('')
    try {
      const response = await fetch(`${API_URL}/images/${selectedImage.id}/annotations`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...draftBox, label: draftLabel.trim() }),
      })
      if (!response.ok) {
        const result = await response.json()
        throw new Error(result.detail || 'Could not save annotation.')
      }
      const savedAnnotation = await response.json()
      setAnnotations((current) => [...current, savedAnnotation])
      setDraftBox(null)
      setDraftLabel('')
    } catch (saveError) {
      setAnnotationError(saveError.message)
    } finally {
      setIsSaving(false)
    }
  }

  async function removeAnnotation(annotationId) {
    setAnnotationError('')
    try {
      const response = await fetch(`${API_URL}/annotations/${annotationId}`, {
        method: 'DELETE',
      })
      if (!response.ok) throw new Error('Could not delete annotation.')
      setAnnotations((current) => current.filter((annotation) => annotation.id !== annotationId))
      setSelectedAnnotationId(null)
    } catch (deleteError) {
      setAnnotationError(deleteError.message)
    }
  }

  function selectAnnotation(annotation) {
    setSelectedAnnotationId(annotation.id)
    setDraftBox(null)
    setDraftLabel('')
  }

  function renderBox(box, isDraft = false) {
    return (
      <div
        className={isDraft ? 'annotation-box draft' : `annotation-box${selectedAnnotationId === box.id ? ' selected' : ''}`}
        key={isDraft ? 'draft' : box.id}
        onPointerDown={(event) => {
          event.stopPropagation()
          if (!isDraft) selectAnnotation(box)
        }}
        style={{
          left: `${box.x * 100}%`,
          top: `${box.y * 100}%`,
          width: `${box.width * 100}%`,
          height: `${box.height * 100}%`,
        }}
      >
        {!isDraft && <span>{box.label}</span>}
      </div>
    )
  }

  return (
    <main className="page">
      <header className="page-header">
        <p className="eyebrow">Research data viewer</p>
        <h1>ImageNet100</h1>
        <p className="description">
          Explore how deep learning models recognize and classify images across
          100 object categories. Using a subset of ImageNet, this project
          investigates image classification, model performance, and the
          challenges of visual recognition.
        </p>
      </header>

      {error && <p className="error-message">{error}</p>}
      {isLoading && <p className="loading">Loading images...</p>}

      {!isLoading && !error && <section className="viewer" aria-label="Image viewer">
        <aside className="image-list">
          <h2>Images ({images.length})</h2>
          <div className="image-list-items">
            {images.map((image, index) => (
              <button
                className={index === selectedIndex ? 'image-item selected' : 'image-item'}
                key={image.id}
                onClick={() => selectImage(index)}
                type="button"
              >
                <img src={`/samples/${image.filename}`} alt="" />
                <span>{image.filename}</span>
              </button>
            ))}
          </div>
        </aside>

        <section className="image-panel">
          {selectedImage ? (
            <>
              <div
                className={`${isDragging ? 'image-stage dragging' : 'image-stage'}${isAnnotating ? ' annotation-mode' : ''}`}
                onPointerDown={isAnnotating ? undefined : handlePointerDown}
                onPointerMove={isAnnotating ? undefined : handlePointerMove}
                onPointerUp={isAnnotating ? undefined : () => setIsDragging(false)}
                onPointerCancel={isAnnotating ? undefined : () => setIsDragging(false)}
                onWheel={(event) => {
                  event.preventDefault()
                  if (!isAnnotating) changeZoom(event.deltaY < 0 ? 0.1 : -0.1)
                }}
              >
                <div
                  className="image-canvas"
                  onPointerDown={isAnnotating ? startDrawing : undefined}
                  onPointerMove={isAnnotating ? drawBox : undefined}
                  onPointerUp={isAnnotating ? finishDrawing : undefined}
                  onPointerCancel={isAnnotating ? (event) => {
                    setIsDrawing(false)
                    setDraftBox(null)
                    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
                      event.currentTarget.releasePointerCapture(event.pointerId)
                    }
                  } : undefined}
                  style={{ transform: `translate(${position.x}px, ${position.y}px) scale(${zoom})` }}
                >
                  <img
                    className="selected-image"
                    src={`/samples/${selectedImage.filename}`}
                    alt={selectedImage.ground_truth_label}
                    draggable="false"
                  />
                  {annotations.map((annotation) => renderBox(annotation, false))}
                  {draftBox && renderBox(draftBox, true)}
                </div>
              </div>

              <div className="image-details">
                <div>
                  <h2>{selectedImage.filename}</h2>
                  <p>Label: {selectedImage.ground_truth_label}</p>
                </div>
                <div className="controls" aria-label="Image controls">
                  <button type="button" onClick={() => changeZoom(-0.1)} disabled={isAnnotating}>
                    -
                  </button>
                  <span>{Math.round(zoom * 100)}%</span>
                  <button type="button" onClick={() => changeZoom(0.1)} disabled={isAnnotating}>
                    +
                  </button>
                  <button type="button" onClick={resetView} disabled={isAnnotating}>
                    Reset
                  </button>
                  <button className="annotation-toggle" onClick={toggleAnnotationMode}>
                    {isAnnotating ? 'Done!' : 'Annotate'}
                  </button>
                </div>
              </div>
              {isAnnotating && (
                <div className="annotation-editor">
                  <p>Draw a box on the image, then add a label.</p>
                  {draftBox && (
                    <div className="annotation-form">
                      <input
                        aria-label="Annotation label"
                        value={draftLabel}
                        onChange={(event) => setDraftLabel(event.target.value)}
                        placeholder="Object label"
                      />
                      <button type="button" onClick={saveAnnotation} disabled={isSaving || !draftLabel.trim()}>
                        {isSaving ? 'Saving...' : 'Save annotation'}
                      </button>
                      <button type="button" onClick={() => { setDraftBox(null); setDraftLabel('') }}>
                        Cancel
                      </button>
                    </div>
                  )}
                  {zoom !== 1 && <p className="hint">Drawing is enabled at 100% zoom.</p>}
                </div>
              )}
              <div className="annotation-summary">
                <strong>{isLoadingAnnotations ? 'Loading annotations...' : `${annotations.length} saved annotation${annotations.length === 1 ? '' : 's'}`}</strong>
                {selectedAnnotationId && (
                  <button type="button" onClick={() => removeAnnotation(selectedAnnotationId)}>
                    Delete selected annotation
                  </button>
                )}
              </div>
              {annotationError && <p className="error-message">{annotationError}</p>}
            </>
          ) : (
            <p className="loading">Loading images...</p>
          )}
        </section>
      </section>}
    </main>
  )
}

export default App
